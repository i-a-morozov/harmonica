# Input arguments flag
import sys
sys.path.append('..')
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_phase', description='Estimate phase for selected plane and BPMs.')
parser.add_argument('-p', '--plane', choices=('x', 'y'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use', default=256)
parser.add_argument('--load', type=int, help='total number of turns to load', default=1024)
select = parser.add_mutually_exclusive_group()
select.add_argument('--skip', metavar='BPM', nargs='+', help='space separated list of valid BPM names to skip')
select.add_argument('--only', metavar='BPM', nargs='+', help='space separated list of valid BPM names to use')
parser.add_argument('-o', '--offset', type=int, help='rise offset for all BPMs', default=0)
parser.add_argument('-r', '--rise', action='store_true', help='flag to use rise data (drop first turns)')
parser.add_argument('-s', '--save', action='store_true', help='flag to save data as numpy array')
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--mean', action='store_true', help='flag to remove mean')
transform.add_argument('--median', action='store_true', help='flag to remove median')
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
parser.add_argument('-f', '--filter', choices=('none', 'svd', 'hankel'), help='filter type', default='none')
parser.add_argument('--rank', type=int, help='rank to use for svd & hankel filter', default=8)
parser.add_argument('--type', choices=('full', 'randomized'), help='computation type for hankel filter', default='randomized')
parser.add_argument('--buffer', type=int, help='buffer size to use for randomized hankel filter', default=16)
parser.add_argument('--count', type=int, help='number of iterations to use for randomized hankel filter', default=16)
parser.add_argument('-w', '--window', type=float, help='window order', default=0.0)
parser.add_argument('--limit', type=int, help='number of columns to use for noise estimation', default=16)
case = parser.add_mutually_exclusive_group()
case.add_argument('--error', action='store_true', help='flag to propagate errors')
case.add_argument('--shift', action='store_true', help='flag to use shifted samples')
parser.add_argument('--size', type=int, help='maximum number of samples to use', default=512)
parser.add_argument('--delta', type=int, help='sample shift step', default=1)
parser.add_argument('--fit', choices=('none', 'noise', 'average', 'fit'), help='fit type', default='none')
parser.add_argument('--dht', action='store_true', help='flag to use DHT estimator')
parser.add_argument('--drop', type=int, help='number of endpoints to drop in DHT', default=32)
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--advance', action='store_true', help='flag to plot phase advance')
parser.add_argument('-H', '--harmonica', action='store_true', help='flag to use harmonica PV names for input')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float64')
parser.add_argument('-u', '--update', action='store_true', help='flag to update harmonica PV')
args = parser.parse_args(args=None if flag else ['--help'])

# Import
import epics
import numpy
import pandas
import torch
from datetime import datetime
from harmonica.util import LIMIT, LENGTH, pv_make
from harmonica.window import Window
from harmonica.data import Data
from harmonica.filter import Filter
from harmonica.frequency import Frequency
from harmonica.decomposition import Decomposition

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit(f'error: CUDA is not avalible')

# Set data plane
plane = args.plane.upper()

# Load frequency and frequency error
value = epics.caget(f'H:FREQUENCY:VALUE:{plane}')
error = epics.caget(f'H:FREQUENCY:ERROR:{plane}')

# Load monitor data
name = epics.caget('H:MONITOR:LIST')[:epics.caget('H:MONITOR:COUNT')]
flag = epics.caget_many([f'H:{name}:FLAG' for name in name])
rise = epics.caget_many([f'H:{name}:RISE' for name in name])

# Set BPM data
bpm = {name: rise for name, flag, rise in zip(name, flag, rise) if flag == 1}

# Check & remove skipped
if args.skip:
  for name in (name.upper() for name in args.skip):
    if not name in bpm:
      exit(f'error: {name} is not a valid BPM to skip')
    bpm.pop(name)

# Check & keep selected
if args.only:
    for name in (name.upper() for name in args.only):
      if not name in bpm:
        exit(f'error: {name} is not a valid BPM to read')
    for name in bpm.copy():
      if not name in (name.upper() for name in args.only):
        bpm.pop(name)

# Check BPM list
if not bpm:
  exit(f'error: BPM list is empty')

# Generate model advance
if args.advance:
  total = epics.caget(f'H:END:MODEL:F{plane}')
  model = torch.tensor(epics.caget_many([f'H:{name}:MODEL:F{plane}' for name in bpm]), dtype=dtype, device=device)
  model, _ = Decomposition.phase_adjacent(total/(2.0*numpy.pi), model)
  model = model.cpu().numpy()
  name = [*bpm.keys()]
  pair = [f'{name[i]}-{name[i+1]}' for i in range(len(bpm)-1)]
  pair.append(f'{name[-1]}-{name[0]}')

# Generate PV names
pv_list = [pv_make(name, args.plane, args.harmonica) for name in bpm]
pv_rise = [*bpm.values()]

# Check length
length = args.load
if length < 0 or length > LIMIT:
  exit(f'error: {length=}, expected a positive value less than {LIMIT=}')

# Check offset
offset = args.offset
if offset < 0:
  exit(f'error: {offset=}, expected a positive value')
if length + offset > LIMIT:
  exit(f'error: sum of {length=} and {offset=}, expected to be less than {LIMIT=}')

# Check rise
if args.rise:
  rise = min(pv_rise)
  if rise < 0:
    exit(f'error: rise values are expected to be positive')
  rise = max(pv_rise)
  if length + offset + rise > LIMIT:
    exit(f'error: sum of {length=}, {offset=} and max {rise=}, expected to be less than {LIMIT=}')
else:
  rise = 0

# Check sample length
if args.length > length:
  exit(f'error: requested sample length {args.length} should be less than {length}')

# Check window order
if args.window < 0.0:
  exit(f'error: window order {args.window} should be greater or equal to zero')

# Load TbT data
size = len(bpm)
count = length + offset + rise
win = Window(length, 'cosine_window', args.window, dtype=dtype, device=device)
tbt = Data.from_epics(win, pv_list, pv_rise if args.rise else None, shift=offset, count=count)

# Remove mean
if args.mean:
  tbt.window_remove_mean()

# Remove median
if args.median:
  tbt.work.sub_(tbt.median())

# Normalize
if args.normalize:
  tbt.normalize()

# Filter (svd)
if args.filter == 'svd':
  flt = Filter(tbt)
  flt.filter_svd(rank=args.rank)
  del flt

# Filter (hankel)
if args.filter == 'hankel':
  flt = Filter(tbt)
  flt.filter_svd(rank=args.rank)
  flt.filter_hankel(rank=args.rank, random=args.type == 'randomized', buffer=args.buffer, count=args.count)
  del flt

# Estimate phase
if not args.dht:
  dec = Decomposition(tbt)
  value, error, table = dec.harmonic_phase(
    value,
    length=args.length,
    name='cosine_window',
    order=args.window,
    error=args.error,
    limit=args.limit,
    sigma_frequency=error,
    shift=args.shift,
    count=args.size,
    step=args.delta,
    fit=args.fit)
else:
  dht = Frequency.dht(tbt.work[:, :args.length])
  table = dht.angle()
  table -= 2.0*numpy.pi*value*torch.linspace(0, args.length - 1, args.length, dtype=dtype, device=device)
  table = Frequency.mod(table, 2.0*numpy.pi, -numpy.pi)
  value = table[:, +args.drop:-args.drop].mean(1)
  error = table[:, +args.drop:-args.drop].std(1)

# Plot
if args.plot:
  from plotly.express import scatter
  if table != None:
    step = len(table[0])
    df = pandas.DataFrame()
    for i, name in enumerate(bpm):
      df = pandas.concat([df, pandas.DataFrame({'STEP':range(1, step + 1), 'BPM':name, 'PHASE':table[i].cpu().numpy()})])
    plot = scatter(df, x='STEP', y='PHASE', color='BPM', title=f'{TIME}: PHASE ({plane})', opacity=0.75, marginal_y='box')
    config = {
      'toImageButtonOptions': {'height':None, 'width':None},
      'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
      'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'],
      'scrollZoom': True
    }
    plot.show(config=config)
  df = pandas.DataFrame()
  df['BPM'] = [*bpm.keys()]
  df['PHASE'] = value.cpu().numpy()
  if error != None:
    df['ERROR'] = error.cpu().numpy()
  title = f'{TIME}: PHASE ({plane})'
  plot = scatter(df, x='BPM', y=f'PHASE', title=title, opacity=0.75, error_y='ERROR' if error != None else None, hover_data=['BPM', 'PHASE', 'ERROR' if error != None else None])
  config = {
    'toImageButtonOptions': {'height':None, 'width':None},
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'],
    'scrollZoom': True
  }
  plot.show(config=config)
  if args.advance:
    total = epics.caget(f'H:FREQUENCY:VALUE:{plane}')
    phase, sigma = Decomposition.phase_adjacent(total, value, sigma_phase=error)
    phase = phase.cpu().numpy()
    sigma = sigma.cpu().numpy() if sigma != None else numpy.zeros(phase.shape)
    df = pandas.DataFrame()
    for case, data, std in zip(['MODEL', 'PHASE'], [model, phase], [0*sigma, sigma]):
      df = pandas.concat([df, pandas.DataFrame({'CASE':case, 'PAIR':pair, 'ADVANCE':data, 'SIGMA':std})])
    title = f'{TIME}: ADJACENT ADVANCE ({plane})'
    plot = scatter(df, x='PAIR', y='ADVANCE', color='CASE', error_y='SIGMA', title=title)
    config = {
      'toImageButtonOptions': {'height':None, 'width':None},
      'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
      'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'],
      'scrollZoom': True
    }
    plot.show(config=config)
    df = pandas.DataFrame()
    df['PAIR'] = pair
    df['ERROR'] = 100*(model - phase)/model
    df['SIGMA'] = 100*sigma/model
    title = f'{TIME}: ADJACENT ADVANCE ERROR ({plane})'
    plot = scatter(df, x='PAIR', y='ERROR', error_y='SIGMA', title=title)
    config = {
      'toImageButtonOptions': {'height':None, 'width':None},
      'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
      'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'],
      'scrollZoom': True
    }
    plot.show(config=config)

# Save to file
if args.save:
  filename = f'phase_plane_{args.plane}_length_{args.length}_time_{TIME}.npy'
  numpy.save(filename, value.cpu().numpy())

# Save to epics
if args.update:
  epics.caput_many([f'H:{name}:PHASE:VALUE:{plane}' for name in bpm], value.cpu().numpy())
  if error != None:
    epics.caput_many([f'H:{name}:PHASE:ERROR:{plane}' for name in bpm], error.cpu().numpy())
  else:
    epics.caput_many([f'H:{name}:PHASE:ERROR:{plane}' for name in bpm], -numpy.ones(value.cpu().numpy().shape))