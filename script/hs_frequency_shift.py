# Input arguments flag
import sys
sys.path.append('..')
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_frequency_shift', description='Save/plot frequency data for selected plane and BPMs using shifts.')
parser.add_argument('-p', '--plane', choices=('x', 'y'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use', default=512)
parser.add_argument('--load', type=int, help='total number of turns to load', default=2048)
parser.add_argument('--shift', type=int, help='shift step', default=8)
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
parser.add_argument('--pad', type=int, help='number of zeros to pad', default=0)
parser.add_argument('--f_min', type=float, help='min frequency value', default=0.0)
parser.add_argument('--f_max', type=float, help='max frequency value', default=0.5)
parser.add_argument('-m', '--method', choices=('fft', 'ffrft', 'parabola'), help='frequency estimation method', default='parabola')
parser.add_argument('--fit', choices=('none', 'std', 'noise'), help='fit type', default='none')
parser.add_argument('--limit', type=int, help='number of columns to use for noise estimation', default=16)
parser.add_argument('--flip', action='store_true', help='flag to flip frequency around 1/2')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--noise', action='store_true', help='flag to plot noise data')
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

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit(f'error: CUDA is not avalible')

# Check and set frequency range & padding
f_min = args.f_min
f_max = args.f_max
if not (0.0 <= f_min <= 0.5):
  exit(f'error: {f_min=}, should be in (0.0, 0.5)')
if not (0.0 <= f_max <= 0.5):
  exit(f'error: {f_max=}, should be in (0.0, 0.5)')
if f_max < f_min:
  exit(f'error: {f_max=} should be greater than {f_min=}')
if (f_min, f_max) != (0.0, 0.5) and args.pad != 0:
  exit(f'error: (f_min, f_max) should be used without padding')

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

# Compute shifted frequencies
f = Frequency(tbt)
frequency = f.task_shift(
  args.length,
  args.shift,
  task=args.method,
  name='cosine_window',
  order=args.window,
  f_range=(f_min, f_max)
)
_, step = frequency.shape

# Estimate noise
if args.fit == 'noise':
  table = []
  win = Window(args.length)
  for signal in tbt.work:
    matrix = Data.from_data(win, Data.make_matrix(args.length, args.shift, signal))
    f = Filter(matrix)
    _, noise = f.estimate_noise(limit=args.limit)
    table.append(noise)
  noise = torch.stack(table).cpu().numpy()

# Clean
del win, tbt, f
if device == 'cuda':
  torch.cuda.empty_cache()

# Convert to numpy
frequency = frequency.cpu().numpy()

# Flip
if args.flip:
  frequency = 1.0 - frequency

# Fit (std)
if args.fit == 'std':
  from statsmodels.api import WLS
  m = frequency.mean(1)
  s = frequency.std(1)
  x = numpy.ones((len(bpm), 1))
  y = m
  w = 1/s**2
  out = WLS(y, x, w).fit()
  f_fit, *_ = out.params
  s_fit, *_ = out.bse

# Fit (noise)
if args.fit == 'noise':
  from statsmodels.api import WLS
  table = []
  x = numpy.ones((step, 1))
  for y, s in zip(frequency, noise):
    w = 1/s**2
    out = WLS(y, x, w).fit()
    f_fit, *_ = out.params
    s_fit, *_ = out.bse
    table.append([f_fit, s_fit])
  x = numpy.ones((len(bpm), 1))
  m, s = numpy.array(table).T
  y = m
  w = 1/s**2
  out = WLS(y, x, w).fit()
  f_fit, *_ = out.params
  s_fit, *_ = out.bse

# Plot
if args.plot:
  df = pandas.DataFrame()
  for i, name in enumerate(bpm):
    df = pandas.concat([df, pandas.DataFrame({'STEP':range(1, step + 1), 'BPM':name, 'FREQUENCY':frequency[i]})])
  from plotly.express import scatter
  title = f'{TIME}: FREQUENCY (SHIFT)<br>SAMPLE={args.length}, SHIFT={args.shift}, COUNT={step}, MEAN={frequency.mean()}, STD={frequency.std()}'
  title = title if not args.fit != 'none' else f'{title}<br>FIT={args.fit.upper()}, VALUE={f_fit}, ERROR={s_fit}'
  plot = scatter(df, x='STEP', y='FREQUENCY', color='BPM', title=title, opacity=0.75, marginal_y='box')
  if args.fit != 'none':
    plot.add_hline(f_fit - s_fit, line_color='red', line_dash='dash', line_width=0.5)
    plot.add_hline(f_fit, line_color='red', line_dash='dash', line_width=0.5)
    plot.add_hline(f_fit + s_fit, line_color='red', line_dash='dash', line_width=0.5)
  config = {
    'toImageButtonOptions': {'height':None, 'width':None},
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'],
    'scrollZoom': True
  }
  plot.show(config=config)
  if args.fit != 'none':
    df = pandas.DataFrame()
    df['BPM'] = [*bpm.keys()]
    df['VALUE'] = m
    df['ERROR'] = s
    plot = scatter(df, x='BPM', y='VALUE', title=title, opacity=0.75, marginal_y='box', error_y='ERROR')
    plot.add_hline(f_fit - s_fit, line_color='red', line_dash='dash', line_width=0.5)
    plot.add_hline(f_fit, line_color='red', line_dash='dash', line_width=0.5)
    plot.add_hline(f_fit + s_fit, line_color='red', line_dash='dash', line_width=0.5)
    plot.show(config=config)
  if args.noise and args.fit == 'noise':
    df = pandas.DataFrame()
    for i, name in enumerate(bpm):
      df = pandas.concat([df, pandas.DataFrame({'STEP':range(1, step + 1), 'BPM':name, 'NOISE':noise[i]})])
    title = f'{TIME}: NOISE (SHIFT)<br>SAMPLE={args.length}, SHIFT={args.shift}, COUNT={step}'
    plot = scatter(df, x='STEP', y='NOISE', color='BPM', title=title, opacity=0.75)
    plot.show(config=config)

# Save to file
if args.save:
  filename = f'frequency_shift_plane_{args.plane}_length_{args.length}_time_{TIME}.npy'
  numpy.save(filename, frequency)

# Save to epics
if args.update:
  plane = args.plane.upper()
  if args.fit == 'none':
    epics.caput(f'H:FREQUENCY:VALUE:{plane}', frequency.flatten().mean())
    epics.caput(f'H:FREQUENCY:ERROR:{plane}', frequency.flatten().std())
    epics.caput_many([f'H:{name}:FREQUENCY:VALUE:{plane}' for name in bpm], frequency.mean(1))
    epics.caput_many([f'H:{name}:FREQUENCY:ERROR:{plane}' for name in bpm], frequency.std(1))
  if args.fit == 'std' or args.fit == 'noise':
    epics.caput(f'H:FREQUENCY:VALUE:{plane}', f_fit)
    epics.caput(f'H:FREQUENCY:ERROR:{plane}', s_fit)
    epics.caput_many([f'H:{name}:FREQUENCY:VALUE:{plane}' for name in bpm], m)
    epics.caput_many([f'H:{name}:FREQUENCY:ERROR:{plane}' for name in bpm], s)