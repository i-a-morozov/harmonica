# Input arguments flag
import sys
sys.path.append('..')
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_noise', description='TbT data noise estimation for selected BPMs and plane.')
parser.add_argument('-p', '--plane', choices=('x', 'y', 'i'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use', default=512)
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
parser.add_argument('--limit', type=int, help='number of columns to use', default=16)
parser.add_argument('--std', action='store_true', help='flag to estimate noise with std')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--auto', action='store_true', help='flag to plot autocorrelation')
parser.add_argument('-H', '--harmonica', action='store_true', help='flag to use harmonica PV names for input')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float64')
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
length = args.length
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

# Load TbT data
size = len(bpm)
count = length + offset + rise
win = Window(length, dtype=dtype, device=device)
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

# Estimate rank & noise
flt = Filter(tbt)
if args.std:
  rnk = -torch.ones(len(bpm), dtype=torch.int32, device=device)
  std = torch.std(tbt.work, 1)
else:
  rnk, std = flt.estimate_noise(limit=args.limit, cpu=True)

# Autocorrelation
if args.auto:
  auto = Frequency.autocorrelation(tbt.work).cpu().numpy()

# Clean
del win, tbt, flt
if device == 'cuda':
  torch.cuda.empty_cache()

# Plot
if args.plot:
  df = pandas.DataFrame()
  df['BPM'] = [*bpm.keys()]
  df['RANK'] = rnk.cpu()
  df['RANK'] = df['RANK'].astype(str)
  df['SIGMA'] = std.cpu()
  from plotly.express import bar
  title = f'{TIME}: NOISE'
  plot = bar(df, x='BPM', y='SIGMA', color='RANK', category_orders={'BPM':df['BPM']}, title=title)
  config = {
    'toImageButtonOptions': {'height':None, 'width':None},
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'],
    'scrollZoom': True
  }
  plot.show(config=config)
  if args.auto:
    from plotly.express import imshow
    plot = imshow(
      auto,
      labels=dict(x="SHIFT", y="BPM", color=f"AUTO"),
      y=[*bpm.keys()],
      aspect=0.5,
      title=f'{TIME}: AUTO'
    )
    plot.show(config=config)

# Save to file
if args.save:
  filename = f'noise_plane_{args.plane}_length_{args.length}_time_{TIME}.npy'
  numpy.save(filename, std.cpu())