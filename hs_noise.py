"""
usage: hs_noise [-h] -p {x,z,i} [-l LENGTH] [--limit LIMIT] [--skip BPM [BPM ...] | --only BPM [BPM ...]] [-o OFFSET] [-r] [-f] [-c | -n] [--print]
                [--mean | --median | --normalize] [--plot] [--device {cpu,cuda}] [--dtype {float32,float64}] [--test]

TbT data noise estimation for selected BPMs and plane.

optional arguments:
  -h, --help            show this help message and exit
  -p {x,z,i}, --plane {x,z,i}
                        data plane
  -l LENGTH, --length LENGTH
                        number of turns to use (integer)
  --limit LIMIT         number of columns to use (integer)
  --skip BPM [BPM ...]  space separated list of valid BPM names to skip
  --only BPM [BPM ...]  space separated list of valid BPM names to use
  -o OFFSET, --offset OFFSET
                        rise offset for all BPMs
  -r, --rise            flag to use rise data from file (drop first turns)
  -f, --file            flag to save data
  -c, --csv             flag to save data as CSV
  -n, --numpy           flag to save data as NUMPY
  --print               flag to print data
  --mean                flag to remove mean
  --median              flag to remove median
  --normalize           flag to normalize data
  --plot                flag to plot data
  --device {cpu,cuda}   data device
  --dtype {float32,float64}
                        data type
  --test                flag to use test PV names
"""

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_noise', description='TbT data noise estimation for selected BPMs and plane.')
parser.add_argument('-p', '--plane', choices=('x', 'z', 'i'), help='data plane', required=True)
parser.add_argument('-l', '--length', type=int, help='number of turns to use (integer)', default=512)
parser.add_argument('--limit', type=int, help='number of columns to use (integer)', default=32)
select = parser.add_mutually_exclusive_group()
select.add_argument('--skip', metavar='BPM', nargs='+', help='space separated list of valid BPM names to skip')
select.add_argument('--only', metavar='BPM', nargs='+', help='space separated list of valid BPM names to use')
parser.add_argument('-o', '--offset', type=int, help='rise offset for all BPMs', default=0)
parser.add_argument('-r', '--rise', action='store_true', help='flag to use rise data from file (drop first turns)')
parser.add_argument('-f', '--file', action='store_true', help='flag to save data')
save = parser.add_mutually_exclusive_group()
save.add_argument('-c', '--csv', action='store_true', help='flag to save data as CSV')
save.add_argument('-n', '--numpy', action='store_true', help='flag to save data as NUMPY')
parser.add_argument('--print', action='store_true', help='flag to print data')
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--mean', action='store_true', help='flag to remove mean')
transform.add_argument('--median', action='store_true', help='flag to remove median')
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float64')
parser.add_argument('--test', action='store_true', help='flag to use test PV names')
args = parser.parse_args()

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

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit(f'error: CUDA is not avalible')

# Import BPM data
try:
  df = pandas.read_json('bpm.json')
except ValueError:
  exit(f'error: problem loading bpm.json')

# Process BPM data
bpm = {name: int(df[name]['RISE']) for name in df if df[name]['FLAG']}

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

# Generate pv names
pv_list = [pv_make(name, args.plane, args.test) for name in bpm]
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
tbt = Data.from_epics(size, win, pv_list, pv_rise if args.rise else None, shift=offset, count=count)

# Remove mean
if args.mean:
  tbt.window_remove_mean()

# Remove median
if args.median:
  tbt.work.sub_(torch.median(tbt.data, 1).values.reshape(-1, 1))

# Normalize
if args.normalize:
  tbt.normalize(window=True)

# Estimate rank & noise
flt = Filter(tbt)
rnk, std = flt.estimate_noise(limit=args.limit, cpu=True)

# Clean
del win, tbt, flt
if device == 'cuda':
  torch.cuda.empty_cache()

# Plot
if args.plot:
  df = pandas.DataFrame()
  df['bpm'] = [*bpm.keys()]
  df['rnk'] = rnk.cpu()
  df['rnk'] = df['rnk'].astype(str)
  df['std'] = std.cpu()
  from plotly.express import bar
  title = f'{TIME}: Noise'
  plot = bar(df, x='bpm', y='std', color='rnk', title=title)
  config = {
    'toImageButtonOptions': {'height':None, 'width':None},
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'],
    'scrollZoom': True
  }
  plot.show(config=config)

# Print data
if args.print:
  fmt = '{:>6}' * 2 + '{:>18.9}'
  print(fmt.format('BPM', 'RNK', 'STD'))
  for name, rank, sigma in zip(bpm, rnk, std):
    print(fmt.format(name, rank, sigma))

# Save to file
if args.file and args.numpy:
  numpy.save(f'noise{TIME}.npy', std.cpu())
if args.file and args.csv:
  numpy.savetxt(f'noise{TIME}.csv', std.cpu(), delimiter=',')