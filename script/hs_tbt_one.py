# Input arguments flag
import sys
sys.path.append('..')
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_tbt_one', description='Print/save/plot TbT data for selected BPMs and plane.')
parser.add_argument('-p', '--plane', choices=('x', 'z', 'i'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to print/save/plot (integer)', default=1024)
parser.add_argument('-b', '--bpm', type=str, help='BPM name', default='stp2')
parser.add_argument('-o', '--offset', type=int, help='rise offset', default=0)
parser.add_argument('-r', '--rise', action='store_true', help='flag to use rise data from file (drop first turns)')
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--mean', action='store_true', help='flag to remove mean')
transform.add_argument('--median', action='store_true', help='flag to remove median')
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
parser.add_argument('--envelope', action='store_true', help='flag to compute envelope')
parser.add_argument('--frequency', action='store_true', help='flag to compute instantaneous frequency')
parser.add_argument('--flip', action='store_true', help='flag to flip frequency around 1/2')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--harmonica', action='store_true', help='flag to use harmonica PV names for input')
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
from harmonica.frequency import Frequency

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit(f'error: CUDA is not avalible')

# Import BPM data
try:
  df = pandas.read_json('../bpm.json')
except ValueError:
  exit(f'error: problem loading bpm.json')

# Process BPM data
bpm = {name: int(df[name]['RISE']) for name in df if df[name]['FLAG']}

# Check & keep only selected BPM
target = args.bpm.upper()
if not target in bpm:
  exit(f'error: {target} in not a valid BPM to read')
for name in bpm.copy():
  if name != target:
    bpm.pop(name)

# Check BPM list
if not bpm:
  exit(f'error: BPM list is empty')

# Generate pv names
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

# Envelope
if args.envelope:
  out = Frequency.dht(tbt.work)
  env = out.abs().flatten().cpu().numpy()
  fre = 1.0/(2.0*numpy.pi)*(out[:, :-1]*out[:, 1:].conj()).angle().abs().flatten().cpu().numpy()
  fre = 1.0 - fre if args.flip else fre

# Convert to numpy
data = tbt.to_numpy().flatten()

# Clean
del win, tbt
if device == 'cuda':
  torch.cuda.empty_cache()

# Set turns
turn = numpy.linspace(0, length - 1, length, dtype=numpy.int32)

# Plot
if args.plot:
  df = pandas.DataFrame()
  df['turn'] = turn
  df[args.plane] = data
  from plotly.express import scatter
  plot = scatter(df, x='turn', y=args.plane, title=f'{TIME}', opacity=0.75)
  if args.envelope:
    df['envelope'] = env
    plot = scatter(df, x='turn', y=[args.plane, 'envelope'], title=f'{TIME}', opacity=0.75)
  config = {
    'toImageButtonOptions': {'height':None, 'width':None},
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'],
    'scrollZoom': True
  }
  plot.show(config=config)
  if args.frequency:
    df = pandas.DataFrame()
    df['turn'] = turn[:-1]
    df['frequency'] = fre
    mean = fre.mean()
    std = fre.std()
    plot = scatter(df, x='turn', y='frequency', title=f'{TIME}: Frequency<br>MEAN: {mean}, STD:{std}', opacity=0.75)
    plot.add_hline(fre.mean(), line_color='red', line_dash="dash", line_width=0.5)
    plot.show(config=config)
