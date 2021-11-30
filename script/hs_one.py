"""
usage: hs_tbt [-h] [-p {x,z,i}] [-l LENGTH] [-b BPM] [-o OFFSET] [-r] [--mean | --median | --normalize] [--envelope] [--frequency] [--flip] [--plot] [--harmonica]
              [--device {cpu,cuda}] [--dtype {float32,float64}]

Print/save/plot TbT data for selected BPMs and plane.

optional arguments:
  -h, --help            show this help message and exit
  -p {x,z,i}, --plane {x,z,i}
                        data plane
  -l LENGTH, --length LENGTH
                        number of turns to print/save/plot (integer)
  -b BPM, --bpm BPM     BPM name
  -o OFFSET, --offset OFFSET
                        rise offset
  -r, --rise            flag to use rise data from file (drop first turns)
  --mean                flag to remove mean
  --median              flag to remove median
  --normalize           flag to normalize data
  --envelope            flag to compute envelope
  --frequency           flag to compute instantaneous frequency
  --flip                flag to flip frequency around 1/2
  --plot                flag to plot data
  --harmonica           flag to use harmonica PV names for input
  --device {cpu,cuda}   data device
  --dtype {float32,float64}
                        data type
"""

# Input arguments flag
import sys
sys.path.append('..')
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_tbt', description='Print/save/plot TbT data for selected BPMs and plane.')
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
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float32')
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

# Envelope
if args.envelope:
  out = Frequency.dht(tbt.work)
  env = out.abs().flatten().cpu().numpy()
  fre = 1/(2*numpy.pi)*(out[:, :-1]*out[:, 1:].conj()).angle().abs().flatten().cpu().numpy()
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
    plot.show(config=config)
