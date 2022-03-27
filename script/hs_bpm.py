# Input arguments flag
import sys
sys.path.append('..')
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_bpm', description='Plot TbT data for selected BPM and plane.')
parser.add_argument('-p', '--plane', choices=('x', 'y', 'i'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use', default=1024)
parser.add_argument('-b', '--bpm', type=str, help='BPM name', default='stp2')
parser.add_argument('-o', '--offset', type=int, help='rise offset', default=0)
parser.add_argument('-r', '--rise', action='store_true', help='flag to use rise data (drop first turns)')
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--mean', action='store_true', help='flag to remove mean')
transform.add_argument('--median', action='store_true', help='flag to remove median')
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
parser.add_argument('-f', '--filter', choices=('none', 'hankel'), help='filter type', default='none')
parser.add_argument('--rank', type=int, help='rank to use for hankel filter', default=8)
parser.add_argument('--type', choices=('full', 'randomized'), help='SVD computation type for hankel filter', default='randomized')
parser.add_argument('--buffer', type=int, help='buffer size to use for randomized hankel filter', default=16)
parser.add_argument('--count', type=int, help='number of iterations to use for randomized hankel filter', default=16)
parser.add_argument('--envelope', action='store_true', help='flag to compute envelope')
parser.add_argument('--frequency', action='store_true', help='flag to compute instantaneous frequency')
parser.add_argument('--flip', action='store_true', help='flag to flip frequency around 1/2')
parser.add_argument('--drop', type=int, help='number of endpoints to drop for mean frequency', default=32)
parser.add_argument('--plot', action='store_true', help='flag to plot data')
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
from harmonica.util import LIMIT, pv_make
from harmonica.window import Window
from harmonica.data import Data
from harmonica.frequency import Frequency
from harmonica.filter import Filter

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

# Filter
if args.filter == 'hankel':
  flt = Filter(tbt)
  flt.filter_hankel(rank=args.rank, random=args.type == 'randomized', buffer=args.buffer, count=args.count)

# Envelope
if args.envelope:
  dht = Frequency.dht(tbt.work)
  envelope = dht.abs().flatten().cpu().numpy()
  frequency = 1.0/(2.0*numpy.pi)*(dht[:, :-1]*dht[:, 1:].conj()).angle().abs().flatten().cpu().numpy()
  frequency = 1.0 - frequency if args.flip else frequency

# Convert to numpy
data = tbt.to_numpy().flatten()

# Set turns
turn = numpy.linspace(0, length - 1, length, dtype=numpy.int32)

# Plot
if args.plot:
  df = pandas.DataFrame()
  df['TURN'] = turn
  df[args.plane.upper()] = data
  from plotly.express import scatter
  plot = scatter(df, x='TURN', y=args.plane.upper(), title=f'{TIME}: TbT ({args.bpm.upper()})', opacity=0.75, labels={'TURN', args.plane.upper()})
  if args.envelope:
    df['ENVELOPE'] = envelope
    plot = scatter(df, x='TURN', y=[args.plane.upper(), 'ENVELOPE'], title=f'{TIME}: TbT ({args.bpm.upper()})', opacity=0.75)
    plot.update_layout(yaxis_title=args.plane.upper())
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)
  if args.frequency:
    df = pandas.DataFrame()
    df['TURN'] = turn[:-1]
    df['FREQUENCY'] = frequency
    drop = args.drop
    mean = frequency[drop:-drop].mean()
    std = frequency[drop:-drop].std()
    plot = scatter(df, x='TURN', y='FREQUENCY', title=f'{TIME}: Frequency ({args.bpm.upper()})<br>MEAN: {mean}, STD:{std}', opacity=0.75)
    plot.add_hline(mean, line_color='red', line_dash="dash", line_width=0.5)
    config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
    plot.show(config=config)
