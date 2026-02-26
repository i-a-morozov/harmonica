#!/bin/env python

# Import
import sys
import argparse
import numpy
import torch
from datetime import datetime
from harmonica.util import LIMIT, pv_make, bpm_select
from harmonica.cs import factory
from harmonica.window import Window
from harmonica.data import Data
from harmonica.filter import Filter
from harmonica.frequency import Frequency

# Input arguments flag
_, *last = sys.argv


# Parse arguments
parser = argparse.ArgumentParser(prog='noise', description='TbT data noise estimation for selected BPMs and plane.')
parser.add_argument('-p', '--plane', choices=('x', 'y', 'i'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use', default=512)
select = parser.add_mutually_exclusive_group()
select.add_argument('--skip', metavar='PATTERN', nargs='+', help='space separated regex patterns for BPM names to skip')
select.add_argument('--only', metavar='PATTERN', nargs='+', help='space separated regex patterns for BPM names to use')
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
parser.add_argument('--prefix', type=str, help='PV prefix', default='BPM')
parser.add_argument('--data', type=str, help='PV data prefix', default='')
parser.add_argument('--tango', action='store_true', help='flag to use tango CS')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float64')
parser.add_argument('-u', '--update', action='store_true', help='flag to update harmonica PV noise data')
parser.add_argument('--verbose', action='store_true', help='verbose flag')
args = parser.parse_args(args=None if last else ['--help'])

# Time
TIME = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
if args.verbose:
  print(f'Time: {TIME}')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit('error: CUDA is not available')

# CS
cs = factory(target=('tango' if args.tango else 'epics'))

# Load monitor data
name = cs.get(f'{args.prefix}:MONITOR:LIST')[:cs.get(f'{args.prefix}:MONITOR:COUNT')]
flag = numpy.asarray([cs.get(f'{args.prefix}:{name}:FLAG') for name in name])
rise = numpy.asarray([cs.get(f'{args.prefix}:{name}:RISE') for name in name])

# Set BPM data
bpm = {name: rise for name, flag, rise in zip(name, flag, rise) if flag == 1}

# Filter BPM list
try:
  bpm = bpm_select(bpm, skip=args.skip, only=args.only)
except ValueError as exception:
  exit(str(exception))

# Check BPM list
if not bpm:
  exit('error: BPM list is empty')

if args.verbose:
  print('Monitor list:')
  for key, value in bpm.items():
    print(f'{key}: {value}')

# Generate PV names
prefix = args.prefix if not args.data else args.data
pv_list = [pv_make(name, args.plane, prefix=prefix) for name in bpm]
pv_rise = [*bpm.values()]
if args.verbose:
  print('PV list:')
  for pv in pv_list:
    print(pv)

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
shift = 0
if args.rise:
  shift = min(pv_rise)
  if shift < 0:
    exit('error: rise values are expected to be positive')
  shift = max(pv_rise)
  if length + offset + shift > LIMIT:
    exit(f'error: sum of {length=}, {offset=} and max {shift=}, expected to be less than {LIMIT=}')

# Load TbT data
size = len(bpm)
count = length + offset + shift
win = Window(length, dtype=dtype, device=device)
matrix = numpy.asarray([cs.get(pv) for pv in pv_list])
data = torch.tensor(matrix, dtype=dtype, device=device)
data = torch.stack([signal[:count] for signal in data])
if args.rise:
  data = torch.stack([signal[offset + rise : offset + rise + length] for signal, rise in zip(data, pv_rise)])
else:
  data = data[:, offset : offset + length]
tbt = Data.from_data(win, data)
if args.verbose:
  print(f'TbT: {tbt}')

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
  rnk = torch.ones(len(bpm), dtype=torch.int64, device=device)
  std = torch.std(tbt.work, 1)
else:
  rnk, std = flt.estimate_noise(limit=args.limit, cpu=True)
output = std.cpu().numpy()

# Autocorrelation
if args.auto:
  auto = Frequency.autocorrelation(tbt.work).cpu().numpy()

# Plot
if args.plot:
  from pandas import DataFrame
  from plotly.express import bar
  df = DataFrame()
  df['BPM'] = [*bpm.keys()]
  df['RANK'] = rnk.cpu().numpy()
  df['RANK'] = df['RANK'].astype(str)
  df['SIGMA'] = output
  title = f'{TIME}: NOISE'
  plot = bar(df, x='BPM', y='SIGMA', color='RANK', category_orders={'BPM':df['BPM']}, title=title)
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)
  if args.auto:
    from plotly.express import imshow
    plot = imshow(auto, labels=dict(x="SHIFT", y="BPM", color="AUTO"), y=[*bpm.keys()], aspect=0.5, title=f'{TIME}: AUTO')
    config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
    plot.show(config=config)

# Save to file
if args.save:
  filename = f'noise_plane_{args.plane}_length_{args.length}_time_({TIME}).npy'
  numpy.save(filename, output)

# Save to cs
if args.update:
  plane = args.plane.upper()
  for name, sigma in zip(bpm, output):
    cs.set(f'{args.prefix}:{name}:NOISE:{plane}', sigma)
