#!/bin/env python

# Import
import sys
import argparse
import epics
import numpy
import pandas
import torch
from datetime import datetime
from harmonica.util import LIMIT, LENGTH, pv_make, bpm_select
from harmonica.window import Window
from harmonica.data import Data
from harmonica.filter import Filter
from harmonica.frequency import Frequency


# Input arguments flag
sys.path.append('../..')
_, *flag = sys.argv

# Parse arguments
parser = argparse.ArgumentParser(prog='frequency_all', description='Save/plot mixed frequency for selected plane and BPMs with optional shifts.')
parser.add_argument('-p', '--plane', choices=('x', 'y'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use', default=64)
parser.add_argument('--load', type=int, help='total number of turns to load', default=128)
parser.add_argument('--shift', type=int, help='shift step', default=-1)
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
parser.add_argument('-f', '--filter', choices=('none', 'svd', 'hankel'), help='filter type', default='none')
parser.add_argument('--rank', type=int, help='rank to use for svd & hankel filter', default=8)
parser.add_argument('--type', choices=('full', 'randomized'), help='computation type for hankel filter', default='randomized')
parser.add_argument('--buffer', type=int, help='buffer size to use for randomized hankel filter', default=16)
parser.add_argument('--count', type=int, help='number of iterations to use for randomized hankel filter', default=16)
parser.add_argument('-w', '--window', type=float, help='window order', default=0.0)
parser.add_argument('--f_min', type=float, help='min frequency value', default=0.0)
parser.add_argument('--f_max', type=float, help='max frequency value', default=0.5)
parser.add_argument('--beta_min', type=float, help='min beta threshold value for x or y', default=0.0E+0)
parser.add_argument('--beta_max', type=float, help='max beta threshold value for x or y', default=1.0E+3)
parser.add_argument('--nufft', action='store_true', help='flag to compute spectum using TYPY-III NUFFT')
parser.add_argument('--time', choices=('position', 'phase'), help='time type to use with NUFFT', default='phase')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('-H', '--harmonica', action='store_true', help='flag to use harmonica PV names for input')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float64')
parser.add_argument('-u', '--update', action='store_true', help='flag to update harmonica PV')
args = parser.parse_args(args=None if flag else ['--help'])

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit('error: CUDA is not available')

# Check and set frequency range
f_min = args.f_min
f_max = args.f_max
if f_min < 0.0:
  exit(f'error: {f_min=}, should be positive')
if f_max < f_min:
  exit(f'error: {f_max=} should be greater than {f_min=}')

# Load monitor data
name = epics.caget('H:MONITOR:LIST')[:epics.caget('H:MONITOR:COUNT')]
flag = epics.caget_many([f'H:{name}:FLAG' for name in name])
join = epics.caget_many([f'H:{name}:JOIN' for name in name])
rise = epics.caget_many([f'H:{name}:RISE' for name in name])
beta = epics.caget_many([f'H:{name}:MODEL:B{args.plane.upper()}' for name in name])
beta = {key: value for key, value in zip(name, beta)}

# Set BPM data
bpm = {name: rise for name, flag, rise, join in zip(name, flag, rise, join) if flag == 1 and join == 1}

# Filter BPM list
try:
  bpm = bpm_select(bpm, skip=args.skip, only=args.only)
except ValueError as exception:
  exit(str(exception))

# Check beta values
if args.beta_min < 0:
  exit(f'error: min beta threshold {args.beta_min} should be positive')
if args.beta_max < 0:
  exit(f'error: max beta threshold {args.beta_max} should be positive')
if args.beta_min > args.beta_max:
  exit(f'error: max beta threshold {args.beta_max} should be greater than min beta threshold {args.beta_min}')

# Filter by beta values
for name in bpm.copy():
  if not (args.beta_min <= beta[name] <= args.beta_max):
    bpm.pop(name)

# Check BPM list
if not bpm:
  exit('error: BPM list is empty')

# Generate PV names
pv_list = [pv_make(name, args.plane, args.harmonica) for name in bpm]
pv_rise = [*bpm.values()]

# Set BPM positions
if args.nufft:
  if args.time == 'position':
    position = epics.caget_many([f'H:{name}:TIME' for name in bpm])
    position = numpy.array(position)/LENGTH
  if args.time == 'phase':
    total = epics.caget(f'H:TAIL:MODEL:F{args.plane.upper()}')
    position = epics.caget_many([f'H:{name}:MODEL:F{args.plane.upper()}' for name in bpm])
    position = numpy.array(position)/total

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
    exit('error: rise values are expected to be positive')
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
tbt = Data.from_epics(win, pv_list, pv_rise=(pv_rise if args.rise else None), shift=offset, count=count)

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

# Number of steps
step = range(1 if args.shift <= 0 else 1 + (length - args.length) // args.shift)

# Loop over steps
shift = args.shift
length = args.length
win = Window(length, 'cosine_window', args.window, dtype=dtype, device=device)
tbt_shift = Data(size, win)
f = Frequency(tbt_shift)
result = []
for i in step:
  tbt_shift.set_data(tbt.work[:, i*shift : length + i*shift])
  frequency = f.compute_joined_frequency(
    length=length,
    f_range=(f_min, f_max),
    name='cosine_window',
    order=args.window,
    normalize=True,
    position=position if args.nufft else None
  )
  result.append(frequency)

# Clean
del win, tbt, tbt_shift, f
if device == 'cuda':
  torch.cuda.empty_cache()

# Format result
frequency = torch.stack(result).cpu().numpy()

# Plot
if args.plot:
  df = pandas.DataFrame()
  for i, name in enumerate(['F1', 'F2', 'F3']):
    df = pandas.concat([df, pandas.DataFrame({'STEP':step, 'CASE':name, 'FREQUENCY':frequency[:, i]})])
  from plotly.express import scatter
  mean = numpy.mean(frequency[:, -1])
  median = numpy.median(frequency[:, -1])
  std = numpy.std(frequency[:, -1])
  title = f'{TIME}: FREQUENCY (ALL)<br>LENGTH={args.length}, SHIFT={args.shift}, COUNT={len(step)}<br>MEAN: {mean}, MEDIAN: {median}, SIGMA: {std}'
  plot = scatter(df, x='STEP', y='FREQUENCY', color='CASE', title=title, opacity=0.75, marginal_y='box')
  config = {
    'toImageButtonOptions': {'height':None, 'width':None},
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'],
    'scrollZoom': True
  }
  plot.show(config=config)

# Save to file
if args.save:
  filename = f'frequency_all_plane_{args.plane}_length_{args.length}_time_{TIME}.npy'
  numpy.save(filename, frequency)

# Save to epics
if args.update:
  plane = args.plane.upper()
  *_, frequency = frequency.T
  epics.caput(f'H:FREQUENCY:VALUE:{plane}', frequency.mean())
  epics.caput(f'H:FREQUENCY:ERROR:{plane}', frequency.std())