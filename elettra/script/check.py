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
from harmonica.decomposition import Decomposition

# Input arguments flag
_, *last = sys.argv

# Parse arguments
parser = argparse.ArgumentParser(prog='check', description='Check phase synchronization for selected BPMs and plane.')
parser.add_argument('-p', '--plane', choices=('x', 'y'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use', default=256)
select = parser.add_mutually_exclusive_group()
select.add_argument('--skip', metavar='PATTERN', nargs='+', help='space separated regex patterns for BPM names to skip')
select.add_argument('--only', metavar='PATTERN', nargs='+', help='space separated regex patterns for BPM names to use')
parser.add_argument('-o', '--offset', type=int, help='rise offset for all BPMs', default=0)
parser.add_argument('-r', '--rise', action='store_true', help='flag to use rise data (drop first turns)')
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--mean', action='store_true', help='flag to remove mean')
transform.add_argument('--median', action='store_true', help='flag to remove median')
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
parser.add_argument('-f', '--filter', choices=('none', 'svd', 'hankel'), help='filter type', default='none')
parser.add_argument('--rank', type=int, help='rank to use for svd & hankel filter', default=8)
parser.add_argument('--type', choices=('full', 'randomized'), help='svd computation type for hankel filter', default='randomized')
parser.add_argument('--buffer', type=int, help='buffer size to use for randomized hankel filter', default=16)
parser.add_argument('--count', type=int, help='number of iterations to use for randomized hankel filter', default=16)
parser.add_argument('-w', '--window', type=float, help='window order', default=0.0)
parser.add_argument('--factor', type=float, help='threshold factor', default=5.0)
parser.add_argument('--load', action='store_true', help='flag to load phase from harmonica PVs instead of estimating from TbT data')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--prefix', type=str, help='PV prefix', default='BPM')
parser.add_argument('--data', type=str, help='PV data prefix', default='')
parser.add_argument('--tango', action='store_true', help='flag to use tango CS')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float64')
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

# Set data plane
plane = args.plane.upper()

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

# Set model phase
PHASE = torch.tensor([cs.get(f'{args.prefix}:{name}:MODEL:F{plane}') for name in bpm], dtype=dtype, device=device)

# Set tunes
q = cs.get(f'{args.prefix}:FREQUENCY:VALUE:{plane}')
Q = cs.get(f'{args.prefix}:FREQUENCY:MODEL:{plane}')

# Load phase
if args.load:
  phase = torch.tensor([cs.get(f'{args.prefix}:{name}:PHASE:VALUE:{plane}') for name in bpm], dtype=dtype, device=device)

if not args.load:
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

  # Check window order
  if args.window < 0.0:
    exit(f'error: window order {args.window} should be greater or equal to zero')

  # Load TbT data
  count = length + offset + shift
  win = Window(length, 'cosine_window', args.window, dtype=dtype, device=device)
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

  # Filter (svd)
  if args.filter == 'svd':
    flt = Filter(tbt)
    flt.filter_svd(rank=args.rank)

  # Filter (hankel)
  if args.filter == 'hankel':
    flt = Filter(tbt)
    flt.filter_svd(rank=args.rank)
    flt.filter_hankel(rank=args.rank, random=args.type == 'randomized', buffer=args.buffer, count=args.count)

  # Estimate phase
  dec = Decomposition(tbt)
  phase, _, _ = dec.harmonic_phase(q, length=args.length, order=args.window, factor=args.factor)
  output = tbt.to_numpy()
else:
  output = phase.cpu().numpy()

# Check
check, table = Decomposition.phase_check(q, Q, phase, PHASE, factor=args.factor)

# Print result
for marked in check:
  index, value = check[marked]
  if index != 0:
    print(marked, [*bpm][marked], check[marked])

# Plot
if args.plot:
  from pandas import DataFrame
  from pandas import concat
  from plotly.express import scatter
  mark = [-1 if key not in check else check[key][0]/2 - 1 for key in range(len(bpm))]
  df = DataFrame()
  for case, data in zip(['PHASE', 'MODEL', 'CHECK', 'FLAG'], [table['phase'], table['model'], table['check'], mark]):
      df = concat([df, DataFrame({'CASE':case, 'BPM':range(len(bpm)), 'ADVANCE':data})])
  plot = scatter(df, x='BPM', y='ADVANCE', color='CASE', title=f'{TIME}: ADVANCE ({plane})', opacity=0.75, color_discrete_sequence = ['red', 'green', 'blue', 'black'])
  plot.update_layout(xaxis = dict(tickmode = 'array', tickvals = list(range(len(bpm))), ticktext = list(bpm.keys())))
  plot.update_traces(marker={'size': 10})
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)
