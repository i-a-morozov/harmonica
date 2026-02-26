#!/bin/env python

# Import
import sys
import argparse
import numpy
import torch
from datetime import datetime
from harmonica.util import LIMIT, pv_make, mod, bpm_select
from harmonica.cs import factory
from harmonica.window import Window
from harmonica.data import Data
from harmonica.filter import Filter
from harmonica.frequency import Frequency
from harmonica.decomposition import Decomposition

# Input arguments flag
_, *last = sys.argv


# Parse arguments
parser = argparse.ArgumentParser(prog='phase', description='Estimate phase for selected BPMs ans plane.')
parser.add_argument('-p', '--plane', choices=('x', 'y'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use', default=256)
parser.add_argument('--load', type=int, help='total number of turns to load', default=512)
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
parser.add_argument('--type', choices=('full', 'randomized'), help='svd computation type for hankel filter', default='randomized')
parser.add_argument('--buffer', type=int, help='buffer size to use for randomized hankel filter', default=16)
parser.add_argument('--count', type=int, help='number of iterations to use for randomized hankel filter', default=16)
parser.add_argument('-w', '--window', type=float, help='window order', default=0.0)
parser.add_argument('--limit', type=int, help='number of columns to use for noise estimation', default=32)
case = parser.add_mutually_exclusive_group()
case.add_argument('--error', action='store_true', help='flag to propagate errors')
case.add_argument('--shift', action='store_true', help='flag to use shifted samples')
parser.add_argument('--size', type=int, help='maximum number of samples to use', default=256)
parser.add_argument('--delta', type=int, help='sample shift step', default=8)
parser.add_argument('-m', '--method', choices=('none', 'noise', 'error'), help='amplitude estimation method (shifted samples)', default='none')
parser.add_argument('--dht', action='store_true', help='flag to use DHT estimator')
parser.add_argument('--drop', type=int, help='number of endpoints to drop in DHT', default=32)
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--advance', action='store_true', help='flag to plot phase advance')
parser.add_argument('--monotonic', action='store_true', help='flag to plot monotonic phase and integer tune estimate')
parser.add_argument('--coupled', action='store_true', help='flag to compute coupled phase using opposite-plane frequency')
parser.add_argument('--prefix', type=str, help='PV prefix', default='BPM')
parser.add_argument('--data', type=str, help='PV data prefix', default='')
parser.add_argument('--tango', action='store_true', help='flag to use tango CS')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float64')
parser.add_argument('-u', '--update', action='store_true', help='flag to update harmonica PV')
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
target = {'X':'Y', 'Y':'X'}[plane] if args.coupled else plane

# Load frequency and frequency error
frequency_value = cs.get(f'{args.prefix}:FREQUENCY:VALUE:{target}')
frequency_error = cs.get(f'{args.prefix}:FREQUENCY:ERROR:{target}')

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

# Generate model advance
if args.advance:
  total = cs.get(f'{args.prefix}:TAIL:MODEL:F{target}')
  model = torch.tensor([cs.get(f'{args.prefix}:{name}:MODEL:F{target}') for name in bpm], dtype=dtype, device=device)
  model, _ = Decomposition.phase_adjacent(total/(2.0*numpy.pi), model)
  model_numpy = model.cpu().numpy()
  name = [*bpm.keys()]
  pair = [f'{name[i]}-{name[i+1]}' for i in range(len(bpm)-1)]
  pair.append(f'{name[-1]}-{name[0]}')

# Generate PV names
prefix = args.prefix if not args.data else args.data
pv_list = [pv_make(name, args.plane, prefix=prefix) for name in bpm]
pv_rise = [*bpm.values()]
if args.verbose:
  print('PV list:')
  for pv in pv_list:
    print(pv)

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
shift = 0
if args.rise:
  shift = min(pv_rise)
  if shift < 0:
    exit('error: rise values are expected to be positive')
  shift = max(pv_rise)
  if length + offset + shift > LIMIT:
    exit(f'error: sum of {length=}, {offset=} and max {shift=}, expected to be less than {LIMIT=}')

# Check sample length
if args.length > length:
  exit(f'error: requested sample length {args.length} should be less than {length}')

# Check window order
if args.window < 0.0:
  exit(f'error: window order {args.window} should be greater or equal to zero')

# Load TbT data
size = len(bpm)
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
if not args.dht:
  dec = Decomposition(tbt)
  value, error, table = dec.harmonic_phase(
    frequency_value,
    length=args.length,
    order=args.window,
    window='cosine_window',
    error=args.error,
    limit=args.limit,
    sigma_frequency=frequency_error,
    shift=args.shift,
    count=args.size,
    step=args.delta,
    clean=True,
    factor=5.0,
    method=args.method)
else:
  dht = Frequency.dht(tbt.work[:, :args.length])
  table = dht.angle()
  table -= 2.0*numpy.pi*frequency_value*torch.linspace(0, args.length - 1, args.length, dtype=dtype, device=device)
  table = mod(table, 2.0*numpy.pi, -numpy.pi)
  value = table[:, +args.drop:-args.drop].mean(-1)
  error = table[:, +args.drop:-args.drop].std(-1)

# Convert phase data to numpy
output = value.cpu().numpy()
error_output = error.cpu().numpy() if error is not None else torch.zeros_like(value).cpu().numpy()

# Plot
if args.plot:
  from pandas import DataFrame
  from pandas import concat
  from plotly.express import scatter
  mode = f'{plane}, FREQUENCY={target}' if args.coupled else f'{plane}'
  if table is not None:
    step = len(table[0])
    df = DataFrame()
    for i, name in enumerate(bpm):
      df = concat([df, DataFrame({'STEP':range(1, step + 1), 'BPM':name, 'PHASE':table[i].cpu().numpy()})])
    plot = scatter(df, x='STEP', y='PHASE', color='BPM', title=f'{TIME}: PHASE ({mode})', opacity=0.75, marginal_y='box')
    config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
    plot.show(config=config)
  df = DataFrame()
  df['BPM'] = [*bpm.keys()]
  df['PHASE'] = output
  df['ERROR'] = error_output
  title = f'{TIME}: PHASE ({mode})'
  plot = scatter(df, x='BPM', y='PHASE', title=title, opacity=0.75, error_y='ERROR', color_discrete_sequence = ['blue'], hover_data=['BPM', 'PHASE', 'ERROR'])
  plot.update_traces(mode='lines+markers', line={'width': 1.5}, marker={'size': 5})
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)
  if args.monotonic:
    phase_table = output.copy()
    phase_start, *_ = phase_table
    phase_table = phase_table - phase_start
    for i in range(1, len(phase_table)):
      while phase_table[i] < phase_table[i - 1]:
        phase_table[i] += 2.0*numpy.pi
    *_, phase_final = phase_table
    tune_integer = round(phase_final/(2.0*numpy.pi))
    df = DataFrame()
    df['BPM'] = [*bpm.keys()]
    df['PHASE'] = phase_table
    title = f'{TIME}: MONOTONIC PHASE ({mode}), INTEGER={tune_integer}'
    plot = scatter(df, x='BPM', y='PHASE', title=title, color_discrete_sequence=['blue'], hover_data=['BPM', 'PHASE'])
    plot.update_traces(mode='lines+markers', line={'width': 1.5}, marker={'size': 5})
    plot.show(config=config)
  if args.advance:
    total = cs.get(f'{args.prefix}:FREQUENCY:VALUE:{target}')
    phase, sigma = Decomposition.phase_adjacent(total, value, sigma_phase=error)
    phase = phase.cpu().numpy()
    sigma = sigma.cpu().numpy()
    tune_model = numpy.sum(model_numpy)/(2.0*numpy.pi)
    tune_phase = numpy.sum(phase)/(2.0*numpy.pi)
    df = DataFrame()
    for case, adv, std in zip(['MODEL', 'PHASE'], [model_numpy, phase], [numpy.zeros_like(sigma), sigma]):
      df = concat([df, DataFrame({'CASE':case, 'PAIR':pair, 'ADVANCE':adv, 'SIGMA':std})])
    title = f'{TIME}: ADJACENT ADVANCE ({mode}), MODEL={tune_model:.9f}, PHASE={tune_phase:.9f}'
    plot = scatter(df, x='PAIR', y='ADVANCE', color='CASE', error_y='SIGMA', title=title, color_discrete_sequence = ['blue', 'red'])
    plot.update_traces(marker={'size': 5})
    config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
    plot.show(config=config)
    df = DataFrame()
    advance_error = 100*(model_numpy - phase)/model_numpy
    sigma_error = 100*sigma/model_numpy
    rms_error = numpy.sqrt(numpy.mean(advance_error**2))
    df['PAIR'] = pair
    df['ERROR'] = advance_error
    df['SIGMA'] = sigma_error
    title = f'{TIME}: ADJACENT ADVANCE ERROR ({mode}), RMS={rms_error:.6g}%'
    plot = scatter(df, x='PAIR', y='ERROR', error_y='SIGMA', title=title, color_discrete_sequence = ['blue'])
    plot.update_traces(marker={'size': 5})
    config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
    plot.show(config=config)

# Save to file
if args.save:
  filename = f'{"phase_coupled" if args.coupled else "phase"}_plane_{args.plane}_length_{args.length}_time_({TIME}).npy'
  numpy.save(filename, output)

# Save to cs
if args.update and args.coupled:
  exit('error: --update with --coupled is not supported for phase data')

if args.update and not args.coupled:
  for name, phase_value, sigma in zip(bpm, output, error_output):
    cs.set(f'{args.prefix}:{name}:PHASE:VALUE:{plane}', phase_value)
    cs.set(f'{args.prefix}:{name}:PHASE:ERROR:{plane}', sigma)
