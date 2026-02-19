#!/bin/env python

# Import
import sys
sys.path.append('../..')
import argparse
import epics
import numpy
import pandas
import torch
from datetime import datetime
from harmonica.util import LIMIT, pv_make, bpm_select
from harmonica.statistics import weighted_mean, weighted_variance
from harmonica.statistics import median, biweight_midvariance, standardize, rescale
from harmonica.anomaly import threshold, score
from harmonica.window import Window
from harmonica.data import Data
from harmonica.filter import Filter
from harmonica.frequency import Frequency

# Input arguments flag
_, *flag = sys.argv

# Parse arguments
parser = argparse.ArgumentParser(prog='frequency_shift', description='Save/plot frequency data for selected plane and BPMs using shifted samples.')
parser.add_argument('-p', '--plane', choices=('x', 'y'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use', default=512)
parser.add_argument('--load', type=int, help='total number of turns to load', default=2048)
parser.add_argument('--shift', type=int, help='shift step', default=8)
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
parser.add_argument('--pad', type=int, help='number of zeros to pad on each side', default=0)
parser.add_argument('--f_min', type=float, help='min frequency value', default=0.0)
parser.add_argument('--f_max', type=float, help='max frequency value', default=0.5)
parser.add_argument('-m', '--method', choices=('fft', 'ffrft', 'parabola'), help='frequency estimation method', default='parabola')
parser.add_argument('--clean', action='store_true', help='flag to clean frequency data')
parser.add_argument('--factor', type=float, help='threshold factor', default=5.0)
parser.add_argument('--process', choices=('none', 'noise'), help='processing type', default='none')
parser.add_argument('--limit', type=int, help='number of columns to use for noise estimation', default=32)
parser.add_argument('--flip', action='store_true', help='flag to flip frequency around 1/2')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--noise', action='store_true', help='flag to plot noise data')
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

# Check and set frequency range & padding
f_min = args.f_min
f_max = args.f_max
if not (0.0 <= f_min <= 0.5):
  exit(f'error: {f_min=}, should be in (0.0, 0.5)')
if not (0.0 <= f_max <= 0.5):
  exit(f'error: {f_max=}, should be in (0.0, 0.5)')
if f_max < f_min:
  exit(f'error: {f_max=} should be greater than {f_min=}')
if args.pad < 0:
  exit(f'error: {args.pad=}, should be positive')
if (f_min, f_max) != (0.0, 0.5) and args.pad != 0:
  exit('error: (f_min, f_max) should be used without padding')

# Load monitor data
name = epics.caget('H:MONITOR:LIST')[:epics.caget('H:MONITOR:COUNT')]
flag = epics.caget_many([f'H:{name}:FLAG' for name in name])
rise = epics.caget_many([f'H:{name}:RISE' for name in name])

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

# Estimate noise
if args.process == 'noise':
  table = []
  win = Window(args.length)
  for signal in tbt.work:
    matrix = Data.from_data(win, Data.make_matrix(args.length, args.shift, signal))
    f = Filter(matrix)
    _, noise = f.estimate_noise(limit=args.limit)
    table.append(noise)
  noise = torch.stack(table)

# Filter (svd)
if args.filter == 'svd':
  flt = Filter(tbt)
  flt.filter_svd(rank=args.rank)

# Filter (hankel)
if args.filter == 'hankel':
  flt = Filter(tbt)
  flt.filter_svd(rank=args.rank)
  flt.filter_hankel(rank=args.rank, random=args.type == 'randomized', buffer=args.buffer, count=args.count)

# Set Frequency instance
f = Frequency(tbt, pad=length + 2*args.pad)

# Compute shifted frequencies
frequency = f.compute_shifted_frequency(
  args.length,
  args.shift,
  method=args.method,
  name='cosine_window',
  order=args.window,
  f_range=(f_min, f_max)
)
_, step = frequency.shape

# Clean
if args.clean:
  data = standardize(frequency.flatten(), center_estimator=median, spread_estimator=biweight_midvariance)
  factor = torch.tensor(args.factor, dtype=dtype, device=device)
  mask = threshold(data, -factor, +factor).reshape_as(frequency)
  mark = 0.5 >= rescale(score(tbt.size, mask.flatten()).to(dtype), scale_min=0.0, scale_max=1.0).nan_to_num()
else:
  mask = torch.ones_like(frequency)
  mark = torch.ones(tbt.size, dtype=dtype, device=device)

# Process (none)
if args.process == 'none':
  signal_center = weighted_mean(frequency, weight=mask)
  signal_spread = weighted_variance(frequency, weight=mask, center=signal_center).sqrt()
  weight = mark/signal_spread**2
  center = weighted_mean(signal_center, weight=weight)
  spread = weighted_variance(signal_center, weight=weight, center=center).sqrt()

# Process (noise)
if args.process == 'noise':
  weight = mask/noise**2
  weight = weight/weight.sum(-1, keepdim=True)
  signal_center = weighted_mean(frequency, weight=weight)
  signal_spread = weighted_variance(frequency, weight=weight, center=signal_center).sqrt()
  weight = mark/signal_spread**2
  center = weighted_mean(signal_center, weight=weight)
  spread = weighted_variance(signal_center, weight=weight, center=center).sqrt()

# Flip
if args.flip:
  frequency = 1.0 - frequency
  signal_center = 1.0 - signal_center
  center = 1.0 - center

# Convert to numpy
frequency = frequency.cpu().numpy()
center, spread = center.cpu().numpy(), spread.cpu().numpy()
signal_center, signal_spread = signal_center.cpu().numpy(), signal_spread.cpu().numpy()

# Plot
if args.plot:
  df = pandas.DataFrame()
  for i, name in enumerate(bpm):
    df = pandas.concat([df, pandas.DataFrame({'STEP':range(1, step + 1), 'BPM':name, 'FREQUENCY':frequency[i]})])
  from plotly.express import scatter
  title = f'{TIME}: FREQUENCY (SHIFTED) ({frequency.mean():12.9}, {frequency.std():12.9})<br>SAMPLE: {args.length}, SHIFT: {args.shift}, COUNT: {step}'
  title = f'{title}<br>CENTER: {center:12.9}, SPREAD: {spread:12.9}'
  plot = scatter(df, x='STEP', y='FREQUENCY', color='BPM', title=title, opacity=0.75, marginal_y='box')
  plot.add_hline(center - spread, line_color='black', line_dash='dash', line_width=1.0)
  plot.add_hline(center, line_color='black', line_dash='dash', line_width=1.0)
  plot.add_hline(center + spread, line_color='black', line_dash='dash', line_width=1.0)
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)
  mark = mark.to(torch.bool).logical_not().cpu().numpy()
  df = pandas.DataFrame()
  df['BPM'] = [*bpm.keys()]
  df['FREQUENCY'] = signal_center
  df['FLAG'] = (~mark).astype(numpy.float64)
  df['ERROR'] = signal_spread
  plot = scatter(df, x='BPM', y='FREQUENCY', error_y='ERROR', title=title, opacity=0.75, marginal_y='box', color_discrete_sequence = ['blue'],  hover_data=['BPM', 'FREQUENCY', 'ERROR', 'FLAG'], symbol_sequence=['circle'])
  plot.add_hline(center - spread, line_color='black', line_dash="dash", line_width=1.0)
  plot.add_hline(center, line_color='black', line_dash="dash", line_width=1.0)
  plot.add_hline(center + spread, line_color='black', line_dash="dash", line_width=1.0)
  plot.update_traces(marker={'size': 10})
  if mark.sum() != 0:
    mark = pandas.DataFrame({'BPM':df.BPM[mark], 'FREQUENCY':df.FREQUENCY[mark], 'FLAG':df.FLAG[mark], 'ERROR':df.ERROR[mark]})
    mark = scatter(mark, x='BPM', y='FREQUENCY', error_y='ERROR', title=title, opacity=0.75, marginal_y='box', color_discrete_sequence = ['red'],  hover_data=['BPM', 'FREQUENCY', 'ERROR', 'FLAG'], symbol_sequence=['circle'])
    mark.update_traces(marker={'size': 10})
    mark, *_ = mark.data
    plot.add_trace(mark)
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)
  if args.process == 'noise':
    df = pandas.DataFrame()
    for i, name in enumerate(bpm):
      df = pandas.concat([df, pandas.DataFrame({'STEP':range(1, step + 1), 'BPM':name, 'NOISE':noise[i]})])
    title = f'{TIME}: NOISE (SHIFTED)<br>SAMPLE: {args.length}, SHIFT: {args.shift}, COUNT: {step}'
    plot = scatter(df, x='STEP', y='NOISE', color='BPM', title=title, opacity=0.75)
    config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
    plot.show(config=config)

# Save to file
if args.save:
  filename = f'frequency_shifted_plane_{args.plane}_length_{args.length}_time_{TIME}.npy'
  numpy.save(filename, frequency)

# Save to epics
if args.update:
  plane = args.plane.upper()
  epics.caput(f'H:FREQUENCY:VALUE:{plane}', center)
  epics.caput(f'H:FREQUENCY:ERROR:{plane}', spread)
  epics.caput_many([f'H:{name}:FREQUENCY:VALUE:{plane}' for name in bpm], signal_center)
  epics.caput_many([f'H:{name}:FREQUENCY:ERROR:{plane}' for name in bpm], signal_spread)
