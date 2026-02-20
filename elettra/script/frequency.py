#!/bin/env python

# Import
import sys
import argparse
import epics
import numpy
import pandas
import torch
from datetime import datetime
from harmonica.util import LIMIT, pv_make, bpm_select
from harmonica.statistics import weighted_mean, weighted_variance
from harmonica.statistics import median, biweight_midvariance, standardize
from harmonica.anomaly import threshold
from harmonica.window import Window
from harmonica.data import Data
from harmonica.filter import Filter
from harmonica.frequency import Frequency

# Input arguments flag
_, *flag = sys.argv

# Parse arguments
parser = argparse.ArgumentParser(prog='hs_frequency', description='Save/plot frequency data for selected BPMs and plane.')
parser.add_argument('-p', '--plane', choices=('x', 'y'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use', default=1024)
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
    exit('error: rise values are expected to be positive')
  rise = max(pv_rise)
  if length + offset + rise > LIMIT:
    exit(f'error: sum of {length=}, {offset=} and max {rise=}, expected to be less than {LIMIT=}')
else:
  rise = 0

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
  flt = Filter(tbt)
  _, noise = flt.estimate_noise(limit=args.limit)

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

# Apply window
if args.window:
  f.data.window_remove_mean()
  f.data.window_apply()

# Compute frequencies
f(args.method, f_range=(f_min, f_max))

# Clean
if args.clean:
  data = standardize(f.frequency, center_estimator=median, spread_estimator=biweight_midvariance)
  factor = torch.tensor(args.factor, dtype=dtype, device=device)
  mask = threshold(data, -factor, +factor).squeeze(0)
else:
  mask = torch.ones_like(f.frequency)

# Process (none)
if args.process == 'none':
  center = weighted_mean(f.frequency, weight=mask)
  spread = weighted_variance(f.frequency, weight=mask, center=center).sqrt()

# Process (noise)
if args.process == 'noise':
  factor = torch.stack([spectrum[index] for index, spectrum in zip(f.ffrft_bin.to(torch.int64), f.ffrft_spectrum)])**2
  weight = mask*factor/noise**2
  weight = weight/weight.sum(-1)
  center = weighted_mean(f.frequency, weight=weight)
  spread = weighted_variance(f.frequency, weight=weight, center=center).sqrt()

# Convert to numpy
frequency = f.frequency.cpu().numpy()
center = center.cpu().numpy()
spread = spread.cpu().numpy()

# Flip
if args.flip:
  frequency = 1.0 - frequency
  center = 1.0 - center

# Plot
if args.plot:
  mask = mask.to(torch.bool).logical_not().cpu().numpy()
  df = pandas.DataFrame()
  df['BPM'] = [*bpm.keys()]
  df['FREQUENCY'] = frequency
  df['WEIGHT'] = (~mask).astype(numpy.float64) if args.process == 'none' else weight.cpu().numpy()
  from plotly.express import scatter
  title = f'{TIME}: FREQUENCY ({frequency.mean():12.9}, {frequency.std():12.9})<br>CENTER: {center:12.9}, SPREAD: {spread:12.9}'
  plot = scatter(df, x='BPM', y='FREQUENCY', title=title, opacity=0.75, marginal_y='box', color_discrete_sequence = ['blue'],  hover_data=['BPM', 'FREQUENCY', 'WEIGHT'], symbol_sequence=['circle'])
  plot.add_hline(center - spread, line_color='black', line_dash="dash", line_width=1.0)
  plot.add_hline(center, line_color='black', line_dash="dash", line_width=1.0)
  plot.add_hline(center + spread, line_color='black', line_dash="dash", line_width=1.0)
  plot.update_traces(marker={'size': 10})
  if mask.sum() != 0:
    mask = pandas.DataFrame({'BPM':df.BPM[mask], 'FREQUENCY':df.FREQUENCY[mask], 'WEIGHT':df.WEIGHT[mask]})
    mask = scatter(mask, x='BPM', y='FREQUENCY', color_discrete_sequence = ['red'], hover_data=['BPM', 'FREQUENCY', 'WEIGHT'], symbol_sequence=['circle'])
    mask.update_traces(marker={'size': 10})
    mask, *_ = mask.data
    plot.add_trace(mask)
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)

# Save to file
if args.save:
  filename = f'frequency_plane_{args.plane}_length_{args.length}_time_{TIME}.npy'
  numpy.save(filename, frequency)

# Save to epics
if args.update:
  plane = args.plane.upper()
  epics.caput(f'H:FREQUENCY:VALUE:{plane}', center)
  epics.caput(f'H:FREQUENCY:ERROR:{plane}', spread)
  epics.caput_many([f'H:{name}:FREQUENCY:VALUE:{plane}' for name in bpm], frequency)
  epics.caput_many([f'H:{name}:FREQUENCY:ERROR:{plane}' for name in bpm], numpy.zeros(frequency.shape))
