# Input arguments flag
import sys
sys.path.append('..')
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_frequency', description='Print/save/plot frequency data for selected plane and BPMs.')
parser.add_argument('-p', '--plane', choices=('x', 'z'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use (integer)', default=1024)
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
parser.add_argument('-w', '--window', action='store_true', help='flag to apply window')
parser.add_argument('--name', choices=('cosine_window', 'kaiser_window'), help='window type', default='cosine_window')
parser.add_argument('--order', type=float, help='window order parameter (float >= 0.0)', default=1.0)
parser.add_argument('--pad', type=int, help='number of zeros to pad (integer)', default=0)
parser.add_argument('--f_min', type=float, help='min frequency value (float)', default=0.0)
parser.add_argument('--f_max', type=float, help='max frequency value (float)', default=0.5)
parser.add_argument('-m', '--method', choices=('fft', 'ffrft', 'parabola'), help='frequency estimation method', default='parabola')
parser.add_argument('--fit', choices=('none', 'noise', 'ols', 'wls'), help='fit type', default='none')
parser.add_argument('--fraction', type=float, help='fraction of points near spectum peak', default=0.05)
parser.add_argument('--limit', type=int, help='number of columns to use for noise estimation', default=32)
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
from harmonica.filter import Filter

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit(f'error: CUDA is not avalible')

# Check and set frequency range
f_min = args.f_min
f_max = args.f_max
if not (0.0 <= f_min <= 0.5):
  exit(f'error: {f_min=}, should be in (0.0, 0.5)')
if not (0.0 <= f_max <= 0.5):
  exit(f'error: {f_max=}, should be in (0.0, 0.5)')
if f_max < f_min:
  exit(f'error: {f_max=} should be greater than {f_min=}')

# Import BPM data
try:
  df = pandas.read_json('../bpm.json')
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
win = Window(length, args.name, args.order, dtype=dtype, device=device)
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

# Estimate noise
if args.fit == 'noise' or args.fit == 'wls':
  f = Filter(tbt)
  _, noise = f.estimate_noise(limit=args.limit)
else:
  noise = None

# Set Frequency instance
f = Frequency(tbt, pad=args.pad)

# Apply window
if args.window:
  f.data.window_remove_mean()
  f.data.window_apply()

# Compute frequencies
f(args.method, f_range=(f_min, f_max))

# Convert to numpy
frequency = f.frequency.cpu().numpy()

# Fit (noise)
if args.fit == 'noise':
  from statsmodels.api import WLS
  x = numpy.ones((len(bpm), 1))
  y = frequency
  w = (1/noise**2).cpu().numpy()
  out = WLS(y, x, w).fit()
  f_fit = out.params.item()
  s_fit = out.bse.item()

# Fit (ols/wls)
if args.fit == 'ols' or args.fit == 'wls':
  fraction = args.fraction
  if fraction >= 1.0 or fraction <= 0.0:
    exit(f'error: fraction is expected to br in (0, 1)')
  from statsmodels.api import WLS
  m, s = f.task_fit(size=int(fraction*length), mode=args.fit, std=noise).T
  x = numpy.ones((len(bpm), 1))
  y = m.cpu().numpy()
  w = 1/s.cpu().numpy()**2
  out = WLS(y, x, w).fit()
  f_fit, *_ = out.params
  s_fit, *_ = out.bse

# Clean
del win, tbt, f
if device == 'cuda':
  torch.cuda.empty_cache()

# Flip
if args.flip:
  frequency = 1.0 - frequency
  if args.fit != 'none':
    f_fit = 1.0 - f_fit

# Plot
if args.plot:
  df = pandas.DataFrame()
  df['bpm'] = [*bpm.keys()]
  df['frequency'] = frequency
  from plotly.express import scatter
  mean = numpy.mean(frequency)
  median = numpy.median(frequency)
  std = numpy.std(frequency)
  title = f'{TIME}: Frequency<br>MEAN: {mean}, MEDIAN: {median}, SIGMA: {std}'
  title = title if not args.fit != 'none' else f'{title}<br>FIT={args.fit.upper()}, VALUE={f_fit}, ERROR={s_fit}'
  if args.fit == 'none':
    plot = scatter(df, x='bpm', y='frequency', title=title, opacity=0.75, marginal_y='box')
    plot.add_hline(mean - std, line_color='black', line_dash="dash", line_width=0.5)
    plot.add_hline(mean, line_color='black', line_dash="dash", line_width=0.5)
    plot.add_hline(mean + std, line_color='black', line_dash="dash", line_width=0.5)
  if args.fit == 'noise':
    df['weight'] = w
    plot = scatter(df, x='bpm', y='frequency', title=title, opacity=0.75, marginal_y='box', hover_data=['bpm', 'frequency', 'weight'])
    plot.add_hline(mean - std, line_color='black', line_dash="dash", line_width=0.5)
    plot.add_hline(mean, line_color='black', line_dash="dash", line_width=0.5)
    plot.add_hline(mean + std, line_color='black', line_dash="dash", line_width=0.5)
  if args.fit == 'ols' or args.fit == 'wls':
    m, s = m.cpu().numpy(), s.cpu().numpy()
    m = 1.0 - m if args.flip else m
    df['f_fit'] = m
    df['s_fit'] = s
    plot = scatter(df, x='bpm', y='f_fit', title=title, opacity=0.75, marginal_y='box', error_y='s_fit')
    plot.add_hline(f_fit - s_fit, line_color='red', line_dash="dash", line_width=0.5)
    plot.add_hline(f_fit, line_color='red', line_dash="dash", line_width=0.5)
    plot.add_hline(f_fit + s_fit, line_color='red', line_dash="dash", line_width=0.5)
  config = {
    'toImageButtonOptions': {'height':None, 'width':None},
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'],
    'scrollZoom': True
  }
  plot.show(config=config)

# Print data
if args.print:
  fmt = '{:>6}' + '{:>18.12}'
  print(fmt.format('BPM', 'FREQUENCY'))
  for name, value in zip(bpm, frequency):
    print(fmt.format(name, value))

# Save to file
if args.file and args.numpy:
  filename = f'frequency_plane_{args.plane}_length_{args.length}_time_{TIME}.npy'
  numpy.save(filename, frequency)
if args.file and args.csv:
  filename = f'frequency_plane_{args.plane}_length_{args.length}_time_{TIME}.csv'
  numpy.savetxt(filename, frequency, delimiter=',')