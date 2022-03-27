# Input arguments flag
import sys
sys.path.append('..')
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_spectrum', description='Save/plot amplitude spectrum data for selected BPMs and plane.')
parser.add_argument('-p', '--plane', choices=('x', 'y'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use', default=1024)
select = parser.add_mutually_exclusive_group()
select.add_argument('--skip', metavar='BPM', nargs='+', help='space separated list of valid BPM names to skip')
select.add_argument('--only', metavar='BPM', nargs='+', help='space separated list of valid BPM names to use')
parser.add_argument('-o', '--offset', type=int, help='rise offset for all BPMs', default=0)
parser.add_argument('-r', '--rise', action='store_true', help='flag to use rise data (drop first turns)')
parser.add_argument('-s', '--save', action='store_true', help='flag to save data as numpy array')
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--mean', action='store_true', help='flag to remove mean')
transform.add_argument('--median', action='store_true', help='flag to remove median')
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
parser.add_argument('-f', '--filter', choices=('none', 'svd', 'hankel'), help='filter type', default='none')
parser.add_argument('--rank', type=int, help='rank to use for svd & hankel filter', default=8)
parser.add_argument('--type', choices=('full', 'randomized'), help='SVD computation type for hankel filter', default='randomized')
parser.add_argument('--buffer', type=int, help='buffer size to use for randomized hankel filter', default=16)
parser.add_argument('--count', type=int, help='number of iterations to use for randomized hankel filter', default=16)
parser.add_argument('-w', '--window', type=float, help='window order', default=0.0)
parser.add_argument('--pad', type=int, help='number of zeros to pad', default=0)
parser.add_argument('--f_min', type=float, help='min frequency value', default=0.0)
parser.add_argument('--f_max', type=float, help='max frequency value', default=0.5)
parser.add_argument('--log', action='store_true', help='flag to apply log10 to amplitude spectra')
parser.add_argument('--flip', action='store_true', help='flag to flip spectra around 1/2')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--map', action='store_true', help='flag to plot heat map')
parser.add_argument('--average', action='store_true', help='flag to plot average spectrum')
parser.add_argument('--peaks', type=int, help='number of peaks to find in average spectrum', default=1)
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
from harmonica.filter import Filter
from harmonica.frequency import Frequency

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit(f'error: CUDA is not avalible')

# Check and set frequency range & padding
f_min = args.f_min
f_max = args.f_max
if not (0.0 <= f_min <= 0.5):
  exit(f'error: {f_min=}, should be in (0.0, 0.5)')
if not (0.0 <= f_max <= 0.5):
  exit(f'error: {f_max=}, should be in (0.0, 0.5)')
if f_max < f_min:
  exit(f'error: {f_max=} should be greater than {f_min=}')
if (f_min, f_max) != (0.0, 0.5) and args.pad != 0:
  exit(f'error: (f_min, f_max) should be used without padding')

# Load monitor data
name = epics.caget('H:MONITOR:LIST')[:epics.caget('H:MONITOR:COUNT')]
flag = epics.caget_many([f'H:{name}:FLAG' for name in name])
rise = epics.caget_many([f'H:{name}:RISE' for name in name])

# Set BPM data
bpm = {name: rise for name, flag, rise in zip(name, flag, rise) if flag == 1}

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

# Check window order
if args.window < 0.0:
  exit(f'error: window order {args.window} should be greater or equal to zero')

# Load TbT data
size = len(bpm)
count = length + offset + rise
win = Window(length, 'cosine_window', args.window, dtype=dtype, device=device)
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
f = Frequency(tbt, pad=args.pad)

# Apply window
if args.window > 0.0:
  f.data.window_remove_mean()
  f.data.window_apply()

# Compute FFT spectrum
f.fft_get_spectrum()
grid = f.fft_grid
data = f.fft_spectrum

# Compute FFRFT spectrum
if (f_min, f_max) != (0.0, 0.5):
  span = (f_max - f_min)
  center = f_min + 0.5*span
  f.ffrft_get_spectrum(center=center, span=span)
  grid = f.ffrft_get_grid()
  data = f.ffrft_spectrum

# Convert to numpy
grid = grid.detach().cpu().numpy()[1:]
data = data.detach().cpu().numpy()[:, 1:]

# Mean spectrum
if args.average:
  f('ffrft')
  mean_grid, mean_data = f.compute_mean_spectrum(log=args.log)
  mean_grid = mean_grid.cpu().numpy()[1:]
  mean_data = mean_data.cpu().numpy()[1:]

# Flip
if args.flip:
  grid = 1.0 - grid[::-1]
  data = data[:, ::-1]
  if args.average:
    mean_grid = 1.0 - mean_grid[::-1]
    mean_data = mean_data[::-1]

# Scale
if args.log:
  data = numpy.log10(data + 1.0E-12)

# Peaks
if args.average and args.peaks > 0:
  from scipy.signal import find_peaks
  peak, *_ = find_peaks(mean_data)
  peak = numpy.array([mean_grid[peak], mean_data[peak]]).T
  peak_grid, peak_data = numpy.array(sorted(peak, key=lambda x: x[1], reverse=True)[:args.peaks]).T

# Plot
if args.plot:
  df = pandas.DataFrame()
  for i, name in enumerate(bpm):
    df = pandas.concat([df, pandas.DataFrame({'FREQUENCY':grid, 'BPM':name, f'DTFT({args.plane.upper()})':data[i]})])
  from plotly.express import scatter
  plot = scatter(df, x='FREQUENCY', y=f'DTFT({args.plane.upper()})', color='BPM', title=f'{TIME}: SPECTRUM', opacity=0.75)
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)
  if args.map:
    from plotly.express import imshow
    plot = imshow(data, labels=dict(x='FREQUENCY', y='BPM', color=f'DTFT({args.plane.upper()})'), x=grid, y=[*bpm.keys()], aspect=0.5, title=f'{TIME}: SPECTRUM (MAP)')
    plot.show()
  if args.average:
    df = pandas.DataFrame()
    df['FREQUENCY'] = mean_grid
    df[f'DTFT({args.plane.upper()})'] = mean_data
    plot = scatter(df, x='FREQUENCY', y=f'DTFT({args.plane.upper()})', title=f'{TIME}: SPECTRUM (AVERAGE)')
    if args.peaks > 0:
      plot.add_scatter(x=peak_grid, y=peak_data, mode='markers', marker=dict(color='red', size=10), showlegend=False, name='PEAK')
    config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
    plot.show(config=config)

# Save to file
if args.save:
  filename = f'spectrum_plane_{args.plane}_length_{args.length}_time_{TIME}.npy'
  numpy.save(filename, data)