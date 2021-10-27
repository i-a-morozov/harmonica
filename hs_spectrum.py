"""
usage: hs_spectrum [-h] [-p {x,z,i}] [-l LENGTH] [-s BPM [BPM ...] | -o BPM [BPM ...]] [-r] [-f] [-c | -n] [--print] [--mean | --normalize] [-w]
                   [--name {cosine_window,kaiser_window}] [--order ORDER] [--pad PAD] [--min MIN] [--max MAX] [--log] [--flip] [--plot] [--map] [--average]
                   [--device {cpu,cuda}] [--dtype {float32,float64}] [--test]

Print/save/plot amplitude spectrum data for selected plane and BPMs.

optional arguments:
  -h, --help            show this help message and exit
  -p {x,z,i}, --plane {x,z,i}
                        data plane
  -l LENGTH, --length LENGTH
                        number of turns to use (integer)
  -s BPM [BPM ...], --skip BPM [BPM ...]
                        space separated list of valid BPM names to skip
  -o BPM [BPM ...], --only BPM [BPM ...]
                        space separated list of valid BPM names to use
  -r, --rise            flag to use rise data (drop first turns)
  -f, --file            flag to save data
  -c, --csv             flag to save data as CSV
  -n, --numpy           flag to save data as NUMPY
  --print               flag to print data
  --mean                flag to remove mean
  --normalize           flag to normalize data
  -w, --window          flag to apply window
  --name {cosine_window,kaiser_window}
                        window type
  --order ORDER         window order parameter (float >= 0.0)
  --pad PAD             number of zeros to pad (integer)
  --min MIN             min frequency value (float)
  --max MAX             max frequency value (float)
  --log                 flag to apply log10 to amplitude spectra
  --flip                flag to flip spectra around 1/2
  --plot                flag to plot data
  --map                 flag to plot heat map
  --average             flag to plot average spectrum
  --device {cpu,cuda}   data device
  --dtype {float32,float64}
                        data type
  --test                flag to use test PV names
"""

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_spectrum', description='Print/save/plot amplitude spectrum data for selected plane and BPMs.')
parser.add_argument('-p', '--plane', choices=('x', 'z', 'i'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use (integer)', default=1024)
select = parser.add_mutually_exclusive_group()
select.add_argument('-s', '--skip', metavar='BPM', nargs='+', help='space separated list of valid BPM names to skip')
select.add_argument('-o', '--only', metavar='BPM', nargs='+', help='space separated list of valid BPM names to use')
parser.add_argument('-r', '--rise', action='store_true', help='flag to use rise data (drop first turns)')
parser.add_argument('-f', '--file', action='store_true', help='flag to save data')
save = parser.add_mutually_exclusive_group()
save.add_argument('-c', '--csv', action='store_true', help='flag to save data as CSV')
save.add_argument('-n', '--numpy', action='store_true', help='flag to save data as NUMPY')
parser.add_argument('--print', action='store_true', help='flag to print data')
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--mean', action='store_true', help='flag to remove mean')
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
parser.add_argument('-w', '--window', action='store_true', help='flag to apply window')
parser.add_argument('--name', choices=('cosine_window', 'kaiser_window'), help='window type', default='cosine_window')
parser.add_argument('--order', type=float, help='window order parameter (float >= 0.0)', default=1.0)
parser.add_argument('--pad', type=int, help='number of zeros to pad (integer)', default=0)
parser.add_argument('--min', type=float, help='min frequency value (float)', default=0.0)
parser.add_argument('--max', type=float, help='max frequency value (float)', default=0.5)
parser.add_argument('--log', action='store_true', help='flag to apply log10 to amplitude spectra')
parser.add_argument('--flip', action='store_true', help='flag to flip spectra around 1/2')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--map', action='store_true', help='flag to plot heat map')
parser.add_argument('--average', action='store_true', help='flag to plot average spectrum')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float64')
parser.add_argument('--test', action='store_true', help='flag to use test PV names')
args = parser.parse_args()

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

# Check and set frequency range & padding
f_min = args.min
f_max = args.max
if not (0.0 <= f_min <= 0.5):
  exit(f'error: MIN = {f_min}, MIN should be in (0.0, 0.5)')
if not (0.0 <= f_max <= 0.5):
  exit(f'error: MAX = {f_max}, MAX should be in (0.0, 0.5)')
if f_max < f_min:
  exit(f'error: (MIN, MAX) = {f_min, f_max}, MAX should be greater than MIN')
if (f_min, f_max) != (0.0, 0.5) and args.pad != 0:
  exit(f'error: (MIN, MAX) = {f_min, f_max} should be used without padding')

# Import BPM data
try:
  df = pandas.read_json('bpm.json')
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
pv_list = [pv_make(name, args.plane, args.test) for name in bpm]
pv_rise = [*bpm.values()]

# Load TbT data
size = len(bpm)
length = min(args.length, LIMIT)
count = min(length + max(pv_rise) if args.rise else length, LIMIT)
win = Window(length, name=args.name, order=args.order, dtype=dtype, device=device)
tbt = Data.from_epics(size, win, pv_list, pv_rise if args.rise else None, count=count)

# Remove mean
if args.mean:
  tbt.window_remove_mean()

# Normalize
if args.normalize:
  tbt.normalize(window=True)

# Set Frequency instance
f = Frequency(tbt, pad=args.pad)

# Apply window
if args.window:
  f.data.window_remove_mean()
  f.data.window_apply()

# Compute FFT spectrum
f.fft_get_spectrum()
grid = f.fft_grid
data = f.fft_spectrum

# Compute FFRFT spectrum
if (f_min, f_max) != (0.0, 0.5):
  span = f_max - f_min
  center = f_min + 0.5*span
  f.ffrft_get_spectrum(center=center, span=span)
  grid = f.ffrft_get_grid()
  data = f.ffrft_spectrum

# Convert to numpy
grid = grid.cpu().numpy()[1:-1]
data = data.cpu().numpy()[:, 1:-1]

# Mean spectrum
if args.average:
  f('ffrft', window=args.window)
  mean_grid, mean_data = f.task_mean_spectrum(window=args.window, log=args.log)
  mean_grid = mean_grid.cpu().numpy()[1:-1]
  mean_data = mean_data.cpu().numpy()[1:-1]

# Clean
del win, tbt, f
if device == 'cuda':
  torch.cuda.empty_cache()

# Flip
if args.flip:
  grid = 1.0 - grid[::-1]
  data = data[:, ::-1]
  if args.average:
    mean_grid = 1.0 - mean_grid[::-1]
    mean_data = mean_data[::-1]

# Scale
if args.log:
  data = numpy.log10(data)

# Plot
if args.plot:
  df = pandas.DataFrame()
  for i, name in enumerate(bpm):
    df = pandas.concat([df, pandas.DataFrame({'frequency':grid, 'bpm':name, f"dtft({args.plane})":data[i]})])
  from plotly.express import scatter
  plot = scatter(df, x='frequency', y=f"dtft({args.plane})", color='bpm', title=TIME, opacity=0.75)
  plot.show()
  if args.map:
    from plotly.express import imshow
    plot = imshow(data, labels=dict(x="frequency", y="bpm", color=f"dtft({args.plane})"), x=grid, y=[*bpm.keys()], aspect=0.5, title=TIME)
    plot.show()
  if args.average:
    plot = scatter(x=mean_grid, y=mean_data, title=TIME)
    plot.show()

# Print data
if args.print:
  fmt = '{:>6}' + '{:>18.12}' * len(bpm)
  print(fmt.format('FREQUENCY', *bpm))
  for i in range(len(grid)):
    print(fmt.format(grid[i], *data[:, i]))

# Save to file
if args.file and args.numpy:
  numpy.save(f'spectrum_{TIME}.npy', data)
if args.file and args.csv:
  numpy.savetxt(f'spetrum_{TIME}.csv', data.transpose(), delimiter=',')