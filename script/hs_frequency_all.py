"""
usage: hs_frequency_all [-h] -p {x,z} [-l LENGTH] [--load LOAD] [--shift SHIFT] [--skip BPM [BPM ...] | --only BPM [BPM ...]] [-o OFFSET] [-r] [-f] [-c | -n] [--print]
                        [--mean | --median | --normalize] [-w] [--name {cosine_window,kaiser_window}] [--order ORDER] --f_min F_MIN --f_max F_MAX [--beta_min BETA_MIN]
                        [--beta_max BETA_MAX] [--nufft] [--time {position,phase}] [--plot] [--harmonica] [--device {cpu,cuda}] [--dtype {float32,float64}]

Compute mixed frequency for selected plane and BPMs with optional shifts.

optional arguments:
  -h, --help            show this help message and exit
  -p {x,z}, --plane {x,z}
                        data plane
  -l LENGTH, --length LENGTH
                        number of turns to use (integer)
  --load LOAD           number of turns to load (integer)
  --shift SHIFT         shift step (integer)
  --skip BPM [BPM ...]  space separated list of valid BPM names to skip
  --only BPM [BPM ...]  space separated list of valid BPM names to use
  -o OFFSET, --offset OFFSET
                        rise offset for all BPMs
  -r, --rise            flag to use rise data from file (drop first turns)
  -f, --file            flag to save data
  -c, --csv             flag to save data as CSV
  -n, --numpy           flag to save data as NUMPY
  --print               flag to print data
  --mean                flag to remove mean
  --median              flag to remove median
  --normalize           flag to normalize data
  -w, --window          flag to apply window
  --name {cosine_window,kaiser_window}
                        window type
  --order ORDER         window order parameter (float >= 0.0)
  --f_min F_MIN         min frequency value (float)
  --f_max F_MAX         max frequency value (float)
  --beta_min BETA_MIN   min beta threshold value for x or z
  --beta_max BETA_MAX   max beta threshold value for x or z
  --nufft               flag to compute spectum using TYPY-III NUFFT
  --time {position,phase}
                        time type to use with NUFFT
  --plot                flag to plot data
  --harmonica           flag to use harmonica PV names for input
  --device {cpu,cuda}   data device
  --dtype {float32,float64}
                        data type
"""

# Input arguments flag
import sys
sys.path.append('..')
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_frequency_all', description='Compute mixed frequency for selected plane and BPMs with optional shifts.')
parser.add_argument('-p', '--plane', choices=('x', 'z'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use (integer)', default=128)
parser.add_argument('--load', type=int, help='number of turns to load (integer)', default=128)
parser.add_argument('--shift', type=int, help='shift step (integer)', default=0)
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
parser.add_argument('--f_min', type=float, help='min frequency value (float)', default=0.0)
parser.add_argument('--f_max', type=float, help='max frequency value (float)', default=0.5)
parser.add_argument('--beta_min', type=float, help='min beta threshold value for x or z', default=0.0E+0)
parser.add_argument('--beta_max', type=float, help='max beta threshold value for x or z', default=1.0E+3)
parser.add_argument('--nufft', action='store_true', help='flag to compute spectum using TYPY-III NUFFT')
parser.add_argument('--time', choices=('position', 'phase'), help='time type to use with NUFFT', default='phase')
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
from harmonica.util import LIMIT, LENGTH, QX, QZ, pv_make
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
f_min = args.f_min
f_max = args.f_max
if f_min < 0.0:
  exit(f'error: {f_min=}, should be positive')
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

# Check beta values
if args.beta_min < 0:
  exit(f'error: min beta threshold {args.beta_min} should be positive')
if args.beta_max < 0:
  exit(f'error: max beta threshold {args.beta_max} should be positive')
if args.beta_min > args.beta_max:
  exit(f'error: max beta threshold {args.beta_max} should be greater than min beta threshold {args.beta_min}')

# Filter for given range
for name in bpm.copy():
    if args.plane == 'x':
      if not (args.beta_min <= df[name]['BX'] <= args.beta_max):
        bpm.pop(name)
    if args.plane == 'z':
      if not (args.beta_min <= df[name]['BZ'] <= args.beta_max):
        bpm.pop(name)

# Check BPM list
if not bpm:
  exit(f'error: BPM list is empty')

# Generate pv names
pv_list = [pv_make(name, args.plane, args.harmonica) for name in bpm]
pv_rise = [*bpm.values()]

# Set BPM positions
if args.nufft:
  if args.time == 'position':
    position = numpy.array([df[name]["S"] for name in bpm])/LENGTH
  if args.time == 'phase':
    if args.plane == 'x':
      tune = QX
      case = 'FX'
    if args.plane == 'z':
      tune = QZ
      case = 'FZ'
    position = numpy.array([df[name][case] for name in df])
    position = numpy.cumsum(position)/(2.0*numpy.pi*tune)
    start, *_ = position
    position = position - start
    position = numpy.array([value for (value, name) in zip(position, df) if name in bpm])

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
    exit(f'error: rise values are expected to be positive')
  rise = max(pv_rise)
  if length + offset + rise > LIMIT:
    exit(f'error: sum of {length=}, {offset=} and max {rise=}, expected to be less than {LIMIT=}')
else:
  rise = 0

# Check sample length
if args.length > length:
  exit(f'error: requested sample length {args.length} should be less than {length}')

# Load TbT data
size = len(bpm)
count = length + offset + rise
win = Window(length, args.name, args.order, dtype=dtype, device=device)
tbt = Data.from_epics(size, win, pv_list, pv_rise if args.rise else None, shift=offset, count=count)

# Remove mean
if args.mean:
  tbt.window_remove_mean()

# Remove median
if args.median:
  tbt.work.sub_(torch.median(tbt.data, 1).values.reshape(-1, 1))

# Normalize
if args.normalize:
  tbt.normalize(window=args.window)

# Number of steps
step = range(1 if args.shift <= 0 else 1 + (length - args.length) // args.shift)

# Loop over steps
shift = args.shift
length = args.length
win = Window(length, args.name, args.order, dtype=dtype, device=device)
tbt_shift = Data(size, win)
f = Frequency(tbt_shift)
result = []
for i in step:
  tbt_shift.set_data(tbt.work[:, i*shift : length + i*shift])
  frequency = f.task_mixed_frequency(
    length=length,
    window=args.window,
    f_range=(f_min, f_max),
    name=args.name,
    order=args.order,
    normalize=args.normalize,
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
    df = pandas.concat([df, pandas.DataFrame({'step':step, 'case':name, 'frequency':frequency[:, i]})])
  from plotly.express import scatter
  title = f'{TIME}: Frequency (all)<br>sample={args.length} & shift={args.shift}'
  plot = scatter(df, x='step', y='frequency', color='case', title=title, opacity=0.75, marginal_y='box')
  config = {
    'toImageButtonOptions': {'height':None, 'width':None},
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'],
    'scrollZoom': True
  }
  plot.show(config=config)

# Print data
if args.print:
  fmt = '{:>6}' + '{:>18.12}' * 3
  print(fmt.format('STEP', 'F1', 'F2', 'F3'))
  for i in step:
    print(fmt.format(i, *frequency[i]))

# Save to file
if args.file and args.numpy:
  filename = f'frequency_all_plane_{args.plane}_length_{args.length}_time_{TIME}.npy'
  numpy.save(filename, frequency)
if args.file and args.csv:
  filename = f'frequency_all_plane_{args.plane}_length_{args.length}_time_{TIME}.csv'
  numpy.savetxt(filename, frequency, delimiter=',')