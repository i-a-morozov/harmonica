"""
usage: hs_tbt_all [-h] [-p {x,z,i}] [-l LENGTH] [--load LOAD] [--skip BPM [BPM ...] | --only BPM [BPM ...]] [-o OFFSET] [-r] [--beta_min BETA_MIN] [--beta_max BETA_MAX]
                  [-f] [-c | -n] [--print] [--mean | --median | --normalize] [--plot] [--harmonica] [--device {cpu,cuda}] [--dtype {float32,float64}]

Print/save/plot mixed TbT data for selected BPMs and plane.

optional arguments:
  -h, --help            show this help message and exit
  -p {x,z,i}, --plane {x,z,i}
                        data plane
  -l LENGTH, --length LENGTH
                        number of turns to print/save/plot (integer)
  --load LOAD           number of turns to load (integer)
  --skip BPM [BPM ...]  space separated list of valid BPM names to skip
  --only BPM [BPM ...]  space separated list of valid BPM names to use
  -o OFFSET, --offset OFFSET
                        rise offset for all BPMs
  -r, --rise            flag to use rise data from file (drop first turns)
  --beta_min BETA_MIN   min beta threshold value for x or z
  --beta_max BETA_MAX   max beta threshold value for x or z
  -f, --file            flag to save data
  -c, --csv             flag to save data as CSV
  -n, --numpy           flag to save data as NUMPY
  --print               flag to print data
  --mean                flag to remove mean
  --median              flag to remove median
  --normalize           flag to normalize data
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
parser = argparse.ArgumentParser(prog='hs_tbt_all', description='Print/save/plot mixed TbT data for selected BPMs and plane.')
parser.add_argument('-p', '--plane', choices=('x', 'z', 'i'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to print/save/plot (integer)', default=4)
parser.add_argument('--load', type=int, help='number of turns to load (integer)', default=1024)
select = parser.add_mutually_exclusive_group()
select.add_argument('--skip', metavar='BPM', nargs='+', help='space separated list of valid BPM names to skip')
select.add_argument('--only', metavar='BPM', nargs='+', help='space separated list of valid BPM names to use')
parser.add_argument('-o', '--offset', type=int, help='rise offset for all BPMs', default=0)
parser.add_argument('-r', '--rise', action='store_true', help='flag to use rise data from file (drop first turns)')
parser.add_argument('--beta_min', type=float, help='min beta threshold value for x or z', default=0.0E+0)
parser.add_argument('--beta_max', type=float, help='max beta threshold value for x or z', default=1.0E+3)
parser.add_argument('-f', '--file', action='store_true', help='flag to save data')
save = parser.add_mutually_exclusive_group()
save.add_argument('-c', '--csv', action='store_true', help='flag to save data as CSV')
save.add_argument('-n', '--numpy', action='store_true', help='flag to save data as NUMPY')
parser.add_argument('--print', action='store_true', help='flag to print data')
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--mean', action='store_true', help='flag to remove mean')
transform.add_argument('--median', action='store_true', help='flag to remove median')
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--harmonica', action='store_true', help='flag to use harmonica PV names for input')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float32')
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

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit(f'error: CUDA is not avalible')

# Import BPM data
try:
  df = pandas.read_json('../bpm.json')
except ValueError:
  exit(f'error: problem loading bpm.json')

# Process BPM data
bpm = {name: int(df[name]['RISE']) for name in df if df[name]['FLAG'] and df[name]['JOIN']}

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

# Set BPM positions
position = numpy.array([df[name]["S"] for name in bpm])

# Generate pv names
pv_list = [pv_make(name, args.plane, args.harmonica) for name in bpm]
pv_rise = [*bpm.values()]

# Check load length
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

# Load TbT data
size = len(bpm)
count = length + offset + rise
win = Window(length, dtype=dtype, device=device)
tbt = Data.from_epics(size, win, pv_list, pv_rise if args.rise else None, shift=offset, count=count)

# Remove mean
if args.mean:
  tbt.window_remove_mean()

# Remove median
if args.median:
  tbt.work.sub_(torch.median(tbt.data, 1).values.reshape(-1, 1))

# Normalize
if args.normalize:
  tbt.normalize(window=True)

# Check mixed length
if args.length < 0 or args.length > args.load:
  exit(f'error: requested length {args.length} is expected to be positive and less than load length {args.load}')

# Generate mixed data
tbt = tbt.make_signal(args.length)

# Convert to numpy
data, *_ = tbt.to_numpy()
name = [name for name in bpm] * args.length
turn = numpy.array([numpy.zeros(len(bpm), dtype=numpy.int32) + i for i in range(args.length)]).flatten()
time = 1/LENGTH*numpy.array([position + LENGTH * i for i in range(args.length)]).flatten()

# Clean
del win, tbt
if device == 'cuda':
  torch.cuda.empty_cache()

# Plot
if args.plot:
  df = pandas.DataFrame()
  df['name'] = name
  df['turn'] = turn.astype(str)
  df['time'] = time
  df[args.plane] = data
  from plotly.express import line
  plot = line(
    df,
    x='time',
    y=args.plane,
    color='turn',
    hover_data=['turn', 'name'],
    title=f'{TIME}: TbT (mixed)',
    markers=True)
  config = {
    'toImageButtonOptions': {'height':None, 'width':None},
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'],
    'scrollZoom': True
  }
  plot.show(config=config)

# Print data
if args.print:
  fmt = '{:>6}' + '{:>6}' + '{:>18.9}' + '{:>18.9}'
  print(fmt.format('BPM', 'T', 'S', args.plane.upper()))
  for i in range(len(data)):
    print(fmt.format(name[i], turn[i], time[i], data[i]))

# Save to file
data = numpy.array([time, data])
if args.file and args.numpy:
  filename = f'tbt_all_plane_{args.plane}_length_{args.length}_time_{TIME}.npy'
  numpy.save(filename, data)
if args.file and args.csv:
  filename = f'tbt_all_plane_{args.plane}_length_{args.length}_time_{TIME}.csv'
  numpy.savetxt(filename, data.transpose(), delimiter=',')