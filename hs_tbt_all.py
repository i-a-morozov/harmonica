"""
usage: hs_tbt_all [-h] [-p {x,z}] [-t TURN] [-l LENGTH] [-s BPM [BPM ...] | -o BPM [BPM ...]] [-r] [--beta_min BETA_MIN] [--beta_max BETA_MAX] [-f]
                  [-c | -n] [--print] [--mean | --normalize] [--plot] [--device {cpu,cuda}] [--dtype {float32,float64}] [--test]

Print/save/plot mixed tbt data for selected BPMs and plane.

optional arguments:
  -h, --help            show this help message and exit
  -p {x,z}, --plane {x,z}
                        data plane
  -t TURN, --turn TURN  number of turns to print/save/plot (integer)
  -l LENGTH, --length LENGTH
                        total number of turns to load (integer)
  -s BPM [BPM ...], --skip BPM [BPM ...]
                        space separated list of valid BPM names to skip
  -o BPM [BPM ...], --only BPM [BPM ...]
                        space separated list of valid BPM names to use
  -r, --rise            flag to use rise data (drop first turns)
  --beta_min BETA_MIN   min beta threshold value for x or z
  --beta_max BETA_MAX   max beta threshold value for x or z
  -f, --file            flag to save data
  -c, --csv             flag to save data as CSV
  -n, --numpy           flag to save data as NUMPY
  --print               flag to print data
  --mean                flag to remove mean
  --normalize           flag to normalize data
  --plot                flag to plot data
  --device {cpu,cuda}   data device
  --dtype {float32,float64}
                        data type
  --test                flag to use test PV names
"""

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_tbt_all', description='Print/save/plot mixed tbt data for selected BPMs and plane.')
parser.add_argument('-p', '--plane', choices=('x', 'z'), help='data plane', default='x')
parser.add_argument('-t', '--turn', type=int, help='number of turns to print/save/plot (integer)', default=1)
parser.add_argument('-l', '--length', type=int, help='total number of turns to load (integer)', default=1024)
select = parser.add_mutually_exclusive_group()
select.add_argument('-s', '--skip', metavar='BPM', nargs='+', help='space separated list of valid BPM names to skip')
select.add_argument('-o', '--only', metavar='BPM', nargs='+', help='space separated list of valid BPM names to use')
parser.add_argument('-r', '--rise', action='store_true', help='flag to use rise data (drop first turns)')
parser.add_argument('--beta_min', type=float, help='min beta threshold value for x or z', default=0.0E+0)
parser.add_argument('--beta_max', type=float, help='max beta threshold value for x or z', default=1.0E+3)
parser.add_argument('-f', '--file', action='store_true', help='flag to save data')
save = parser.add_mutually_exclusive_group()
save.add_argument('-c', '--csv', action='store_true', help='flag to save data as CSV')
save.add_argument('-n', '--numpy', action='store_true', help='flag to save data as NUMPY')
parser.add_argument('--print', action='store_true', help='flag to print data')
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--mean', action='store_true', help='flag to remove mean')
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float32')
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

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit(f'error: CUDA is not avalible')

# Import BPM data
try:
  df = pandas.read_json('bpm.json')
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
pv_list = [pv_make(name, args.plane, args.test) for name in bpm]
pv_rise = [*bpm.values()]

# Load TbT data
size = len(bpm)
length = min(args.length, LIMIT)
count = min(length + max(pv_rise) if args.rise else length, LIMIT)
win = Window(length, dtype=dtype, device=device)
tbt = Data.from_epics(size, win, pv_list, pv_rise if args.rise else None, count=count)

# Remove mean
if args.mean:
  tbt.window_remove_mean()

# Normalize
if args.normalize:
  tbt.normalize(window=True)

# Generate mixed data
tbt = tbt.make_signal(args.turn)

# Convert to numpy
data, *_ = tbt.to_numpy()
name = [name for name in bpm] * args.turn
turn = numpy.array([numpy.zeros(len(bpm), dtype=numpy.int32) + i for i in range(args.turn)]).flatten()
time = 1/LENGTH*numpy.array([position + LENGTH * i for i in range(args.turn)]).flatten()

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
  df['data'] = data
  from plotly.express import line
  plot = line(df, x='time', y='data', color='turn', hover_data=['turn', 'name'], title=TIME, markers=True)
  plot.show()

# Print data
if args.print:
  fmt = '{:>6}' + '{:>6}' + '{:>18.9}' + '{:>18.9}'
  print(fmt.format('BPM', 'T', 'S', args.plane.upper()))
  for i in range(len(data)):
    print(fmt.format(name[i], turn[i], time[i], data[i]))

# Save to file
data = numpy.array([time, data])
if args.file and args.numpy:
  numpy.save(f'tbt_all_{TIME}.npy', data)
if args.file and args.csv:
  numpy.savetxt(f'tbt_all_{TIME}.csv', data.transpose(), delimiter=',')