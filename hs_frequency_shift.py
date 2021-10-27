"""
usage: hs_frequency_shift [-h] [-p {x,z}] [-l LENGTH] [-t TURN] [--shift SHIFT] [-s BPM [BPM ...] | -o BPM [BPM ...]] [-r] [-f] [-c | -n] [--print]
                          [--mean | --normalize] [-w] [--name {cosine_window,kaiser_window}] [--order ORDER] [--pad PAD] [--min MIN] [--max MAX]
                          [-m {fft,ffrft,parabola}] [--flip] [--plot] [--device {cpu,cuda}] [--dtype {float32,float64}] [--test]

Print/save/plot frequency data for selected plane and BPMs using shifts.

optional arguments:
  -h, --help            show this help message and exit
  -p {x,z}, --plane {x,z}
                        data plane
  -l LENGTH, --length LENGTH
                        total number of turns to use (integer)
  -t TURN, --turn TURN  sample length (integer)
  --shift SHIFT         shift step (integer)
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
  --min MIN             min frequency (float)
  --max MAX             max frequency (float)
  -m {fft,ffrft,parabola}, --method {fft,ffrft,parabola}
                        frequency estimation method
  --flip                flag to flip frequency around 1/2
  --plot                flag to plot data
  --device {cpu,cuda}   data device
  --dtype {float32,float64}
                        data type
  --test                flag to use test PV names
"""

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_frequency_shift', description='Print/save/plot frequency data for selected plane and BPMs using shifts.')
parser.add_argument('-p', '--plane', choices=('x', 'z'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='total number of turns to use (integer)', default=1024)
parser.add_argument('-t', '--turn', type=int, help='sample length (integer)', default=512)
parser.add_argument('--shift', type=int, help='shift step (integer)', default=8)
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
parser.add_argument('--min', type=float, help='min frequency (float)', default=0.0)
parser.add_argument('--max', type=float, help='max frequency (float)', default=0.5)
parser.add_argument('-m', '--method', choices=('fft', 'ffrft', 'parabola'), help='frequency estimation method', default='parabola')
parser.add_argument('--flip', action='store_true', help='flag to flip frequency around 1/2')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
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
  tbt.normalize(window=args.window)

# Generate shifted TbT data
data = torch.cat([tbt.make_matrix(idx, args.turn, args.shift).data for idx in range(len(bpm))])
size, length = data.shape
step = size//len(bpm)
win = Window(length, name=args.name, order=args.order, dtype=dtype, device=device)
tbt = Data.from_tensor(win, data)

del data
if device == 'cuda':
  torch.cuda.empty_cache()

# Set Frequency instance
f = Frequency(tbt, pad=args.pad)

# Compute frequencies
f(args.method, window=args.window, f_range=(f_min, f_max))

# Convert to numpy
frequency = f.frequency.cpu().numpy()

# Clean
del win, tbt, f
if device == 'cuda':
  torch.cuda.empty_cache()

# Flip
if args.flip:
  frequency = 1.0 - frequency

# Partition frequency data
frequency = frequency.reshape(-1, step)

# Plot
if args.plot:
  df = pandas.DataFrame()
  for i, name in enumerate(bpm):
    df = pandas.concat([df, pandas.DataFrame({'step':range(1, step + 1), 'bpm':name, 'frequency':frequency[i]})])
  from plotly.express import scatter
  plot = scatter(df, x='step', y='frequency', color='bpm', title=TIME, opacity=0.75, marginal_y='box')
  plot.show()

# Print data
if args.print:
  fmt = '{:>6}' + '{:>18.12}' * len(bpm)
  print(fmt.format('STEP', *bpm))
  for i in range(step):
    print(fmt.format(i, *frequency[:, i]))

# Save to file
if args.file and args.numpy:
  numpy.save(f'frequency_shift_{TIME}.npy', frequency)
if args.file and args.csv:
  numpy.savetxt(f'frequency_shift_{TIME}.csv', frequency.transpose(), delimiter=',')