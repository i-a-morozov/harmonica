"""
usage: hs_frequency_all [-h] [-p {x,z}] [-l LENGTH] [--shift SHIFT] [-s BPM [BPM ...] | -o BPM [BPM ...]] [-r] [--beta_min BETA_MIN] [--beta_max BETA_MAX] [--normalize]
                        [-w] [--name {cosine_window,kaiser_window}] [--order ORDER] [--min MIN] [--max MAX] [--nufft] [--device {cpu,cuda}] [--dtype {float32,float64}]
                        [--test]

Compute mixed frequency for selected plane and BPMs.

optional arguments:
  -h, --help            show this help message and exit
  -p {x,z}, --plane {x,z}
                        data plane
  -l LENGTH, --length LENGTH
                        number of turns to use (integer)
  --shift SHIFT         start shift for all BPMs (integer)
  -s BPM [BPM ...], --skip BPM [BPM ...]
                        space separated list of valid BPM names to skip
  -o BPM [BPM ...], --only BPM [BPM ...]
                        space separated list of valid BPM names to use
  -r, --rise            flag to use rise data (drop first turns)
  --beta_min BETA_MIN   min beta threshold value for x or z
  --beta_max BETA_MAX   max beta threshold value for x or z
  --normalize           flag to normalize data
  -w, --window          flag to apply window
  --name {cosine_window,kaiser_window}
                        window type
  --order ORDER         window order parameter (float >= 0.0)
  --min MIN             min frequency value (float)
  --max MAX             max frequency value (float)
  --nufft               flag to compute spectum using TYPY-III NUFFT
  --device {cpu,cuda}   data device
  --dtype {float32,float64}
                        data type
  --test                flag to use test PV names
"""

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_frequency_all', description='Compute mixed frequency for selected plane and BPMs.')
parser.add_argument('-p', '--plane', choices=('x', 'z'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use (integer)', default=64)
parser.add_argument('--shift', type=int, help='start shift for all BPMs (integer)', default=0)
select = parser.add_mutually_exclusive_group()
select.add_argument('-s', '--skip', metavar='BPM', nargs='+', help='space separated list of valid BPM names to skip')
select.add_argument('-o', '--only', metavar='BPM', nargs='+', help='space separated list of valid BPM names to use')
parser.add_argument('-r', '--rise', action='store_true', help='flag to use rise data (drop first turns)')
parser.add_argument('--beta_min', type=float, help='min beta threshold value for x or z', default=0.0E+0)
parser.add_argument('--beta_max', type=float, help='max beta threshold value for x or z', default=1.0E+3)
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
parser.add_argument('-w', '--window', action='store_true', help='flag to apply window')
parser.add_argument('--name', choices=('cosine_window', 'kaiser_window'), help='window type', default='cosine_window')
parser.add_argument('--order', type=float, help='window order parameter (float >= 0.0)', default=1.0)
parser.add_argument('--min', type=float, help='min frequency value (float)', default=0.0)
parser.add_argument('--max', type=float, help='max frequency value (float)', default=16.0)
parser.add_argument('--nufft', action='store_true', help='flag to compute spectum using TYPY-III NUFFT')
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

# Check and set frequency range
f_min = args.min
f_max = args.max
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
position = numpy.array([df[name]["S"] for name in bpm])/LENGTH

# Generate pv names
pv_list = [pv_make(name, args.plane, args.test) for name in bpm]
pv_rise = [*bpm.values()]

# Load TbT data
size = len(bpm)
length = min(args.length, LIMIT)
count = min(length + max(pv_rise) if args.rise else length, LIMIT)
win = Window(length, name=args.name, order=args.order, dtype=dtype, device=device)
tbt = Data.from_epics(size, win, pv_list, pv_rise if args.rise else None, count=count, shift=args.shift)

# Set Frequency instance
f = Frequency(tbt)

# Compute frequency
frequency = f.task_mixed_frequency(
  length=args.length,
  window=args.window,
  f_range=(f_min, f_max),
  name=args.name,
  order=args.order,
  normalize=args.normalize,
  position=position if args.nufft else None
)

# Print result
fmt = 3*'{:>18.16}'
print(fmt.format('F1', 'F2', 'F3'))
print(fmt.format(*frequency.cpu().numpy()))