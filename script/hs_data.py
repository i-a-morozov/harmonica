"""
usage: hs_data [-h] [-p {x,z,i}] [-l LENGTH] [--skip BPM [BPM ...] | --only BPM [BPM ...]] [-o OFFSET] [-r] [--pv | --data] [-f FILE] [--out {pv,file}]
               [--mean | --median | --normalize] [--harmonica] [--device {cpu,cuda}] [--dtype {float32,float64}]

Save/load TbT data for selected BPMs and plane.

optional arguments:
  -h, --help            show this help message and exit
  -p {x,z,i}, --plane {x,z,i}
                        data plane
  -l LENGTH, --length LENGTH
                        number of turns to save (integer)
  --skip BPM [BPM ...]  space separated list of valid BPM names to skip
  --only BPM [BPM ...]  space separated list of valid BPM names to use
  -o OFFSET, --offset OFFSET
                        rise offset for all BPMs
  -r, --rise            flag to use rise data from file (drop first turns)
  --pv                  flag to load data from PVs
  --data                flag to load data from file
  -f FILE, --file FILE  input file name
  --out {pv,file}       output target
  --mean                flag to remove mean
  --median              flag to remove median
  --normalize           flag to normalize data
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
parser = argparse.ArgumentParser(prog='hs_data', description='Save/load TbT data for selected BPMs and plane.')
parser.add_argument('-p', '--plane', choices=('x', 'z', 'i'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to save (integer)', default=1024)
select = parser.add_mutually_exclusive_group()
select.add_argument('--skip', metavar='BPM', nargs='+', help='space separated list of valid BPM names to skip')
select.add_argument('--only', metavar='BPM', nargs='+', help='space separated list of valid BPM names to use')
parser.add_argument('-o', '--offset', type=int, help='rise offset for all BPMs', default=0)
parser.add_argument('-r', '--rise', action='store_true', help='flag to use rise data from file (drop first turns)')
origin = parser.add_mutually_exclusive_group()
origin.add_argument('--pv', action='store_true', help='flag to load data from PVs')
origin.add_argument('--data', action='store_true', help='flag to load data from file')
parser.add_argument('-f', '--file', type=str, help='input file name')
parser.add_argument('--out', choices=('pv', 'file'), help='output target', default='file')
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--mean', action='store_true', help='flag to remove mean')
transform.add_argument('--median', action='store_true', help='flag to remove median')
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
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
win = Window(length, dtype=dtype, device=device)
if args.pv:
  tbt = Data.from_epics(size, win, pv_list, pv_rise if args.rise else None, shift=offset, count=count)
pv_list = [pv.replace('VEPP4', 'HARMONICA') for pv in pv_list]
if args.data:
  filename = args.file
  try:
    df = pandas.read_feather(filename)
  except FileNotFoundError:
    exit(f'error: input file not found')
  except:
    exit(f'error: problem loading input file')
  try:
    data = df[pv_list].to_numpy().T
  except KeyError:
    exit(f'error: requested keys are missing')
  tensor = torch.zeros((size, length), dtype=dtype, device=device)
  for i, value in enumerate(data):
    rise = pv_rise[i] if args.rise else 0
    tensor[i].copy_(torch.tensor(value[offset + rise : offset + rise + length]))
  tbt = Data.from_tensor(win, tensor)
  tbt.pv_list = pv_list
  tbt.pv_rise = pv_rise

# Remove mean
if args.mean:
  tbt.window_remove_mean()

# Remove median
if args.median:
  tbt.work.sub_(torch.median(tbt.data, 1).values.reshape(-1, 1))

# Normalize
if args.normalize:
  tbt.normalize(window=True)

# Convert to numpy
data = tbt.to_numpy()

# Save
if args.out == 'file':
  df = pandas.DataFrame()
  for idx, pv in enumerate(pv_list):
    df[pv] = data[idx]
  filename = f'data_plane_{args.plane}_length_{args.length}_time_{TIME}.ft'
  df.to_feather(filename)
if args.out == 'pv':
  tbt.pv_list = pv_list
  tbt.save_epics()

# Clean
del win, tbt
if device == 'cuda':
  torch.cuda.empty_cache()