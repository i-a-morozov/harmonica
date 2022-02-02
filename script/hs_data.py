# Input arguments flag
import sys
sys.path.append('..')
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_data', description='Save/load TbT data for selected BPMs')
parser.add_argument('-l', '--length', type=int, help='number of turns to save/load (integer)', default=1024)
select = parser.add_mutually_exclusive_group()
select.add_argument('--skip', metavar='BPM', nargs='+', help='space separated list of valid BPM names to skip')
select.add_argument('--only', metavar='BPM', nargs='+', help='space separated list of valid BPM names to use')
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--mean', action='store_true', help='flag to remove mean')
transform.add_argument('--median', action='store_true', help='flag to remove median')
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
parser.add_argument('-f', '--filter', choices=('none', 'svd', 'hankel'), help='filter type', default='none')
parser.add_argument('--rank', type=int, help='rank to use for svd & hankel filter', default=8)
parser.add_argument('--type', choices=('full', 'randomized'), help='computation type for hankel filter', default='randomized')
parser.add_argument('--buffer', type=int, help='buffer size to use for randomized hankel filter', default=16)
parser.add_argument('--count', type=int, help='number of iterations to use for randomized hankel filter', default=16)
parser.add_argument('--input', choices=('data', 'file'), help='input target', default='data')
parser.add_argument('--file', type=str, help='input file name', default=None)
parser.add_argument('--output', choices=('data', 'file'), help='output target', default='file')
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
from harmonica.util import LIMIT, LENGTH, pv_make
from harmonica.window import Window
from harmonica.data import Data
from harmonica.filter import Filter

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit(f'error: CUDA is not avalible')

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

# Check length
length = args.length
if length < 0 or length > LIMIT:
  exit(f'error: {length=}, expected a positive value less than {LIMIT=}')

# Input (epics)
if args.input == 'data':

  pv_x = [pv_make(name, 'x', args.harmonica) for name in bpm]
  pv_y = [pv_make(name, 'y', args.harmonica) for name in bpm]
  pv_i = [pv_make(name, 'i', args.harmonica) for name in bpm]

  win = Window(length, dtype=dtype, device=device)
  tbt_x = Data.from_epics(win, pv_x)
  tbt_y = Data.from_epics(win, pv_y)
  tbt_i = Data.from_epics(win, pv_i)

# Input (file)
if args.input == 'file':

  try:
    frame = pandas.read_pickle(args.file)
  except FileNotFoundError as exception:
    exit(exception)

  tbt_x = torch.tensor([frame['X'][name][:length] for name in bpm], dtype=dtype, device=device)
  tbt_y = torch.tensor([frame['Y'][name][:length] for name in bpm], dtype=dtype, device=device)
  tbt_i = torch.tensor([frame['I'][name][:length] for name in bpm], dtype=dtype, device=device)

  win = Window(length, dtype=dtype, device=device)
  tbt_x = Data.from_data(win, tbt_x)
  tbt_y = Data.from_data(win, tbt_y)
  tbt_i = Data.from_data(win, tbt_i)

# Remove mean
if args.mean:
  tbt_x.window_remove_mean()
  tbt_y.window_remove_mean()

# Remove median
if args.median:
  tbt_x.work.sub_(tbt_x.median())
  tbt_y.work.sub_(tbt_y.median())

# Normalize
if args.normalize:
  tbt_x.normalize()
  tbt_y.normalize()

# Filter (svd)
if args.filter == 'svd':
  flt_x = Filter(tbt_x)
  flt_y = Filter(tbt_y)
  flt_x.filter_svd(rank=args.rank)
  flt_y.filter_svd(rank=args.rank)

# Filter (hankel)
if args.filter == 'hankel':
  flt_x = Filter(tbt_x)
  flt_y = Filter(tbt_y)
  flt_x.filter_svd(rank=args.rank)
  flt_x.filter_hankel(rank=args.rank, random=args.type == 'randomized', buffer=args.buffer, count=args.count)
  flt_y.filter_svd(rank=args.rank)
  flt_y.filter_hankel(rank=args.rank, random=args.type == 'randomized', buffer=args.buffer, count=args.count)

# Output (epics)
if args.output == 'data':

  pv_x = [pv_make(name, 'x', True) for name in bpm]
  pv_y = [pv_make(name, 'y', True) for name in bpm]
  pv_i = [pv_make(name, 'i', True) for name in bpm]

  tbt_x.source = 'epics'
  tbt_y.source = 'epics'
  tbt_i.source = 'epics'

  tbt_x.pv_list = pv_x
  tbt_y.pv_list = pv_y
  tbt_i.pv_list = pv_i

  tbt_x.save_epics()
  tbt_y.save_epics()
  tbt_i.save_epics()

# Output (file)
if args.output == 'file':

  data_x = {}
  data_y = {}
  data_i = {}

  for i, name in enumerate(bpm):
    data_x[name] = list(tbt_x.work[i].cpu().numpy())
    data_y[name] = list(tbt_y.work[i].cpu().numpy())
    data_i[name] = list(tbt_i.work[i].cpu().numpy())

  data = {'X': data_x, 'Y': data_y, 'I': data_i}

  filename = f'tbt_size_{len(bpm)}_length_{args.length}_time_{TIME}.pkl.gz'
  df = pandas.DataFrame.from_dict(data)
  df.to_pickle(filename, compression='gzip')