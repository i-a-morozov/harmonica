#!/bin/env python

# Import
import sys
import argparse
import epics
import numpy
import pandas
import torch
from datetime import datetime
from harmonica.util import LIMIT, pv_make, bpm_select
from harmonica.window import Window
from harmonica.data import Data
from harmonica.filter import Filter

# Input arguments flag
_, *flag = sys.argv

# Parse arguments
parser = argparse.ArgumentParser(prog='trajectory', description='Save/plot trajectory TbT data for selected BPMs and plane.')
parser.add_argument('-p', '--plane', choices=('x', 'y', 'i'), help='data plane', default='x')
parser.add_argument('-l', '--length', type=int, help='number of turns to use', default=4)
parser.add_argument('--load', type=int, help='number of turns to load (integer)', default=128)
select = parser.add_mutually_exclusive_group()
select.add_argument('--skip', metavar='PATTERN', nargs='+', help='space separated regex patterns for BPM names to skip')
select.add_argument('--only', metavar='PATTERN', nargs='+', help='space separated regex patterns for BPM names to use')
parser.add_argument('-o', '--offset', type=int, help='rise offset for all BPMs', default=0)
parser.add_argument('-r', '--rise', action='store_true', help='flag to use rise data (drop first turns)')
parser.add_argument('-s', '--save', action='store_true', help='flag to save data as numpy array')
parser.add_argument('--trajectory', action='store_true', help='flag to save trajectory matrix')
parser.add_argument('--compare', action='store_true', help='flag to compare current trajectory against saved')
transform = parser.add_mutually_exclusive_group()
transform.add_argument('--mean', action='store_true', help='flag to remove mean')
transform.add_argument('--median', action='store_true', help='flag to remove median')
transform.add_argument('--normalize', action='store_true', help='flag to normalize data')
parser.add_argument('-f', '--filter', choices=('none', 'svd', 'hankel'), help='filter type', default='none')
parser.add_argument('--rank', type=int, help='rank to use for svd & hankel filter', default=8)
parser.add_argument('--type', choices=('full', 'randomized'), help='SVD computation type for hankel filter', default='randomized')
parser.add_argument('--buffer', type=int, help='buffer size to use for randomized hankel filter', default=16)
parser.add_argument('--count', type=int, help='number of iterations to use for randomized hankel filter', default=16)
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--difference', action='store_true', help='flag to plot pairwise BPM differences (bpm1-bpm2, bpm3-bpm4, ...)')
parser.add_argument('-H', '--harmonica', action='store_true', help='flag to use harmonica PV names for input')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float64')
args = parser.parse_args(args=None if flag else ['--help'])

# Length
LENGTH = 259.2

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit('error: CUDA is not available')

# Load monitor data
name = epics.caget('H:MONITOR:LIST')[:epics.caget('H:MONITOR:COUNT')]
flag = epics.caget_many([f'H:{name}:FLAG' for name in name])
rise = epics.caget_many([f'H:{name}:RISE' for name in name])

# Set BPM data
bpm = {name: rise for name, flag, rise in zip(name, flag, rise) if flag == 1}

# Filter BPM list
try:
  bpm = bpm_select(bpm, skip=args.skip, only=args.only)
except ValueError as exception:
  exit(str(exception))

# Check BPM list
if not bpm:
  exit('error: BPM list is empty')

# Set BPM positions
position = numpy.array(epics.caget_many([f'H:{name}:TIME' for name in bpm]))

# Generate PV names
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
    exit('error: rise values are expected to be positive')
  rise = max(pv_rise)
  if length + offset + rise > LIMIT:
    exit(f'error: sum of {length=}, {offset=} and max {rise=}, expected to be less than {LIMIT=}')
else:
  rise = 0

# Load TbT data
size = len(bpm)
count = length + offset + rise
win = Window(length, dtype=dtype, device=device)
tbt = Data.from_epics(win, pv_list, pv_rise=(pv_rise if args.rise else None), shift=offset, count=count)

# Remove mean
if args.mean:
  tbt.window_remove_mean()

# Remove median
if args.median:
  tbt.work.sub_(tbt.median())

# Normalize
if args.normalize:
  tbt.normalize()

# Filter (none)
if args.filter == 'none':
  data = tbt.to_numpy()

# Filter (svd)
if args.filter == 'svd':
  flt = Filter(tbt)
  flt.filter_svd(rank=args.rank)
  data = tbt.to_numpy()

# Filter (hankel)
if args.filter == 'hankel':
  flt = Filter(tbt)
  flt.filter_svd(rank=args.rank)
  flt.filter_hankel(rank=args.rank, random=args.type == 'randomized', buffer=args.buffer, count=args.count)
  data = tbt.to_numpy()

# Check mixed length
if args.length < 0 or args.length > args.load:
  exit(f'error: requested length {args.length} is expected to be positive and less than load length {args.load}')

# Generate mixed data
data = tbt.make_signal(args.length, tbt.work)

# Convert to numpy
data = data.cpu().numpy()
trajectory = data.reshape(args.length, size)
bpm_name = [name for name in bpm]
name = bpm_name * args.length
turn = numpy.array([numpy.zeros(len(bpm), dtype=numpy.int32) + i for i in range(args.length)]).flatten()
time = 1/LENGTH*numpy.array([position + LENGTH * i for i in range(args.length)]).flatten()

# Compare with saved trajectory
if args.compare:
  try:
    reference = numpy.load('trajectory.npy')
  except FileNotFoundError:
    exit('error: trajectory.npy is not found')
  except Exception as exception:
    exit(f'error: failed to load trajectory.npy: {exception}')
  reference = numpy.asarray(reference)
  difference = trajectory[:args.length] - reference[:args.length]
  df = pandas.DataFrame()
  df['BPM'] = bpm_name * args.length
  df['TURN'] = numpy.array([numpy.zeros(size, dtype=numpy.int32) + i for i in range(args.length)]).flatten().astype(str)
  df['TIME'] = 1/LENGTH*numpy.array([position + LENGTH * i for i in range(args.length)]).flatten()
  df['DIFF'] = difference.flatten()
  from plotly.express import line
  plot = line(df, x='TIME', y='DIFF', color='TURN', hover_data=['TURN', 'BPM'], title=f'{TIME}: TbT (TRAJECTORY DIFF)', markers=True)
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)

# Plot
if args.plot:
  from plotly.express import line
  if args.difference:
    pair_size = size//2
    if pair_size == 0:
      exit('error: at least two BPMs are required for --difference')
    trajectory_diff = trajectory[:, 0:2*pair_size:2] - trajectory[:, 1:2*pair_size:2]
    pair_name = [f'{name_1}-{name_2}' for name_1, name_2 in zip(bpm_name[0:2*pair_size:2], bpm_name[1:2*pair_size:2])]
    pair_position = 0.5*(position[0:2*pair_size:2] + position[1:2*pair_size:2])
    pair_turn = numpy.array([numpy.zeros(pair_size, dtype=numpy.int32) + i for i in range(args.length)]).flatten()
    pair_time = 1/LENGTH*numpy.array([pair_position + LENGTH * i for i in range(args.length)]).flatten()
    df = pandas.DataFrame()
    df['BPM'] = pair_name * args.length
    df['TURN'] = pair_turn.astype(str)
    df['TIME'] = pair_time
    df['DIFF'] = trajectory_diff.flatten()
    plot = line(df, x='TIME', y='DIFF', color='TURN', hover_data=['TURN', 'BPM'], title=f'{TIME}: TbT (TRAJECTORY PAIR DIFF)', markers=True)
  else:
    df = pandas.DataFrame()
    df['BPM'] = name
    df['TURN'] = turn.astype(str)
    df['TIME'] = time
    df[args.plane.upper()] = data
    plot = line(df, x='TIME', y=args.plane.upper(), color='TURN', hover_data=['TURN', 'BPM'], title=f'{TIME}: TbT (TRAJECTORY)', markers=True)
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)

# Save to file
if args.save:
  filename = f'tbt_trajectory_plane_{args.plane}_length_{args.length}_time_{TIME}.npy'
  numpy.save(filename, numpy.array([time, data]))

# Save trajectory matrix to file
if args.trajectory:
  numpy.save('trajectory.npy', trajectory)
