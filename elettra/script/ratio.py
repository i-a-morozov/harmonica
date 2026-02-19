#!/bin/env python

# Import
import sys
sys.path.append('../..')
import argparse
import epics
import numpy
import pandas
from datetime import datetime
from harmonica.util import bpm_select

# Input arguments flag
_, *flag = sys.argv

# Parse arguments
parser = argparse.ArgumentParser(prog='ratio', description='Compute/plot BX_A/BX and BY_A/BY ratios from twiss PV data.')
select = parser.add_mutually_exclusive_group()
select.add_argument('--skip', metavar='PATTERN', nargs='+', help='space separated regex patterns for BPM names to skip')
select.add_argument('--only', metavar='PATTERN', nargs='+', help='space separated regex patterns for BPM names to use')
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('-s', '--save', action='store_true', help='flag to save data as numpy array')
parser.add_argument('-u', '--update', action='store_true', help='flag to update harmonica PV ratio data')
args = parser.parse_args(args=None if flag else ['--help'])

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Load monitor data
name = epics.caget('H:MONITOR:LIST')[:epics.caget('H:MONITOR:COUNT')]
flag = epics.caget_many([f'H:{name}:FLAG' for name in name])

# Set BPM data
bpm = {name: None for name, flag in zip(name, flag) if flag == 1}

# Filter BPM list
try:
  bpm = bpm_select(bpm, skip=args.skip, only=args.only)
except ValueError as exception:
  exit(str(exception))

# Check BPM list
if not bpm:
  exit('error: BPM list is empty')

# Keep BPM order
bpm = [*bpm.keys()]

# Load twiss data from PVs
bx_a = numpy.array(epics.caget_many([f'H:{name}:AMPLITUDE:BX:VALUE' for name in bpm]), dtype=numpy.float64)
sx_a = numpy.array(epics.caget_many([f'H:{name}:AMPLITUDE:BX:ERROR' for name in bpm]), dtype=numpy.float64)
by_a = numpy.array(epics.caget_many([f'H:{name}:AMPLITUDE:BY:VALUE' for name in bpm]), dtype=numpy.float64)
sy_a = numpy.array(epics.caget_many([f'H:{name}:AMPLITUDE:BY:ERROR' for name in bpm]), dtype=numpy.float64)

bx = numpy.array(epics.caget_many([f'H:{name}:PHASE:BX:VALUE' for name in bpm]), dtype=numpy.float64)
sx = numpy.array(epics.caget_many([f'H:{name}:PHASE:BX:ERROR' for name in bpm]), dtype=numpy.float64)
by = numpy.array(epics.caget_many([f'H:{name}:PHASE:BY:VALUE' for name in bpm]), dtype=numpy.float64)
sy = numpy.array(epics.caget_many([f'H:{name}:PHASE:BY:ERROR' for name in bpm]), dtype=numpy.float64)

# Load current waveform data and compute per-BPM current mean/spread
def current_stat(value):
  if value is None:
    return numpy.nan, numpy.nan
  data = numpy.array(value, dtype=numpy.float64).reshape(-1)
  data = data[numpy.isfinite(data)]
  if data.size == 0:
    return numpy.nan, numpy.nan
  return data.mean(), data.std()

current = epics.caget_many([f'H:{name}:DATA:I' for name in bpm])
value_i, error_i = zip(*(current_stat(value) for value in current))
value_i = numpy.array(value_i, dtype=numpy.float64)
error_i = numpy.array(error_i, dtype=numpy.float64)

# Compute robust mask and weighted center/spread for current
valid_i = numpy.isfinite(value_i)
if valid_i.any():
  median_i = numpy.nanmedian(value_i)
  mad_i = numpy.nanmedian(numpy.abs(value_i - median_i))
  scale_i = 1.4826*mad_i if mad_i > 0.0 else numpy.nanstd(value_i)
  if numpy.isfinite(scale_i) and scale_i > 0.0:
    z_i = (value_i - median_i)/scale_i
    mask_i = numpy.abs(z_i) > 5.0
  else:
    mask_i = numpy.zeros_like(value_i, dtype=bool)
else:
  mask_i = numpy.zeros_like(value_i, dtype=bool)

weight_i = numpy.zeros_like(value_i, dtype=numpy.float64)
good_i = (~mask_i) & numpy.isfinite(value_i) & numpy.isfinite(error_i) & (error_i > 0.0)
weight_i[good_i] = 1.0/error_i[good_i]**2
if weight_i.sum() > 0.0:
  center_i = numpy.average(value_i, weights=weight_i)
  spread_i = numpy.sqrt(numpy.average((value_i - center_i)**2, weights=weight_i))
else:
  center_i = numpy.nanmean(value_i)
  spread_i = numpy.nanstd(value_i)

# Compute ratios and propagated errors
with numpy.errstate(divide='ignore', invalid='ignore'):
  rx = bx_a/bx
  ry = by_a/by
  sigma_rx = numpy.sqrt((sx_a/bx)**2 + (bx_a*sx/bx**2)**2)
  sigma_ry = numpy.sqrt((sy_a/by)**2 + (by_a*sy/by**2)**2)

# Summary
center_rx = numpy.nanmean(rx)
spread_rx = numpy.nanstd(rx)
center_ry = numpy.nanmean(ry)
spread_ry = numpy.nanstd(ry)
print(f'RX: center={center_rx:12.9f}, spread={spread_rx:12.9f}')
print(f'RY: center={center_ry:12.9f}, spread={spread_ry:12.9f}')
print(f'I:  center={center_i:12.9f}, spread={spread_i:12.9f}')

# Plot
if args.plot:
  from plotly.subplots import make_subplots
  from plotly.express import scatter
  df = pandas.DataFrame({
    'BPM': bpm,
    'I': value_i,
    'SIGMA_I': error_i,
    'MASK_I': mask_i,
    'RX': rx,
    'SIGMA_RX': sigma_rx,
    'RY': ry,
    'SIGMA_RY': sigma_ry
  })
  plot = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
      f'I center={center_i:12.9f}, spread={spread_i:12.9f}',
      f'RX center={center_rx:12.9f}, spread={spread_rx:12.9f}',
      f'RY center={center_ry:12.9f}, spread={spread_ry:12.9f}'
    )
  )
  plot.update_layout(title_text=f'{TIME}: RATIO')

  plti = scatter(df, x='BPM', y='I', error_y='SIGMA_I', color_discrete_sequence=['blue'], hover_data=['BPM', 'I', 'SIGMA_I'])
  plti.update_traces(mode='lines+markers', line={'width': 1.5}, marker={'size': 8})
  plti, *_ = plti.data
  plot.add_trace(plti, row=1, col=1)
  plot.update_yaxes(title_text='I', row=1, col=1)
  plot.add_hline(center_i - spread_i, line_color='black', line_dash='dash', line_width=1.0, row=1, col=1)
  plot.add_hline(center_i, line_color='black', line_dash='dash', line_width=1.0, row=1, col=1)
  plot.add_hline(center_i + spread_i, line_color='black', line_dash='dash', line_width=1.0, row=1, col=1)
  if mask_i.sum() != 0:
    bad = pandas.DataFrame({'BPM': df.BPM[mask_i], 'I': df.I[mask_i], 'SIGMA_I': df.SIGMA_I[mask_i]})
    bad = scatter(bad, x='BPM', y='I', error_y='SIGMA_I', color_discrete_sequence=['red'], hover_data=['BPM', 'I', 'SIGMA_I'])
    bad.update_traces(mode='markers', marker={'size': 8})
    bad, *_ = bad.data
    plot.add_trace(bad, row=1, col=1)

  pltx = scatter(df, x='BPM', y='RX', error_y='SIGMA_RX', color_discrete_sequence=['blue'], hover_data=['BPM', 'RX', 'SIGMA_RX'])
  pltx.update_traces(mode='lines+markers', line={'width': 1.5}, marker={'size': 8})
  pltx, *_ = pltx.data
  plot.add_trace(pltx, row=2, col=1)
  plot.update_yaxes(title_text='BX_A/BX', row=2, col=1)
  plot.add_hline(1.0, line_color='black', line_dash='dash', line_width=1.0, row=2, col=1)

  plty = scatter(df, x='BPM', y='RY', error_y='SIGMA_RY', color_discrete_sequence=['blue'], hover_data=['BPM', 'RY', 'SIGMA_RY'])
  plty.update_traces(mode='lines+markers', line={'width': 1.5}, marker={'size': 8})
  plty, *_ = plty.data
  plot.add_trace(plty, row=3, col=1)
  plot.update_xaxes(title_text='BPM', row=3, col=1)
  plot.update_yaxes(title_text='BY_A/BY', row=3, col=1)
  plot.add_hline(1.0, line_color='black', line_dash='dash', line_width=1.0, row=3, col=1)

  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)

# Save to file
if args.save:
  filename = f'ratio_time_{TIME}.npy'
  numpy.save(filename, numpy.array([rx, sigma_rx, ry, sigma_ry]))

# Save to epics
if args.update:
  epics.caput_many([f'H:{name}:RATIO:VALUE:X' for name in bpm], rx)
  epics.caput_many([f'H:{name}:RATIO:ERROR:X' for name in bpm], sigma_rx)
  epics.caput_many([f'H:{name}:RATIO:VALUE:Y' for name in bpm], ry)
  epics.caput_many([f'H:{name}:RATIO:ERROR:Y' for name in bpm], sigma_ry)
