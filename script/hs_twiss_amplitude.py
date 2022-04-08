# Input arguments flag
import sys
sys.path.append('..')
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_frequency', description='Compute and plot twiss from amplitude for x any y planes')
parser.add_argument('--unit', choices=('m', 'mm', 'mk'), help='amplitude units', default='m')
parser.add_argument('--clean', action='store_true', help='flag to clean frequency data')
parser.add_argument('--factor', type=float, help='threashold factor', default=5.0)
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--action', action='store_true', help='flag to plot action data')
parser.add_argument('-H', '--harmonica', action='store_true', help='flag to use harmonica PV names for input')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float64')
parser.add_argument('-u', '--update', action='store_true', help='flag to update harmonica PV')
args = parser.parse_args(args=None if flag else ['--help'])

# Import
import epics
import numpy
import pandas
import torch
from datetime import datetime
from harmonica.window import Window
from harmonica.data import Data
from harmonica.frequency import Frequency
from harmonica.decomposition import Decomposition
from harmonica.model import Model
from harmonica.table import Table
from harmonica.twiss import Twiss

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit(f'error: CUDA is not avalible')

# Amplitude factor
factor = {'m':1.0E+0, 'mm':1.0E-2, 'mk':1.0E-6}[args.unit]

# Load monitor data
name = epics.caget('H:MONITOR:LIST')[:epics.caget('H:MONITOR:COUNT')]
flag = torch.tensor(epics.caget_many([f'H:{name}:FLAG' for name in name]), dtype=torch.int64, device=device)

# Load x frequency data
value_nux = torch.tensor(epics.caget('H:FREQUENCY:VALUE:X'), dtype=dtype, device=device)
error_nux = torch.tensor(epics.caget('H:FREQUENCY:ERROR:X'), dtype=dtype, device=device)

# Load y frequency data
value_nuy = torch.tensor(epics.caget('H:FREQUENCY:VALUE:Y'), dtype=dtype, device=device)
error_nuy = torch.tensor(epics.caget('H:FREQUENCY:ERROR:Y'), dtype=dtype, device=device)

# Load x amplitude data
value_ax = factor*torch.tensor(epics.caget_many([f'H:{name}:AMPLITUDE:VALUE:X' for name in name]), dtype=dtype, device=device)
error_ax = factor*torch.tensor(epics.caget_many([f'H:{name}:AMPLITUDE:ERROR:X' for name in name]), dtype=dtype, device=device)

# Load y amplitude data
value_ay = factor*torch.tensor(epics.caget_many([f'H:{name}:AMPLITUDE:VALUE:Y' for name in name]), dtype=dtype, device=device)
error_ay = factor*torch.tensor(epics.caget_many([f'H:{name}:AMPLITUDE:ERROR:Y' for name in name]), dtype=dtype, device=device)

# Load x phase data
value_fx = torch.tensor(epics.caget_many([f'H:{name}:PHASE:VALUE:X' for name in name]), dtype=dtype, device=device)
error_fx = torch.tensor(epics.caget_many([f'H:{name}:PHASE:ERROR:X' for name in name]), dtype=dtype, device=device)

# Load y phase data
value_fy = torch.tensor(epics.caget_many([f'H:{name}:PHASE:VALUE:Y' for name in name]), dtype=dtype, device=device)
error_fy = torch.tensor(epics.caget_many([f'H:{name}:PHASE:ERROR:Y' for name in name]), dtype=dtype, device=device)

# Set model
model = Model(path='../config.yaml', dtype=dtype, device=device)
model.flag[model.monitor_index] = flag

# Set table
table = Table(name, value_nux, value_nuy, value_ax, value_ay, value_fx, value_fy,
              error_nux, error_nuy, error_ax, error_ay, error_fx, error_fy,
              dtype=dtype, device=device)

# Set twiss
twiss = Twiss(model, table)

# Compute actions
twiss.get_action(data_threshold={'use': args.clean, 'factor': args.factor})

# Comute twiss
twiss.get_twiss_from_amplitude()

# Set data for each monitor
value_jx, error_jx = twiss.action['jx'].cpu().numpy(), twiss.action['sigma_jx'].cpu().numpy()
value_jy, error_jy = twiss.action['jy'].cpu().numpy(), twiss.action['sigma_jy'].cpu().numpy()

# Set processed data
center_jx, spread_jx = twiss.action['center_jx'].cpu().numpy(), twiss.action['spread_jx'].cpu().numpy()
center_jy, spread_jy = twiss.action['center_jy'].cpu().numpy(), twiss.action['spread_jy'].cpu().numpy()

# Set mask
mask_x, mask_y = twiss.action['mask']
mask_x, mask_y = mask_x.logical_not().cpu().numpy(), mask_y.logical_not().cpu().numpy()

# Set model twiss
bx_m = twiss.model.bx[twiss.model.monitor_index]
by_m = twiss.model.by[twiss.model.monitor_index]

# Set twiss
bx, sigma_bx = twiss.data_amplitude['bx'].cpu().numpy(), twiss.data_amplitude['sigma_bx'].cpu().numpy()
by, sigma_by = twiss.data_amplitude['by'].cpu().numpy(), twiss.data_amplitude['sigma_by'].cpu().numpy()

# Plot
if args.plot:
  df = pandas.DataFrame()
  df['BPM'] = name
  df['JX'] = value_jx
  df['SIGMA_JX'] = error_jx
  df['BX'] = bx
  df['BX_M'] = bx_m
  df['SIGMA_BX'] = sigma_bx
  df['ERROR_BX'] = (bx_m - bx)/bx_m
  df['DELTA_BX'] = sigma_bx/bx_m
  df['JY'] = value_jy
  df['SIGMA_JY'] = error_jy
  df['BY'] = by
  df['BY_M'] = by_m
  df['SIGMA_BY'] = sigma_by
  df['ERROR_BY'] = (by_m - by)/by_m
  df['DELTA_BY'] = sigma_by/by_m
  from plotly.subplots import make_subplots
  from plotly.express import scatter
  if args.action:
    plot = make_subplots(rows=2, cols=1, vertical_spacing=0.1, subplot_titles=(f'JX: {center_jx:9.6}, SIGMA_JX: {spread_jx:9.6}', f'JY: {center_jy:12.9}, SIGMA_JY: {spread_jy:12.9}'))
    plot.update_layout(title_text=f'{TIME}: ACTION')
    pltx = scatter(df, x='BPM', y='JX', error_y='SIGMA_JX', color_discrete_sequence=['blue'], hover_data=['BPM', 'JX', 'SIGMA_JX'])
    pltx.update_traces(marker={'size': 10})
    plty = scatter(df, x='BPM', y='JY', error_y='SIGMA_JY', color_discrete_sequence=['blue'], hover_data=['BPM', 'JY', 'SIGMA_JY'])
    plty.update_traces(marker={'size': 10})
    pltx, *_ = pltx.data
    plty, *_ = plty.data
    plot.add_trace(pltx, row=1, col=1)
    plot.add_trace(plty, row=2, col=1)
    plot.update_xaxes(title_text='BPM', row=1, col=1)
    plot.update_xaxes(title_text='BPM', row=2, col=1)
    plot.update_yaxes(title_text='JX', row=1, col=1)
    plot.update_yaxes(title_text='JY', row=2, col=1)
    plot.add_hline(center_jx - spread_jx, line_color='black', line_dash="dash", line_width=1.0, row=1, col=1)
    plot.add_hline(center_jx, line_color='black', line_dash="dash", line_width=1.0, row=1, col=1)
    plot.add_hline(center_jx + spread_jx, line_color='black', line_dash="dash", line_width=1.0, row=1, col=1)
    plot.add_hline(center_jy - spread_jy, line_color='black', line_dash="dash", line_width=1.0, row=2, col=1)
    plot.add_hline(center_jy, line_color='black', line_dash="dash", line_width=1.0, row=2, col=1)
    plot.add_hline(center_jy + spread_jy, line_color='black', line_dash="dash", line_width=1.0, row=2, col=1)
    if mask_x.sum() != 0:
      mask = pandas.DataFrame({'BPM':df.BPM[mask_x], 'JX':df.JX[mask_x], 'SIGMA_JX':df.SIGMA_JX[mask_x]})
      mask = scatter(mask, x='BPM', y='JX', error_y='SIGMA_JX', color_discrete_sequence=['red'], hover_data=['BPM', 'JX', 'SIGMA_JX'])
      mask.update_traces(marker={'size': 10})
      mask, *_ = mask.data
      plot.add_trace(mask, row=1, col=1)
    if mask_y.sum() != 0:
      mask = pandas.DataFrame({'BPM':df.BPM[mask_y], 'JY':df.JY[mask_y], 'SIGMA_JY':df.SIGMA_JY[mask_y]})
      mask = scatter(mask, x='BPM', y='JY', error_y='SIGMA_JY', color_discrete_sequence=['red'], hover_data=['BPM', 'JY', 'SIGMA_JY'])
      mask.update_traces(marker={'size': 10})
      mask, *_ = mask.data
      plot.add_trace(mask, row=2, col=1)
    config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
    plot.show(config=config)
  plot = make_subplots(rows=2, cols=2, vertical_spacing=0.1, horizontal_spacing=0.05, subplot_titles=(f'sum(abs(BX_M-BX)/BX_M)={sum(abs(bx_m - bx)/bx_m):9.6}', f'sum(abs(BY_M-BY)/BY_M)={sum(abs(by_m - by)/by_m):9.6}', f' ', f' '))
  plot.update_layout(title_text=f'{TIME}: TWISS (AMPLITUDE)')
  pltx1 = scatter(df, x='BPM', y='BX', error_y='SIGMA_BX', color_discrete_sequence=['blue'], hover_data=['BPM', 'BX', 'SIGMA_BX'])
  pltx1.update_traces(marker={'size': 10})
  pltx1, *_ = pltx1.data
  plot.add_trace(pltx1, row=1, col=1)
  pltx1 = scatter(df, x='BPM', y='BX_M', color_discrete_sequence=['black'], hover_data=['BPM', 'BX_M'], symbol_sequence=['cross'])
  pltx1.update_traces(marker={'size': 5})
  pltx1, *_ = pltx1.data
  plot.add_trace(pltx1, row=1, col=1)
  plot.update_xaxes(title_text='BPM', row=1, col=1)
  plot.update_yaxes(title_text='BX', row=1, col=1)
  plty1 = scatter(df, x='BPM', y='BY', error_y='SIGMA_BY', color_discrete_sequence=['blue'], hover_data=['BPM', 'BY', 'SIGMA_BY'])
  plty1.update_traces(marker={'size': 10})
  plty1, *_ = plty1.data
  plot.add_trace(plty1, row=1, col=2)
  plty1 = scatter(df, x='BPM', y='BY_M', color_discrete_sequence=['black'], hover_data=['BPM', 'BY_M'], symbol_sequence=['cross'])
  plty1.update_traces(marker={'size': 5})
  plty1, *_ = plty1.data
  plot.add_trace(plty1, row=1, col=2)
  plot.update_xaxes(title_text='BPM', row=1, col=2)
  plot.update_yaxes(title_text='BY', row=1, col=2)
  pltx2 = scatter(df, x='BPM', y='ERROR_BX', error_y='DELTA_BX', color_discrete_sequence=['blue'], hover_data=['BPM', 'ERROR_BX', 'DELTA_BX'])
  pltx2.update_traces(marker={'size': 10})
  pltx2, *_ = pltx2.data
  plot.add_trace(pltx2, row=2, col=1)
  plot.update_xaxes(title_text='BPM', row=2, col=1)
  plot.update_yaxes(title_text='(BX_M-BX)/BX_M', row=2, col=1)
  plty2 = scatter(df, x='BPM', y='ERROR_BY', error_y='DELTA_BY', color_discrete_sequence=['blue'], hover_data=['BPM', 'ERROR_BX', 'DELTA_BY'])
  plty2.update_traces(marker={'size': 10})
  plty2, *_ = plty2.data
  plot.add_trace(plty2, row=2, col=2)
  plot.update_xaxes(title_text='BPM', row=2, col=2)
  plot.update_yaxes(title_text='(BY_M-BY)/BY_M', row=2, col=2)
  if mask_x.sum() != 0:
    mask = pandas.DataFrame({'BPM':df.BPM[mask_x], 'BX':df.BX[mask_x], 'SIGMA_BX':df.SIGMA_BX[mask_x]})
    mask = scatter(mask, x='BPM', y='BX', error_y='SIGMA_BX', color_discrete_sequence=['red'], hover_data=['BPM', 'BX', 'SIGMA_BX'])
    mask.update_traces(marker={'size': 10})
    mask, *_ = mask.data
    plot.add_trace(mask, row=1, col=1)
    mask = pandas.DataFrame({'BPM':df.BPM[mask_x], 'ERROR_BX':df.ERROR_BX[mask_x], 'DELTA_BX':df.DELTA_BX[mask_x]})
    mask = scatter(mask, x='BPM', y='ERROR_BX', error_y='DELTA_BX', color_discrete_sequence=['red'], hover_data=['BPM', 'ERROR_BX', 'DELTA_BX'])
    mask.update_traces(marker={'size': 10})
    mask, *_ = mask.data
    plot.add_trace(mask, row=2, col=1)
  if mask_y.sum() != 0:
    mask = pandas.DataFrame({'BPM':df.BPM[mask_y], 'BY':df.BY[mask_y], 'SIGMA_BY':df.SIGMA_BY[mask_y]})
    mask = scatter(mask, x='BPM', y='BY', error_y='SIGMA_BY', color_discrete_sequence=['red'], hover_data=['BPM', 'BY', 'SIGMA_BY'])
    mask.update_traces(marker={'size': 10})
    mask, *_ = mask.data
    plot.add_trace(mask, row=1, col=2)
    mask = pandas.DataFrame({'BPM':df.BPM[mask_y], 'ERROR_BY':df.ERROR_BY[mask_y], 'DELTA_BY':df.DELTA_BY[mask_y]})
    mask = scatter(mask, x='BPM', y='ERROR_BY', error_y='DELTA_BY', color_discrete_sequence=['red'], hover_data=['BPM', 'ERROR_BY', 'DELTA_BY'])
    mask.update_traces(marker={'size': 10})
    mask, *_ = mask.data
    plot.add_trace(mask, row=2, col=2)
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)

# Save to epics
if args.update:
  epics.caput(f'H:ACTION:LIST:VALUE:X', value_jx)
  epics.caput(f'H:ACTION:LIST:ERROR:X', error_jx)
  epics.caput(f'H:ACTION:VALUE:X', center_jx)
  epics.caput(f'H:ACTION:ERROR:X', spread_jx)
  epics.caput(f'H:ACTION:LIST:VALUE:Y', value_jy)
  epics.caput(f'H:ACTION:LIST:ERROR:Y', error_jy)
  epics.caput(f'H:ACTION:VALUE:Y', center_jy)
  epics.caput(f'H:ACTION:ERROR:Y', spread_jy)
  epics.caput_many([f'H:{bpm}:AMPLITUDE:BX:VALUE' for bpm in name], bx)
  epics.caput_many([f'H:{bpm}:AMPLITUDE:BX:ERROR' for bpm in name], sigma_bx)
  epics.caput_many([f'H:{bpm}:AMPLITUDE:BY:VALUE' for bpm in name], by)
  epics.caput_many([f'H:{bpm}:AMPLITUDE:BY:ERROR' for bpm in name], sigma_by)