#!/bin/env python

# Import
import sys
import argparse
import os
import epics
import numpy
import pandas
import torch
from datetime import datetime
from harmonica.model import Model
from harmonica.table import Table
from harmonica.twiss import Twiss

# Input arguments flag
_, *flag = sys.argv

# Parse arguments
parser = argparse.ArgumentParser(prog='twiss_phase', description='Compute and plot twiss from phase for x and y planes')
parser.add_argument('-m', '--model', type=str, help='model config path', default='elettra.yaml')
parser.add_argument('-l', '--limit', type=int, help='range limit', default=4)
parser.add_argument('--unit', choices=('m', 'mm', 'mk'), help='amplitude units', default='m')
parser.add_argument('--clean', action='store_true', help='flag to clean data')
parser.add_argument('--factor', type=float, help='threshold factor', default=5.0)
parser.add_argument('--plot', action='store_true', help='flag to plot data')
parser.add_argument('--amplitude', action='store_true', help='flag to plot twiss from amplitude data')
parser.add_argument('--phase', action='store_true', help='flag to plot phase advance data')
parser.add_argument('-H', '--harmonica', action='store_true', help='flag to use harmonica PV names for input')
parser.add_argument('--device', choices=('cpu', 'cuda'), help='data device', default='cpu')
parser.add_argument('--dtype', choices=('float32', 'float64'), help='data type', default='float64')
parser.add_argument('-u', '--update', action='store_true', help='flag to update harmonica PV')
args = parser.parse_args(args=None if flag else ['--help'])

# Time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Check and set device & data type
dtype = {'float32': torch.float32, 'float64': torch.float64}[args.dtype]
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
  exit('error: CUDA is not available')

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
model_path = args.model
if not os.path.isabs(model_path) and not os.path.exists(model_path):
  local = os.path.join(os.path.dirname(__file__), model_path)
  parent = os.path.join(os.path.dirname(__file__), '..', model_path)
  if os.path.exists(local):
    model_path = local
  elif os.path.exists(parent):
    model_path = parent
model = Model(path=model_path, dtype=dtype, device=device)
model.flag[model.monitor_index] = flag

# Set table
table = Table(name, value_nux, value_nuy, value_ax, value_ay, value_fx, value_fy,
              error_nux, error_nuy, error_ax, error_ay, error_fx, error_fy,
              dtype=dtype, device=device)

# Set twiss
twiss = Twiss(model, table, limit=args.limit)

# Compute actions
twiss.get_action(dict_threshold={'use': args.clean, 'factor': args.factor})

# Compute twiss from amplitude
twiss.get_twiss_from_amplitude()

# Compute twiss from phase
twiss.phase_virtual()
twiss.get_twiss_from_phase()
mask_x = twiss.filter_twiss(plane='x')
mask_y = twiss.filter_twiss(plane='y')
if args.limit != 1:
  twiss.process_twiss(plane='x', mask=mask_x, weight=True)
  twiss.process_twiss(plane='y', mask=mask_y, weight=True)
else:
  twiss.process_twiss(plane='x')
  twiss.process_twiss(plane='y')

# Set model twiss
ax_m, bx_m = twiss.model.ax[1:-1].cpu().numpy(), twiss.model.bx[1:-1].cpu().numpy()
ay_m, by_m = twiss.model.ay[1:-1].cpu().numpy(), twiss.model.by[1:-1].cpu().numpy()

# Set model phase data
fx_m, fy_m = twiss.model.monitor_phase_x.cpu().numpy(), twiss.model.monitor_phase_y.cpu().numpy()

# Set twiss from amplitude
bx_m_a = twiss.model.bx[twiss.model.monitor_index].cpu().numpy()
by_m_a = twiss.model.by[twiss.model.monitor_index].cpu().numpy()
bx_a, sigma_bx_a = twiss.data_amplitude['bx'].cpu().numpy(), twiss.data_amplitude['sigma_bx'].cpu().numpy()
by_a, sigma_by_a = twiss.data_amplitude['by'].cpu().numpy(), twiss.data_amplitude['sigma_by'].cpu().numpy()

# Set phase advance data
fx, sigma_fx = twiss.table.phase_x.cpu().numpy(), twiss.table.sigma_x.cpu().numpy()
fy, sigma_fy = twiss.table.phase_y.cpu().numpy(), twiss.table.sigma_y.cpu().numpy()

# Set twiss from phase
ax, sigma_ax = twiss.ax[1:-1].cpu().numpy(), twiss.sigma_ax[1:-1].cpu().numpy()
bx, sigma_bx = twiss.bx[1:-1].cpu().numpy(), twiss.sigma_bx[1:-1].cpu().numpy()
ay, sigma_ay = twiss.ay[1:-1].cpu().numpy(), twiss.sigma_ay[1:-1].cpu().numpy()
by, sigma_by = twiss.by[1:-1].cpu().numpy(), twiss.sigma_by[1:-1].cpu().numpy()

# Plot
if args.plot:
  from plotly.subplots import make_subplots
  from plotly.express import scatter
  from plotly.express import line
  if args.phase:
    df = pandas.DataFrame()
    df['BPM'] = name
    df['FX'] = fx
    df['FX_M'] = fx_m
    df['ERROR_FX'] = fx_m - fx
    df['SIGMA_FX'] = sigma_fx
    df['FY'] = fy
    df['FY_M'] = fy_m
    df['ERROR_FY'] = fy_m - fy
    df['SIGMA_FY'] = sigma_fy
    plot = make_subplots(rows=2, cols=2, vertical_spacing=0.1, shared_xaxes=True, horizontal_spacing=0.05, subplot_titles=(f'sum(abs(FX_M-FX))={sum(abs(fx_m - fx)):9.6}', f'sum(abs(FY_M-FY))={sum(abs(fy_m - fy)):9.6}', ' ', ' '))
    plot.update_layout(title_text=f'{TIME}: TWISS (PHASE): LIMIT={twiss.limit}')
    pltx1 = scatter(df, x='BPM', y='FX', error_y='SIGMA_FX', color_discrete_sequence=['blue'], hover_data=['BPM', 'FX', 'SIGMA_FX'])
    pltx1.update_traces(marker={'size': 10})
    pltx1, *_ = pltx1.data
    plot.add_trace(pltx1, row=1, col=1)
    pltx1 = line(df, x='BPM', y='FX_M', color_discrete_sequence=['black'], markers=True, hover_data=['BPM', 'FX_M'], symbol_sequence=['square-open'])
    pltx1.update_traces(marker={'size': 10})
    pltx1, *_ = pltx1.data
    plot.add_trace(pltx1, row=1, col=1)
    plot.update_xaxes(title_text='BPM', row=1, col=1)
    plot.update_yaxes(title_text='FX', row=1, col=1)
    plty1 = scatter(df, x='BPM', y='FY', error_y='SIGMA_FY', color_discrete_sequence=['blue'], hover_data=['BPM', 'FY', 'SIGMA_FY'])
    plty1.update_traces(marker={'size': 10})
    plty1, *_ = plty1.data
    plot.add_trace(plty1, row=1, col=2)
    plty1 = line(df, x='BPM', y='FY_M', color_discrete_sequence=['black'], markers=True, hover_data=['BPM', 'FY_M'], symbol_sequence=['square-open'])
    plty1.update_traces(marker={'size': 10})
    plty1, *_ = plty1.data
    plot.add_trace(plty1, row=1, col=2)
    plot.update_xaxes(title_text='BPM', row=1, col=2)
    plot.update_yaxes(title_text='FY', row=1, col=2)
    pltx2 = scatter(df, x='BPM', y='ERROR_FX', error_y='SIGMA_FX', color_discrete_sequence=['blue'], hover_data=['BPM', 'ERROR_FX', 'SIGMA_FX'])
    pltx2.update_traces(marker={'size': 10})
    pltx2, *_ = pltx2.data
    plot.add_trace(pltx2, row=2, col=1)
    plot.update_xaxes(title_text='BPM', row=2, col=1)
    plot.update_yaxes(title_text='FX_M-FX', row=2, col=1)
    plty2 = scatter(df, x='BPM', y='ERROR_FY', error_y='SIGMA_FY', color_discrete_sequence=['blue'], hover_data=['BPM', 'ERROR_FX', 'SIGMA_FY'])
    plty2.update_traces(marker={'size': 10})
    plty2, *_ = plty2.data
    plot.add_trace(plty2, row=2, col=2)
    plot.update_xaxes(title_text='BPM', row=2, col=2)
    plot.update_yaxes(title_text='FY_M-FY', row=2, col=2)
    config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
    plot.show(config=config)

  df = pandas.DataFrame()
  df['BPM'] = twiss.model.name[1:-1]
  df['AX'] = ax
  df['AX_M'] = ax_m
  df['SIGMA_AX'] = sigma_ax
  df['ERROR_AX'] = (ax_m - ax)
  df['BX'] = bx
  df['BX_M'] = bx_m
  df['SIGMA_BX'] = sigma_bx
  df['ERROR_BX'] = (bx_m - bx)/bx_m
  df['DELTA_BX'] = sigma_bx/bx_m
  df['AY'] = ay
  df['AY_M'] = ay_m
  df['SIGMA_AY'] = sigma_ay
  df['ERROR_AY'] = (ay_m - ay)
  df['BY'] = by
  df['BY_M'] = by_m
  df['SIGMA_BY'] = sigma_by
  df['ERROR_BY'] = (by_m - by)/by_m
  df['DELTA_BY'] = sigma_by/by_m
  monitor_index = numpy.array([index - 1 for index in model.monitor_index], dtype=numpy.int64)
  virtual_index = numpy.array([index - 1 for index in model.virtual_index if 0 < index < model.size - 1], dtype=numpy.int64)
  if monitor_index.size == 0:
    rms_bx = 100.0*numpy.sqrt(numpy.nanmean(df['ERROR_BX'].to_numpy()**2))
    rms_by = 100.0*numpy.sqrt(numpy.nanmean(df['ERROR_BY'].to_numpy()**2))
  else:
    rms_bx = 100.0*numpy.sqrt(numpy.nanmean(df['ERROR_BX'].to_numpy()[monitor_index]**2))
    rms_by = 100.0*numpy.sqrt(numpy.nanmean(df['ERROR_BY'].to_numpy()[monitor_index]**2))
  plot = make_subplots(rows=2, cols=2, vertical_spacing=0.1, shared_xaxes=True, horizontal_spacing=0.05, subplot_titles=(f'RMS X = {rms_bx:9.6f} %', f'RMS Y = {rms_by:9.6f} %', ' ', ' '))
  plot.update_layout(title_text=f'{TIME}: TWISS (PHASE): LIMIT={twiss.limit}')
  pltx1 = scatter(df, x='BPM', y='BX', error_y='SIGMA_BX', color_discrete_sequence=['blue'], hover_data=['BPM', 'BX', 'SIGMA_BX'], symbol_sequence=['circle-open'])
  pltx1.update_traces(marker={'size': 10})
  pltx1.update_traces(name='BX', legendgroup='1', showlegend=True)
  pltx1, *_ = pltx1.data
  plot.add_trace(pltx1, row=1, col=1)
  pltx1 = line(df, x='BPM', y='BX_M', color_discrete_sequence=['black'], markers=True, hover_data=['BPM', 'BX_M'], symbol_sequence=['square-open'])
  pltx1.update_traces(marker={'size': 10})
  pltx1.update_traces(name='BX_M', legendgroup='2', showlegend=True)
  pltx1, *_ = pltx1.data
  plot.add_trace(pltx1, row=1, col=1)
  plot.update_xaxes(title_text='BPM', row=1, col=1)
  plot.update_yaxes(title_text='BX', row=1, col=1)
  plty1 = scatter(df, x='BPM', y='BY', error_y='SIGMA_BY', color_discrete_sequence=['blue'], hover_data=['BPM', 'BY', 'SIGMA_BY'], symbol_sequence=['circle-open'])
  plty1.update_traces(marker={'size': 10})
  plty1.update_traces(name='BY', legendgroup='3', showlegend=True)
  plty1, *_ = plty1.data
  plot.add_trace(plty1, row=1, col=2)
  plty1 = line(df, x='BPM', y='BY_M', color_discrete_sequence=['black'], markers=True, hover_data=['BPM', 'BY_M'], symbol_sequence=['square-open'])
  plty1.update_traces(marker={'size': 10})
  plty1.update_traces(name='BY_M', legendgroup='4', showlegend=True)
  plty1, *_ = plty1.data
  plot.add_trace(plty1, row=1, col=2)
  plot.update_xaxes(title_text='BPM', row=1, col=2)
  plot.update_yaxes(title_text='BY', row=1, col=2)
  pltx2 = scatter(df, x='BPM', y='ERROR_BX', error_y='DELTA_BX', color_discrete_sequence=['blue'], hover_data=['BPM', 'ERROR_BX', 'DELTA_BX'], symbol_sequence=['circle-open'])
  pltx2.update_traces(marker={'size': 10})
  pltx2.update_traces(name='(BX_M-BX)/BX_M', legendgroup='5', showlegend=True)
  pltx2, *_ = pltx2.data
  plot.add_trace(pltx2, row=2, col=1)
  plot.update_xaxes(title_text='BPM', row=2, col=1)
  plot.update_yaxes(title_text='(BX_M-BX)/BX_M', row=2, col=1)
  plty2 = scatter(df, x='BPM', y='ERROR_BY', error_y='DELTA_BY', color_discrete_sequence=['blue'], hover_data=['BPM', 'ERROR_BY', 'DELTA_BY'], symbol_sequence=['circle-open'])
  plty2.update_traces(marker={'size': 10})
  plty2.update_traces(name='(BY_M-BY)/BY_M', legendgroup='6', showlegend=True)
  plty2, *_ = plty2.data
  plot.add_trace(plty2, row=2, col=2)
  plot.update_xaxes(title_text='BPM', row=2, col=2)
  plot.update_yaxes(title_text='(BY_M-BY)/BY_M', row=2, col=2)
  if virtual_index.size != 0:
    virtual = pandas.DataFrame({'BPM': df.BPM[virtual_index], 'BX': df.BX[virtual_index], 'SIGMA_BX': df.SIGMA_BX[virtual_index]})
    virtual = scatter(virtual, x='BPM', y='BX', error_y='SIGMA_BX', color_discrete_sequence=['green'], hover_data=['BPM', 'BX', 'SIGMA_BX'], symbol_sequence=['circle-open'])
    virtual.update_traces(marker={'size': 10})
    virtual.update_traces(name='BX (VIRTUAL)', legendgroup='virtual', showlegend=True)
    virtual, *_ = virtual.data
    plot.add_trace(virtual, row=1, col=1)

    virtual = pandas.DataFrame({'BPM': df.BPM[virtual_index], 'BY': df.BY[virtual_index], 'SIGMA_BY': df.SIGMA_BY[virtual_index]})
    virtual = scatter(virtual, x='BPM', y='BY', error_y='SIGMA_BY', color_discrete_sequence=['green'], hover_data=['BPM', 'BY', 'SIGMA_BY'], symbol_sequence=['circle-open'])
    virtual.update_traces(marker={'size': 10})
    virtual.update_traces(name='BY (VIRTUAL)', legendgroup='virtual', showlegend=False)
    virtual, *_ = virtual.data
    plot.add_trace(virtual, row=1, col=2)

    virtual = pandas.DataFrame({'BPM': df.BPM[virtual_index], 'ERROR_BX': df.ERROR_BX[virtual_index], 'DELTA_BX': df.DELTA_BX[virtual_index]})
    virtual = scatter(virtual, x='BPM', y='ERROR_BX', error_y='DELTA_BX', color_discrete_sequence=['green'], hover_data=['BPM', 'ERROR_BX', 'DELTA_BX'], symbol_sequence=['circle-open'])
    virtual.update_traces(marker={'size': 10})
    virtual.update_traces(name='(BX_M-BX)/BX_M (VIRTUAL)', legendgroup='virtual', showlegend=False)
    virtual, *_ = virtual.data
    plot.add_trace(virtual, row=2, col=1)

    virtual = pandas.DataFrame({'BPM': df.BPM[virtual_index], 'ERROR_BY': df.ERROR_BY[virtual_index], 'DELTA_BY': df.DELTA_BY[virtual_index]})
    virtual = scatter(virtual, x='BPM', y='ERROR_BY', error_y='DELTA_BY', color_discrete_sequence=['green'], hover_data=['BPM', 'ERROR_BY', 'DELTA_BY'], symbol_sequence=['circle-open'])
    virtual.update_traces(marker={'size': 10})
    virtual.update_traces(name='(BY_M-BY)/BY_M (VIRTUAL)', legendgroup='virtual', showlegend=False)
    virtual, *_ = virtual.data
    plot.add_trace(virtual, row=2, col=2)
  if args.amplitude:
    mask = pandas.DataFrame({'BPM':name, 'BX':bx_a, 'SIGMA_BX':sigma_bx_a})
    mask = scatter(mask, x='BPM', y='BX', error_y='SIGMA_BX', color_discrete_sequence=['red'], hover_data=['BPM', 'BX', 'SIGMA_BX'], symbol_sequence=['circle-open'])
    mask.update_traces(marker={'size': 10})
    mask.update_traces(name='BX_A', legendgroup='7', showlegend=True)
    mask, *_ = mask.data
    plot.add_trace(mask, row=1, col=1)
    mask = pandas.DataFrame({'BPM':name, 'BY':by_a, 'SIGMA_BY':sigma_by_a})
    mask = scatter(mask, x='BPM', y='BY', error_y='SIGMA_BY', color_discrete_sequence=['red'], hover_data=['BPM', 'BY', 'SIGMA_BY'], symbol_sequence=['circle-open'])
    mask.update_traces(marker={'size': 10})
    mask.update_traces(name='BY_A', legendgroup='8', showlegend=True)
    mask, *_ = mask.data
    plot.add_trace(mask, row=1, col=2)
    mask = pandas.DataFrame({'BPM':name, 'ERROR_BX':((bx_m_a) - bx_a)/(bx_m_a), 'DELTA_BX':sigma_bx_a/(bx_m_a)})
    mask = scatter(mask, x='BPM', y='ERROR_BX', error_y='DELTA_BX', color_discrete_sequence=['red'], hover_data=['BPM', 'ERROR_BX', 'DELTA_BX'], symbol_sequence=['circle-open'])
    mask.update_traces(marker={'size': 10})
    mask.update_traces(name='(BX_M-BX_A)/BX_M', legendgroup='9', showlegend=True)
    mask, *_ = mask.data
    plot.add_trace(mask, row=2, col=1)
    mask = pandas.DataFrame({'BPM':name, 'ERROR_BY':((by_m_a) - by_a)/(by_m_a), 'DELTA_BY':sigma_by_a/(by_m_a)})
    mask = scatter(mask, x='BPM', y='ERROR_BY', error_y='DELTA_BY', color_discrete_sequence=['red'], hover_data=['BPM', 'ERROR_BY', 'DELTA_BY'], symbol_sequence=['circle-open'])
    mask.update_traces(marker={'size': 10})
    mask.update_traces(name='(BY_M-BY_A)/BY_M', legendgroup='10', showlegend=True)
    mask, *_ = mask.data
    plot.add_trace(mask, row=2, col=2)
  plot.update_layout(legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5))
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)

  plot = make_subplots(rows=2, cols=2, vertical_spacing=0.1, shared_xaxes=True, horizontal_spacing=0.05, subplot_titles=(f'sum(abs(AX_M-AX))={sum(abs(ax_m - ax)):9.6}', f'sum(abs(AY_M-AY))={sum(abs(ay_m - ay)):9.6}', ' ', ' '))
  plot.update_layout(title_text=f'{TIME}: TWISS (PHASE): LIMIT={twiss.limit}')
  pltx1 = scatter(df, x='BPM', y='AX', error_y='SIGMA_AX', color_discrete_sequence=['blue'], hover_data=['BPM', 'AX', 'SIGMA_AX'], symbol_sequence=['circle-open'])
  pltx1.update_traces(marker={'size': 10})
  pltx1.update_traces(name='AX', legendgroup='11', showlegend=True)
  pltx1, *_ = pltx1.data
  plot.add_trace(pltx1, row=1, col=1)
  pltx1 = line(df, x='BPM', y='AX_M', color_discrete_sequence=['black'], markers=True, hover_data=['BPM', 'AX_M'], symbol_sequence=['square-open'])
  pltx1.update_traces(marker={'size': 10})
  pltx1.update_traces(name='AX_M', legendgroup='12', showlegend=True)
  pltx1, *_ = pltx1.data
  plot.add_trace(pltx1, row=1, col=1)
  plot.update_xaxes(title_text='BPM', row=1, col=1)
  plot.update_yaxes(title_text='AX', row=1, col=1)
  plty1 = scatter(df, x='BPM', y='AY', error_y='SIGMA_AY', color_discrete_sequence=['blue'], hover_data=['BPM', 'AY', 'SIGMA_AY'], symbol_sequence=['circle-open'])
  plty1.update_traces(marker={'size': 10})
  plty1.update_traces(name='AY', legendgroup='13', showlegend=True)
  plty1, *_ = plty1.data
  plot.add_trace(plty1, row=1, col=2)
  plty1 = line(df, x='BPM', y='AY_M', color_discrete_sequence=['black'], markers=True, hover_data=['BPM', 'AY_M'], symbol_sequence=['square-open'])
  plty1.update_traces(marker={'size': 10})
  plty1.update_traces(name='AY_M', legendgroup='14', showlegend=True)
  plty1, *_ = plty1.data
  plot.add_trace(plty1, row=1, col=2)
  plot.update_xaxes(title_text='BPM', row=1, col=2)
  plot.update_yaxes(title_text='AY', row=1, col=2)
  pltx2 = scatter(df, x='BPM', y='ERROR_AX', error_y='SIGMA_AX', color_discrete_sequence=['blue'], hover_data=['BPM', 'ERROR_AX', 'SIGMA_BX'], symbol_sequence=['circle-open'])
  pltx2.update_traces(marker={'size': 10})
  pltx2, *_ = pltx2.data
  plot.add_trace(pltx2, row=2, col=1)
  plot.update_xaxes(title_text='BPM', row=2, col=1)
  plot.update_yaxes(title_text='AX_M-BX', row=2, col=1)
  plty2 = scatter(df, x='BPM', y='ERROR_AY', error_y='SIGMA_AY', color_discrete_sequence=['blue'], hover_data=['BPM', 'ERROR_AY', 'SIGMA_BY'], symbol_sequence=['circle-open'])
  plty2.update_traces(marker={'size': 10})
  plty2, *_ = plty2.data
  plot.add_trace(plty2, row=2, col=2)
  plot.update_xaxes(title_text='BPM', row=2, col=2)
  plot.update_yaxes(title_text='AY_M-AY', row=2, col=2)
  if virtual_index.size != 0:
    virtual = pandas.DataFrame({'BPM': df.BPM[virtual_index], 'AX': df.AX[virtual_index], 'SIGMA_AX': df.SIGMA_AX[virtual_index]})
    virtual = scatter(virtual, x='BPM', y='AX', error_y='SIGMA_AX', color_discrete_sequence=['green'], hover_data=['BPM', 'AX', 'SIGMA_AX'], symbol_sequence=['circle-open'])
    virtual.update_traces(marker={'size': 10})
    virtual.update_traces(name='AX (VIRTUAL)', legendgroup='virtual', showlegend=True)
    virtual, *_ = virtual.data
    plot.add_trace(virtual, row=1, col=1)

    virtual = pandas.DataFrame({'BPM': df.BPM[virtual_index], 'AY': df.AY[virtual_index], 'SIGMA_AY': df.SIGMA_AY[virtual_index]})
    virtual = scatter(virtual, x='BPM', y='AY', error_y='SIGMA_AY', color_discrete_sequence=['green'], hover_data=['BPM', 'AY', 'SIGMA_AY'], symbol_sequence=['circle-open'])
    virtual.update_traces(marker={'size': 10})
    virtual.update_traces(name='AY (VIRTUAL)', legendgroup='virtual', showlegend=False)
    virtual, *_ = virtual.data
    plot.add_trace(virtual, row=1, col=2)

    virtual = pandas.DataFrame({'BPM': df.BPM[virtual_index], 'ERROR_AX': df.ERROR_AX[virtual_index], 'SIGMA_AX': df.SIGMA_AX[virtual_index]})
    virtual = scatter(virtual, x='BPM', y='ERROR_AX', error_y='SIGMA_AX', color_discrete_sequence=['green'], hover_data=['BPM', 'ERROR_AX', 'SIGMA_AX'], symbol_sequence=['circle-open'])
    virtual.update_traces(marker={'size': 10})
    virtual.update_traces(name='AX_M-AX (VIRTUAL)', legendgroup='virtual', showlegend=False)
    virtual, *_ = virtual.data
    plot.add_trace(virtual, row=2, col=1)

    virtual = pandas.DataFrame({'BPM': df.BPM[virtual_index], 'ERROR_AY': df.ERROR_AY[virtual_index], 'SIGMA_AY': df.SIGMA_AY[virtual_index]})
    virtual = scatter(virtual, x='BPM', y='ERROR_AY', error_y='SIGMA_AY', color_discrete_sequence=['green'], hover_data=['BPM', 'ERROR_AY', 'SIGMA_AY'], symbol_sequence=['circle-open'])
    virtual.update_traces(marker={'size': 10})
    virtual.update_traces(name='AY_M-AY (VIRTUAL)', legendgroup='virtual', showlegend=False)
    virtual, *_ = virtual.data
    plot.add_trace(virtual, row=2, col=2)
  plot.update_layout(legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5))
  config = {'toImageButtonOptions': {'height':None, 'width':None}, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'modeBarButtonsToAdd':['drawopenpath', 'eraseshape'], 'scrollZoom': True}
  plot.show(config=config)

# Save to epics
if args.update:
  epics.caput_many([f'H:{bpm}:AMPLITUDE:BX:VALUE' for bpm in name], bx_a)
  epics.caput_many([f'H:{bpm}:AMPLITUDE:BX:ERROR' for bpm in name], sigma_bx_a)
  epics.caput_many([f'H:{bpm}:AMPLITUDE:BY:VALUE' for bpm in name], by_a)
  epics.caput_many([f'H:{bpm}:AMPLITUDE:BY:ERROR' for bpm in name], sigma_by_a)
  epics.caput_many([f'H:{bpm}:PHASE:BX:VALUE' for bpm in model.name[1:-1]], bx)
  epics.caput_many([f'H:{bpm}:PHASE:BX:ERROR' for bpm in model.name[1:-1]], sigma_bx)
  epics.caput_many([f'H:{bpm}:PHASE:BY:VALUE' for bpm in model.name[1:-1]], by)
  epics.caput_many([f'H:{bpm}:PHASE:BY:ERROR' for bpm in model.name[1:-1]], sigma_by)
  epics.caput_many([f'H:{bpm}:PHASE:AX:VALUE' for bpm in model.name[1:-1]], ax)
  epics.caput_many([f'H:{bpm}:PHASE:AX:ERROR' for bpm in model.name[1:-1]], sigma_ax)
  epics.caput_many([f'H:{bpm}:PHASE:AY:VALUE' for bpm in model.name[1:-1]], ay)
  epics.caput_many([f'H:{bpm}:PHASE:AY:ERROR' for bpm in model.name[1:-1]], sigma_ay)
