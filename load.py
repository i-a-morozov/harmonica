"""
Load configuration data from config.yaml into harmonica epics process variables
Optionaly load test TbT data
Note, run softIoc -d harmonica.db first

"""

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='load', description='Load configuration data into harmonica epics process variables.')
parser.add_argument('-t', '--test', action='store_true', help='flag to load test TbT data')
args = parser.parse_args()

# Import
import yaml
import epics
import numpy
import pandas
from tqdm import tqdm

# Load configuration
with open('config.yaml', 'r') as stream:
  try:
    config = yaml.safe_load(stream)
  except yaml.YAMLError as exception:
    exit(exception)

# Sort locations by position
config = {key: config[key] for key in sorted(config.keys(), key=lambda name: config[name]['TIME'])}

# Check HEAD & TAIN
for name in ['HEAD', 'TAIL']:
  if name not in config:
    exit(f'error: {name} location is missing')

# Load model frequency
epics.caput('H:FREQUENCY:MODEL:X', config['TAIL']['FX']/(2.0*numpy.pi))
epics.caput('H:FREQUENCY:MODEL:Y', config['TAIL']['FY']/(2.0*numpy.pi))

# Load monitor & virtual
table = [*config.keys()]
epics.caput('H:LOCATION:COUNT', len(table))
epics.caput('H:LOCATION:LIST', table)

# Load monitor
table = [name for name, data in config.items() if data['TYPE'] == 'MONITOR']
epics.caput('H:MONITOR:COUNT', len(table))
epics.caput('H:MONITOR:LIST', table)

# Load test data
if args.test:
  try:
    frame = pandas.read_pickle('virtual_tbt.pkl.gz')
  except FileNotFoundError as exception:
    exit(exception)

# Load data for each location
for name, data in tqdm(config.items(), unit='record'):
  epics.caput(f'H:{name}:TYPE', data['TYPE'])
  epics.caput(f'H:{name}:FLAG', data['FLAG'])
  epics.caput(f'H:{name}:JOIN', data['JOIN'])
  epics.caput(f'H:{name}:RISE', data['RISE'])
  epics.caput(f'H:{name}:TIME', data['TIME'])
  epics.caput(f'H:{name}:MODEL:BX', data['BX'])
  epics.caput(f'H:{name}:MODEL:AX', data['AX'])
  epics.caput(f'H:{name}:MODEL:FX', data['FX'])
  epics.caput(f'H:{name}:MODEL:BY', data['BY'])
  epics.caput(f'H:{name}:MODEL:AY', data['AY'])
  epics.caput(f'H:{name}:MODEL:FY', data['FY'])
  epics.caput(f'H:{name}:MODEL:SIGMA_BX', data['SIGMA_BX'])
  epics.caput(f'H:{name}:MODEL:SIGMA_AX', data['SIGMA_AX'])
  epics.caput(f'H:{name}:MODEL:SIGMA_FX', data['SIGMA_FX'])
  epics.caput(f'H:{name}:MODEL:SIGMA_BY', data['SIGMA_BY'])
  epics.caput(f'H:{name}:MODEL:SIGMA_AY', data['SIGMA_AY'])
  epics.caput(f'H:{name}:MODEL:SIGMA_FY', data['SIGMA_FY'])
  if args.test and name in table:
    epics.caput(f'H:{name}:DATA:X', frame['X'][name])
    epics.caput(f'H:{name}:DATA:Y', frame['Y'][name])
    epics.caput(f'H:{name}:DATA:I', frame['I'][name])
