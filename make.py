"""
Generate harmonica.db

"""

# Import
import yaml
from harmonica.util import record_make

# Load configuration
with open('config.yaml', 'r') as stream:
  try:
    config = yaml.safe_load(stream)
  except yaml.YAMLError as exception:
    exit(exception)

# Common
common = """
    # LOCATION
    record(waveform, "H:LOCATION:COUNT")         {field(NELM, "1")    field(FTVL, "SHORT")}
    record(waveform, "H:LOCATION:LIST")          {field(NELM, "1024") field(FTVL, "STRING")}
    record(waveform, "H:MONITOR:COUNT")          {field(NELM, "1")    field(FTVL, "SHORT")}
    record(waveform, "H:MONITOR:LIST")           {field(NELM, "1024") field(FTVL, "STRING")}

    # FREQUENCY
    record(waveform, "H:FREQUENCY:MODEL:X")      {field(NELM, "1")    field(FTVL, "DOUBLE")}
    record(waveform, "H:FREQUENCY:MODEL:Y")      {field(NELM, "1")    field(FTVL, "DOUBLE")}
    record(waveform, "H:FREQUENCY:VALUE:X")      {field(NELM, "1")    field(FTVL, "DOUBLE")}
    record(waveform, "H:FREQUENCY:VALUE:Y")      {field(NELM, "1")    field(FTVL, "DOUBLE")}
    record(waveform, "H:FREQUENCY:VALUE:Z")      {field(NELM, "1")    field(FTVL, "DOUBLE")}
    record(waveform, "H:FREQUENCY:ERROR:X")      {field(NELM, "1")    field(FTVL, "DOUBLE")}
    record(waveform, "H:FREQUENCY:ERROR:Y")      {field(NELM, "1")    field(FTVL, "DOUBLE")}
    record(waveform, "H:FREQUENCY:ERROR:Z")      {field(NELM, "1")    field(FTVL, "DOUBLE")}

    # ACTION (MONITOR)
    record(waveform, "H:ACTION:LIST:VALUE:X")    {field(NELM, "1024")    field(FTVL, "DOUBLE")}
    record(waveform, "H:ACTION:LIST:ERROR:X")    {field(NELM, "1024")    field(FTVL, "DOUBLE")}
    record(waveform, "H:ACTION:LIST:VALUE:Y")    {field(NELM, "1024")    field(FTVL, "DOUBLE")}
    record(waveform, "H:ACTION:LIST:ERROR:Y")    {field(NELM, "1024")    field(FTVL, "DOUBLE")}
    record(waveform, "H:ACTION:VALUE:X")         {field(NELM, "1")    field(FTVL, "DOUBLE")}
    record(waveform, "H:ACTION:ERROR:X")         {field(NELM, "1")    field(FTVL, "DOUBLE")}
    record(waveform, "H:ACTION:VALUE:Y")         {field(NELM, "1")    field(FTVL, "DOUBLE")}
    record(waveform, "H:ACTION:ERROR:Y")         {field(NELM, "1")    field(FTVL, "DOUBLE")}
"""

# Generate records
records = [record_make(name) for name in config]

# Generate file
with open('harmonica.db', 'w') as stream:
  stream.write('# Start all PVs (attached to console)\n')
  stream.write('# softIoc -d harmonica.db\n')
  stream.write(common)
  for record in records:
    stream.write(record)