"""
Utility constants & functions.

"""

import epics
import numpy
import pandas

# Maximum length to read from TbT PVs
LIMIT:int = 8192

# Ring circumference
LENGTH:float = 366.075015600006

# Generate TbT PV name
def pv_make(name:str, plane:str, flag:bool=False) -> str:
    """
    Generate PV name for given BPM name and plane.

    Use harmonica PV name prefix if test flag is True

    Parameters
    ----------
    name: str
        BPM name
    plane: str
        selected plane (x, y or i)
    flag: bool
        flag to use harmonica PV name prefix

    Returns
    -------
    PV name (str)

    """
    if not flag:
        plane = {'X':'x', 'Y':'z', 'I':'i'}[plane.upper()]

    return f'H:{name}:DATA:{plane.upper()}' if flag else f'VEPP4:{name}:turns_{plane}-I'

# Generate location record
def record_make(name:str) -> str:
    """
    Generate DB location record for given location name.

    Parameters
    ----------
    name: str
        location name

    Returns
    -------
    DB location record (str)

    """
    return f'''
    # {name}
    record(waveform, "H:{name}:TYPE")              {{field(NELM, "1")    field(FTVL, "STRING")}}
    record(waveform, "H:{name}:FLAG")              {{field(NELM, "1")    field(FTVL, "SHORT")}}
    record(waveform, "H:{name}:JOIN")              {{field(NELM, "1")    field(FTVL, "SHORT")}}
    record(waveform, "H:{name}:RISE")              {{field(NELM, "1")    field(FTVL, "SHORT")}}
    record(waveform, "H:{name}:S")                 {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:BX")          {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:AX")          {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:FX")          {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:BY")          {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:AY")          {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:FY")          {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:DATA:X")            {{field(NELM, "8192") field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:DATA:Y")            {{field(NELM, "8192") field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:DATA:I")            {{field(NELM, "8192") field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:FREQUENCY:VALUE:X") {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:FREQUENCY:VALUE:Y") {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:FREQUENCY:ERROR:X") {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:FREQUENCY:ERROR:Y") {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:AMPLITUDE:VALUE:X") {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:AMPLITUDE:VALUE:Y") {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:AMPLITUDE:ERROR:X") {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:AMPLITUDE:ERROR:Y") {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:PHASE:VALUE:X")     {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:PHASE:VALUE:Y")     {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:PHASE:ERROR:X")     {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:PHASE:ERROR:Y")     {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:AMPLITUDE:IX:VALUE"){{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:AMPLITUDE:IY:VALUE"){{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:AMPLITUDE:IX:ERROR"){{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:AMPLITUDE:IY:ERROR"){{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:AMPLITUDE:BX:VALUE"){{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:AMPLITUDE:BY:VALUE"){{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:AMPLITUDE:BX:ERROR"){{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:AMPLITUDE:BY:ERROR"){{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:PHASE:AX:VALUE")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:PHASE:BX:VALUE")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:PHASE:AY:VALUE")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:PHASE:BY:VALUE")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:PHASE:AX:ERROR")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:PHASE:BX:ERROR")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:PHASE:AY:ERROR")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:PHASE:BY:ERROR")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    '''

def record_load(name:str, data:dict, connection_timeout:float=1.0) -> None:
    """
    Load data into epics process variables for given location.

    Parameters
    ----------
    name: str
        location name
    data: dict
        location data
        {type:str, flag:int, join:int, rise:int, s:float, *:float}

    Returns
    -------
    None

    """
    epics.caput(f'H:{name}:TYPE',     data['TYPE'], connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:FLAG',     data['FLAG'], connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:JOIN',     data['JOIN'], connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:RISE',     data['RISE'], connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:S',        data['S'],    connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:BX', data['BX'],   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:AX', data['AX'],   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:FX', data['FX'],   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:BY', data['BY'],   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:AY', data['AY'],   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:FY', data['FY'],   connection_timeout=connection_timeout)


def data_load(case:str, file:str) -> numpy.ndarray:
    """
    Return TbT data for selected plane.

    Parameters
    ----------
    case: str
        plane case ('X', 'Y' or 'I')
    file: str
        input file name (picked dataframe)

    Returns
    -------
    TbT data for selected plane (numpy.ndarray)

    """
    return numpy.array(pandas.read_pickle(file)[case].tolist())


def mod(x:float, y:float, d:float=0.0) -> float:
    """
    Returns the remainder on division of x by y with offset d.

    """
    return x - ((x - d) - (x - d) % y)


def fst(array):
    """
    Returns the first elemet.

    """
    x, *_ = array
    return x


def lst(array):
    """
    Returns the last element.

    """
    *_, x = array
    return x


def rst(array):
    """
    Returns all but the last element.

    """
    *x, _ = array
    return x


def mst(array):
    """
    Returns all but the first element.

    """
    _, *x = array
    return x


def main():
    pass


if __name__ == '__main__':
    main()