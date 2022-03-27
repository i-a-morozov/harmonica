"""
Utility module.
Constants & auxiliary functions.

"""

import epics
import numpy
import torch
import pandas

from itertools import combinations


# Maximum allowed length to read from TbT PVs
LIMIT:int = 8192


# Circumference
LENGTH:float = 366.075015600006


# Generate TbT PV name
def pv_make(name:str, plane:str, flag:bool=False) -> str:
    """
    Generate PV name for given BPM name and data plane.

    Note, use harmonica PV prefix if flag is True

    Parameters
    ----------
    name: str
        BPM name
    plane: str
        selected plane (x, y or i)
    flag: bool
        flag to use harmonica PV prefix

    Returns
    -------
    PV name (str)

    """
    if not flag:
        plane = {'X':'x', 'Y':'z', 'I':'i'}[plane.upper()]

    return f'H:{name}:DATA:{plane.upper()}' if flag else f'VEPP4:{name}:turns_{plane}-I'


# Generate DB location record
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
    record(waveform, "H:{name}:TIME")              {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:BX")          {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:AX")          {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:FX")          {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:BY")          {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:AY")          {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:FY")          {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:SIGMA_BX")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:SIGMA_AX")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:SIGMA_FX")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:SIGMA_BY")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:SIGMA_AY")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
    record(waveform, "H:{name}:MODEL:SIGMA_FY")    {{field(NELM, "1")    field(FTVL, "DOUBLE")}}
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


# Load data to location record
def record_load(name:str, data:dict, connection_timeout:float=1.0) -> None:
    """
    Load data into epics process variables for given location.

    Parameters
    ----------
    name: str
        location name
    data: dict
        location data
        {TYPE:str, FLAG:int, JOIN:int, RISE:int, TIME:float, *:float}

    Returns
    -------
    None

    """
    epics.caput(f'H:{name}:TYPE',     data.get('TYPE'), connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:FLAG',     data.get('FLAG'), connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:JOIN',     data.get('JOIN'), connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:RISE',     data.get('RISE'), connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:TIME',     data.get('TIME'), connection_timeout=connection_timeout)

    epics.caput(f'H:{name}:MODEL:BX', data.get('BX'),   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:AX', data.get('AX'),   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:FX', data.get('FX'),   connection_timeout=connection_timeout)

    epics.caput(f'H:{name}:MODEL:BY', data.get('BY'),   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:AY', data.get('AY'),   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:FY', data.get('FY'),   connection_timeout=connection_timeout)

    epics.caput(f'H:{name}:MODEL:SIGMA_BX', data.get('SIGMA_BX'),   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:SIGMA_AX', data.get('SIGMA_AX'),   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:SIGMA_FX', data.get('SIGMA_FX'),   connection_timeout=connection_timeout)

    epics.caput(f'H:{name}:MODEL:SIGMA_BY', data.get('SIGMA_BY'),   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:SIGMA_AY', data.get('SIGMA_AY'),   connection_timeout=connection_timeout)
    epics.caput(f'H:{name}:MODEL:SIGMA_FY', data.get('SIGMA_FY'),   connection_timeout=connection_timeout)


# Load TbT data from file
def data_load(case:str, file:str) -> numpy.ndarray:
    """
    Load TbT data for selected case from file (pickled dataframe).

    Parameters
    ----------
    plane: str
        selected plane ('X', 'Y' or 'I')
    file: str
        input file name (picked dataframe)

    Returns
    -------
    TbT data for selected plane (numpy.ndarray)

    """
    return numpy.array(pandas.read_pickle(file)[plane].tolist())


# Remainder with offset
def mod(x:float, y:float, z:float=0.0) -> float:
    """
    Return the remainder on division of x by y with offset z.

    Note, float value is returned

    """
    return x - ((x - z) - (x - z) % y)


# Chain
def chain(pairs):
    """
    Generate chain from location pairs.

    """
    table, chain = [], []

    for i in numpy.unique(numpy.array(pairs).flatten()):
        if chain == []:
            chain.append(i)
            value = i
            continue
        if i == value + 1:
            chain.append(i)
            value = i
            continue
        table.append(chain)
        chain = []
        chain.append(i)
        value = i
    else:
        table.append(chain)

    return table


# Generate pairs
def generate_pairs(limit:int, count:int, *, probe:int=0, table:list=None) -> list:
    """
    Generate combinations of unique pairs of the probed location with other locations.

    Note, the probed location has index 0, other locations are in range defined by limit

    Parameters
    ----------
    limit: int
        maximum distance from the probed location
    count: int
        number of unique locations in combination
    table: list
        list of other indices

    Returns
    -------
    [..., [combination_i], ...]

    """
    other = [i for i in range(-limit, 1 + limit) if i != 0] if table is None else table
    pairs = [(probe, i) for i in other]
    stock = {pair: abs(sum(pair)) for pair in pairs}
    combo = list(combinations(pairs, count))
    stock = [sum(stock[j] for j in i) for i in combo]
    combo = [[stock, list(map(list, combo))] for combo, stock in zip(combo, stock)]
    return [pair for _, pair in sorted(combo)]


# Generate indices of other locations
def generate_other(probe:int, limit:int, flags:list, *, inverse:bool=True, forward:bool=True) -> list:
    """
    Generate indices of other locations for given probe location, limit range and flags.


    Parameters
    ----------
    probe: int
        probe index
    limit: int
        maximum distance from the probed location
    flags: list
        virtual/monitor flags 0/1 for each location
    inverse: bool
        flag to move in the inverse direction
    forward: bool
        flag to move in the forward direction

    Returns
    -------
    other (list)

    """
    other = []
    total = len(flags)

    local = []
    index = probe
    while inverse:
        index -= 1
        if flags[int(mod(index, total))] == 1:
            local.append(index)
        if len(local) == limit:
            break
    other.extend(reversed(local))

    local = []
    index = probe
    while forward:
        index += 1
        if flags[int(mod(index, total))] == 1:
            local.append(index)
        if len(local) == limit:
            break
    other.extend(local)

    return other

# Make mark from mask
def make_mark(size:int, mask:torch.Tensor, *,
              dtype:torch.dtype=torch.int64, device:torch.device='cpu') -> torch.Tensor:
    """
    Compute mark for given mask.

    Parameters
    ----------
    size: int
        data size
    mask: torch.Tensor
        bool mask

    Returns
    -------
    mark (torch.Tensor)

    """
    return torch.arange(size, dtype=dtype, device=device)[mask]


# Make mask from mark
def make_mask(size:int, mark:torch.Tensor, *,
              dtype:torch.dtype=torch.bool, device:torch.device='cpu') -> torch.Tensor:
    """
    Compute mask for given mark.

    Parameters
    ----------
    size: int
        data size
    mark: torch.Tensor
        data indices

    Returns
    -------
    mask (torch.Tensor)

    """
    mask = torch.zeros(size, dtype=dtype, device=device)
    mask[mark] = True
    return mask


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