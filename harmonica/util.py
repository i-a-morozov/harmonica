"""
Utility constants & functions.

"""

# Number of BPMs
BPM = 54

# Maximum allowed number of turns to read from PVs
LIMIT = 8192

# Ring circumference
LENGTH = 366.075015600006

# Model tunes
QX = 8.53688309873731
QZ = 7.57677562060534

# BPM PV name generator
def pv_make(name: str, plane: str, flag: bool = False) -> str:
    """
    Generate PV name (str) for given BPM name (str) and plane (str).

    Use test PV names if test flag.

    Parameters
    ----------
    name: str
        BPM name
    plane: str
        selected plane (x, z or i)
    flag: bool
        flag to use harmonica PV names, replace prefix

    Returns
    -------
    None

    """
    return f'HARMONICA:{name}:turns_{plane}-I' if flag else f'VEPP4:{name}:turns_{plane}-I'


def fst(array):
    x, *_ = array
    return x

def lst(array):
    *_, x = array
    return x

def rst(array):
    *x, _ = array
    return x

def mst(array):
    _, *x = array
    return x

def main():
    pass

if __name__ == '__main__':
    main()