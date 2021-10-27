"""
Utility constants & functions.

"""

# Number of BPMs
NBPM = 54

# Maximum allowed number of turns to read from PVs
LIMIT = 8192

# Ring circumference
LENGTH = 366.075015600006

# BPM PV name generator
def pv_make(name: str, plane: str, test: bool = False) -> str:
    """
    Generate PV name (str) for given BPM name (str) and plane (str).

    Use test PV names if test flag.

    Parameters
    ----------
    name: str
        BPM name
    plane: str
        selected plane (x, z or i)
    test: bool
        flag to use test PV names, replace prefix

    Returns
    -------
    None

    """
    return f"TEST:{name}:turns_{plane}-I" if test else f"VEPP4:{name}:turns_{plane}-I"
