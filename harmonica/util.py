"""
Utility constants & functions.

"""

# Maximum length to read from TbT PVs
LIMIT:int = 8192

# Ring circumference
LENGTH:float = 366.075015600006

# Generate TbT PV name
def pv_make(name:str, plane:str, flag:bool=False) -> str:
    """
    Generate PV name for given BPM name and plane.

    Use harmonica PV name prefix if test flag is True.

    Parameters
    ----------
    name: str
        BPM name, not checked on input
    plane: str
        selected plane (x, z or i), not checked on input
    flag: bool
        flag to use harmonica PV name prefix

    Returns
    -------
    PV name (str)

    """
    return f'HARMONICA:{name}:turns_{plane}-I' if flag else f'VEPP4:{name}:turns_{plane}-I'


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