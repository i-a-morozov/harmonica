"""
Model module.

"""

import torch
import numpy
import pandas
import yaml
import json

from .util import mod, make_mask, generate_other, generate_pairs
from .decomposition import Decomposition

class Model():
    """
    Returns
    ----------
    Model class instance.

    Parameters
    ----------
    path: str
        path to config file
    limit: int
        maximum range limit
    error: bool
        flag to compute model advance errors
    dtype: torch.dtype
        data type
    device: torch.device
        data device

    Attributes
    ----------
    _epsilon: float
        float epsilon
    _head: str
        head location name
    _tail: str
        tail location name
    _virtual: str
        virtual type
    _monitor: str
        monitor type
    dtype: torch.dtype
        torch data type
    device: torch.device
        torch device
    path: str
        path to yaml file
    limit: int
        maximum range limit
    error: bool
        flag to compute model advance errors
    model: str
        model type ('uncoupled')
    dict: dict
        yaml file as dict
    data_frame: pandas.DataFrame
        yaml file as data frame
    size: int
        number of locations
    name: list
        location names
    kind: list
        location kinds (expected kinds _virtual or _monitor)
    flag: torch.Tensor
        flags
    join: list
        join flags
    rise: list
        starting turn
    time: torch.Tensor
        position
    length: torch.Tensor
        total length
    monitor_index: list
        list of monitor location indices
    virtual_index: list
        list of virtual location indices
    monitor_count: int
        number of monitor locations
    virtual_count: int
        number of virtual locations
    monitor_name: list
        list of monitor location names
    virtual_name: list
        list of virtual location names
    bx: torch.Tensor
        (uncoupled) bx
    ax: torch.Tensor
        (uncoupled) ax
    fx: torch.Tensor
        (uncoupled) fx
    sigma_bx: torch.Tensor
        (uncoupled) bx error
    sigma_ax: torch.Tensor
        (uncoupled) ax error
    sigma_fx: torch.Tensor
        (uncoupled) fx error
    by: torch.Tensor
        (uncoupled) by
    ay: torch.Tensor
        (uncoupled) ay
    fy: torch.Tensor
        (uncoupled) fy
    sigma_by: torch.Tensor
        (uncoupled) by error
    sigma_ay: torch.Tensor
        (uncoupled) ay error
    sigma_fy: torch.Tensor
        (uncoupled) fy error
    mux: torch.Tensor
        (uncoupled) total x advance
    muy: torch.Tensor
        (uncoupled) total y advance
    sigma_mux: torch.Tensor
        (uncoupled) total x advance error
    sigma_muy: torch.Tensor
        (uncoupled) total y advance error
    nux: torch.Tensor
        (uncoupled) total x tune
    nuy: torch.Tensor
        (uncoupled) total y tune
    sigma_nux: torch.Tensor
        (uncoupled) total x tune error
    sigma_nuy: torch.Tensor
        (uncoupled) total y tune error
    phase_x: torch.Tensor
        (uncoupled) x phase advance from each location to the next one
    sigma_x: torch.Tensor
        (uncoupled) x phase advance error from each location to the next one
    phase_y: torch.Tensor
        (uncoupled) y phase advance from each location to the next one
    sigma_y: torch.Tensor
        (uncoupled) y phase advance error from each monitor location to the next one
    monitor_phase_x: torch.Tensor
        (uncoupled) x phase advance from each monitor location to the next one
    monitor_sigma_x: torch.Tensor
        (uncoupled) x phase advance error from each monitor location to the next one
    monitor_phase_y: torch.Tensor
        (uncoupled) y phase advance from each monitor location to the next one
    monitor_sigma_y: torch.Tensor
        (uncoupled) y phase advance error from each monitor location to the next one
    count: torch.Tensor
        (uncoupled) range limit endpoints [1, 6, 15, 28, 45, 66, 91, 120, ...]
    combo: torch.Tensor
        (uncoupled) index combinations [..., [..., [[i, j], [i, k]], ...], ...]
    index: torch.Tensor
        (uncoupled) index combimations mod number of locations
    fx_ij: torch.Tensor
        (uncoupled) x model advance i to j
    fx_ik: torch.Tensor
        (uncoupled) x model advance i to k
    sigma_fx_ij: torch.Tensor
        (uncoupled) x model advance error i to j
    sigma_fx_ik: torch.Tensor
        (uncoupled) x model advance error i to k
    fy_ij: torch.Tensor
        (uncoupled) y model advance i to j
    fy_ik: torch.Tensor
        (uncoupled) y model advance i to k
    sigma_fy_ij: torch.Tensor
        (uncoupled) y model advance error i to j
    sigma_fy_ik: torch.Tensor
        (uncoupled) y model advance error i to k

    Methods
    ----------
    __init__(self, path:str=None, limit:int=8, error:bool=False, model:str='uncoupled', *, dtype:torch.dtype=torch.float64, device:torch.device='cpu') -> None
        Model instance initialization.
    uncoupled(self) -> None
        Set attributes for uncoupled model.
    save_uncoupled(self, file:str='model.json') -> None
        Save uncoupled model.
    load_uncoupled(self, file:str='model.json') -> None
        Load uncoupled model.
    from_uncoupled(cls, file:str='model.json', *, dtype:torch.dtype=torch.float64, device:torch.device='cpu') -> 'Model'
        Initialize uncoupled model from file.
    get_name(self, index:int) -> str:
        Return name of given location index.
    get_index(self, name:str) -> int
        Return index of given location name.
    is_monitor(self, index:int) -> bool
        Return True, if location is a monitor.
    is_virtual(self, index:int) -> bool
        Return True, if location is a virtual.
    is_same(self, probe:int, other:int) -> bool
        Return True, if locations are at the same place.
    __len__(self) -> int
        Return total number of locations (monitor & virtual).
    __getitem__(self, index:int) -> dict
        Return corresponding self.dict value for given location index.
    __call__(self, index:int) -> dict
        Return corresponding self.dict value for given location index or name.
    __repr__(self) -> str
        String representation.
    get_next(self, probe:int) -> list
        Return next location index and name.
    get_next_monitor(self, probe:int) -> list
        Return next monitor location index and name.
    get_next_virtual(self, probe:int) -> list
        Return next virtual location index and name.
    count(self, probe:int, other:int) -> int
        Count number of locations between probed and other including endpoints.
    count_monitor(self, probe:int, other:int) -> int
        Count number of monitor locations between probed and other including endpoints.
    count_virtual(self, probe:int, other:int) -> int
        Count number of virtual locations between probed and other including endpoints.

    """
    def __init__(self, path:str=None, limit:int=None, error:bool=False, model:str='uncoupled', *,
                 dtype:torch.dtype=torch.float64, device:torch.device='cpu') -> None:
        """
        Model instance initialization.

        Parameters
        ----------
        path: str
            path to config file
        limit: int
            maximum rangle limit
        model: str
            model type ('uncoupled')
        dtype: torch.dtype
            data type
        device: torch.device
            data device

        Returns
        -------
        None

        """
        self._epsilon = 1.0E-12

        self._head = 'HEAD'
        self._tail = 'TAIL'

        self._virtual = 'VIRTUAL'
        self._monitor = 'MONITOR'

        self.dtype, self.device = dtype, device

        self.path = path
        self.limit = limit
        self.error = error
        self.model = model
        self.dict = None

        if self.path is not None:

            with open(self.path) as path:
                self.dict = yaml.safe_load(path)

            self.data_frame = pandas.DataFrame.from_dict(self.dict)

            if self._head not in self.dict:
                raise Exception(f'MODEL: {self._head} record is not found in {self.path}')

            if self._tail not in self.dict:
                raise Exception(f'MODEL: {self._tail} record is not found in {self.path}')

            _, self.size = self.data_frame.shape

            self.name = [*self.data_frame.columns]

            self.kind = [*self.data_frame.loc['TYPE'].values]
            self.flag = [flag if kind == self._monitor else 0 for flag, kind in zip([*self.data_frame.loc['FLAG'].values], self.kind)]
            self.join = [*self.data_frame.loc['JOIN'].values]
            self.rise = [*self.data_frame.loc['RISE'].values]
            self.time = [*self.data_frame.loc['TIME'].values]

            self.flag = torch.tensor(self.flag, dtype=torch.int64, device=self.device)
            self.time = torch.tensor(self.time, dtype=self.dtype, device=self.device)

            *_, self.length = self.time

            self.monitor_index = [index for index, kind in enumerate(self.kind) if kind == self._monitor]
            self.virtual_index = [index for index, kind in enumerate(self.kind) if kind == self._virtual]
            self.monitor_count = len(self.monitor_index)
            self.virtual_count = len(self.virtual_index)
            self.monitor_name = [name for name, kind in zip(self.name, self.kind) if kind == self._monitor]
            self.virtual_name = [name for name, kind in zip(self.name, self.kind) if kind == self._virtual]

            if self.model == 'uncoupled':
                self.uncoupled()


    def uncoupled(self) -> None:
        """
        Set attributes for uncoupled model.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.bx = torch.tensor(self.data_frame.loc['BX'], dtype=self.dtype, device=self.device)
        self.ax = torch.tensor(self.data_frame.loc['AX'], dtype=self.dtype, device=self.device)
        self.fx = torch.tensor(self.data_frame.loc['FX'], dtype=self.dtype, device=self.device)

        self.by = torch.tensor(self.data_frame.loc['BY'], dtype=self.dtype, device=self.device)
        self.ay = torch.tensor(self.data_frame.loc['AY'], dtype=self.dtype, device=self.device)
        self.fy = torch.tensor(self.data_frame.loc['FY'], dtype=self.dtype, device=self.device)

        self.sigma_bx = torch.tensor(self.data_frame.loc['SIGMA_BX'], dtype=self.dtype, device=self.device)
        self.sigma_ax = torch.tensor(self.data_frame.loc['SIGMA_AX'], dtype=self.dtype, device=self.device)
        self.sigma_fx = torch.tensor(self.data_frame.loc['SIGMA_FX'], dtype=self.dtype, device=self.device)

        self.sigma_by = torch.tensor(self.data_frame.loc['SIGMA_BY'], dtype=self.dtype, device=self.device)
        self.sigma_ay = torch.tensor(self.data_frame.loc['SIGMA_AY'], dtype=self.dtype, device=self.device)
        self.sigma_fy = torch.tensor(self.data_frame.loc['SIGMA_FY'], dtype=self.dtype, device=self.device)

        *_, self.mux = self.fx
        *_, self.muy = self.fy

        self.sigma_mux = torch.sqrt(torch.sum(self.sigma_fx**2))
        self.sigma_muy = torch.sqrt(torch.sum(self.sigma_fy**2))

        self.nux = self.mux/(2.0*numpy.pi)
        self.nuy = self.muy/(2.0*numpy.pi)

        self.sigma_nux = self.sigma_mux/(2.0*numpy.pi)
        self.sigma_nuy = self.sigma_mux/(2.0*numpy.pi)

        probe = torch.tensor(range(self.size), dtype=torch.int64, device=self.device)
        other = probe + 1
        self.phase_x, self.sigma_x = Decomposition.phase_advance(probe, other, self.nux, self.fx, error=True, sigma_frequency=self.sigma_nux, sigma_phase=self.sigma_fx, model=True)
        self.phase_y, self.sigma_y = Decomposition.phase_advance(probe, other, self.nuy, self.fy, error=True, sigma_frequency=self.sigma_nuy, sigma_phase=self.sigma_fy, model=True)

        probe = torch.tensor(self.monitor_index, dtype=torch.int64, device=self.device)
        flags = make_mask(self.size, self.monitor_index)
        other = torch.tensor([generate_other(index.item(), 1, flags, inverse=False) for index in probe], dtype=torch.int64, device=self.device).flatten()
        self.monitor_phase_x, self.monitor_sigma_x = Decomposition.phase_advance(probe, other, self.nux, self.fx, error=True, sigma_frequency=self.sigma_nux, sigma_phase=self.sigma_fx, model=True)
        self.monitor_phase_y, self.monitor_sigma_y = Decomposition.phase_advance(probe, other, self.nuy, self.fy, error=True, sigma_frequency=self.sigma_nuy, sigma_phase=self.sigma_fy, model=True)

        if self.limit != None:

            self.count = torch.tensor([limit*(2*limit - 1) for limit in range(1, self.limit + 1)], dtype=torch.int64, device=self.device)
            self.combo = [generate_other(probe, self.limit, self.flag) for probe in range(self.size)]
            self.combo = torch.stack([generate_pairs(self.limit, 1 + 1, probe=probe, table=table, dtype=torch.int64, device=self.device) for probe, table in enumerate(self.combo)])
            self.index = mod(self.combo, self.size).to(torch.int64)

            index = self.combo.swapaxes(0, -1)

            value, sigma = Decomposition.phase_advance(*index, self.nux, self.fx, error=self.error, model=True, sigma_frequency=self.sigma_nux, sigma_phase=self.sigma_fx)
            self.fx_ij, self.fx_ik = value.swapaxes(0, 1)
            self.sigma_fx_ij, self.sigma_fx_ik = sigma.swapaxes(0, 1)

            value, sigma = Decomposition.phase_advance(*index, self.nuy, self.fy, error=self.error, model=True, sigma_frequency=self.sigma_nuy, sigma_phase=self.sigma_fy)
            self.fy_ij, self.fy_ik = value.swapaxes(0, 1)
            self.sigma_fy_ij, self.sigma_fy_ik = sigma.swapaxes(0, 1)


    def save_uncoupled(self, file:str='model.json') -> None:
        """
        Save uncoupled model.

        Parameters
        ----------
        file: str
            file name

        Returns
        -------
        None

        """
        skip = ['dtype', 'device', 'data_frame']

        data = {}
        for key, value in self.__dict__.items():
            if key not in skip:
                data[key] = value.cpu().tolist() if isinstance(value, torch.Tensor) else value

        with open(file, 'w') as stream:
            json.dump(data, stream)


    def load_uncoupled(self, file:str='model.json') -> None:
        """
        Load uncoupled model.

        Parameters
        ----------
        file: str
            file name

        Returns
        -------
        None

        """
        with open(file) as stream:
            data = json.load(stream)

        for key, value in data.items():
            setattr(self, key, value)

        self.data_frame = pandas.DataFrame.from_dict(self.dict)

        self.flag = torch.tensor(self.flag, dtype=torch.int64, device=self.device)
        self.time = torch.tensor(self.time, dtype=self.dtype, device=self.device)

        self.length = torch.tensor(self.length, dtype=self.dtype, device=self.device)

        self.bx = torch.tensor(self.bx, dtype=self.dtype, device=self.device)
        self.ax = torch.tensor(self.ax, dtype=self.dtype, device=self.device)
        self.fx = torch.tensor(self.fx, dtype=self.dtype, device=self.device)
        self.sigma_bx = torch.tensor(self.sigma_bx, dtype=self.dtype, device=self.device)
        self.sigma_ax = torch.tensor(self.sigma_ax, dtype=self.dtype, device=self.device)
        self.sigma_fx = torch.tensor(self.sigma_fx, dtype=self.dtype, device=self.device)

        self.by = torch.tensor(self.by, dtype=self.dtype, device=self.device)
        self.ay = torch.tensor(self.ay, dtype=self.dtype, device=self.device)
        self.fy = torch.tensor(self.fy, dtype=self.dtype, device=self.device)
        self.sigma_by = torch.tensor(self.sigma_by, dtype=self.dtype, device=self.device)
        self.sigma_ay = torch.tensor(self.sigma_ay, dtype=self.dtype, device=self.device)
        self.sigma_fy = torch.tensor(self.sigma_fy, dtype=self.dtype, device=self.device)

        self.mux = torch.tensor(self.mux, dtype=self.dtype, device=self.device)
        self.muy = torch.tensor(self.muy, dtype=self.dtype, device=self.device)
        self.sigma_mux = torch.tensor(self.sigma_mux, dtype=self.dtype, device=self.device)
        self.sigma_muy = torch.tensor(self.sigma_muy, dtype=self.dtype, device=self.device)

        self.nux = torch.tensor(self.nux, dtype=self.dtype, device=self.device)
        self.nuy = torch.tensor(self.nuy, dtype=self.dtype, device=self.device)
        self.sigma_nux = torch.tensor(self.sigma_nux, dtype=self.dtype, device=self.device)
        self.sigma_nuy = torch.tensor(self.sigma_nuy, dtype=self.dtype, device=self.device)

        self.phase_x = torch.tensor(self.phase_x, dtype=self.dtype, device=self.device)
        self.sigma_x = torch.tensor(self.sigma_x, dtype=self.dtype, device=self.device)
        self.phase_y = torch.tensor(self.phase_y, dtype=self.dtype, device=self.device)
        self.sigma_y = torch.tensor(self.sigma_y, dtype=self.dtype, device=self.device)

        self.monitor_phase_x = torch.tensor(self.monitor_phase_x, dtype=self.dtype, device=self.device)
        self.monitor_sigma_x = torch.tensor(self.monitor_sigma_x, dtype=self.dtype, device=self.device)
        self.monitor_phase_y = torch.tensor(self.monitor_phase_y, dtype=self.dtype, device=self.device)
        self.monitor_sigma_y = torch.tensor(self.monitor_sigma_y, dtype=self.dtype, device=self.device)

        if self.limit != None:

            self.count = torch.tensor(self.count, dtype=torch.int64, device=self.device)
            self.combo = torch.tensor(self.combo, dtype=torch.int64, device=self.device)
            self.index = torch.tensor(self.index, dtype=torch.int64, device=self.device)

            self.fx_ij = torch.tensor(self.fx_ij, dtype=self.dtype, device=self.device)
            self.fx_ik = torch.tensor(self.fx_ik, dtype=self.dtype, device=self.device)
            self.sigma_fx_ij = torch.tensor(self.sigma_fx_ij, dtype=self.dtype, device=self.device)
            self.sigma_fx_ik = torch.tensor(self.sigma_fx_ik, dtype=self.dtype, device=self.device)

            self.fy_ij = torch.tensor(self.fy_ij, dtype=self.dtype, device=self.device)
            self.fy_ik = torch.tensor(self.fy_ik, dtype=self.dtype, device=self.device)
            self.sigma_fy_ij = torch.tensor(self.sigma_fy_ij, dtype=self.dtype, device=self.device)
            self.sigma_fy_ik = torch.tensor(self.sigma_fy_ik, dtype=self.dtype, device=self.device)


    @classmethod
    def from_uncoupled(cls, file:str='model.json', *,
                       dtype:torch.dtype=torch.float64, device:torch.device='cpu') -> 'Model':
        """
        Initialize uncoupled model from file.

        Parameters
        ----------
        file: str
            file name
        dtype: torch.dtype
            data type
        device: torch.device
            data device

        Returns
        -------
        Model

        """
        model = cls(path=None, dtype=dtype, device=device)
        model.load_uncoupled(file=file)
        return model


    def get_name(self, index:int) -> str:
        """
        Return name of given location index.

        """
        index = int(mod(index, self.size))

        return self.name[index]


    def get_index(self, name:str) -> int:
        """
        Return index of given location name.

        """
        return self.name.index(name)


    def is_monitor(self, index:int) -> bool:
        """
        Return True, if location is a monitor.

        """
        index = int(mod(index, self.size))
        return self[index].get('TYPE') == self._monitor


    def is_virtual(self, index:int) -> bool:
        """
        Return True, if location is a virtual.

        """
        index = int(mod(index, self.size))
        return self[index].get('TYPE') == self._virtual


    def is_same(self, probe:int, other:int) -> bool:
        """
        Return True, if locations are at the same place.

        """
        probe = int(mod(probe, self.size))
        other = int(mod(other, self.size))
        delta = abs(self[probe].get('TIME') - self[other].get('TIME'))
        if delta < self._epsilon:
            return True
        if abs(delta - self.length) < self._epsilon:
            return True
        return False


    def __len__(self) -> int:
        """
        Return total number of locations (monitor & virtual).

        """
        return self.size


    def __getitem__(self, index:int) -> dict:
        """
        Return corresponding self.dict value for given location index.

        """
        if self.dict is None:
            return None
        return self.dict[self.name[index]]


    def __call__(self, index:int) -> dict:
        """
        Return corresponding self.dict value for given location index or name.

        """
        if self.dict is None:
            return None
        index = int(mod(index, self.size))
        if isinstance(index, int):
            return self[index]
        return self.dict[index] if index in self.name else None


    def __repr__(self) -> str:
        """
        String representation.

        """
        return f'Model(path={self.path}, model={self.model})'


    def get_next(self, probe:int) -> list:
        """
        Return next location index and name.

        """
        if isinstance(probe, int):
            index = int(mod(probe + 1, self.size))
            return [probe + 1, index, self.name[index]]
        return self.get_next(self.name.index(probe))


    def get_next_monitor(self, probe:int) -> list:
        """
        Return next monitor location index and name.

        Note, probe itself must be a monitor location.

        """
        if isinstance(probe, int):
            if probe not in self.monitor_index:
                return []
            other = self.monitor_index.index(probe) + 1
            count = 1 if other >= self.monitor_count else 0
            other = self.monitor_index[int(mod(other, self.monitor_count))]
            return [other + count*self.size, other, self.name[other]]
        return self.get_next_monitor(self.name.index(probe))


    def get_next_virtual(self, probe:int) -> list:
        """
        Return next virtual location index and name.

        Note, probe itself must be a virtual location.

        """
        if isinstance(probe, int):
            if probe not in self.virtual_index:
                return []
            other = self.virtual_index.index(probe) + 1
            count = 1 if other >= self.virtual_count else 0
            other = self.virtual_index[int(mod(other, self.virtual_count))]
            return [other + count*self.size, other, self.name[other]]
        return self.get_next_virtual(self.name.index(probe))


    def count(self, probe:int, other:int) -> int:
        """
        Count number of locations between probed and other including endpoints.

        Note, both can be negative

        Parameters
        ----------
        probe: int
            first location
        other: int
            second location

        Returns
        -------
        number of locations (int)

        """
        return len(range(probe, other + 1) if probe < other else range(other, probe + 1))


    def count_monitor(self, probe:int, other:int) -> int:
        """
        Count number of monitor locations between probed and other including endpoints.

        Note, both can be negative

        Parameters
        ----------
        probe: int
            first location
        other: int
            second location

        Returns
        -------
        number of monitor locations (int)

        """
        index = torch.tensor(range(probe, other + 1) if probe < other else range(other, probe + 1), dtype=self.dtype, device=self.device)
        index = mod(index, self.size).to(torch.int64)
        flag = [flag if kind == self._monitor else 0 for flag, kind in zip(self.flag, self.kind)]
        flag = torch.tensor(flag, dtype=torch.int64, device=self.device)
        count = flag[index]
        return count.sum().item()


    def count_virtual(self, probe:int, other:int) -> int:
        """
        Count number of virtual locations between probed and other including endpoints.

        Note, both can be negative

        Parameters
        ----------
        probe: int
            first location
        other: int
            second location

        Returns
        -------
        number of virual locations (int)

        """
        return self.count(probe, other) - self.count_monitor(probe, other)


def main():
    pass

if __name__ == '__main__':
    main()