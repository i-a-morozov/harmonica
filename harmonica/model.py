"""
Model module.

"""

import torch
import functorch
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


    @staticmethod
    @functorch.vmap
    def matrix_uncoupled(ax1:torch.Tensor, bx1:torch.Tensor, ax2:torch.Tensor, bx2:torch.Tensor, fx12:torch.Tensor,
                         ay1:torch.Tensor, by1:torch.Tensor, ay2:torch.Tensor, by2:torch.Tensor, fy12:torch.Tensor) -> torch.Tensor:
        """
        Generate uncoupled transport matrices using twiss data between given locations.

        Input twiss parameters should be 1D tensors with matching length

        Parameters
        ----------
        ax1, bx1, ay1, by1: torch.Tensor
            twiss parameters at the 1st location(s)
        ax2, bx2, ay2, by2: torch.Tensor
            twiss parameters at the 2nd location(s)
        fx12, fy12: torch.Tensor
            twiss phase advance between locations

        Returns
        -------
        uncoupled transport matrices (torch.Tensor)

        """
        cx = torch.cos(fx12)
        sx = torch.sin(fx12)

        mx11 = torch.sqrt(bx2/bx1)*(cx + ax1*sx)
        mx12 = torch.sqrt(bx1*bx2)*sx
        mx21 = -(1 + ax1*ax2)/torch.sqrt(bx1*bx2)*sx + (ax1 - ax2)/torch.sqrt(bx1*bx2)*cx
        mx22 = torch.sqrt(bx1/bx2)*(cx - ax2*sx)

        rx1 = torch.stack([mx11, mx12])
        rx2 = torch.stack([mx21, mx22])

        mx = torch.stack([rx1, rx2])

        cy = torch.cos(fy12)
        sy = torch.sin(fy12)

        my11 = torch.sqrt(by2/by1)*(cy + ay1*sy)
        my12 = torch.sqrt(by1*by2)*sy
        my21 = -(1 + ay1*ay2)/torch.sqrt(by1*by2)*sy + (ay1 - ay2)/torch.sqrt(by1*by2)*cy
        my22 = torch.sqrt(by1/by2)*(cy - ay2*sy)

        ry1 = torch.stack([my11, my12])
        ry2 = torch.stack([my21, my22])

        my = torch.stack([ry1, ry2])

        return torch.block_diag(mx, my)


    @staticmethod
    @functorch.vmap
    def matrix_kick(kn:torch.Tensor, ks:torch.Tensor) -> torch.Tensor:
        """
        Generate thin quadrupole kick matrices.

        Input parameters should be 1D tensors with matching length

        Parameters
        ----------
        kn, ks: torch.Tensor
            kn, ks

        Returns
        -------
        thin quadrupole kick matrices (torch.Tensor)

        """
        i = torch.ones_like(kn)
        o = torch.zeros_like(kn)

        m = torch.stack([
                torch.stack([i,   o,   o, o]),
                torch.stack([-kn, i, +ks, o]),
                torch.stack([  o, o,   i, o]),
                torch.stack([+ks, o, +kn, i])
            ])

        return m


    @staticmethod
    @functorch.vmap
    def matrix_roll(angle:torch.Tensor) -> torch.Tensor:
        """
        Generate roll rotation matrices.

        Input parameter should be a 1D tensor

        Parameters
        ----------
        angle: torch.Tensor
            roll angle

        Returns
        -------
        roll rotation matrices (torch.Tensor)

        """
        o = torch.zeros_like(angle)

        c = torch.cos(angle)
        s = torch.sin(angle)

        m = torch.stack([
                torch.stack([ c,  o, s, o]),
                torch.stack([ o,  c, o, s]),
                torch.stack([-s,  o, c, o]),
                torch.stack([ o, -s, o, c])
            ])

        return m


    def matrix(self, probe:torch.Tensor, other:torch.Tensor) -> torch.Tensor:
        """
        Generate uncoupled transport matrix (or matrices) for given locations.

        Matrices are generated from probe to other
        One-turn matrices are generated where probe == other
        Input parameters should be 1D tensors with matching length
        Additionaly probe and/or other input parameter can be an int or str in self.name (not checked)

        Parameters
        ----------
        probe: torch.Tensor
            probe locations
        other: torch.Tensor
            other locations

        Returns
        -------
        uncoupled transport matrices (torch.Tensor)

        """
        if isinstance(probe, int):
            probe = torch.tensor([probe], dtype=torch.int64, device=self.device)

        if isinstance(probe, str):
            probe = torch.tensor([self.name.index(probe)], dtype=torch.int64, device=self.device)

        if isinstance(other, int):
            other = torch.tensor([other], dtype=torch.int64, device=self.device)

        if isinstance(other, str):
            other = torch.tensor([self.name.index(other)], dtype=torch.int64, device=self.device)

        other[probe == other] += self.size

        fx, _ = Decomposition.phase_advance(probe, other, self.nux, self.fx)
        fy, _ = Decomposition.phase_advance(probe, other, self.nuy, self.fy)

        probe = mod(probe, self.size).to(torch.int64)
        other = mod(other, self.size).to(torch.int64)

        return self.matrix_uncoupled(self.ax[probe], self.bx[probe], self.ax[other], self.bx[other], fx,
                                self.ay[probe], self.by[probe], self.ay[other], self.by[other], fy).squeeze()

    def make_transport(self) -> None:
        """
        Set transport matrices between adjacent locations.

        self.transport[i] is a transport matrix from i to i + 1

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        probe = torch.arange(self.size, dtype=torch.int64, device=self.device)
        other = 1 + probe
        self.transport = self.matrix(probe, other)


    def make_kick(self, kn:torch.Tensor=None, ks:torch.Tensor=None) -> None:
        """
        Generate thin quadrupole kick matrices for each segment.

        self.kick[i] is a kick matrix at the end of segment i

        Parameters
        ----------
        kn, ks: torch.Tensor
            kn, ks

        Returns
        -------
        None

        """
        if isinstance(kn, float):
            kn = kn*torch.randn(self.size, dtype=self.dtype, device=self.device)

        if isinstance(ks, float):
            ks = ks*torch.randn(self.size, dtype=self.dtype, device=self.device)

        if len(kn) == self.size and len(ks) == self.size:
            self.kick = self.matrix_kick(kn, ks)
            self.error_kn = kn
            self.error_ks = ks


    def make_roll(self, angle:torch.Tensor=None) -> None:
        """
        Generate roll matrices for each segment.

        self.roll[i] is a roll matrix at the end of segment i

        Parameters
        ----------
        angle: torch.Tensor
            roll angle

        Returns
        -------
        None

        """
        if isinstance(angle, float):
            angle = angle*torch.randn(self.size, dtype=self.dtype, device=self.device)

        if len(angle) == self.size:
            self.roll = self.matrix_roll(angle)
            self.error_angle = angle


    def apply_error(self) -> None:
        """
        Apply errors to each transport segment.

        self.transport[i] = self.roll[i] @ self.kick[i] @ self.transport[i]

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if hasattr(self, 'kick'):
            self.transport = torch.matmul(self.kick, self.transport)

        if hasattr(self, 'roll'):
            self.transport = torch.matmul(self.roll, self.transport)


    def make_turn(self) -> None:
        """
        Generate one-turn matrix at 'HEAD' location.

        Set self.turn attribute

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.turn, *_ = self.transport
        for i in range(1, self.size):
            self.turn = self.transport[i] @ self.turn


    @staticmethod
    def matrix_rotation(angle:torch.Tensor) -> torch.Tensor:
        """
        Generate rotation matrix for given angles.

        Parameters
        ----------
        angle: torch.Tensor
            rotation angles

        Returns
        -------
        rotation matrix (torch.Tensor)

        """
        return torch.block_diag(*[torch.tensor([[value.cos(), +value.sin()], [-value.sin(), value.cos()]]) for value in angle])


    @staticmethod
    def twiss(matrix:torch.Tensor, *, epsilon:float=1.0E-12) -> tuple:
        """
        Compute Wolski twiss parameters for given one-turn input matrix.

        Input matrix can have arbitrary even dimension
        In-plane 'beta' is used for ordering
        If input matrix is unstable, return None for each output

        Symplectic block is [[0, 1], [-1, 0]]
        Complex block is 1/sqrt(2)*[[1, 1j], [1, -1j]]
        Rotation block is [[cos(t), sin(t)], [-sin(t), cos(t)]]

        Parameters
        ----------
        matrix: torch.Tensor
            one-turn matrix
        epsilon: float
            tolerance epsilon

        Returns
        -------
        tunes [T_1, ..., T_k], normalization matrix N and Wolski twiss matrices W = [W_1, ..., W_k] (tuple)
        M = N R N^-1 = ... + W_i S sin(T_i) - (W_i S)^2 cos(T_i) + ..., i = 1, ..., k

        """
        dtype = matrix.dtype
        device = matrix.device

        rdtype = torch.tensor(1, dtype=dtype).abs().dtype
        cdtype = (1j*torch.tensor(1, dtype=dtype)).dtype

        dimension = len(matrix) // 2

        b_p = torch.tensor([[1, 0], [0, 1]], dtype=rdtype, device=device)
        b_s = torch.tensor([[0, 1], [-1, 0]], dtype=rdtype, device=device)
        b_c = 0.5**0.5*torch.tensor([[1, +1j], [1, -1j]], dtype=cdtype, device=device)

        m_p = torch.stack([torch.block_diag(*[b_p*(i == j) for i in range(dimension)]) for j in range(dimension)])
        m_s = torch.block_diag(*[b_s for _ in range(dimension)])
        m_c = torch.block_diag(*[b_c for i in range(dimension)])

        l, v = torch.linalg.eig(matrix)

        if (l.abs() - epsilon > 1).sum():
            return None, None, None

        l, v = l.reshape(dimension, -1), v.T.reshape(dimension, -1, 2*dimension)
        for i, (v1, v2) in enumerate(v):
            v[i] /= (-1j*(v1 @ m_s.to(cdtype) @ v2)).abs().sqrt()

        for i in range(dimension):
            order = torch.imag(l[i].log()).argsort()
            l[i], v[i] = l[i, order], v[i, order]

        t = 1.0 - l.log().abs().mean(-1)/(2.0*numpy.pi)

        n = torch.cat([*v]).T.conj()
        n = (n @ m_c).real
        w = torch.zeros_like(m_p)
        for i in range(dimension):
            w[i] = n @ m_p[i] @ n.T

        order = torch.tensor([w[i].diag().argmax() for i in range(dimension)]).argsort()
        t, v = t[order], v[order]
        n = torch.cat([*v]).T.conj()
        n = (n @ m_c).real

        flag = torch.stack(torch.hsplit(n.T @ m_s @ n - m_s, dimension)).abs().sum((1, -1)) > epsilon
        for i in range(dimension):
            if flag[i]:
                t[i] = (1 - t[i]).abs()
                v[i] = v[i].conj()

        n = torch.cat([*v]).T.conj()
        n = (n @ m_c).real

        rotation = []
        for i in range(dimension):
            angle = (n[2*i, 2*i + 1] + 1j*n[2*i, 2*i]).angle() - 0.5*numpy.pi
            block = torch.tensor([[angle.cos(), angle.sin()], [-angle.sin(), angle.cos()]])
            rotation.append(block)

        n = n @ torch.block_diag(*rotation)
        for i in range(dimension):
            w[i] = n @ m_p[i] @ n.T

        return t, n, w


    @staticmethod
    def propagate_twiss(twiss:torch.Tensor, matrix:torch.Tensor) -> torch.Tensor:
        """
        Propagate Wolski twiss parameters.

        Parameters
        ----------
        wolski: torch.Tensor
            initial wolski twiss parameters
        matrix: torch.Tensor
            batch of transport matrices

        Returns
        -------
        final wolski twiss parameters for each matrix (torch.Tensor)

        """
        data = torch.zeros((len(matrix), *twiss.shape), dtype=twiss.dtype, device=twiss.device).swapaxes(0, 1)

        for i in torch.arange(len(data), device=twiss.device):
            data[i] = torch.matmul(matrix, torch.matmul(twiss[i], matrix.swapaxes(1, -1)))

        return data


    @staticmethod
    def advance_twiss(normal:torch.Tensor, matrix:torch.Tensor) -> tuple:
        """
        Compute phase advance and final normalization matrix.

        Phase advance is mod 2*pi

        Parameters
        ----------
        normal: torch.Tensor
            initial normalization matrix
        matrix: torch.Tensor
            transport matrix

        Returns
        -------
        phase advance and final normalization matrix (tuple)

        """
        dimension = len(normal) // 2

        local = torch.matmul(matrix, normal)
        table = torch.tensor([torch.arctan2(local[2*i, 2*i + 1], local[2*i, 2*i]) for i in range(dimension)])

        rotation = []
        for angle in table:
            block = torch.tensor([[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]])
            rotation.append(block)

        return table, local @ torch.block_diag(*rotation)


    @staticmethod
    def lb_normal(a1x:torch.Tensor, b1x:torch.Tensor, a2x:torch.Tensor, b2x:torch.Tensor,
                  a1y:torch.Tensor, b1y:torch.Tensor, a2y:torch.Tensor, b2y:torch.Tensor,
                  u:torch.Tensor, v1:torch.Tensor, v2:torch.Tensor, *,
                  epsilon:float=1.0E-12) -> torch.Tensor:
        """
        Generate Lebedev-Bogacz normalization matrix.

        a1x, b1x, a2y, b2y are 'in-plane' twiss parameters

        Parameters
        ----------
        a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2: torch.Tensor
            Lebedev-Bogacz twiss parameters
        epsilon: float
            tolerance epsilon

        Returns
        -------
        normalization matrix (torch.Tensor)
        M = N R N^-1

        """
        cv1, sv1 = v1.cos(), v1.sin()
        cv2, sv2 = v2.cos(), v2.sin()
        if b1x < epsilon: b1x *= 0.0
        if b2x < epsilon: b2x *= 0.0
        if b1y < epsilon: b1y *= 0.0
        if b2y < epsilon: b2y *= 0.0
        return torch.tensor(
            [
                [b1x.sqrt(), 0, b2x.sqrt()*cv2, -b2x.sqrt()*sv2],
                [-a1x/b1x.sqrt(), (1-u)/b1x.sqrt(), (-a2x*cv2 + u*sv2)/b2x.sqrt(), (a2x*sv2 + u*cv2)/b2x.sqrt()],
                [b1y.sqrt()*cv1, -b1y.sqrt()*sv1, b2y.sqrt(), 0],
                [(-a1y*cv1 + u*sv1)/b1y.sqrt(), (a1y*sv1 + u*cv1)/b1y.sqrt(), -a2y/b2y.sqrt(), (1-u)/b2y.sqrt()]
            ]
        ).nan_to_num(posinf=0.0, neginf=0.0)


    @classmethod
    def cs_normal(cls, ax:torch.Tensor, bx:torch.Tensor, ay:torch.Tensor, by:torch.Tensor) -> torch.Tensor:
        """
        Generate Courant-Snyder normalization matrix.

        Parameters
        ----------
        ax, bx, ay, by: torch.Tensor
            Courant-Snyder twiss parameters
        epsilon: float
            tolerance epsilon

        Returns
        -------
        normalization matrix (torch.Tensor)
        M = N R N^-1

        """
        return cls.lb_normal(*torch.tensor([ax, bx, 0, 0, 0, 0, ay, by, 0, 0, 0]))


    @classmethod
    def convert_wolski_lb(cls, twiss:torch.Tensor) -> torch.Tensor:
        """
        Convert Wolski twiss to Lebedev-Bogacz twiss.

        """
        a1x = -twiss[0, 0, 1]
        b1x = +twiss[0, 0, 0]
        a2x = -twiss[1, 0, 1]
        b2x = +twiss[1, 0, 0]

        a1y = -twiss[0, 2, 3]
        b1y = +twiss[0, 2, 2]
        a2y = -twiss[1, 2, 3]
        b2y = +twiss[1, 2, 2]

        u = 1/2*(1 + a1x**2 - a1y**2 - b1x*twiss[0, 1, 1] + b1y*twiss[0, 3, 3])

        cv1 = (1/torch.sqrt(b1x*b1y)*twiss[0, 0, 2]).nan_to_num(nan=-1.0)
        sv1 = (1/u*(a1y*cv1 + 1/torch.sqrt(b1x)*(torch.sqrt(b1y)*twiss[0, 0, 3]))).nan_to_num(nan=0.0)

        cv2 = (1/torch.sqrt(b2x*b2y)*twiss[1, 0, 2]).nan_to_num(nan=+1.0)
        sv2 = (1/u*(a2x*cv2 + 1/torch.sqrt(b2y)*(torch.sqrt(b2x)*twiss[1, 1, 2]))).nan_to_num(nan=0.0)

        v1 = torch.arctan2(sv1, cv1)
        v2 = torch.arctan2(sv2, cv2)

        return torch.tensor([a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2])


    @classmethod
    def convert_lb_wolski(cls,
                          a1x:torch.Tensor, b1x:torch.Tensor, a2x:torch.Tensor, b2x:torch.Tensor,
                          a1y:torch.Tensor, b1y:torch.Tensor, a2y:torch.Tensor, b2y:torch.Tensor,
                          u:torch.Tensor, v1:torch.Tensor, v2:torch.Tensor) -> torch.Tensor:
        """
        Convert Lebedev-Bogacz twiss to Wolski twiss.

        """
        n = cls.lb_normal(a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2)

        p1 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=n.dtype, device=n.device)
        p2 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=n.dtype, device=n.device)

        w1 = torch.matmul(n, torch.matmul(p1, n.T))
        w2 = torch.matmul(n, torch.matmul(p2, n.T))

        return torch.stack([w1, w2])


    @classmethod
    def convert_wolski_cs(cls, twiss):
        """
        Convert Wolski twiss to Courant-Snyder twiss.

        """
        return cls.convert_wolski_lb(twiss)[[0, 1, 6, 7]]


    @classmethod
    def convert_cs_wolski(cls, ax:torch.Tensor, bx:torch.Tensor, ay:torch.Tensor, by:torch.Tensor) -> torch.Tensor:
        """
        Convert Courant-Snyder twiss to Wolski twiss.

        """
        return cls.convert_lb_wolski(*torch.tensor([ax, bx, 0, 0, 0, 0, ay, by, 0, 0, 0]))


    def make_twiss(self) -> bool:
        """
        Compute Wolski twiss parameters.

        Set self.is_stable attribute and if self.stable is True

        Set Wolski twiss parameters self.wolski, self.woski[i] are parameters at location i
        Note, Wolski twiss parameters can be converted to LB/CS using self.convert_wolski_lb/cs(self.wolski[i])

        Set normalization matrices self.normal, self.normal[i] is a normalization matrix at location i

        Set phase advance between adjacent locations self.advance, self.advance[i] is a phase advance from i to i + 1
        And self.tune is accumulated phase advance over 2*pi

        self.transport[i] = self.normal[i + 1] @ self.make_roration(self.advance[i]) @ self.normal[i].inverse()

        Parameters
        ----------
        None

        Returns
        -------
        stable flag (bool)

        """
        if not hasattr(self, 'transport'):
            self.make_transport()

        if not hasattr(self, 'turn'):
            self.make_turn()

        tune, normal, wolski = self.twiss(self.turn)

        self.is_stable = tune is not None

        if not self.is_stable:
            return self.is_stable

        self.tune = tune

        self.normal = torch.zeros((self.size, *self.turn.shape), dtype=self.dtype, device=self.device)
        self.wolski = torch.zeros((self.size, *wolski.shape), dtype=self.dtype, device=self.device)
        self.advance = torch.zeros((self.size, *self.tune.shape), dtype=self.dtype, device=self.device)

        for i in range(self.size):

            self.wolski[i] = torch.clone(wolski)
            wolski = self.propagate_twiss(wolski, self.transport[i].unsqueeze(0)).squeeze()

            self.normal[i] = torch.clone(normal)
            self.advance[i], normal = self.advance_twiss(normal, self.transport[i])

        self.tune = self.advance.sum(0)/(2.0*numpy.pi)

        return self.is_stable


    def make_trajectory(self, length:int, initial:torch.Tensor) -> torch.Tensor:
        """
        Generate test trajectories for given initial condition.

        Parameters
        ----------
        length: int
            number of iterations
        initial: torch.Tensor
            initial condition at 'HEAD' location

        Returns
        -------
        trajectories (torch.Tensor)

        """
        if not hasattr(self, 'transport'):
            self.make_transport()

        if not hasattr(self, 'turn'):
            self.make_turn()

        trajectory = torch.zeros((self.size, length, *initial.shape), dtype=self.dtype, device=self.device)

        trajectory[0, 0] = initial

        for i in range(1, length):
            trajectory[0, i] = self.turn @ trajectory[0, i - 1]

        for i in range(1, self.size):
            trajectory[i] = (self.transport[i - 1] @ trajectory[i - 1].T).T

        return trajectory


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