"""
Model module.
Setup linear model from CS twiss or normalization matrices at locations of interest.

"""
from __future__ import annotations

import torch
import functorch
import numpy
import pandas
import yaml
import json

from .util import mod, make_mask, generate_other, generate_pairs
from .decomposition import Decomposition
from .parameterization import matrix_uncoupled, matrix_coupled, matrix_rotation
from .parameterization import twiss_compute, twiss_propagate, twiss_phase_advance
from .parameterization import cs_normal


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
    model: str
        model type ('uncoupled' or 'coupled')
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
        model type ('uncoupled' or 'coupled')
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
    ax: torch.Tensor
        ax
    bx: torch.Tensor
        bx
    fx: torch.Tensor
        fx
    ay: torch.Tensor
        ay
    by: torch.Tensor
        by
    fy: torch.Tensor
        fy
    sigma_ax: torch.Tensor
        ax error
    sigma_bx: torch.Tensor
        bx error
    sigma_fx: torch.Tensor
        fx error
    sigma_ay: torch.Tensor
        ay error
    sigma_by: torch.Tensor
        by error
    sigma_fy: torch.Tensor
        fy error
    mux: torch.Tensor
        total x advance
    muy: torch.Tensor
        total y advance
    sigma_mux: torch.Tensor
        total x advance error
    sigma_muy: torch.Tensor
        total y advance error
    nux: torch.Tensor
        total x tune
    nuy: torch.Tensor
        total y tune
    sigma_nux: torch.Tensor
        total x tune error
    sigma_nuy: torch.Tensor
        total y tune error
    phase_x: torch.Tensor
        x phase advance from each location to the next one
    sigma_x: torch.Tensor
        x phase advance error from each location to the next one
    phase_y: torch.Tensor
        y phase advance from each location to the next one
    sigma_y: torch.Tensor
        y phase advance error from each monitor location to the next one
    normal: torch.Tensor
        normalization matrix for each location
    sigma_normal: torch.Tensor
        error normalization matrix for each location
    monitor_phase_x: torch.Tensor
        x phase advance from each monitor location to the next one
    monitor_sigma_x: torch.Tensor
        x phase advance error from each monitor location to the next one
    monitor_phase_y: torch.Tensor
        y phase advance from each monitor location to the next one
    monitor_sigma_y: torch.Tensor
        y phase advance error from each monitor location to the next one
    count: torch.Tensor
        range limit endpoints [1, 6, 15, 28, 45, 66, 91, 120, ...]
    table: torch.Tensor
        index combinations
    combo: torch.Tensor
        index combinations (triplets) [..., [..., [[i, j], [i, k]], ...], ...]
    index: torch.Tensor
        index combimations mod number of locations
    fx_ij: torch.Tensor
        x model advance i to j
    fx_ik: torch.Tensor
        x model advance i to k
    sigma_fx_ij: torch.Tensor
        x model advance error i to j
    sigma_fx_ik: torch.Tensor
        x model advance error i to k
    fy_ij: torch.Tensor
        y model advance i to j
    fy_ik: torch.Tensor
        y model advance i to k
    sigma_fy_ij: torch.Tensor
        y model advance error i to j
    sigma_fy_ik: torch.Tensor
        y model advance error i to k
    transport: torch.Tensor
        transport matrices between locations
    kick: torch.Tensor
        error kick matrices
    error_kn: torch.Tensor
        normal kick errors
    error_ks: torch.Tensor
        skew kick errors
    turn: torch.Tensor
        one-turn matrix
    is_stable: bool
        one-turn matrix stability flag (make_twiss)
    twiss: torch.Tensor
        Wolski twiss matrices (make_twiss)
    advance: torch.Tensor
        phase advance between locations (make_twiss)
    tune: torch.Tensor
        accumulated tunes (make_twiss)
    normalization: torch.Tensor
         normalization matrices (make_twiss)

    Methods
    ----------
    __init__(self, path:str=None, limit:int=None, error:bool=False, model:str='uncoupled', *, dtype:torch.dtype=torch.float64, device:torch.device=torch.device('cpu')) -> None
        Model instance initialization.
    uncoupled(self) -> None
        Set attributes for uncoupled model.
    coupled(self) -> None
        Set attributes for coupled model.
    config_uncoupled(self, file:str, *, name:list=None, kind:list=None, flag:list=None, join:list=None, rise:list=None, time:list=None, ax:torch.Tensor=None, bx:torch.Tensor=None, fx:torch.Tensor=None, ay:torch.Tensor=None, by:torch.Tensor=None, fy:torch.Tensor=None, sigma_ax:torch.Tensor=None, sigma_bx:torch.Tensor=None, sigma_fx:torch.Tensor=None, sigma_ay:torch.Tensor=None, sigma_by:torch.Tensor=None, sigma_fy:torch.Tensor=None, epsilon:float=1.0E-12) -> None
        Save uncoupled model configuration to file.
    config_coupled(self, file:str, *, name:list=None, kind:list=None, flag:list=None, join:list=None, rise:list=None, time:list=None, normal:torch.Tensor=None, fx:torch.Tensor=None, fy:torch.Tensor=None, sigma_normal:torch.Tensor=None, sigma_fx:torch.Tensor=None, sigma_fy:torch.Tensor=None, epsilon:float=1.0E-12) -> None
        Save coupled model configuration to file.
    save(self, file:str='model.json') -> None
        Save model.
    load(self, file:str='model.json') -> None
        Load model.
    from_file(cls, file:str='model.json', *, dtype:torch.dtype=torch.float64, device:torch.device=torch.device('cpu')) -> Model
        Initialize model from file.
    matrix(self, probe:torch.Tensor, other:torch.Tensor) -> torch.Tensor
        Generate transport matrices between given probe and other locations.
    make_transport(self) -> None
        Set transport matrices between adjacent locations.
    matrix_transport(self, probe:int, other:int) -> torch.Tensor
        Generate transport matrix from probe to other using self.transport.
    matrix_kick(kn:torch.Tensor, ks:torch.Tensor) -> torch.Tensor
        Generate thin quadrupole kick matrices.
    make_kick(self, kn:torch.Tensor=None, ks:torch.Tensor=None) -> None
        Generate thin quadrupole kick errors for each segment.
    apply_error(self) -> None
        Apply errors to each transport segment.
    make_turn(self) -> None
        Generate one-turn matrix at the 'HEAD' location using self.transport matrices.
    make_twiss(self) -> bool
        Compute and set twiss using self.transport matrices.
    make_trajectory(self, length:int, initial:torch.Tensor, *, full:bool=True) -> torch.Tensor
        Generate test trajectories for given initial condition.
    make_momenta(matrix:torch.Tensor, qx1:torch.Tensor, qx2:torch.Tensor, qy1:torch.Tensor, qy2:torch.Tensor) -> torch.Tensor
        Compute momenta at position 1 for given transport matrix and coordinates at 1 & 2.
    make_momenta_error(matrix:torch.Tensor, sigma_qx1:torch.Tensor, sigma_qx2:torch.Tensor, sigma_qy1:torch.Tensor, sigma_qy2:torch.Tensor, *, sigma_matrix:torch.Tensor=None) -> torch.Tensor
        Compute momenta errors at position 1 for given transport matrix and coordinates at 1 & 2.
    get_name(self, index:int) -> str
        Return name of given location index.
    get_index(self, name:str) -> int
        Return index of given location name.
    is_monitor(self, index:int) -> bool
        Return True, if location is a monitor.
    is_virtual(self, index:int) -> bool
        Return True, if location is a virtual.
    is_same(self, probe:int, other:int) -> bool
        Return True, if locations are at the same place.
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
     __len__(self) -> int
        Return total number of locations (monitor & virtual).
    __getitem__(self, index:int) -> dict
        Return corresponding self.dict value for given location index.
    __call__(self, index:int) -> dict
        Return corresponding self.dict value for given location index or name.
    __repr__(self) -> str
        String representation.

    """
    def __init__(self,
                 path:str=None,
                 limit:int=None,
                 error:bool=False,
                 model:str='uncoupled',
                 *,
                 dtype:torch.dtype=torch.float64,
                 device:torch.device=torch.device('cpu')) -> None:
        """
        Model instance initialization.

        Parameters
        ----------
        path: str
            path to config file
        limit: int
            maximum rangle limit
        model: str
            model type ('uncoupled' or 'coupled')
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

            if self.model == 'coupled':
                self.coupled()

            self.fx = torch.tensor(self.data_frame.loc['FX'], dtype=self.dtype, device=self.device)
            self.fy = torch.tensor(self.data_frame.loc['FY'], dtype=self.dtype, device=self.device)

            self.sigma_fx = torch.tensor(self.data_frame.loc['SIGMA_FX'], dtype=self.dtype, device=self.device)
            self.sigma_fy = torch.tensor(self.data_frame.loc['SIGMA_FY'], dtype=self.dtype, device=self.device)

            *_, self.mux = self.fx
            *_, self.muy = self.fy

            self.sigma_mux = torch.sqrt(torch.sum(self.sigma_fx**2))
            self.sigma_muy = torch.sqrt(torch.sum(self.sigma_fy**2))

            self.nux = self.mux/(2.0*numpy.pi)
            self.nuy = self.muy/(2.0*numpy.pi)

            self.sigma_nux = self.sigma_mux/(2.0*numpy.pi)
            self.sigma_nuy = self.sigma_mux/(2.0*numpy.pi)

            probe = torch.arange(self.size, dtype=torch.int64, device=self.device)
            other = probe + 1
            self.phase_x, self.sigma_x = Decomposition.phase_advance(probe, other, self.nux, self.fx, error=True, sigma_frequency=self.sigma_nux, sigma_phase=self.sigma_fx, model=True)
            self.phase_y, self.sigma_y = Decomposition.phase_advance(probe, other, self.nuy, self.fy, error=True, sigma_frequency=self.sigma_nuy, sigma_phase=self.sigma_fy, model=True)

            self.phase_x = mod(self.phase_x + self._epsilon, 2.0*numpy.pi) - self._epsilon
            self.phase_y = mod(self.phase_y + self._epsilon, 2.0*numpy.pi) - self._epsilon

            probe = torch.tensor(self.monitor_index, dtype=torch.int64, device=self.device)
            flags = make_mask(self.size, self.monitor_index)
            other = torch.tensor([generate_other(index.item(), 1, flags, inverse=False) for index in probe], dtype=torch.int64, device=self.device).flatten()

            self.monitor_phase_x, self.monitor_sigma_x = Decomposition.phase_advance(probe, other, self.nux, self.fx, error=True, sigma_frequency=self.sigma_nux, sigma_phase=self.sigma_fx, model=True)
            self.monitor_phase_y, self.monitor_sigma_y = Decomposition.phase_advance(probe, other, self.nuy, self.fy, error=True, sigma_frequency=self.sigma_nuy, sigma_phase=self.sigma_fy, model=True)

            self.monitor_phase_x = mod(self.monitor_phase_x + self._epsilon, 2.0*numpy.pi) - self._epsilon
            self.monitor_phase_y = mod(self.monitor_phase_y + self._epsilon, 2.0*numpy.pi) - self._epsilon

            if self.limit is not None:

                self.count = torch.tensor([limit*(2*limit - 1) for limit in range(1, self.limit + 1)], dtype=torch.int64, device=self.device)
                self.table = [generate_other(probe, self.limit, self.flag) for probe in range(self.size)]
                self.combo = torch.stack([generate_pairs(self.limit, 1 + 1, probe=probe, table=table, dtype=torch.int64, device=self.device) for probe, table in enumerate(self.table)])
                self.table = torch.tensor(self.table, dtype=torch.int64, device=self.device)
                self.index = mod(self.combo, self.size).to(torch.int64)

                index = self.combo.swapaxes(0, -1)

                value, sigma = Decomposition.phase_advance(*index, self.nux, self.fx, error=self.error, model=True, sigma_frequency=self.sigma_nux, sigma_phase=self.sigma_fx)
                self.fx_ij, self.fx_ik = value.swapaxes(0, 1)
                self.sigma_fx_ij, self.sigma_fx_ik = sigma.swapaxes(0, 1)

                value, sigma = Decomposition.phase_advance(*index, self.nuy, self.fy, error=self.error, model=True, sigma_frequency=self.sigma_nuy, sigma_phase=self.sigma_fy)
                self.fy_ij, self.fy_ik = value.swapaxes(0, 1)
                self.sigma_fy_ij, self.sigma_fy_ik = sigma.swapaxes(0, 1)


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
        self.ax = torch.tensor(self.data_frame.loc['AX'], dtype=self.dtype, device=self.device)
        self.bx = torch.tensor(self.data_frame.loc['BX'], dtype=self.dtype, device=self.device)

        self.ay = torch.tensor(self.data_frame.loc['AY'], dtype=self.dtype, device=self.device)
        self.by = torch.tensor(self.data_frame.loc['BY'], dtype=self.dtype, device=self.device)

        self.sigma_ax = torch.tensor(self.data_frame.loc['SIGMA_AX'], dtype=self.dtype, device=self.device)
        self.sigma_bx = torch.tensor(self.data_frame.loc['SIGMA_BX'], dtype=self.dtype, device=self.device)

        self.sigma_ay = torch.tensor(self.data_frame.loc['SIGMA_AY'], dtype=self.dtype, device=self.device)
        self.sigma_by = torch.tensor(self.data_frame.loc['SIGMA_BY'], dtype=self.dtype, device=self.device)

        self.normal = torch.stack([self.ax, self.bx, self.ay, self.by]).T
        self.normal = torch.stack([cs_normal(*twiss, dtype=self.dtype, device=self.device) for twiss in self.normal])


    def coupled(self) -> None:
        """
        Set attributes for coupled model.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.normal = self.data_frame.loc[['N00', 'N01', 'N02', 'N03', 'N10', 'N11', 'N12', 'N13', 'N20', 'N21', 'N22', 'N23', 'N30', 'N31', 'N32', 'N33']].to_numpy().tolist()
        self.normal = torch.tensor(self.normal, dtype=self.dtype, device=self.device)
        self.normal[self.normal.abs() < self._epsilon] = 0.0
        self.normal = self.normal.T.reshape(self.size, 4, 4)

        self.sigma_normal = self.data_frame.loc[['SIGMA_N00', 'SIGMA_N01', 'SIGMA_N02', 'SIGMA_N03', 'SIGMA_N10', 'SIGMA_N11', 'SIGMA_N12', 'SIGMA_N13', 'SIGMA_N20', 'SIGMA_N21', 'SIGMA_N22', 'SIGMA_N23', 'SIGMA_N30', 'SIGMA_N31', 'SIGMA_N32', 'SIGMA_N33']].to_numpy().tolist()
        self.sigma_normal = torch.tensor(self.sigma_normal, dtype=self.dtype, device=self.device)
        self.sigma_normal[self.sigma_normal.abs() < self._epsilon] = 0.0
        self.sigma_normal = self.sigma_normal.T.reshape(self.size, 4, 4)


    def config_uncoupled(self,
                         file:str,
                         *,
                         name:list=None,
                         kind:list=None,
                         flag:list=None,
                         join:list=None,
                         rise:list=None,
                         time:list=None,
                         ax:torch.Tensor=None,
                         bx:torch.Tensor=None,
                         fx:torch.Tensor=None,
                         ay:torch.Tensor=None,
                         by:torch.Tensor=None,
                         fy:torch.Tensor=None,
                         sigma_ax:torch.Tensor=None,
                         sigma_bx:torch.Tensor=None,
                         sigma_fx:torch.Tensor=None,
                         sigma_ay:torch.Tensor=None,
                         sigma_by:torch.Tensor=None,
                         sigma_fy:torch.Tensor=None,
                         epsilon:float=1.0E-12) -> None:
        """
        Save uncoupled model configuration to file.

        Note, instance attribute is used corresponding named parameter is None.

        Parameters
        ----------
        file: str
            output file name
        name: list
            location names
        kind: list
            location kinds (expected kinds _virtual or _monitor)
        flag: list
            flags
        join: list
            join flags
        rise: list
            starting turn
        time: torch.Tensor
            position
        ax, bx, fx: torch.Tensor
            x plane twiss data
        ay, by, fy: torch.Tensor
            y plane twiss data
        sigma_ax, sigma_bx, sigma_fx: torch.Tensor
            x plane twiss error
        sigma_ay, sigma_by, sigma_fy: torch.Tensor
            y plane twiss error
        epsilon: float
            epsilon

        Returns
        -------
        None

        """
        size = self.size

        if name is not None:
            if not isinstance(name, list):
                raise TypeError(f'MODEL: name parameter type mismatch, expected list of strings')
            if len(name) != size:
                raise ValueError(f'MODEL: name parameter size mismatch')
            head, *_, tail = name
            if head != self._head:
                raise ValueError(f'MODEL: expected {self._head} at the first position')
            if tail != self._tail:
                raise ValueError(f'MODEL: expected {self._tail} at the last position')
            if not all(isinstance(location, str) for location in name):
                raise ValueError(f'MODEL: name parameter type mismatch, expected list of strings')
        else:
            name = self.name

        if kind is not None:
            if not isinstance(kind, list):
                raise TypeError(f'MODEL: kind parameter type mismatch, expected list of strings')
            if len(kind) != size:
                raise ValueError(f'MODEL: kind parameter size mismatch')
            if not all(isinstance(location, str) for location in kind):
                raise ValueError(f'MODEL: kind parameter type mismatch, expected list of strings')
            if set(kind) != set([self._monitor, self._virtual]):
                raise ValueError(f'MODEL: kind parameter value mismatch, expected {self._monitor} or {self._virtual}')
        else:
            kind = self.kind

        if flag is not None:
            if not isinstance(flag, list):
                raise TypeError(f'MODEL: flag parameter type mismatch, expected list of integers')
            if len(flag) != size:
                raise ValueError(f'MODEL: flag parameter size mismatch')
            if not all(isinstance(location, int) for location in flag):
                raise TypeError(f'MODEL: flag parameter type mismatch, expected list of integers')
            if set(flag) != set([0, 1]):
                raise ValueError(f'MODEL: flag parameter value mismatch, expected 0 or 1')
        else:
            flag = self.flag.cpu().numpy().tolist()

        if join is not None:
            if not isinstance(join, list):
                raise TypeError(f'MODEL: join parameter type mismatch, expected list of integers')
            if len(join) != size:
                raise ValueError(f'MODEL: join parameter size mismatch')
            if not all(isinstance(location, int) for location in join):
                raise TypeError(f'MODEL: join parameter type mismatch, expected list of integers')
            if set(join) != set([0, 1]):
                raise ValueError(f'MODEL: join parameter value mismatch, expected 0 or 1')
        else:
            join = self.join

        if rise is not None:
            if not isinstance(rise, list):
                raise TypeError(f'MODEL: rise parameter type mismatch, expected list of integers')
            if len(rise) != size:
                raise ValueError(f'MODEL: rise parameter size mismatch')
            if not all(isinstance(location, int) for location in rise):
                raise TypeError(f'MODEL: rise parameter type mismatch, expected list of integers')
        else:
            rise = self.rise

        if time is not None:
            if not isinstance(time, list):
                raise TypeError(f'MODEL: time parameter type mismatch, expected list of floats')
            if len(time) != size:
                raise ValueError(f'MODEL: time parameter size mismatch')
            if not all(isinstance(location, float) for location in time):
                raise TypeError(f'MODEL: time parameter type mismatch, expected list of floats')
        else:
            time = self.time.cpu().numpy().tolist()

        if ax is not None:
            if not isinstance(ax, torch.Tensor):
                raise TypeError(f'MODEL: ax parameter type mismatch, tensor expected')
            if len(ax) != size:
                raise ValueError(f'MODEL: ax parameter size mismatch')
            ax = ax.cpu().numpy().tolist()
        else:
            ax = self.ax.cpu().numpy().tolist()

        if bx is not None:
            if not isinstance(bx, torch.Tensor):
                raise TypeError(f'MODEL: bx parameter type mismatch, tensor expected')
            if len(bx) != size:
                raise ValueError(f'MODEL: bx parameter size mismatch')
            bx = bx.cpu().numpy().tolist()
        else:
            bx = self.bx.cpu().numpy().tolist()

        if fx is not None:
            if not isinstance(fx, torch.Tensor):
                raise TypeError(f'MODEL: fx parameter type mismatch, tensor expected')
            if len(fx) != size:
                raise ValueError(f'MODEL: fx parameter size mismatch')
            fx = fx.cpu().numpy().tolist()
        else:
            fx = self.fx.cpu().numpy().tolist()

        if ay is not None:
            if not isinstance(ay, torch.Tensor):
                raise TypeError(f'MODEL: ay parameter type mismatch, tensor expected')
            if len(ay) != size:
                raise ValueError(f'MODEL: ay parameter size mismatch')
        else:
            ay = self.ay.cpu().numpy().tolist()

        if by is not None:
            if not isinstance(by, torch.Tensor):
                raise TypeError(f'MODEL: by parameter type mismatch, tensor expected')
            if len(by) != size:
                raise ValueError(f'MODEL: by parameter size mismatch')
            by = by.cpu().numpy().tolist()
        else:
            by = self.by.cpu().numpy().tolist()

        if fy is not None:
            if not isinstance(fy, torch.Tensor):
                raise TypeError(f'MODEL: fy parameter type mismatch, tensor expected')
            if len(fy) != size:
                raise ValueError(f'MODEL: fy parameter size mismatch')
            fy = fy.cpu().numpy().tolist()
        else:
            fy = self.fy.cpu().numpy().tolist()

        if sigma_ax is not None:
            if not isinstance(sigma_ax, torch.Tensor):
                raise TypeError(f'MODEL: sigma_ax parameter type mismatch, tensor expected')
            if len(sigma_ax) != size:
                raise ValueError(f'MODEL: sigma_ax parameter size mismatch')
            sigma_ax = sigma_ax.cpu().numpy().tolist()
        else:
            sigma_ax = self.sigma_ax.cpu().numpy().tolist()

        if sigma_bx is not None:
            if not isinstance(sigma_bx, torch.Tensor):
                raise TypeError(f'MODEL: sigma_bx parameter type mismatch, tensor expected')
            if len(sigma_bx) != size:
                raise ValueError(f'MODEL: sigma_bx parameter size mismatch')
            sigma_bx = sigma_bx.cpu().numpy().tolist()
        else:
            sigma_bx = self.sigma_bx.cpu().numpy().tolist()

        if sigma_fx is not None:
            if not isinstance(sigma_fx, torch.Tensor):
                raise TypeError(f'MODEL: sigma_fx parameter type mismatch, tensor expected')
            if len(sigma_fx) != size:
                raise ValueError(f'MODEL: sigma_fx parameter size mismatch')
            sigma_fx = sigma_fx.cpu().numpy().tolist()
        else:
            sigma_fx = self.sigma_fx.cpu().numpy().tolist()

        if sigma_ay is not None:
            if not isinstance(sigma_ay, torch.Tensor):
                raise TypeError(f'MODEL: sigma_ay parameter type mismatch, tensor expected')
            if len(sigma_ay) != size:
                raise ValueError(f'MODEL: sigma_ay parameter size mismatch')
            sigma_ay = sigma_ay.cpu().numpy().tolist()
        else:
            sigma_ay = self.sigma_ay.cpu().numpy().tolist()

        if sigma_by is not None:
            if not isinstance(sigma_by, torch.Tensor):
                raise TypeError(f'MODEL: sigma_by parameter type mismatch, tensor expected')
            if len(sigma_by) != size:
                raise ValueError(f'MODEL: sigma_by parameter size mismatch')
            sigma_by = sigma_by.cpu().numpy().tolist()
        else:
            sigma_by = self.sigma_by.cpu().numpy().tolist()

        if sigma_fy is not None:
            if not isinstance(sigma_fy, torch.Tensor):
                raise TypeError(f'MODEL: sigma_fy parameter type mismatch, tensor expected')
            if len(sigma_fy) != size:
                raise ValueError(f'MODEL: sigma_fy parameter size mismatch')
            sigma_fy = sigma_fy.cpu().numpy().tolist()
        else:
            sigma_fy = self.sigma_fy.cpu().numpy().tolist()

        model = ''
        for i in range(size):
            model += f'{name[i]:16}:  '
            model += '{'
            model += f'TYPE: {kind[i]:16}, '
            model += f'FLAG: {flag[i]:2}, '
            model += f'JOIN: {join[i]:2}, '
            model += f'RISE: {rise[i]:2}, '
            model += f'TIME: {time[i]:24.16E}, '
            model += f'AX: {ax[i]:24.16E}, '
            model += f'BX: {bx[i]:24.16E}, '
            model += f'FX: {fx[i]:24.16E}, '
            model += f'AY: {ay[i]:24.16E}, '
            model += f'BY: {by[i]:24.16E}, '
            model += f'FY: {fy[i]:24.16E}, '
            model += f'SIGMA_AX: {sigma_ax[i]:24.16E}, '
            model += f'SIGMA_BX: {sigma_bx[i]:24.16E}, '
            model += f'SIGMA_FX: {sigma_fx[i]:24.16E}, '
            model += f'SIGMA_AY: {sigma_ay[i]:24.16E}, '
            model += f'SIGMA_BY: {sigma_by[i]:24.16E}, '
            model += f'SIGMA_FY: {sigma_fy[i]:24.16E}'
            model += '}\n'

        with open(file, mode='w') as stream:
            stream.write(model)


    def config_coupled(self,
                       file:str,
                       *,
                       name:list=None,
                       kind:list=None,
                       flag:list=None,
                       join:list=None,
                       rise:list=None,
                       time:list=None,
                       normal:torch.Tensor=None,
                       fx:torch.Tensor=None,
                       fy:torch.Tensor=None,
                       sigma_normal:torch.Tensor=None,
                       sigma_fx:torch.Tensor=None,
                       sigma_fy:torch.Tensor=None,
                       epsilon:float=1.0E-12) -> None:
        """
        Save coupled model configuration to file.

        Note, instance attribute is used corresponding named parameter is None.

        Parameters
        ----------
        file: str
            output file name
        name: list
            location names
        kind: list
            location kinds (expected kinds _virtual or _monitor)
        flag: list
            flags
        join: list
            join flags
        rise: list
            starting turn
        time: torch.Tensor
            position
        fx: torch.Tensor
            x plane phase data
        fy: torch.Tensor
            y plane phase data
        sigma_fx: torch.Tensor
            x plane phase error
        sigma_fy: torch.Tensor
            y plane phase error
            epsilon: float
                epsilon

        Returns
        -------
        None

        """
        size = self.size

        if name is not None:
            if not isinstance(name, list):
                raise TypeError(f'MODEL: name parameter type mismatch, expected list of strings')
            if len(name) != size:
                raise ValueError(f'MODEL: name parameter size mismatch')
            head, *_, tail = name
            if head != self._head:
                raise ValueError(f'MODEL: expected {self._head} at the first position')
            if tail != self._tail:
                raise ValueError(f'MODEL: expected {self._tail} at the last position')
            if not all(isinstance(location, str) for location in name):
                raise ValueError(f'MODEL: name parameter type mismatch, expected list of strings')
        else:
            name = self.name

        if kind is not None:
            if not isinstance(kind, list):
                raise TypeError(f'MODEL: kind parameter type mismatch, expected list of strings')
            if len(kind) != size:
                raise ValueError(f'MODEL: kind parameter size mismatch')
            if not all(isinstance(location, str) for location in kind):
                raise ValueError(f'MODEL: kind parameter type mismatch, expected list of strings')
            if set(kind) != set([self._monitor, self._virtual]):
                raise ValueError(f'MODEL: kind parameter value mismatch, expected {self._monitor} or {self._virtual}')
        else:
            kind = self.kind

        if flag is not None:
            if not isinstance(flag, list):
                raise TypeError(f'MODEL: flag parameter type mismatch, expected list of integers')
            if len(flag) != size:
                raise ValueError(f'MODEL: flag parameter size mismatch')
            if not all(isinstance(location, int) for location in flag):
                raise TypeError(f'MODEL: flag parameter type mismatch, expected list of integers')
            if set(flag) != set([0, 1]):
                raise ValueError(f'MODEL: flag parameter value mismatch, expected 0 or 1')
        else:
            flag = self.flag.cpu().numpy().tolist()

        if join is not None:
            if not isinstance(join, list):
                raise TypeError(f'MODEL: join parameter type mismatch, expected list of integers')
            if len(join) != size:
                raise ValueError(f'MODEL: join parameter size mismatch')
            if not all(isinstance(location, int) for location in join):
                raise TypeError(f'MODEL: join parameter type mismatch, expected list of integers')
            if set(join) != set([0, 1]):
                raise ValueError(f'MODEL: join parameter value mismatch, expected 0 or 1')
        else:
            join = self.join

        if rise is not None:
            if not isinstance(rise, list):
                raise TypeError(f'MODEL: rise parameter type mismatch, expected list of integers')
            if len(rise) != size:
                raise ValueError(f'MODEL: rise parameter size mismatch')
            if not all(isinstance(location, int) for location in rise):
                raise TypeError(f'MODEL: rise parameter type mismatch, expected list of integers')
        else:
            rise = self.rise

        if time is not None:
            if not isinstance(time, list):
                raise TypeError(f'MODEL: time parameter type mismatch, expected list of floats')
            if len(time) != size:
                raise ValueError(f'MODEL: time parameter size mismatch')
            if not all(isinstance(location, float) for location in time):
                raise TypeError(f'MODEL: time parameter type mismatch, expected list of floats')
        else:
            time = self.time.cpu().numpy().tolist()

        if normal is not None:
            if not isinstance(normal, torch.Tensor):
                    raise TypeError(f'MODEL: normal parameter type mismatch, tensor expected')
            if len(normal) != size:
                raise ValueError(f'MODEL: normal parameter size mismatch')
            normal = normal.cpu().numpy().tolist()
        else:
            normal = self.normal.cpu().numpy().tolist()

        if fx is not None:
            if not isinstance(fx, torch.Tensor):
                raise TypeError(f'MODEL: fx parameter type mismatch, tensor expected')
            if len(fx) != size:
                raise ValueError(f'MODEL: fx parameter size mismatch')
            fx = fx.cpu().numpy().tolist()
        else:
            fx = self.fx.cpu().numpy().tolist()

        if fy is not None:
            if not isinstance(fy, torch.Tensor):
                raise TypeError(f'MODEL: fy parameter type mismatch, tensor expected')
            if len(fy) != size:
                raise ValueError(f'MODEL: fy parameter size mismatch')
            fy = fy.cpu().numpy().tolist()
        else:
            fy = self.fy.cpu().numpy().tolist()

        if sigma_normal is not None:
            if not isinstance(sigma_normal, torch.Tensor):
                    raise TypeError(f'MODEL: sigma_normal parameter type mismatch, tensor expected')
            if len(sigma_normal) != size:
                raise ValueError(f'MODEL: sigma_normal parameter size mismatch')
            sigma_normal = sigma_normal.cpu().numpy().tolist()
        else:
            if hasattr(self, 'sigma_normal'):
                sigma_normal = self.sigma_normal.cpu().numpy().tolist()
            else:
                sigma_normal = torch.zeros_like(self.normal).cpu().numpy().tolist()

        if sigma_fx is not None:
            if not isinstance(sigma_fx, torch.Tensor):
                raise TypeError(f'MODEL: sigma_fx parameter type mismatch, tensor expected')
            if len(sigma_fx) != size:
                raise ValueError(f'MODEL: sigma_fx parameter size mismatch')
            sigma_fx = sigma_fx.cpu().numpy().tolist()
        else:
            sigma_fx = self.sigma_fx.cpu().numpy().tolist()

        if sigma_fy is not None:
            if not isinstance(sigma_fy, torch.Tensor):
                raise TypeError(f'MODEL: sigma_fy parameter type mismatch, tensor expected')
            if len(sigma_fy) != size:
                raise ValueError(f'MODEL: sigma_fy parameter size mismatch')
            sigma_fy = sigma_fy.cpu().numpy().tolist()
        else:
            sigma_fy = self.sigma_fy.cpu().numpy().tolist()

        model = ''
        for i in range(size):
            model += f'{name[i]:16}:  '
            model += '{'
            model += f'TYPE: {kind[i]:16}, '
            model += f'FLAG: {flag[i]:2}, '
            model += f'JOIN: {join[i]:2}, '
            model += f'RISE: {rise[i]:2}, '
            model += f'TIME: {time[i]:24.16E}, '
            model += ''.join(f'N{q}{p}: {normal[i][q][p]:24.16E}, ' for q in range(4) for p in range(4))
            model += f'FX: {fx[i]:24.16E}, '
            model += f'FY: {fy[i]:24.16E}, '
            model += ''.join(f'SIGMA_N{q}{p}: {sigma_normal[i][q][p]:24.16E}, ' for q in range(4) for p in range(4))
            model += f'SIGMA_FX: {sigma_fx[i]:24.16E}, '
            model += f'SIGMA_FY: {sigma_fy[i]:24.16E}'
            model += '}\n'

        with open(file, mode='w') as stream:
            stream.write(model)


    def save(self,
             file:str='model.json') -> None:
        """
        Save model.

        Parameters
        ----------
        file: str
            output file name

        Returns
        -------
        None

        """
        skip = ['dtype', 'device', 'data_frame', 'transport', 'kick', 'error_kn', 'error_ks', 'turn', 'is_stable', 'out_twiss', 'out_normal', 'out_advance', 'out_tune']

        data = {}
        for key, value in self.__dict__.items():
            if key not in skip:
                data[key] = value.cpu().tolist() if isinstance(value, torch.Tensor) else value

        with open(file, 'w') as stream:
            json.dump(data, stream)


    def load(self,
             file:str='model.json') -> None:
        """
        Load model.

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

        if self.model == 'uncoupled':

            self.ax = torch.tensor(self.ax, dtype=self.dtype, device=self.device)
            self.bx = torch.tensor(self.bx, dtype=self.dtype, device=self.device)
            self.ay = torch.tensor(self.ay, dtype=self.dtype, device=self.device)
            self.by = torch.tensor(self.by, dtype=self.dtype, device=self.device)

            self.sigma_bx = torch.tensor(self.sigma_bx, dtype=self.dtype, device=self.device)
            self.sigma_ax = torch.tensor(self.sigma_ax, dtype=self.dtype, device=self.device)
            self.sigma_by = torch.tensor(self.sigma_by, dtype=self.dtype, device=self.device)
            self.sigma_ay = torch.tensor(self.sigma_ay, dtype=self.dtype, device=self.device)

        if self.model == 'coupled':

            self.normal = torch.tensor(self.normal, dtype=self.dtype, device=self.device)

            self.sigma_normal = torch.tensor(self.sigma_normal, dtype=self.dtype, device=self.device)

        self.fx = torch.tensor(self.fx, dtype=self.dtype, device=self.device)
        self.fy = torch.tensor(self.fy, dtype=self.dtype, device=self.device)
        self.sigma_fx = torch.tensor(self.sigma_fx, dtype=self.dtype, device=self.device)
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
    def from_file(cls,
                  file:str='model.json',
                  *,
                  dtype:torch.dtype=torch.float64,
                  device:torch.device=torch.device('cpu')) -> Model:
        """
        Initialize model from file.

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
        model.load(file=file)
        return model


    def matrix(self,
               probe:torch.Tensor,
               other:torch.Tensor) -> torch.Tensor:
        """
        Generate transport matrices between given probe and other locations.

        Matrices are generated from probe to other
        One-turn matrices are generated if probe == other
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
        transport matrices (torch.Tensor)

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

        fx, _ = Decomposition.phase_advance(probe, other, self.nux, self.fx, error=False)
        fy, _ = Decomposition.phase_advance(probe, other, self.nuy, self.fy, error=False)

        probe = mod(probe, self.size).to(torch.int64)
        other = mod(other, self.size).to(torch.int64)

        if self.model == 'uncoupled':
            matrix = matrix_uncoupled(self.ax[probe], self.bx[probe], self.ax[other], self.bx[other], fx,
                                      self.ay[probe], self.by[probe], self.ay[other], self.by[other], fy)

        if self.model == 'coupled':
                matrix = matrix_coupled(self.normal[probe], self.normal[other], torch.stack([fx, fy]).T)

        return matrix.squeeze()


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
        self.transport = self.matrix(probe, probe + 1)


    def matrix_transport(self,
                         probe:int,
                         other:int) -> torch.Tensor:
        """
        Generate transport matrix from probe to other using self.transport.

        Parameters
        ----------
        probe: int
            probe location
        other: int
            other location

        Returns
        -------
        transport matrix (torch.Tensor)

        """
        if isinstance(probe, str):
            probe = self.name.index(probe)

        if isinstance(other, str):
            other = self.name.index(other)

        if probe < other:
            matrix = self.transport[probe]
            for i in range(probe + 1, other):
                matrix = self.transport[int(mod(i, self.size))] @ matrix
            return matrix

        if probe > other:
            matrix = self.transport[other]
            for i in range(other + 1, probe):
                matrix = self.transport[int(mod(i, self.size))] @ matrix
            return torch.inverse(matrix)


    @staticmethod
    @functorch.vmap
    def matrix_kick(kn:torch.Tensor,
                    ks:torch.Tensor) -> torch.Tensor:
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


    def make_kick(self,
                  kn:torch.Tensor=None,
                  ks:torch.Tensor=None) -> None:
        """
        Generate thin quadrupole kick errors for each segment.

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
            self.error_kn = kn
            self.error_ks = ks


    def apply_error(self,
                    *,
                    split:bool=False) -> None:
        """
        Apply errors to each transport segment.

        self.transport[i] = self.kick[i] @ self.transport[i]

        Parameters
        ----------
        split: bool
            flag to split error
            self.transport[i] = self.kick[i] @ self.transport[i] self.kick[i - 1]

        Returns
        -------
        None

        """
        if hasattr(self, 'error_kn') and hasattr(self, 'error_ks'):
            if split:
                kick = self.matrix_kick(0.5*self.error_kn, 0.5*self.error_ks)
                self.transport = torch.matmul(torch.matmul(kick, self.transport), kick.roll(1, 0))
            else:
                kick = self.matrix_kick(self.error_kn, self.error_ks)
                self.transport = torch.matmul(kick, self.transport)


    def make_turn(self) -> None:
        """
        Generate one-turn matrix at the 'HEAD' location using self.transport matrices.

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


    def make_twiss(self) -> bool:
        """
        Compute and set twiss using self.transport matrices.

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

        tune, normal, twiss = twiss_compute(self.turn)

        self.is_stable = tune is not None

        if not self.is_stable:
            return self.is_stable

        transport = torch.clone(self.transport)
        for i in range(1, self.size):
            transport[i] @= transport[i - 1]

        self.out_twiss = twiss_propagate(twiss, transport).roll(1, 0)

        self.out_advance, self.out_normal = twiss_phase_advance(normal, transport)

        self.out_advance = self.out_advance.T
        for i, advance in enumerate(self.out_advance):
            self.out_advance[i], _ = Decomposition.phase_adjacent(tune[i], advance, error=False)
            self.out_advance[i] = mod(self.out_advance[i] + self._epsilon, 2.0*numpy.pi) - self._epsilon
        self.out_advance = self.out_advance.roll(1, 1)
        self.out_tune = self.out_advance.sum(1)/(2.0*numpy.pi)
        self.out_advance = self.out_advance.T

        self.out_normal = self.out_normal.roll(1, 0)

        return self.is_stable


    def make_trajectory(self,
                        length:int,
                        initial:torch.Tensor,
                        *,
                        full:bool=True) -> torch.Tensor:
        """
        Generate test trajectories for given initial condition.

        Parameters
        ----------
        length: int
            number of iterations
        initial: torch.Tensor
            initial condition at 'HEAD' location
        full: bool
            flag to generate trajectories at all locations (else only at monitor locations)

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

        return trajectory if full else trajectory[self.monitor_index]


    @staticmethod
    def make_momenta(matrix:torch.Tensor,
                     qx1:torch.Tensor,
                     qx2:torch.Tensor,
                     qy1:torch.Tensor,
                     qy2:torch.Tensor) -> torch.Tensor:
        """
        Compute momenta at position 1 for given transport matrix and coordinates at 1 & 2.

        Parameters
        ----------
        matrix: torch.Tensor
            transport matrix between
        qx1, qx2, qy1, qy2: torch.Tensor
            x & y coordinates at 1 & 2

        Returns
        -------
        px and py at 1 (torch.Tensor)

        """
        m11, m12, m13, m14 = matrix[0]
        m21, m22, m23, m24 = matrix[1]
        m31, m32, m33, m34 = matrix[2]
        m41, m42, m43, m44 = matrix[3]

        px1  = qx1*(m11*m34 - m14*m31)/(m14*m32 - m12*m34)
        px1 += qx2*m34/(m12*m34 - m14*m32)
        px1 += qy1*(m13*m34 - m14*m33)/(m14*m32 - m12*m34)
        px1 += qy2*m14/(m14*m32 - m12*m34)

        py1  = qx1*(m11*m32 - m12*m31)/(m12*m34 - m14*m32)
        py1 += qx2*m32/(m14*m32 - m12*m34)
        py1 += qy1*(m12*m33 - m13*m32)/(m14*m32 - m12*m34)
        py1 += qy2*m12/(m12*m34 - m14*m32)

        return torch.stack([px1, py1])


    @staticmethod
    def make_momenta_error(matrix:torch.Tensor,
                           sigma_qx1:torch.Tensor,
                           sigma_qx2:torch.Tensor,
                           sigma_qy1:torch.Tensor,
                           sigma_qy2:torch.Tensor,
                           *,
                           sigma_matrix:torch.Tensor=None) -> torch.Tensor:
        """
        Compute momenta errors at position 1 for given transport matrix and coordinates at 1 & 2.

        Parameters
        ----------
        matrix: torch.Tensor
            transport matrix between
        sigma_qx1, sigma_qx2, sigma_qy1, sigma_qy2: torch.Tensor
            x & y coordinates errors at 1 & 2

        Returns
        -------
        sigma_px and sigma_py at 1 (torch.Tensor)

        """
        m11, m12, m13, m14 = matrix[0]
        m21, m22, m23, m24 = matrix[1]
        m31, m32, m33, m34 = matrix[2]
        m41, m42, m43, m44 = matrix[3]

        sigma_px1  = sigma_qx1**2*(m11*m34 - m14*m31)**2/(m14*m32 - m12*m34)**2
        sigma_px1 += sigma_qx2**2*m34**2/(m12*m34 - m14*m32)**2
        sigma_px1 += sigma_qy1**2*(m13*m34 - m14*m33)**2/(m14*m32 - m12*m34)**2
        sigma_px1 += sigma_qy2**2*m14**2/(m14*m32 - m12*m34)**2

        sigma_py1  = sigma_qx1**2*(m11*m32 - m12*m31)**2/(m12*m34 - m14*m32)**2
        sigma_py1 += sigma_qx2**2*m32**2/(m14*m32 - m12*m34)**2
        sigma_py1 += sigma_qy1**2*(m12*m33 - m13*m32)**2/(m14*m32 - m12*m34)**2
        sigma_py1 += sigma_qy2**2*m12**2/(m12*m34 - m14*m32)**2

        return torch.stack([sigma_px1, sigma_py1]).sqrt()


    def get_name(self,
                 index:int) -> str:
        """
        Return name of given location index.

        """
        index = int(mod(index, self.size))

        return self.name[index]


    def get_index(self,
                  name:str) -> int:
        """
        Return index of given location name.

        """
        return self.name.index(name)


    def is_monitor(self,
                   index:int) -> bool:
        """
        Return True, if location is a monitor.

        """
        index = int(mod(index, self.size))
        return self[index].get('TYPE') == self._monitor


    def is_virtual(self,
                   index:int) -> bool:
        """
        Return True, if location is a virtual.

        """
        index = int(mod(index, self.size))
        return self[index].get('TYPE') == self._virtual


    def is_same(self,
                probe:int,
                other:int) -> bool:
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


    def get_next(self,
                 probe:int) -> list:
        """
        Return next location index and name.

        """
        if isinstance(probe, int):
            index = int(mod(probe + 1, self.size))
            return [probe + 1, index, self.name[index]]
        return self.get_next(self.name.index(probe))


    def get_next_monitor(self,
                         probe:int) -> list:
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


    def get_next_virtual(self,
                         probe:int) -> list:
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


    def count(self,
              probe:int,
              other:int) -> int:
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


    def count_monitor(self,
                      probe:int,
                      other:int) -> int:
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


    def count_virtual(self,
                      probe:int,
                      other:int) -> int:
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


    def __len__(self) -> int:
        """
        Return total number of locations (monitor & virtual).

        """
        return self.size


    def __getitem__(self,
                    index:int) -> dict:
        """
        Return corresponding self.dict value for given location index.

        """
        if self.dict is None:
            return None
        return self.dict[self.name[index]]


    def __call__(self,
                 index:int) -> dict:
        """
        Return corresponding self.dict value for given location index or name.

        """
        if self.dict is None:
            return None
        if isinstance(index, int):
            index = int(mod(index, self.size))
            return self[index]
        return self.dict[index] if index in self.name else None


    def __repr__(self) -> str:
        """
        String representation.

        """
        return f'Model(path={self.path}, model={self.model})'


def main():
    pass

if __name__ == '__main__':
    main()