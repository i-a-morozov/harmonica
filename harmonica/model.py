"""
Model module.

"""

import epics
import numpy
import pandas
import torch
import nufft
import yaml

from .util import mod
from .decomposition import Decomposition

class Model():

    def __init__(self, path:str=None, decomposition:'Decomposition'=Decomposition(), *,
                 dtype:torch.dtype=torch.float64, device:torch.device='cpu') -> None:
        """
        Model instance initialization.

        Parameters
        ----------
        path: str
            path to config file
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
        self.decomposition = decomposition
        self.dict = None

        if self.path is not None:

            with open(self.path) as path:
                self.dict = yaml.safe_load(path)

            if self._head not in self.dict:
                raise Exception(f'{self._head} record is not found in {self.path}')

            if self._tail not in self.dict:
                raise Exception(f'{self._tail} record is not found in {self.path}')

            self.size = len(self.dict)
            self.name = list(self.dict.keys())
            self.next = {name: self.name[int(mod(index + 1, self.size))] for index, name in enumerate(self.name)}

            self.kind = [value.get('TYPE') for value in self.dict.values()]
            self.flag = [value.get('FLAG') for value in self.dict.values()]
            self.join = [value.get('JOIN') for value in self.dict.values()]
            self.rise = [value.get('RISE') for value in self.dict.values()]
            self.time = [value.get('TIME') for value in self.dict.values()]

            *_, self.length = self.time

            self.monitor_index = [index for index, kind in enumerate(self.kind) if kind == self._monitor]
            self.virtual_index = [index for index, kind in enumerate(self.kind) if kind == self._virtual]
            self.monitor_count = len(self.monitor_index)
            self.virtual_count = len(self.virtual_index)
            self.monitor_name = [name for name, kind in zip(self.name, self.kind) if kind == self._monitor]
            self.virtual_name = [name for name, kind in zip(self.name, self.kind) if kind == self._virtual]

            self.bx = torch.tensor([value.get('BX') for value in self.dict.values()], dtype=self.dtype, device=self.device)
            self.ax = torch.tensor([value.get('AX') for value in self.dict.values()], dtype=self.dtype, device=self.device)
            self.fx = torch.tensor([value.get('FX') for value in self.dict.values()], dtype=self.dtype, device=self.device)

            self.by = torch.tensor([value.get('BY') for value in self.dict.values()], dtype=self.dtype, device=self.device)
            self.ay = torch.tensor([value.get('AY') for value in self.dict.values()], dtype=self.dtype, device=self.device)
            self.fy = torch.tensor([value.get('FY') for value in self.dict.values()], dtype=self.dtype, device=self.device)

            self.sigma_fx = torch.tensor([value.get('EX') for value in self.dict.values()], dtype=self.dtype, device=self.device)

            self.sigma_fy = torch.tensor([value.get('EY') for value in self.dict.values()], dtype=self.dtype, device=self.device)

            *_, self.mux = self.fx
            *_, self.muy = self.fy

            self.sigma_mux = torch.sqrt(torch.sum(self.sigma_fx**2))
            self.sigma_muy = torch.sqrt(torch.sum(self.sigma_fy**2))

            self.nux = self.mux/(2.0*numpy.pi)
            self.nuy = self.muy/(2.0*numpy.pi)

            self.sigma_nux = self.sigma_mux/(2.0*numpy.pi)
            self.sigma_nuy = self.sigma_mux/(2.0*numpy.pi)

            self.phase_x, self.sigma_x = [], []
            self.phase_y, self.sigma_y = [], []

            for index in range(self.size):

                phase_x, sigma_x = self.decomposition.phase_advance(index, index + 1, self.nux, self.fx, True, self.sigma_nux, self.sigma_fx)
                self.phase_x.append(phase_x)
                self.sigma_x.append(sigma_x)

                phase_y, sigma_y = self.decomposition.phase_advance(index, index + 1, self.nuy, self.fy, True, self.sigma_nuy, self.sigma_fy)
                self.phase_y.append(phase_y)
                self.sigma_y.append(sigma_y)

            self.phase_x = mod(torch.stack(self.phase_x), 2.0*numpy.pi)
            self.phase_y = mod(torch.stack(self.phase_y), 2.0*numpy.pi)

            self.sigma_x = torch.stack(self.sigma_x)
            self.sigma_y = torch.stack(self.sigma_y)

            self.monitor_phase_x, self.monitor_sigma_x = [], []
            self.monitor_phase_y, self.monitor_sigma_y = [], []

            for index in self.monitor_index:

                probe = index
                other = self.monitor_index.index(probe) + 1
                count = 1 if other >= self.monitor_count else 0
                other = self.monitor_index[int(mod(other, self.monitor_count))] + count*self.size

                phase_x, sigma_x = self.decomposition.phase_advance(probe, other, self.nux, self.fx, True, self.sigma_nux, self.sigma_fx)
                self.monitor_phase_x.append(phase_x)
                self.monitor_sigma_x.append(sigma_x)

                phase_y, sigma_y = self.decomposition.phase_advance(probe, other, self.nuy, self.fy, True, self.sigma_nuy, self.sigma_fy)
                self.monitor_phase_y.append(phase_y)
                self.monitor_sigma_y.append(sigma_y)

            self.monitor_phase_x = mod(torch.stack(self.monitor_phase_x), 2.0*numpy.pi)
            self.monitor_phase_y = mod(torch.stack(self.monitor_phase_y), 2.0*numpy.pi)

            self.monitor_sigma_x = torch.stack(self.monitor_sigma_x)
            self.monitor_sigma_y = torch.stack(self.monitor_sigma_y)


    def get_name(self, idx:int):
        """
        Return name of given location index.

        """
        idx = int(mod(idx, self.size))

        return self.name[idx]


    def get_index(self, name:str):
        """
        Return index of given location name.

        """
        return self.name.index(name)


    def is_monitor(self, idx):
        """
        Return True, if location is a monitor.

        """
        idx = int(mod(idx, self.size))

        return self[idx].get('TYPE') == self._monitor


    def is_same(self, idx, idy):
        """
        Return True, if locations are at the same location.

        """
        idx = int(mod(idx, self.size))
        idy = int(mod(idy, self.size))

        delta = abs(self[idx].get('TIME') - self[idy].get('TIME'))

        if delta < self._epsilon:
            return True

        if abs(delta - self.length) < self._epsilon:
            return True

        return False

    def __len__(self) -> int:
        """
        Return total number of lacations (monitor& virtual).

        """
        return self.size


    def __getitem__(self, index:int) -> dict:
        """
        Return corresponding self.dict entry for given location index.

        """
        if self.dict is None:
            return None

        return self.dict[self.name[index]]


    def __call__(self, index:int) -> dict:
        """
        Return corresponding self.dict entry for given location index.

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
        return f'Model(path={self.path}, dtype={self.dtype}, device={self.device})'


    def __str__(self) -> str:
        """
        String representation.

        """
        return self.__repr__()


    def get_next(self, probe:int) -> list:
        """
        Return next location index and name.

        """
        if isinstance(probe, int):
            index = int(mod(probe + 1, self.size))
            return [probe + 1, index, self.name[index]]

        return self.get_next(self.name.index(probe))


    def get_next_monitor(self, probe:int):
        """
        Return next monitor location index and name.

        """
        if isinstance(probe, int):

            if probe not in self.monitor_index:
                return []

            other = self.monitor_index.index(probe) + 1
            count = 1 if other >= self.monitor_count else 0
            other = self.monitor_index[int(mod(other, self.monitor_count))]
            return [other + count*self.size, other, self.name[other]]

        return self.get_next_monitor(self.name.index(probe))


    def get_next_virtual(self, probe:int):
        """
        Return next virtual location index and name.

        """
        if isinstance(probe, int):

            if probe not in self.virtual_index:
                return []

            other = self.virtual_index.index(probe) + 1
            count = 1 if other >= self.virtual_count else 0
            other = self.virtual_index[int(mod(other, self.virtual_count))]
            return [other + count*self.size, other, self.name[other]]

        return self.get_next_virtual(self.name.index(probe))