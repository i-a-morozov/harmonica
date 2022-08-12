"""
Data module.
Generate, save and load TbT data.
Data processing (remove mean, apply window & perform data normalization).

"""
from __future__ import annotations

import torch
import numpy
import pandas
import epics

from .util import LIMIT
from .window import Window

class Data():
    """
    Returns
    ----------
    Data class instance.

    Data(size:int, window:Window, dtype:torch.dtype=None, device:torch.device=None)
    Data.from_data(window:Window, data:torch.Tensor)
    Data.from_file(size:int, window:Window, file:str, dtype:torch.dtype=None, device:torch.device=None)
    Data.from_epics(window:Window, pv_list:list, pv_rise:list=None, shift:int=0, count:int=LIMIT, dtype:torch.dtype=None, device:torch.device=None)

    Data type and device are inherited from window if dtype == None and device == None
    Data type and device are inherited from input data for Data.from_data
    Data type can be complex

    Parameters
    ----------
    size: int
        number of signals
    window: Window
        window instance
    data: torch.Tensor
        input TbT data with matching length (from_data)
    file: str
        input file name (from_file)
    pv_list: list
        list of TbT PV names (from_epics)
    pv_rise: list
        list of TbT PV rise indices (from_epics)
    shift: int
        rise shift for all TbT PVs (from_epics)
    count: int
        maximum length to read from TbT PVs (from_epics)
    dtype: torch.dtype
        data type for self.data and self.work
    device: torch.device
        device for self.data and self.work

    Attributes
    ----------
    size: int
        number of signals
    window: Window
        window instance
    length: int
        window/signal length
    dtype: torch.dtype
        data type for self.data and self.work
    device: torch.device
        device for self.data and self.work
    data: torch.tensor
        data container
    work: torch.tensor
        work container (working copy of data container)
    source: str
        data source ('empty', 'data', 'file' or 'epics')
    file: str
        input file (from_file)
    pv_list: list
        list of TbT PV names (from_epics)
    pv_rise: list
        list of TbT PV rise indices (from_epics)

    Methods
    ----------
    __init__(self, size:int, window:Window, dtype:torch.dtype=None, device:torch.device=None) -> None
        Data instance initialization.
    reset(self) -> None
        Reset self.work container (copy self.data container to self.work container).
    set_data(self, data:torch.Tensor) -> None
        Copy input data with matching shape to self.data container and reset self.work container.
    from_data(cls, window:Window, data:torch.Tensor) -> Data
        Generate Data instance for given window and data.
    save_data(self, file:str) -> None
        Save self.work to file (numpy array).
    load_data(self, file:str) -> None
        Load data from file (numpy array) into self.data container and reset self.work container.
    from_file(cls, size:int, window:Window, file:str, dtype:torch.dtype=None, device:torch.device=None) -> Data
        Generate Data instance from file (numpy array).
    pv_put(pv:str, data:torch.Tensor, *, wait:bool=True) -> None
        Put data to PV.
    pv_get(pv:str, *, count:int=None, **kwargs) -> torch.Tensor
        Get PV data.
    save_epics(self, *, wait:bool=True) -> None
        Save data from self.work into self.pv_list.
    load_epics(self, *, shift:int=0, count:int=LIMIT) -> None
        Load data from self.pv_list into self.data container and reset self.work container.
    from_epics(cls, window:Window, pv_list:list, *, pv_rise:list=None, shift:int=0, count:int=LIMIT, dtype:torch.dtype=None, device:torch.device=None) -> Data
        Generate Data instance from epics.
    make_harmonic(length:int, f:float, *, m:float=0.0, a:float=1.0, b:float=0.0, c:float=None, s:float=None, dtype:torch.dtype=torch.float64, device:torch.device=torch.device('cpu')) -> torch.Tensor
        Generate harmonic signal for given parameters.
    make_noise(length:int, sigma:torch.Tensor, *, dtype:torch.dtype=torch.float64, device:torch.device=torch.device('cpu')) -> torch.Tensor
        Generate normal noise for given length and list of noise parameters.
    add_noise(self, sigma:torch.tensor) -> None
        Add noise to self.work container.
    make_matrix(length:int, shift:int, signal:torch.Tensor) -> torch.Tensor
        Generate matrix from given signal.
    make_signal(length:int, matrix:torch.Tensor) -> torch.Tensor
        Generate mixed signal from given matrix.
    window_mean(self) -> torch.Tensor
        Return window weighted mean for each signal in self.work container.
    window_remove_mean(self) -> None
        Remove window weighted mean from self.work container.
    window_apply(self) -> None
        Apply window to self.work container.
    mean(self) -> torch.Tensor
        Return mean for each signal in self.work container.
    median(self) -> torch.Tensor
        Return median  for each signal in self.work container.
    normalize(self, window:bool = False) -> None
        Normalize (standardize) self.work container.
    to_tensor(self) -> torch.Tensor
        Return self.work container as tensor.
    to_numpy(self) -> numpy.ndarray
        Return self.work container as numpy.
    to_dict(self) -> dict
        Return self.work container as dict if epics.
    to_frame(self) -> pandas.DataFrame
        Return self.work container as data frame if epics.
    __repr__(self) -> str
        String representation.
    __len__(self) -> int
        Return number of signals.
    __getitem__(self, idx:int) -> torch.Tensor
        Return signal for given index or PV name if epics.
    __call__(self) -> None
        Reset self.work container (copy self.data container to self.work container).

    """
    def __init__(self,
                 size:int,
                 window:Window,
                 *,
                 dtype:torch.dtype=None,
                 device:torch.device=None) -> None:
        """
        Data instance initialization.

        If dtype/device != None, use input dtype/device to set self.data and self.work containers
        Else use window dtype/device

        Parameters
        ----------
        size: int
            number of signals
        window: Window
            window instance
        dtype: torch.dtype
            data type for self.data and self.work
        device: torch.device
            device for self.data and self.work

        Returns
        -------
        None

        """
        self.size = size
        self.window = window
        self.length = self.window.length
        self.dtype = dtype if dtype != None else self.window.dtype
        self.device = device if device != None else self.window.device
        self.data = torch.zeros((self.size, self.length), dtype=self.dtype, device=self.device)
        self.work = torch.zeros((self.size, self.length), dtype=self.dtype, device=self.device)
        self.source = 'empty'
        self.file = None
        self.pv_list = None
        self.pv_rise = None


    def reset(self) -> None:
        """
        Reset self.work container (copy self.data to self.work).

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.work.copy_(self.data)


    def set_data(self,
                 data:torch.Tensor) -> None:
        """
        Copy input data with matching shape to self.data container and reset self.work container.

        Parameters
        ----------
        data: torch.Tensor
            input data with matching shape

        Returns
        -------
        None

        """
        if data.shape == self.data.shape:
            self.source = 'data'
            self.file = None
            self.pv_list = None
            self.pv_rise = None
            self.data.copy_(data)
            self.reset()
            return

        raise ValueError()(f'DATA: expected input data shape {self.data.shape}, got {data.shape}')


    @classmethod
    def from_data(cls,
                  window:Window,
                  data:torch.Tensor) -> Data:
        """
        Generate Data instance for given window and data.

        Data type and device are inherited from input data

        Parameters
        ----------
        window: Window
            Window instance
        data: torch.Tensor
            input data with matching length

        Returns
        -------
        Data instance

        """
        size, length = data.shape

        if length != window.length:
            raise ValueError(f'DATA: expected input data length {window.length}, got {length}')

        out = cls(size, window, dtype=data.dtype, device=data.device)
        out.set_data(data)
        return out


    def save_data(self,
                  file:str) -> None:
        """
        Save self.work to file (numpy array).

        Parameters
        ----------
        file: str
            output filename

        Returns
        -------
        None

        """
        numpy.save(file, self.work.cpu().numpy())


    def load_data(self,
                  file:str) -> None:
        """
        Load data from file (numpy array) into self.data container and reset self.work container.

        Parameters
        ----------
        file: str
            input filename

        Returns
        -------
        None

        """
        data = numpy.load(file)
        size, length = data.shape

        if size < self.size:
            raise ValueError(f'DATA: expected input data size > {self.size}, got {size}')

        if length < self.length:
            raise ValueError(f'DATA: expected input data length > {self.length}, got {length}')

        self.source = 'file'
        self.file = file
        self.pv_list = None
        self.pv_rise = None
        self.data.copy_(torch.tensor(data[:self.size, :self.length], dtype=self.dtype, device=self.device))
        self.reset()


    @classmethod
    def from_file(cls,
                  size:int,
                  window:Window,
                  file:str,
                  dtype:torch.dtype=None,
                  device:torch.device=None) -> Data:
        """
        Generate Data instance from file (numpy array).

        Parameters
        ----------
        size: int
            number of signals
        window:
            window instance
        file: str
            input filename
        dtype: torch.dtype
            data type for self.data and self.work
        device: torch.device
            device for self.data and self.work

        Returns
        -------
        Data instance

        """
        out = cls(size, window, dtype=dtype, device=device)
        out.load_data(file)
        return out


    @staticmethod
    def pv_put(pv:str,
               data:torch.Tensor,
               *,
               wait:bool=True) -> None:
        """
        Put data to PV.

        Parameters
        ----------
        pv: str
            PV name
        data: torch.Tensor
            data to put
        wait: bool
            flag to wait for processing to complete

        Returns
        -------
        None

        """
        epics.caput(pv, data.cpu().numpy(), wait=wait)


    @staticmethod
    def pv_get(pv:str,
               *,
               count:int=None,
               **kwargs) -> torch.Tensor:
        """
        Get PV data.

        Parameters
        ----------
        pv: str
            PV name
        count: int
            length to read from PV
        **kwargs:
            dtype, device

        Returns
        -------
        PV value (torch.Tensor)

        """
        data = epics.caget(pv, count=count)
        if data is None:
            raise TypeError()(f'DATA: {pv=} is None.')
        return torch.tensor(data, **kwargs)


    def save_epics(self,
                   *,
                   wait:bool=True) -> None:
        """
        Save self.work container into self.pv_list.

        Parameters
        ----------
        wait: bool
            flag to wait for processing to complete

        Returns
        -------
        None

        """
        if self.pv_list is None:
            raise ValueError(f'DATA: pv_list is not defined')

        if len(self.pv_list) != self.size:
            raise ValueError(f'DATA: self.pv_list length {len(self.pv_list)} and self.work length {self.size} expected to match')

        epics.caput_many(self.pv_list, self.work.cpu().numpy(), wait=wait)


    def load_epics(self,
                   *,
                   shift:int=0,
                   count:int=LIMIT) -> None:
        """
        Load data from self.pv_list into self.data container and reset self.work container.

        Parameters
        ----------
        shift: int
            common start shift for all PVs
        count: int
            maximum length to load

        Returns
        -------
        None

        """
        if self.pv_list is None:
            raise ValueError(f'DATA: pv_list is not defined')

        if len(self.pv_list) != self.size:
            raise ValueError(f'DATA: self.pv_list length {len(self.pv_list)} and self.work length {self.size} expected to match')

        self.source = 'epics'
        self.file = None
        data = epics.caget_many(self.pv_list, count=count)

        if any(_ is None for _ in data):
            raise ValueError(f'DATA: None in data')
        data = numpy.array(data).reshape(self.size, -1)

        if data.shape != (self.size, count):
            raise ValueError(f'DATA: expected shape {(self.size, self.length)}, got {pv_data.shape}')

        if self.pv_rise != None and len(self.pv_list) == len(self.pv_rise):
            for i, (rise, signal) in enumerate(zip(self.pv_rise, data)):
                self.data[i].copy_(torch.tensor(signal[shift + rise : shift + rise + self.length]))
            self.reset()
            return

        elif self.pv_rise == None:
            self.data.copy_(torch.tensor(data[:, shift : shift + self.length]))
            self.reset()
            return

        raise ValueError(f'DATA: pv_list length {len(self.pv_list)} and pv_rise length {len(self.pv_rise)} expected to match')


    @classmethod
    def from_epics(cls,
                   window:Window,
                   pv_list:list,
                   *,
                   pv_rise:list=None,
                   shift:int=0,
                   count:int=LIMIT,
                   dtype:torch.dtype=None,
                   device:torch.device=None) -> Data:
        """
        Generate Data instance from epics.

        If pv_rise == None, data is loaded from index defined by shift parameter
        Note, if pv_rise != None, it is expected to match pv_list length

        Parameters
        ----------
        window:
            Window instance
        pv_list:
            list of PV names
        pv_rise:
            list of starting indices for each PV
        shift: int
            common start shift for all PVs
        count: int
            maximum length to load
        dtype: torch.dtype
            data type for self.data and self.work
        device: torch.device
            device for self.data and self.work

        Returns
        -------
        Data instance

        """
        out = cls(len(pv_list), window, dtype=dtype, device=device)
        out.pv_list = pv_list
        out.pv_rise = pv_rise
        out.load_epics(shift=shift, count=count)
        return out


    @staticmethod
    def make_harmonic(length:int,
                      f:float,
                      *,
                      m:float=0.0,
                      a:float=1.0,
                      b:float=0.0,
                      c:float=None,
                      s:float=None,
                      dtype:torch.dtype=torch.float64,
                      device:torch.device=torch.device('cpu')) -> torch.Tensor:
        """
        Generate harmonic signal for given parameters.

        h(n) = m + a*cos(2*pi*f*n + b) = m + c*cos(2*pi*f*n) + s*sin(2*pi*f*n)
        n = 0, ..., length - 1

        If c & s parameters are passed, amplitude and phase are ignored.

        Parameters
        ----------
        length: int
            data length
        f: float
            frequency
        m: float
            mean
        a: float
            amplitude
        b: float
            phase
        c: float
            cos amplitude
        s: float
            sin amplitude
        dtype: torch.dtype
            data type
        device: torch.device
            device

        Returns
        -------
        harmonic signal (torch.Tensor)

        """
        time = torch.linspace(0, length - 1, length, dtype=dtype, device=device)

        if c == None and s == None:
            return m + a*torch.cos(2.0*numpy.pi*f*time + b)

        return m + c*torch.cos(2.0*numpy.pi*f*time) + s*torch.sin(2.0*numpy.pi*f*time)


    @staticmethod
    def make_noise(length:int,
                   sigma:torch.Tensor,
                   *,
                   dtype:torch.dtype=torch.float64,
                   device:torch.device=torch.device('cpu')) -> torch.Tensor:
        """
        Generate normal noise for given length and list of noise parameters.

        Parameters
        ----------
        length: int
            data length
        sigma: torch.Tensor
            sigma values for each signal

        Returns
        -------
        noise data (torch.Tensor) with shape (len(sigma), length)

        """
        return sigma.reshape(-1, 1)*torch.randn((len(sigma), length), dtype=dtype, device=device)


    def add_noise(self,
                  sigma:torch.tensor) -> None:
        """
        Add noise to self.work container.

        Parameters
        ----------
        sigma: torch.Tensor
            sigma values for each signal

        Returns
        -------
        None

        """
        if len(sigma) != self.size:
            raise ValueError(f'DATA: expected sigma length {self.length}, got {len(sigma)}')

        self.work.add_(self.make_noise(self.length, sigma.to(self.device), dtype=self.dtype, device=self.device))


    @staticmethod
    def make_matrix(length:int,
                    shift:int,
                    signal:torch.Tensor) -> torch.Tensor:
        """
        Generate matrix from given signal.

        Parameters
        ----------
        length: int
            sample length
        shift: int
            sample shift
        signal: torch.Tensor
            input signal

        Returns
        -------
        matrix (torch.Tensor)

        """
        size = 1 + (len(signal) - length)//shift
        data = torch.zeros((size, length), dtype=signal.dtype, device=signal.device)
        for i in range(size):
            data[i].copy_(signal[i*shift : i*shift + length])
        return data


    @staticmethod
    def make_signal(length:int,
                    matrix:torch.Tensor) -> torch.Tensor:
        """
        Generate mixed signal from given matrix.

        Parameters
        ----------
        length: int
            sample length
        matrix: torch.Tensor
            matrix

        Returns
        -------
        signal (torch.Tensor)

        """
        data = torch.zeros(len(matrix)*length, dtype=matrix.dtype, device=matrix.device)
        data.copy_(matrix[:, :length].T.flatten())
        return data


    def window_mean(self) -> torch.Tensor:
        """
        Return window weighted mean for each signal in self.work container.

        Parameters
        ----------
        None

        Returns
        ----------
        mean values for each signal (torch.Tensor) with shape (self.size, 1)

        """
        return 1/self.window.total*torch.sum(self.work*self.window.window, 1).reshape(-1, 1)


    def window_remove_mean(self) -> None:
        """
        Remove window weighted mean from self.work container.

        Parameters
        ----------
        None

        Returns
        ----------
        None

        """
        self.work.sub_(self.window_mean())


    def window_apply(self) -> None:
        """
        Apply window to self.work container.

        Parameters
        ----------
        None

        Returns
        ----------
        None

        """
        self.work.mul_(self.window.window)


    def mean(self) -> torch.Tensor:
        """
        Return mean for each signal in self.work container.

        Parameters
        ----------
        None

        Returns
        ----------
        mean values for each signal (torch.Tensor) with shape (self.size, 1)

        """
        return torch.mean(self.work, 1).reshape(-1, 1)


    def median(self) -> torch.Tensor:
        """
        Return median  for each signal in self.work container.

        Parameters
        ----------
        None

        Returns
        ----------
        median values for each signal (torch.Tensor) with shape (self.size, 1)

        """
        return torch.median(self.work, 1).values.reshape(-1, 1)


    def normalize(self,
                  window:bool = False) -> None:
        """
        Normalize (standardize) self.work container.

        Parameters
        ----------
        window: bool
            flag to use window weighted mean istead of regular mean

        Returns
        -------
        None

        """
        mean = self.window_mean() if window else self.mean()
        std = torch.std(self.work, 1).reshape(-1, 1)
        self.work.sub_(mean).div_(std)


    def to_tensor(self) -> torch.Tensor:
        """
        Return self.work container as tensor.

        Parameters
        ----------
        None

        Returns
        ----------
        data (torch.Tensor)

        """
        return self.work.detach().cpu()


    def to_numpy(self) -> numpy.ndarray:
        """
        Return self.work container as numpy.

        Parameters
        ----------
        None

        Returns
        ----------
        data (numpy.ndarray)

        """
        return self.to_tensor().numpy()


    def to_dict(self) -> dict:
        """
        Return self.work container as dict if epics.

        Parameters
        ----------
        None

        Returns
        ----------
        data (dict)

        """
        if self.source == 'epics':
            name = self.pv_list
            data = self.to_numpy()
            return {pv: signal for pv, signal in zip(name, data)}

        raise ValueError(f'DATA: expected source epics, got {self.source}')


    def to_frame(self) -> pandas.DataFrame:
        """
        Return self.work container as data frame if epics.

        Parameters
        ----------
        None

        Returns
        ----------
        data (pandas.DataFrame)

        """
        if self.source == 'epics':
            frame = pandas.DataFrame()
            name = self.pv_list
            data = self.to_numpy()
            for pv, value in zip(name, data):
                frame = pandas.concat([frame, pandas.DataFrame({"PV": pv, 'DATA': value})])
            return frame

        raise ValueError(f'DATA: expected source epics, got {self.source}')


    def __repr__(self) -> str:
        """
        String representation.

        """
        return f'{self.__class__.__name__}{self.size, self.window}'


    def __len__(self) -> int:
        """
        Return number of signals.

        """
        return self.size


    def __getitem__(self,
                    idx:int) -> torch.Tensor:
        """
        Return signal for given index or PV name if epics.

        """
        if type(idx) == int:
            return self.data[idx]

        if self.source == 'epics' and type(idx) == str:
            if idx in self.pv_list:
                return self[self.pv_list.index(idx)]


    def __call__(self) -> None:
        """
        Reset self.work container (copy self.data container to self.work container).

        """
        self.reset()


def main():
    pass

if __name__ == '__main__':
    main()