"""
Data module.
Generate, save and load TbT data.
Apply window to data.
Data normalization.

"""

import epics
import numpy
import pandas
import torch

from .util import LIMIT
from .window import Window

class Data:
    """
    Returns
    ----------
    Data class instance.

    Data(size:int, window:'Window')
    Data.from_tensor(window:'Window', tensor:torch.Tensor)
    Data.from_harmonics(size:int, window:'Window', mean:torch.Tensor, frequency:torch.Tensor, c_amp:torch.Tensor, s_amp:torch.Tensor, std:torch.Tensor=None)
    Data.from_file(size:int, window:'Window', file_name:str)
    Data.from_epics(size:int, window:'Window', pv_list:list, pv_rise:list, shift:int=0, count:int=LIMIT)

    Data type and device are inherited from window.

    Parameters
    ----------
    size: int
        Number of signals. Data length. TbT size.
    window: 'Window'
        Window instance.
    tensor: torch.Tensor:
        Input TbT data tensor with matching length (from_tensor).
    mean: torch.Tensor
        Mean harmonics values (from_harmonics).
    frequency: torch.Tensor
        Harmonics frequencies (from_harmonics).
    c_amp: torch.Tensor
        Harmonics cos amplitudes (from_harmonics).
    s_amp: torch.Tensor
        Harmonics sin amplitudes (from_harmonics).
    std: torch.Tensor
        Noise std.
    file_name: str
        Input file name (from_file).
    pv_list: list
        List of PV names (from_epics)
    pv_rise: list
        List of PV starting indices (from_epics).
    shift: int
        Shift for all PVs (from_epics).
    count: int
        Length to read from PVs (from_epics).

    Attributes
    ----------
    size: int
        Number of signals. Data length. TbT size.
    window: 'Window'
        Window instance.
    length: int
        Window length. Signal length.
    dtype: torch.dtype
        Data type (inherited from window).
    device: torch.device
        Data device (inherited from window).
    data: torch.tensor
        Data container.
    work: torch.tensor
        Work container. Copy of TbT data.
        Normalization is done on work. Window is applied only to work.
    source: str
        Data source ('empty', 'tensor', 'file' or 'epics').
    file_name: str
        Input file (from_file).
    pv_list: list
        List of PV names (from_epics).
    pv_rise: list
        List of PV starting indices (from_epics).

    Methods
    ----------
    reset(self) -> None
        Reset work. Copy data to work.
    set_data(self, tensor:torch.Tensor) -> None
        Copy input tensor with matching shape to data and reset work.
    from_tensor(cls, window:'Window', tensor:torch.Tensor) -> 'Data'
        Generate Data instance from given window (length) and tensor (size x length).
    generate_harmonics(tensor:torch.Tensor, mean:torch.Tensor, frequency:torch.Tensor, c_amp:torch.Tensor, s_amp:torch.Tensor) -> None
        Generate tensor from harmonics.
    add_noise(tensor:torch.Tensor, std:torch.Tensor) -> None
        Add normal noise.
    from_harmonics(cls, size:int, window:'Window', mean:torch.Tensor, frequency:torch.Tensor, c_amp:torch.Tensor, s_amp:torch.Tensor, std:torch.Tensor=None) -> 'Data'
        Generate Data instance from given harmonics parameters.
    save_data(self, file_name:str) -> None
        Save data to file (numpy).
    load_data(self, file_name:str) -> None
        Load data from file (numpy). Reset work.
    from_file(cls, size:int, window:'Window', file_name:str) -> 'Data'
        Generate Data instance from file (numpy).
    pv_get(pv_name:str, count:int=None, **kwargs) -> torch.Tensor
        Get PV value as tensor.
    pv_put(pv_name:str, tensor:torch.Tensor) -> None
        Put tensor to PV.
    load_epics(self, shift:int=0, count:int=LIMIT) -> None
        Get data from PVs. Reset work.
    save_epics(self) -> None
        Put data to PVs.
    from_epics(cls, size:int, window:'Window', pv_list:list, pv_rise:list=None, shift:int=0, count:int=LIMIT) -> 'Data'
        Generate Data instance from epics.
    make_matrix(self, idx:int, length:int, shift:int, name:str=None, order:float=None) -> 'Data'
        Generate matrix from signal (from work).
    make_signal(self, length:int) -> 'Data'
        Generate mixed signal (from work).
    window_mean(self) -> torch.Tensor
        Return window weighted mean.
    window_remove_mean(self) -> None
        Remove window weighted mean. Result in work.
    window_apply(self) -> None
        Apply window. Result in work.
    mean(self) -> torch.Tensor
        Return mean.
    median(self) -> torch.Tensor
        Return median.
    normalize(self, window:bool=False) -> None
        Normalize (standardize). Result in work.
    to_tensor(self) -> torch.Tensor
        Return data as tensor.
    to_numpy(self) -> numpy.ndarray
        Return data as numpy.
    to_dict(self) -> dict
        Return data as dict if epics.
    to_data_frame(self) -> pandas.DataFrame
        Return data as data frame if epics.
    __repr__(self) -> str
        String representation.
    __len__(self) -> int
        Return number of signals. TbT size.
    __getitem__(self, idx:int) -> torch.Tensor
        Return signal for given index or PV name if epics.
    __call__(self, shift:int=0, count:int=LIMIT) -> None
        Reload epics data. Reset work.

    """

    def __init__(self, size:int, window:'Window') -> None:
        self.size = size
        self.window = window
        self.length = self.window.length
        self.dtype = self.window.dtype
        self.device = self.window.device
        self.data = torch.zeros((self.size, self.length), dtype=self.dtype, device=self.device)
        self.work = torch.clone(self.data)
        self.source = 'empty'
        self.file_name = None
        self.pv_name = None
        self.pv_rise = None


    def reset(self) -> None:
        """
        Reset work. Copy data to work.

        """
        self.work.copy_(self.data)


    def set_data(self, tensor:torch.Tensor) -> None:
        """
        Copy input tensor with matching shape to data and reset work.

        Parameters
        ----------
        tensor: torch.Tensor
            input tensor with matching shape

        Returns
        -------
        None

        """
        if self.data.shape == tensor.shape:
            self.data.copy_(tensor)
            self.reset()
            return

        raise Exception(f'DATA: expected shape {self.data.shape}, got {tensor.shape} on input.')


    @classmethod
    def from_tensor(cls, window:'Window', tensor:torch.Tensor) -> 'Data':
        """
        Generate Data instance from given window (length) and tensor (size x length).

        Data type and device are inherited from window.

        Parameters
        ----------
        window: 'Window'
            Window instance
        tensor: torch.Tensor
            input tensor with matching length

        Returns
        -------
        Data instance generated from given window and input tensor

        """
        size, length = tensor.shape

        if length != window.length:
            raise Exception(f'DATA: expected length {window.length}, got {length} on input.')

        out = cls(size, window)
        out.source = 'tensor'
        out.set_data(tensor)
        return out


    @staticmethod
    @torch.jit.script
    def generate_harmonics(tensor:torch.Tensor, mean:torch.Tensor,
                           frequency:torch.Tensor, c_amp:torch.Tensor, s_amp:torch.Tensor) -> None:
        """
        Generate tensor from harmonics.

        Modify tensor container inplace.

        Parameters
        ----------
        tensor: torch.Tensor
            input data container
        mean: torch.Tensor
            mean values for each signal
        frequency: torch.Tensor
            frequency values for each signal and each harmonic
        c_amp: torch.Tensor
            cos amplitude values for each signal and each harmonic
        s_amp: torch.Tensor
            sin amplitude values for each signal and each harmonic

        Returns
        -------
        None

        """
        size, length = tensor.shape
        pi = 2.0*torch.acos(torch.zeros(1, dtype=tensor.dtype, device=tensor.device))
        time = 2.0*pi*frequency*torch.linspace(1, length, length, dtype=tensor.dtype, device=tensor.device)
        torch.sum(c_amp*torch.cos(time) + s_amp*torch.sin(time), 1, out=tensor)
        tensor.add_(mean)


    @staticmethod
    @torch.jit.script
    def add_noise(tensor:torch.Tensor, std:torch.Tensor) -> None:
        """
        Add normal noise.

        Modify tensor container inplace.

        Parameters
        ----------
        tensor: torch.Tensor
            input data container
        std: torch.Tensor
            std values for each signal

        Returns
        -------
        None

        """
        size, length = tensor.shape
        tensor.add_(torch.normal(mean=0.0, std=std.repeat((length, 1)).T))


    @classmethod
    def from_harmonics(cls, size:int, window:'Window', mean:torch.Tensor,
                       frequency:torch.Tensor, c_amp:torch.Tensor, s_amp:torch.Tensor, std:torch.Tensor=None) -> 'Data':
        """
        Generate Data instance from given harmonics parameters.

        Parameters
        ----------
        size: int
            number of signals
        window: 'Window'
            Window instance
        mean: torch.Tensor
            mean values for each signal
        frequency: torch.Tensor
            frequency values for each signal and each harmonic
        c_amp: torch.Tensor
            cos amplitude values for each signal and each harmonic
        s_amp: torch.Tensor
            sin amplitude values for each signal and each harmonic
        std: torch.Tensor
            std values for each signal

        Returns
        -------
        Data instance generated from harmonics

        """
        out = cls(size, window)
        out.source = "harmonics"
        cls.generate_harmonics(out.data, mean, frequency, c_amp, s_amp)
        if std != None:
            cls.add_noise(out.data, std)
        out.reset()
        return out


    def save_data(self, file_name:str) -> None:
        """
        Save data to file (numpy).

        Note, only work is saved.

        Parameters
        ----------
        file_name: str
            file name

        Returns
        -------
        None

        """
        numpy.save(file_name, self.work.detach().cpu().numpy())


    def load_data(self, file_name:str) -> None:
        """
        Load data from file (numpy). Reset work.

        Parameters
        ----------
        file_name: str
            file name

        Returns
        -------
        None

        """
        data = numpy.load(file_name)
        if (self.size, self.length) == data.shape:
            self.source = 'file'
            self.data.copy_(torch.tensor(data, dtype=self.dtype, device=self.device))
            self.reset()
            return
        else:
            raise Exception(f'DATA: expected shape {(self.size, self.length)}, got {data.shape} on input.')


    @classmethod
    def from_file(cls, size:int, window:'Window', file_name:str) -> 'Data':
        """
        Generate Data instance from file (numpy).

        File data should have matching shape.

        Parameters
        ----------
        size: int
            number of signals in a file
        window:
            window instance
        file_name: str
            file name

        Returns
        -------
        Data instance from file

        """
        out = cls(size, window)
        out.file_name = file_name
        out.load_data(out.file_name)
        return out


    @staticmethod
    def pv_get(pv_name:str, count:int=None, **kwargs) -> torch.Tensor:
        """
        Get PV value as tensor.

        Parameters
        ----------
        pv_name: str
            PV name
        count: int
            length to read from PV
        **kwargs:
            passed to torch.tensor

        Returns
        -------
        torch.Tensor
            PV value as tensor

        """
        return torch.tensor(epics.caget(pv_name, count=count), **kwargs)


    @staticmethod
    def pv_put(pv_name:str, tensor:torch.Tensor) -> None:
        """
        Put tensor to PV.

        Parameters
        ----------
        pv_name: str
            PV name
        tensor: torch.Tensor
            tensor to put

        Returns
        -------
        None

        """
        epics.caput(pv_name, tensor.cpu().numpy())


    def load_epics(self, shift:int=0, count:int=LIMIT) -> None:
        """
        Get data from PVs. Reset work.

        pv_list and pv_rise attibutes should be defined.

        Parameters
        ----------
        shift: int
            shift for all PVs
        count: int
            length to load

        Returns
        -------
        None

        """
        self.source = 'epics'
        data = numpy.array(epics.caget_many(self.pv_list, count=count)).reshape(self.size, -1)
        if data.shape == (self.size, count):
            if self.pv_rise != None:
                for i, value in enumerate(data):
                    self.data[i].copy_(torch.tensor(value[shift + self.pv_rise[i] : shift + self.pv_rise[i] + self.length]))
            else:
                self.data.copy_(torch.tensor(data[:, shift : shift + self.length]))

            self.reset()
            return

        raise Exception(f'DATA: expected shape {(self.size, self.length)}, got {data.shape} on input.')


    def save_epics(self) -> None:
        """
        Put data to PVs.

        pv_list attibute should be defined.

        """
        epics.caput_many(self.pv_list, self.work.cpu().numpy())


    @classmethod
    def from_epics(cls, size:int, window:'Window',
                   pv_list:list, pv_rise:list=None, shift:int=0, count:int=LIMIT) -> 'Data':
        """
        Generate Data instance from epics.

        If pv_rise == None, data is loaded from index zero.

        Parameters
        ----------
        size: int
            number of signals
        window:
            Window instance
        pv_list:
            list of PV names
        pv_rise:
            list of statring indices for each PV
        shift: int
            shift for all PVs
        count: int
            length to load

        Returns
        -------
        Data instance from epics

        """
        out = Data(size, window)
        out.pv_list = pv_list
        out.pv_rise = pv_rise
        out.load_epics(shift=shift, count=count)
        return out


    def make_matrix(self, idx:int, length:int, shift:int, name:str=None, order:float=None) -> 'Data':
        """
        Generate matrix from signal (from work).

        Parameters
        ----------
        idx: int
            signal id
        length: int
            sample length
        shift: int
            sample shift
        name: str
            window name
        order: float
            window order

        Returns
        -------
        Data instance with samples generated as shifts of given length

        """
        array = self.work[idx]
        size = 1 + (len(array) - length)//shift
        window = Window(length, name, order, dtype=self.dtype, device=self.device)
        out = Data(size, window)
        out.source = 'tensor'
        for i in range(size):
            out.data[i].copy_(array[i*shift : i*shift + length])
        out.reset()
        return out


    def make_signal(self, length:int, name:str=None, order:float=None) -> 'Data':
        """
        Generate mixed signal (from work).

        Parameters
        ----------
        length: int
            sample length
        name: str
            window name
        order: float
            window order

        Returns
        -------
        Data instance from mixed data.

        """
        window = Window(length*self.size, name, order, dtype=self.dtype, device=self.device)
        out = Data(1, window)
        out.source = 'tensor'
        out.data.copy_(self.work[:, :length].T.flatten())
        out.reset()
        return out


    def window_mean(self) -> torch.Tensor:
        """
        Return window weighted mean.

        Parameters
        ----------
        None

        Returns
        ----------
        torch.Tensor
            mean values for each signal

        """
        return 1/self.window.total*torch.sum(self.work*self.window.data, 1).reshape(-1, 1)


    def window_remove_mean(self) -> None:
        """
        Remove window weighted mean. Result in work.

        """
        self.work.sub_(self.window_mean())


    def window_apply(self) -> None:
        """
        Apply window. Result in work.

        """
        self.work.mul_(self.window.data)


    def mean(self) -> torch.Tensor:
        """
        Return mean.

        Parameters
        ----------
        None

        Returns
        ----------
        torch.Tensor
            mean values for each signal

        """
        return torch.mean(self.work, 1).reshape(-1, 1)


    def median(self) -> torch.Tensor:
        """
        Return median.

        Parameters
        ----------
        None

        Returns
        ----------
        torch.Tensor
            median values for each signal

        """
        return torch.median(self.work, 1).values.reshape(-1, 1)


    def normalize(self, window:bool = False) -> None:
        """
        Normalize (standardize). Result in work.

        Parameters
        ----------
        window: bool
            flag to use window weighted mean

        Returns
        -------
        None

        """
        mean = self.window_mean().reshape(-1, 1) if window else self.mean()
        std = torch.std(self.work, 1).reshape(-1, 1)
        self.work.sub_(mean).div_(std)


    def to_tensor(self) -> torch.Tensor:
        """
        Return work as tensor.

        Parameters
        ----------
        None

        Returns
        ----------
        torch.Tensor
            work tensor (cpu)

        """
        return self.work.cpu()


    def to_numpy(self) -> numpy.ndarray:
        """
        Return data as numpy.

                Parameters
        ----------
        None

        Returns
        ----------
        numpy.ndarray
            work tensor (numpy)

        """
        return self.to_tensor().numpy()


    def to_dict(self) -> dict:
        """
        Return data as dict if epics.

        """
        if self.source == 'epics':
            name = self.pv_list
            data = self.to_numpy()
            return {pv: signal for pv, signal in zip(name, data)}

        raise Exception(f'DATA: expected source epics, got {self.source} on input.')


    def to_data_frame(self) -> pandas.DataFrame:
        """
        Return data as data frame if epics.

        """
        if self.source == 'epics':
            frame = pandas.DataFrame()
            name = self.pv_list
            data = self.to_numpy()
            for pv, value in zip(name, data):
                frame = pandas.concat(
                    [frame, pandas.DataFrame({"PV": pv, 'DATA': value})]
                )
            return frame

        raise Exception(f'DATA: expected source epics, got {self.source} on input.')


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


    def __getitem__(self, idx:int) -> torch.Tensor:
        """
        Return signal for given index or PV name if epics.

        """
        if type(idx) == int:
            return self.data[idx]

        if self.source == 'epics' and type(idx) == str:
            if idx in self.pv_list:
                return self[self.pv_list.index(idx)]


    def __call__(self, shift:int=0, count:int=LIMIT) -> None:
        """
        Reload epics data. Reset work.

        """
        if self.source == 'epics':
            self.load_epics(shift=shift, count=count)
            return

        raise Exception(f'DATA: expected source epics, got {self.source} on input.')


def main():
    pass

if __name__ == '__main__':
    main()
