"""
Frequency module.
Compute amplitude spectrum and frequency estimation (maximum peak in given range).
Compute amplitude spectrum and frequency estimation for mixed data (maximum peak in given range).

"""

import epics
import numpy
import pandas
import torch
import nufft

from .util import LIMIT
from .window import Window
from .data import Data

class Frequency():
    """
    Returns
    ----------
    Frequency class instance.

    Frequency(data:'Data', *, pad:int=0, f_range:tuple=(0.0, 0.5), fraction:float=2.0)

    Parameters
    ----------
    data: 'Data'
        Data instance.
    pad: int
        Padded length.
        Pad zeros for FFT spectrum and frequency computation.
    f_range: tuple
        Frequency range used for initial frequency guess location.
        FFT frequency estimation range.
    fraction: float
        FFRFT frequency range fraction in units of FFT bin size (one over signal length).

    Attributes
    ----------
    data: 'Data'
        Data instance.
    size: int
        Number of signals.
    length: int
        Signal length.
    dtype: torch.dtype
        Data type.
    cdtype: torch.dtype
        Complex data type.
    device: torch.device
        Device.
    pad: int
        Padded length.
    f_min: float
        Min frequency.
    f_max: float
        Max frequency.
    fft_grid: torch.Tensor
        FFT grid frequencies.
    fft_step: torch.Tensor
        FFT frequency step (bin size, one over signal length).
    fft_spectrum: torch.Tensor
        FFT amplitude spectrum.
    fft_min_bin: int
        FFT bin closest to f_min.
    fft_max_bin: int
        FFT bin closest to f_max.
    fft_min: torch.Tensor
        FFT frequency closest to f_min.
    fft_max: torch.Tensor
        FFT frequency closest to f_max.
    fft_bin: torch.Tensor
        Max FFT bins for all signals.
    fft_frequency: torch.Tensor
        FFT frequency estimation for all signals.
    ffrft_flag: bool
        FFRFT initialization flag.
    ffrft_start: torch.Tensor
        FFRFT starting frequencies for all signals.
        By default FFT frequency estimations are used. i.e. might be different across signals.
    ffrft_span: float
        FFRFT frequency range (same for all signals).
    ffrft_range: torch.Tensor
        FFRFT auxiliary.
    ffrft_data: torch.Tensor
        FFRFT auxiliary.
    ffrft_trig: torch.Tensor
        FFRFT auxiliary.
    ffrft_work: torch.Tensor
        FFRFT auxiliary.
    ffrft_table: torch.Tensor
        FFRFT data.
    ffrft_spectrum: torch.Tensor
        FFRFT amplitude spectum.
    ffrft_bin: torch.Tensor
        FFRFT max bin for all signals.
    ffrft_frequency: torch.Tensor
        FFRFT frequency estimation for all signals.
    parabola_bin: torch.Tensor
        Parabola max 'bin' for all signals.
    parabola_frequency: torch.Tensor
        Parabola frequency estimation for all signals.
    frequency: torch.Tensor
        Frequency estimation for all signals (method dependent).

    Methods
    ----------
    fft_get_spectrum(self) -> None
        Compute FFT amplitude spectrum.
    fft_get_frequency(self, *, f_range:tuple=(None, None)) -> None
        Compute FFT frequency estimation with optional frequency range.
    ffrft_set(length:int, span:float, trig:torch.Tensor, work:torch.Tensor) -> None
        FFRFT auxiliary data initialization (staticmethod).
    ffrft_get(length:int, data:torch.Tensor, trig:torch.Tensor, work:torch.Tensor, table:torch.Tensor) -> None
        FFRFT computation (staticmethod).
    ffrft_set_spectrum(self, *, span:float=None) -> None
        FFRFT auxiliary data initialization.
    ffrft_get_spectrum(self, *, center:float=None, span:float=None) -> None
        Compute FFRFT amplitude spectrum for given central frequency and span.
    ffrft_get_grid(self, idx:int=0) -> torch.Tensor
        Compute FFRFT frequency grid for given signal index.
        Frequency grid might be different across signals if starting frequencies are different.
    ffrft_get_frequency(self) -> None
        Compute FFRFT frequency estimation.
    parabola_get_frequency(self) -> None
        Compute parabola frequency estimation.
    task_fft(self, *, reload:bool=False, window:bool=True, f_range:tuple=(None, None), shift:int=0, count:int=LIMIT) -> None
        Compute spectrum (FFT) and frequency estimation using FFT.
        Optionaly reload TbT data (if epics), apply window and pass other parameters.
    task_ffrft(self, *, reload:bool=False, window:bool=True, f_range:tuple=(None, None), center:float=None, span:float=None, shift:int=0, count:int=LIMIT) -> None
        Compute spectrum (FFT & FFRFT) and frequency estimation using FFRFT.
        Optionaly reload TbT data (if epics), apply window and pass other parameters.
    task_parabola(self, *, reload:bool=False, window:bool=True, f_range:tuple=(None, None), center:float=None, span:float=None, shift:int=0, count:int=LIMIT) -> None
        Compute spectrum (FFT & FFRFT) and frequency estimation using parabola.
        Optionaly reload TbT data (if epics), apply window and pass other parameters.
    task_mean_spectrum(self, *, window:bool=False, log:bool=False) -> tuple
        Compure mean normalized spectrum.
        Computed FFT spectra are normalized using estimated frequencies and averaged over TbT signals.
    task_mixed_spectrum(self, *, length:int=1024, window:bool=True, f_range:tuple=(None, None), name:str='cosine_window', order:float=1.0, normalize:bool=True, position:list=None, log:bool=False) -> tuple
        Compute normalized mixed spectrum for given range.
    task_mixed_frequency(self, *, length:int=1024, window:bool=True, f_range:tuple=(None, None), name:str='cosine_window', order:float=1.0, position:list=None) -> torch.Tensor
        Estimate frequency using mixed TbT data.
    __repr__(self) -> str
        String representation.
    __call__(self, task='parabola', *, reload:bool=False, window:bool=True, f_range:tuple=(None, None), center:float=None, span:float=None, shift:int=0, count:int=LIMIT) -> None
        Compute spectrum and frequency using selected method.
        Optionaly reload TbT data (if epics), apply window and pass other parameters.

    """

    def __init__(self, data:'Data', *, pad:int=0, f_range:tuple=(0.0, 0.5), fraction:float=2.0) -> None:
        self.data = data
        self.size = data.size
        self.length = data.length
        self.dtype = self.data.dtype
        self.cdtype = (1j*torch.tensor(1, dtype=self.dtype)).dtype
        self.device = self.data.device
        self.pi = 2.0*torch.acos(torch.zeros(1, dtype=self.dtype, device=self.device))
        self.pad = pad if pad else self.length
        self.f_min, self.f_max = f_range
        self.fft_grid = torch.fft.rfftfreq(self.pad, dtype=self.dtype, device=self.device)
        self.fft_step = torch.tensor(1.0/self.pad, dtype=self.dtype, device=self.device)
        self.fft_spectrum = torch.zeros((self.size, len(self.fft_grid)), dtype=self.dtype, device=self.device)
        self.fft_min_bin = torch.floor(self.f_min/self.fft_step).to(torch.long).item()
        self.fft_max_bin = torch.floor(self.f_max/self.fft_step).to(torch.long).item()
        self.fft_min = self.fft_step*self.fft_min_bin
        self.fft_max = self.fft_step*self.fft_max_bin
        self.fft_bin = torch.zeros(self.size, dtype=self.dtype, device=self.device)
        self.fft_frequency = torch.zeros(self.size, dtype=self.dtype, device=self.device)
        self.ffrft_flag = False
        self.ffrft_start = torch.zeros(self.size, dtype=self.dtype, device=self.device)
        self.ffrft_span = fraction/self.length
        self.ffrft_range = torch.linspace(0.0, self.length - 1.0, self.length, dtype=self.dtype, device=self.device)
        self.ffrft_data = torch.zeros((self.size, self.length), dtype=self.cdtype, device=self.device)
        self.ffrft_trig = torch.zeros(self.length, dtype=self.cdtype, device=self.device)
        self.ffrft_work = torch.zeros(2*self.length, dtype=self.cdtype, device=self.device)
        self.ffrft_table = torch.zeros((self.size, 2*self.length), dtype=self.cdtype, device=self.device)
        self.ffrft_spectrum = torch.zeros((self.size, self.length), dtype=self.dtype, device=self.device)
        self.ffrft_bin = torch.zeros(self.size, dtype=self.dtype, device=self.device)
        self.ffrft_frequency = torch.zeros(self.size, dtype=self.dtype, device=self.device)
        self.parabola_bin = torch.zeros(self.size, dtype=self.dtype, device=self.device)
        self.parabola_frequency = torch.zeros(self.size, dtype=self.dtype, device=self.device)
        self.frequency = torch.zeros(self.size, dtype=self.dtype, device=self.device)


    def fft_get_spectrum(self) -> None:
        """
        Compute FFT amplitude spectrum.

        Modify fft_spectrum container.

        """
        torch.abs(torch.fft.rfft(self.data.work, self.pad), out=self.fft_spectrum)


    def fft_get_frequency(self, *, f_range:tuple=(None, None)) -> None:
        """
        Compute FFT frequency estimation with optional frequency range.

        Modify fft_frequency container.

        Parameters
        ----------
        f_range: tuple
            frequency range in (0.0, 0.5)

        Returns
        -------
        None

        """
        if f_range == (None, None):
            self.fft_bin.copy_(self.fft_min_bin + torch.argmax(self.fft_spectrum[:, self.fft_min_bin:self.fft_max_bin], 1))
            self.fft_frequency.copy_(self.fft_step*self.fft_bin)
            return

        self.f_min, self.f_max = f_range
        self.fft_min_bin = torch.floor(self.f_min/self.fft_step).to(torch.long).item()
        self.fft_max_bin = torch.floor(self.f_max/self.fft_step).to(torch.long).item()
        self.fft_min = self.fft_step*self.fft_min_bin
        self.fft_max = self.fft_step*self.fft_max_bin
        self.fft_get_frequency()


    @staticmethod
    def ffrft_set(length:int, span:float, trig:torch.Tensor, work:torch.Tensor) -> None:
        """
        FFRFT auxiliary data initialization (staticmethod).

        Modify trig and work containers.

        Parameters
        ----------
        length: int
            signal length
        span: float
            frequency span
        trig: torch.Tensor
            auxiliary container
        work: torch.Tensor
            auxiliary container

        Returns
        -------
        None

        """
        pi = 2.0*torch.acos(torch.zeros(1, dtype=trig.dtype, device=trig.device))
        torch.linspace(0.0, length - 1.0, length, out=trig)
        trig.pow_(2.0).mul_(1.0j*pi*span/length).exp_()
        work[:length].copy_(torch.conj(trig))
        torch.linspace(-length, -1.0, length, out=work[length:])
        work[length:].pow_(2.0).mul_(-1.0j*pi*span/length).exp_()
        torch.fft.fft(work, out=work)


    @staticmethod
    @torch.jit.script
    def ffrft_get(length:int, data:torch.Tensor, trig:torch.Tensor,
                  work:torch.Tensor, table:torch.Tensor) -> None:
        """
        FFRFT computation (staticmethod).

        Modify table container.

        Parameters
        ----------
        length: int
            signal length
        data: torch.Tensor
            data container
        trig: torch.Tensor
            auxiliary container
        work: torch.Tensor
            auxiliary container
        table:
            target container

        Returns
        -------
        None

        """
        table[:, length:].zero_()
        table[:, :length] = trig*data
        torch.fft.fft(table, out=table)
        table.mul_(work)
        torch.fft.ifft(table, out=table)
        table[:, :length].mul_(trig)


    def ffrft_set_spectrum(self, *, span:float=None) -> None:
        """
        FFRFT auxiliary data initialization.

        Since signal length is fixed, initialization is done only one time.
        TbT data can be changed, but length should be fixed.

        Parameters
        ----------
        span: float
            frequency span

        Returns
        -------
        None

        """
        self.ffrft_flag = True
        self.ffrft_span = span if span else self.ffrft_span
        self.__class__.ffrft_set(self.length, self.ffrft_span, self.ffrft_trig, self.ffrft_work)


    def ffrft_get_spectrum(self, *, center:float=None, span:float=None) -> None:
        """
        Compute FFRFT amplitude spectrum for given central frequency and span.

        Note, f_min == center - 0.5*span and f_max == center + 0.5*span

        Parameters
        ----------
        center: float
            center frequency
        span: float
            frequency span in (0.0, 0.5)

        Returns
        -------
        None

        """
        if not self.ffrft_flag or span:
            self.ffrft_set_spectrum(span=span)

        if center:
            torch.full(self.ffrft_start.shape, center - 0.5*self.ffrft_span, out=self.ffrft_start)
        else:
            self.ffrft_start.copy_(self.fft_frequency - 0.5*self.ffrft_span)

        torch.mul(self.data.work, torch.exp(self.ffrft_start.reshape(-1, 1)*2j*self.pi*self.ffrft_range), out=self.ffrft_data)
        self.__class__.ffrft_get(self.length, self.ffrft_data, self.ffrft_trig, self.ffrft_work, self.ffrft_table)
        self.ffrft_spectrum.copy_(torch.abs(self.ffrft_table[:, :self.length]))


    def ffrft_get_grid(self, idx:int=0) -> torch.Tensor:
        """
        Compute FFRFT frequency grid for given signal index.

        Frequency grid might be different across signals if starting frequencies are different.
        Note, since starting frequency can be different across signals, grid is computed for given id.

        Parameters
        ----------
        idx: ind
            signal id

        Returns
        -------
        torch.Tensor

        """
        return torch.linspace(self.ffrft_start[idx],
                              self.ffrft_start[idx] + self.ffrft_span-self.ffrft_span/self.length,
                              self.length, dtype=self.dtype, device=self.device)


    def ffrft_get_frequency(self) -> None:
        """
        Compute FFRFT frequency estimation.

        Modify ffrft_frequency container.

        """
        self.ffrft_bin.copy_(torch.argmax(self.ffrft_spectrum, 1))
        self.ffrft_frequency.copy_(self.ffrft_start + self.ffrft_bin*self.ffrft_span/self.length)


    def parabola_get_frequency(self) -> None:
        """
        Compute parabola frequency estimation.

        Modify parabola_frequency container.

        """
        index = [*range(self.size)]
        position = self.ffrft_bin.to(torch.long)
        y1 = torch.log(self.ffrft_spectrum[index, position - 1])
        y2 = torch.log(self.ffrft_spectrum[index, position])
        y3 = torch.log(self.ffrft_spectrum[index, position + 1])
        self.parabola_bin.copy_(position - 0.5 + (y1 - y2)/(y1 - 2.0*y2 + y3))
        self.parabola_frequency.copy_(self.ffrft_start + self.parabola_bin*self.ffrft_span/self.length)


    def task_fft(self, *, reload:bool=False, window:bool=True, f_range:tuple=(None, None),
                 shift:int=0, count:int=LIMIT) -> None:
        """
        Compute spectrum (FFT) and frequency estimation using FFT.
        Optionaly reload TbT data (if epics), apply window and pass other parameters.

        Parameters
        ----------
        reload: bool
            flag to reload epics
        window: bool
            flag to apply window
        f_range: tuple
            frequency range in (0.0, 0.5)
        shift: int
            shift for epics data
        count: int
            count for epics data

        Returns
        -------
        None

        """
        if reload and self.data.source == 'epics':
            self.data(shift=shift, count=count)

        if window:
            self.data.window_apply()

        self.fft_get_spectrum()
        self.fft_get_frequency(f_range=f_range)

        if window:
            self.data.reset()


    def task_ffrft(self, *, reload:bool=False, window:bool=True, f_range:tuple=(None, None),
                 center:float=None, span:float=None, shift:int=0, count:int=LIMIT) -> None:
        """
        Compute spectrum (FFT & FFRFT) and frequency estimation using FFRFT.
        Optionaly reload TbT data (if epics), apply window and pass other parameters.

        Parameters
        ----------
        reload: bool
            flag to reload epics
        window: bool
            flag to apply window
        f_range: tuple
            frequency range in (0.0, 0.5)
        center: float
            center frequency
        span: float
            frequency span in (0.0, 0.5)
        shift: int
            shift for epics data
        count: int
            count for epics data

        Returns
        -------
        None

        """
        if reload and self.data.source == 'epics':
            self.data(shift=shift, count=count)

        if window:
            self.data.window_apply()

        self.task_fft(reload=False, window=False, f_range=f_range)
        self.ffrft_get_spectrum(center=center, span=span)
        self.ffrft_get_frequency()

        if window:
            self.data.reset()


    def task_parabola(self, *, reload:bool=False, window:bool=True, f_range:tuple=(None, None),
                 center:float=None, span:float=None, shift:int=0, count:int=LIMIT) -> None:
        """
        Compute spectrum (FFT & FFRFT) and frequency estimation using parabola.
        Optionaly reload TbT data (if epics), apply window and pass other parameters.

        Parameters
        ----------
        reload: bool
            flag to reload epics
        window: bool
            flag to apply window
        f_range: tuple
            frequency range in (0.0, 0.5)
        center: float
            center frequency
        span: float
            frequency span in (0.0, 0.5)
        shift: int
            shift for epics data
        count: int
            count for epics data

        Returns
        -------
        None

        """
        if reload and self.data.source == 'epics':
            self.data(shift=shift, count=count)

        if window:
            self.data.window_apply()

        self.task_ffrft(reload=False, window=False, f_range=f_range, center=center, span=span)
        self.parabola_get_frequency()

        if window:
            self.data.reset()


    def task_mean_spectrum(self, *, window:bool=False, log:bool=False) -> tuple:
        """
        Compure mean normalized spectrum.
        Computed FFT spectra are normalized using estimated frequencies and averaged over TbT signals.

        Parameters
        ----------
        window: bool
            flag to apply window
        log: bool
            flag to apply log10

        Returns
        -------
        tuple
            frequency grid and amplitude spectum

        """
        if window:
            self.data.window_apply()

        time = torch.linspace(0.0, self.length - 1.0, self.length, dtype=self.dtype, device=self.device)
        norm = torch.abs(torch.sum(self.data.work*torch.exp(2.0j*numpy.pi*time*self.frequency.reshape(-1, 1)), 1))

        if window:
            self.data.reset()

        mean = torch.mean(self.fft_spectrum/norm.reshape(-1, 1), 0)
        return (self.fft_grid, torch.log10(mean) if log else mean)


    def task_mixed_spectrum(self, *, length:int=1024, window:bool=True, f_range:tuple=(None, None),
                            name:str='cosine_window', order:float=1.0, normalize:bool=True,
                            position:list=None, log:bool=False, **kwargs) -> tuple:
        """
        Compute normalized mixed spectrum for given range.

        Without positions, spectrum is computed from mixed signal using normal loop and given range.
        Full frequency range is equal to the total number of signals in TbT divided by two.
        Spectrum is computed using TYPE-III NUFFT if BPM positions are given.
        Positions are assumed to be in [0, 1).

        Parameters
        ----------
        length: int
            length to use
        window: bool
            flag to apply window
        f_range: tuple
            frequency range
        name: str
            window name
        order: float
            window order
        normalize: bool
            flag to normalize data before mixing
        position: list
            BPM positions
        log: bool
            flag to apply log10

        Returns
        -------
        tuple
            frequency grid and amplitude spectum

        """
        rate = self.size
        f_min, f_max = (0.0, 0.5*rate) if f_range == (None, None) else f_range

        if normalize:
            self.data.normalize()

        data = self.data.make_signal(length)

        if window:
            data.window(name=name, order=order)
            data.window_apply()

        if normalize:
            self.data.reset()

        if position is None:
            frequency = Frequency(data)
            f_min, f_max = f_min/rate, f_max/rate
            span = (f_max - f_min)
            center = f_min + 0.5*span
            frequency.ffrft_get_spectrum(center=center, span=span)
            grid = rate*frequency.ffrft_get_grid()
            spectrum, *_ = frequency.ffrft_spectrum
            spectrum /= torch.max(spectrum)
            grid = grid.cpu().numpy()
            spectrum = spectrum.cpu().numpy()
        else:
            data = data.work.cpu().numpy().flatten()
            time = numpy.array([position + i for i in range(1, 1 + length)]).flatten()
            grid = 2.0*numpy.pi*numpy.linspace(f_min, f_max, len(time) + 1)
            spectrum = numpy.abs(nufft.nufft1d3(time, data, grid, **kwargs))
            grid /= 2.0*numpy.pi
            spectrum /= numpy.max(spectrum)

        if self.device.type != 'cpu' and position is None:
            del data
            del frequency
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return (grid, numpy.log10(spectrum) if log else spectrum)


    def task_mixed_frequency(self, *, length:int=1024, window:bool=True, f_range:tuple=(None, None),
                             name:str='cosine_window', order:float=1.0, normalize:bool=True,
                             position:list=None, **kwargs) -> torch.Tensor:
        """
        Estimate frequency using mixed TbT data.

        Without positions, frequency is estimated from mixed signal using normal loop and given range.
        Full frequency range is equal to the total number of signals in TbT divided by two.
        Frequency is estimated using TYPE-III NUFFT if BPM positions are given.
        Positions are assumed to be in [0, 1).

        Parameters
        ----------
        length: int
            length to use
        window: bool
            flag to apply window
        f_range: tuple
            frequency range
        name: str
            window name
        order: float
            window order
        normalize: bool
            flag to normalize data before mixing
        position: list
            BPM positions

        Returns
        -------
        torch.Tensor
            1st, 2nd and 3rd frequency approximations

        """
        rate = self.size
        f_min, f_max = (0.0, 0.5*rate) if f_range == (None, None) else f_range

        if normalize:
            self.data.normalize()

        data = self.data.make_signal(length)

        if normalize:
            self.data.reset()

        if window:
            data.window(name=name, order=order)

        if position is None:
            frequency = Frequency(data)
            frequency('parabola', window=window, f_range=(f_min/rate, f_max/rate))
            f1, f2, f3 = frequency.fft_frequency, frequency.ffrft_frequency, frequency.parabola_frequency
            result = rate*torch.hstack([f1, f2, f3])
        else:
            data.window_apply()
            data = data.work.cpu().numpy().flatten()
            time = numpy.array([position + i for i in range(1, 1 + length)]).flatten()
            grid = 2.0*numpy.pi*numpy.linspace(f_min, f_max, len(time) + 1)
            spectrum = numpy.abs(nufft.nufft1d3(time, data, grid, **kwargs))
            f1 = grid[numpy.argmax(spectrum)]/(2*numpy.pi)
            grid = 2.0*numpy.pi*numpy.linspace(f1 - rate/len(time), f1 + rate/len(time), len(time) + 1)
            spectrum = numpy.abs(nufft.nufft1d3(time, data, grid))
            index = numpy.argmax(spectrum)
            f2 = grid[index]/(2*numpy.pi)
            y1 = numpy.log(spectrum[index - 1])
            y2 = numpy.log(spectrum[index])
            y3 = numpy.log(spectrum[index + 1])
            index = (index - 0.5 + (y1 - y2)/(y1 - 2.0*y2 + y3))
            f3 = f1 - rate/len(time) + 2*index*rate/len(time)**2
            result = torch.tensor([f1, f2, f3], dtype=self.dtype, device=self.device)

        if self.device.type != 'cpu' and position is None:
            del data
            del frequency
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return result


    def __repr__(self) -> str:
        """
        String representation.

        """
        return f'{self.__class__.__name__}({self.data})'


    def __call__(self, task:str='parabola', *, reload:bool=False, window:bool=True, f_range:tuple=(None, None),
                 center:float=None, span:float=None, shift:int=0, count:int=LIMIT) -> None:
        """
        Compute spectrum and frequency using selected method.
        Optionaly reload TbT data (if epics), apply window and pass other parameters.

        Parameters
        ----------
        reload: bool
            flag to reload epics
        window: bool
            flag to apply window
        f_range: tuple
            frequency range in (0.0, 0.5)
        center: float
            center frequency
        span: float
            frequency span in (0.0, 0.5)
        shift: int
            shift for epics data
        count: int
            count for epics data

        Returns
        -------
        None

        """
        if task == 'fft':
            self.task_fft(reload=reload, window=window, f_range=f_range, shift=shift, count=count)
            self.frequency.copy_(self.fft_frequency)
            return

        if task == 'ffrft':
            self.task_ffrft(reload=reload, window=window, f_range=f_range, center=center, span=span, shift=shift, count=count)
            self.frequency.copy_(self.ffrft_frequency)
            return

        if task == 'parabola':
            self.task_parabola(reload=reload, window=window, f_range=f_range, center=center, span=span, shift=shift, count=count)
            self.frequency.copy_(self.parabola_frequency)
            return


def main():
    pass

if __name__ == '__main__':
    main()