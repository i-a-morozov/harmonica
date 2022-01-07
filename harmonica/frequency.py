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
        Data instance
    pad: int
        padded length
        pad zeros for FFT spectrum and frequency computation
    f_range: tuple
        frequency range used for initial frequency guess location
        FFT frequency estimation range
    fraction: float
        FFRFT frequency range fraction in units of FFT bin size (one over signal length)

    Attributes
    ----------
    data: 'Data'
        data instance
    size: int
        number of signals
    length: int
        signal length
    dtype: torch.dtype
        data type
    rdtype: torch.dtype
        real data type
    cdtype: torch.dtype
        complex data type
    device: torch.device
        device
    pad: int
        padded length
    fft: callable
        FFT function, torch.fft.rfft/torch.fft.fft for real/complex input data
    fft_freq: callable
        FFT grid function, torch.fft.rfftfreq/torch.fft.fftfreq for real/complex input data
    fft_grid: torch.Tensor
        FFT grid frequencies
    fft_step: torch.Tensor
        FFT frequency step (bin size, one over padded length)
    fft_spectrum: torch.Tensor
        FFT amplitude spectrum container
    fft_min_bin: int
        FFT bin closest to f_min
    fft_max_bin: int
        FFT bin closest to f_max
    fft_min: torch.Tensor
        FFT frequency closest to f_min
    fft_max: torch.Tensor
        FFT frequency closest to f_max
    fft_bin: torch.Tensor
        Max FFT bins for all signals
    fft_frequency: torch.Tensor
        FFT frequency estimation for all signals
    ffrft_flag: bool
        FFRFT initialization flag
    ffrft_start: torch.Tensor
        FFRFT starting frequencies for all signals
        By default FFT frequency estimations are used. i.e. might be different across signals
    ffrft_span: float
        FFRFT frequency range (same for all signals)
    ffrft_range: torch.Tensor
        FFRFT auxiliary
    ffrft_data: torch.Tensor
        FFRFT auxiliary
    ffrft_trig: torch.Tensor
        FFRFT auxiliary
    ffrft_work: torch.Tensor
        FFRFT auxiliary
    ffrft_table: torch.Tensor
        FFRFT data
    ffrft_spectrum: torch.Tensor
        FFRFT amplitude spectum container
    ffrft_bin: torch.Tensor
        FFRFT max bin for all signals
    ffrft_frequency: torch.Tensor
        FFRFT frequency estimation for all signals
    parabola_bin: torch.Tensor
        parabola max 'bin' for all signals
    parabola_frequency: torch.Tensor
        parabola frequency estimation for all signals
    frequency: torch.Tensor
        frequency estimation for all signals (method dependent)

    Methods
    ----------
    __init__(self, data:'Data', *, pad:int=None, f_range:tuple=(0.0, 0.5), fraction:float=2.0) -> None
        Frequency instance initialization.
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
        Compute FFRFT amplitude spectrum for given central frequency and frequency span.
    ffrft_get_grid(self, idx:int=0) -> torch.Tensor
        Compute FFRFT frequency grid for given signal index.
    ffrft_get_frequency(self) -> None
        Compute FFRFT frequency estimation.
    parabola_get_frequency(self) -> None
        Compute parabola frequency estimation.
    task_fft(self, *, f_range:tuple=(None, None)) -> None
        Compute amplitude spectrum (FFT) and frequency estimation (FFT) using FFT max bin.
    task_ffrft(self, *, f_range:tuple=(None, None), center:float=None, span:float=None) -> None
        Compute amplitude spectrum (FFT & FFRFT) and frequency estimation (FFT & FFRFT) using FFRFT max bin.
    task_parabola(self, *, f_range:tuple=(None, None), center:float=None, span:float=None) -> None
        Compute amplitude spectrum (FFT & FFRFT) and frequency estimation (FFT & FFRFT & parabola) using parabola interpolation.
    __call__(self, task:str='parabola', *, f_range:tuple=(None, None), center:float=None, span:float=None) -> None
        Compute amplitude spectrum and frequency using selected method.
    task_mean_spectrum(self, *, log:bool=False) -> tuple
        Compure normalized mean spectrum.
    task_mixed_spectrum(self, *, length:int=1024, f_range:tuple=(None, None), name:str='cosine_window', order:float=1.0, normalize:bool=True, position:list=None, log:bool=False, **kwargs) -> tuple
        Compute normalized mixed spectrum and given parameters.
    task_mixed_frequency(self, *, length:int=1024, f_range:tuple=(None, None), name:str='cosine_window', order:float=1.0, normalize:bool=True, position:list=None, **kwargs) -> torch.Tensor
        Estimate frequency using mixed data and given parameters.
    task_fit(self, *, size:int=32, mode:str='ols', std:torch.Tensor=None) -> torch.Tensor
        Estimate frequency and its uncertainty with OLS (or WLS) parabola fit.
    task_shift(self, length:int, shift:int, *, task:str='parabola', name:str='cosine_window', order:float=1.0, f_range:tuple=(None, None), center:float=None, span:float=None) -> torch.Tensor
        Estimate frequency using shifted signals.
    __repr__(self) -> str
        String representation.
    mod(x: float, y:float, d:float=0.0) -> float
        Return the remainder on division of x by y with offset d.
    harmonics(cls, order:int, basis:list, *, limit:float=1.0, offset:float=-0.5) -> dict
        Generate list of harmonics up to given order for list of given basis frequencies.
    identify(cls, order:int, basis:list, frequencies:list, *, limit:float=1.0, offset:float=-0.5) -> dict
        Identify list of frequencies up to maximum order for given frequency basis.
    autocorrelation(data:torch.Tensor) -> torch.Tensor
        Compute the autocorrelation for a given batch of signals.
    dht(data:torch.Tensor) -> torch.Tensor
        Compute discrete Hilbert transform for a given batch of signals.

    """

    def __init__(self, data:'Data', *, pad:int=None, f_range:tuple=(0.0, 0.5), fraction:float=2.0) -> None:
        """
        Frequency instance initialization.

        Parameters
        ----------
        data: 'Data'
            Data instance
        pad: int
            padded length
            pad zeros for FFT spectrum and frequency computation
        f_range: tuple
            frequency range used for initial frequency guess location
            FFT frequency estimation range
        fraction: float
            FFRFT frequency range fraction in units of FFT bin size (one over signal length)

        Returns
        -------
        None

        """
        self.data = data
        self.size = data.size
        self.length = data.length
        self.dtype = self.data.dtype
        self.cdtype = (1j*torch.tensor(1, dtype=self.dtype)).dtype
        self.rdtype = torch.tensor(1, dtype=self.dtype).abs().dtype
        self.device = self.data.device
        self.pad = pad if pad else self.length
        self.pi = torch.tensor(numpy.pi, dtype=self.rdtype, device=self.device)
        self.fft = torch.fft.rfft if self.dtype != self.cdtype else torch.fft.fft
        self.fft_freq = torch.fft.rfftfreq if self.dtype != self.cdtype else torch.fft.fftfreq
        self.fft_grid = self.fft_freq(self.pad, dtype=self.rdtype, device=self.device)
        self.fft_grid[self.fft_grid < 0] += 1.0
        self.fft_step = torch.tensor(1.0/self.pad, dtype=self.rdtype, device=self.device)
        self.fft_spectrum = torch.zeros((self.size, len(self.fft_grid)), dtype=self.rdtype, device=self.device)
        self.fft_min, self.fft_max = f_range
        self.fft_min_bin = torch.floor(self.fft_min/self.fft_step).to(torch.long).item()
        self.fft_max_bin = torch.floor(self.fft_max/self.fft_step).to(torch.long).item()
        self.fft_min = self.fft_step*self.fft_min_bin
        self.fft_max = self.fft_step*self.fft_max_bin
        self.fft_bin = torch.zeros(self.size, dtype=self.rdtype, device=self.device)
        self.fft_frequency = torch.zeros(self.size, dtype=self.rdtype, device=self.device)
        self.ffrft_flag = False
        self.ffrft_start = torch.zeros(self.size, dtype=self.rdtype, device=self.device)
        self.ffrft_span = fraction/self.length
        self.ffrft_range = torch.linspace(0.0, self.length - 1.0, self.length, dtype=self.rdtype, device=self.device)
        self.ffrft_data = torch.zeros((self.size, self.length), dtype=self.cdtype, device=self.device)
        self.ffrft_trig = torch.zeros(self.length, dtype=self.cdtype, device=self.device)
        self.ffrft_work = torch.zeros(2*self.length, dtype=self.cdtype, device=self.device)
        self.ffrft_table = torch.zeros((self.size, 2*self.length), dtype=self.cdtype, device=self.device)
        self.ffrft_spectrum = torch.zeros((self.size, self.length), dtype=self.rdtype, device=self.device)
        self.ffrft_bin = torch.zeros(self.size, dtype=self.rdtype, device=self.device)
        self.ffrft_frequency = torch.zeros(self.size, dtype=self.rdtype, device=self.device)
        self.parabola_bin = torch.zeros(self.size, dtype=self.rdtype, device=self.device)
        self.parabola_frequency = torch.zeros(self.size, dtype=self.rdtype, device=self.device)
        self.frequency = torch.zeros(self.size, dtype=self.rdtype, device=self.device)


    def fft_get_spectrum(self) -> None:
        """
        Compute FFT amplitude spectrum.

        Note, self.data.work container is used to compute FFT amplitude spectrum

        """
        torch.abs(self.fft(self.data.work, self.pad), out=self.fft_spectrum)


    def fft_get_frequency(self, *, f_range:tuple=(None, None)) -> None:
        """
        Compute FFT frequency estimation with optional frequency range.

        Note, self.fft_spectrum is assumed to be precomputed.

        Parameters
        ----------
        f_range: tuple
            frequency range in (0.0, 0.5) for real data and (0.0, 1.0) for complex data

        Returns
        -------
        None

        """
        if f_range == (None, None):
            self.fft_bin.copy_(self.fft_min_bin + torch.argmax(self.fft_spectrum[:, self.fft_min_bin:self.fft_max_bin], 1))
            self.fft_frequency.copy_(self.fft_step*self.fft_bin)
            return

        self.fft_min, self.fft_max = f_range
        self.fft_min_bin = torch.floor(self.fft_min/self.fft_step).to(torch.long).item()
        self.fft_max_bin = torch.floor(self.fft_max/self.fft_step).to(torch.long).item()
        self.fft_min = self.fft_step*self.fft_min_bin
        self.fft_max = self.fft_step*self.fft_max_bin
        self.fft_get_frequency(f_range=(None, None))


    @staticmethod
    def ffrft_set(length:int, span:float, trig:torch.Tensor, work:torch.Tensor) -> None:
        """
        FFRFT auxiliary data initialization (staticmethod).

        Compute trig and work containers for given length and span

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
        torch.linspace(0.0, length - 1.0, length, out=trig)
        trig.pow_(2.0).mul_(1.0j*numpy.pi*span/length).exp_()
        work[:length].copy_(torch.conj(trig))
        torch.linspace(-length, -1.0, length, out=work[length:])
        work[length:].pow_(2.0).mul_(-1.0j*numpy.pi*span/length).exp_()
        torch.fft.fft(work, out=work)


    @staticmethod
    def ffrft_get(length:int, data:torch.Tensor, trig:torch.Tensor,
                  work:torch.Tensor, table:torch.Tensor) -> None:
        """
        FFRFT computation (staticmethod).

        Compute table container

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

        Note, since signal length is fixed, initialization is done only one time
        self.data can be changed, but length should be fixed

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
        Compute FFRFT amplitude spectrum for given central frequency and frequency span.

        Note, frequency range is (center - 0.5*span, center + 0.5*span)
        Note, self.fft_frequency is assumed to be precomputed if center==None and span==None

        Parameters
        ----------
        center: float
            center frequency
        span: float
            frequency span

        Returns
        -------
        None

        """
        if not self.ffrft_flag or span != None:
            self.ffrft_set_spectrum(span=span)

        if center:
            torch.full(self.ffrft_start.shape, center - 0.5*self.ffrft_span, out=self.ffrft_start)
        else:
            self.ffrft_start.copy_(self.fft_frequency - 0.5*self.ffrft_span)
        if self.dtype == self.rdtype:
            torch.mul(self.data.work, torch.exp(self.ffrft_start.reshape(-1, 1)*2j*self.pi*self.ffrft_range), out=self.ffrft_data)
        else:
            torch.mul(self.data.work.conj(), torch.exp(self.ffrft_start.reshape(-1, 1)*2j*self.pi*self.ffrft_range), out=self.ffrft_data)

        self.__class__.ffrft_get(self.length, self.ffrft_data, self.ffrft_trig, self.ffrft_work, self.ffrft_table)
        self.ffrft_spectrum.copy_(torch.abs(self.ffrft_table[:, :self.length]))


    def ffrft_get_grid(self, idx:int=0) -> torch.Tensor:
        """
        Compute FFRFT frequency grid for given signal index.

        Frequency grid might be different across signals if starting frequencies are different
        Note, since starting frequency can be different across signals, grid is computed for given id

        Parameters
        ----------
        idx: ind
            signal id

        Returns
        -------
        torch.Tensor

        """
        start = self.ffrft_start[idx]
        end = self.ffrft_start[idx] + self.ffrft_span - self.ffrft_span/self.length
        return torch.linspace(start, end, self.length, dtype=self.rdtype, device=self.device)


    def ffrft_get_frequency(self) -> None:
        """
        Compute FFRFT frequency estimation.

        Note, self.ffrft_spectrum is assumed to be precomputed

        """
        self.ffrft_bin.copy_(torch.argmax(self.ffrft_spectrum, 1))
        self.ffrft_frequency.copy_(self.ffrft_start + self.ffrft_bin*self.ffrft_span/self.length)


    def parabola_get_frequency(self) -> None:
        """
        Compute parabola frequency estimation.

        Note, self.ffrft_spectrum is assumed to be precomputed

        """
        index = [*range(self.size)]
        x = self.ffrft_bin.to(torch.long)
        y1 = self.ffrft_spectrum[index, x - 1]
        y2 = self.ffrft_spectrum[index, x]
        y3 = self.ffrft_spectrum[index, x + 1]
        self.parabola_bin.copy_(x + 0.5*(y1 - y3)/(y1 - 2.0*y2 + y3))
        self.parabola_frequency.copy_(self.ffrft_start + self.parabola_bin*self.ffrft_span/self.length)


    def task_fft(self, *, f_range:tuple=(None, None)) -> None:
        """
        Compute amplitude spectrum (FFT) and frequency estimation (FFT) using FFT max bin.

        Optionaly pass initial guess frequency range
        Note, self.data.work container is used, not altered

        Parameters
        ----------
        f_range: tuple
            frequency range in (0.0, 0.5) for real or in (0.0, 1.0) for complex data

        Returns
        -------
        None

        """
        self.fft_get_spectrum()
        self.fft_get_frequency(f_range=f_range)


    def task_ffrft(self, *, f_range:tuple=(None, None), center:float=None, span:float=None) -> None:
        """
        Compute amplitude spectrum (FFT & FFRFT) and frequency estimation (FFT & FFRFT) using FFRFT max bin.

        Optionaly pass initial guess frequency range and other parameters
        Note, self.data.work container is used, not altered
        Invoke task_fft()

        Parameters
        ----------
        f_range: tuple
            frequency range in (0.0, 0.5) for real or in (0.0, 1.0) for complex data
        center: float
            center frequency
        span: float
            frequency span

        Returns
        -------
        None

        """
        self.task_fft(f_range=f_range)
        self.ffrft_get_spectrum(center=center, span=span)
        self.ffrft_get_frequency()


    def task_parabola(self, *, f_range:tuple=(None, None), center:float=None, span:float=None) -> None:
        """
        Compute amplitude spectrum (FFT & FFRFT) and frequency estimation (FFT & FFRFT & parabola) using parabola interpolation.

        Optionaly pass initial guess frequency range and other parameters
        Note, self.data.work container is used, not altered
        Invoke task_ffrft()

        Parameters
        ----------
        f_range: tuple
            frequency range in (0.0, 0.5) for real or in (0.0, 1.0) for complex data
        center: float
            center frequency
        span: float
            frequency span

        Returns
        -------
        None

        """
        self.task_ffrft(f_range=f_range, center=center, span=span)
        self.parabola_get_frequency()


    def __call__(self, task:str='parabola', *, f_range:tuple=(None, None), center:float=None, span:float=None) -> None:
        """
        Compute amplitude spectrum and frequency using selected method.

        Optionaly pass initial guess frequency range and other parameters
        Note, self.data.work container is used, not altered
        Invoke task_fft(), task_ffrft() or task_parabola()
        Set self.frequency container

        Parameters
        ----------
        f_range: tuple
            frequency range in (0.0, 0.5) for real or in (0.0, 1.0) for complex data
        center: float
            center frequency
        span: float
            frequency span

        Returns
        -------
        None

        """
        if task == 'fft':
            self.task_fft(f_range=f_range)
            self.frequency.copy_(self.fft_frequency)
            return

        if task == 'ffrft':
            self.task_ffrft(f_range=f_range, center=center, span=span)
            self.frequency.copy_(self.ffrft_frequency)
            return

        if task == 'parabola':
            self.task_parabola(f_range=f_range, center=center, span=span)
            self.frequency.copy_(self.parabola_frequency)
            return

        raise Exception(f'FREQUENCY: unknown task {task}')


    def task_mean_spectrum(self, *, log:bool=False) -> tuple:
        """
        Compure normalized mean spectrum.

        Computed FFT spectra are normalized using amplitudes at estimated frequencies and averaged over signals
        Note, self.fft_spectrum and self.frequency are assumed to be precomputed

        Parameters
        ----------
        log: bool
            flag to apply log10 to amplitude spectrum

        Returns
        -------
        frequency grid and normalized mean amplitude spectrum (numpy.ndarray, numpy.ndarray)

        """
        time = self.ffrft_range
        norm = torch.abs(torch.sum(self.data.work*torch.exp(2.0j*numpy.pi*time*self.frequency.reshape(-1, 1)), 1))
        mean = torch.mean(self.fft_spectrum/norm.reshape(-1, 1), 0)
        return (self.fft_grid, torch.log10(mean) if log else mean)


    def task_mixed_spectrum(self, *, length:int=1024, f_range:tuple=(None, None),
                            name:str='cosine_window', order:float=1.0, normalize:bool=True,
                            position:list=None, log:bool=False, **kwargs) -> tuple:
        """
        Compute normalized mixed spectrum and given parameters.

        Without positions, spectrum is computed from mixed signal using FFRFT for given frequency range
        Full frequency range is equal to the total number of signals divided by two for real data (total number of signals for complex data)
        Spectrum is computed using TYPE-III NUFFT if BPM positions are given
        Positions are assumed to be in [0, 1)
        Normalized location along the ring or normalized accumulateed phase advance can be used as position
        Note, self.data.work container is used, not altered

        Parameters
        ----------
        length: int
            length to use
        f_range: tuple
            frequency range
        name: str
            window name
        order: float
            window order
        normalize: bool
            flag to normalize data before mixing
        position: list
            positions list
        log: bool
            flag to apply log10

        Returns
        -------
        frequency grid and normalized mixed amplitude spectrum (numpy.ndarray, numpy.ndarray)

        """
        rate = self.size*(2.0 if self.dtype == self.cdtype else 1.0)
        f_min, f_max = (0.0, 0.5*rate) if f_range == (None, None) else f_range

        window = self.data.window.__class__(length, dtype=self.rdtype, device=self.device)
        data = self.data.from_data(window, self.data.work[:, :length])
        if normalize:
            data.normalize(window=True)

        data = data.make_signal(length, data.work)
        window = self.data.window.__class__(len(data), name, order, dtype=self.rdtype, device=self.device)
        data = self.data.from_data(window, data.reshape(1, -1))
        data.window_apply()

        if position is None:
            frequency = self.__class__(data)
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
            data = data.work.detach().cpu().numpy().flatten()
            time = numpy.array([position + i for i in range(0, length)]).flatten()
            grid = 2.0*numpy.pi*numpy.linspace(f_min, f_max, len(time))
            spectrum = numpy.abs(nufft.nufft1d3(time, data, grid, **kwargs))
            grid /= 2.0*numpy.pi
            spectrum /= numpy.max(spectrum)

        return (grid, numpy.log10(spectrum) if log else spectrum)


    def task_mixed_frequency(self, *, length:int=1024, f_range:tuple=(None, None),
                             name:str='cosine_window', order:float=1.0, normalize:bool=True,
                             position:list=None, **kwargs) -> torch.Tensor:
        """
        Estimate frequency using mixed data and given parameters.

        Without positions, spectrum is computed from mixed signal using FFRFT for given frequency range
        Full frequency range is equal to the total number of signals divided by two for real data (total number of signals for complex data)
        Frequency is estimated using TYPE-III NUFFT if BPM positions are given
        Positions are assumed to be in [0, 1)
        Normalized location along the ring or normalized accumulateed phase advance can be used as position
        Note, self.data.work container is used in computation, not altered

        Parameters
        ----------
        length: int
            length to use
        f_range: tuple
            frequency range
        name: str
            window name
        order: float
            window order
        normalize: bool
            flag to normalize data before mixing
        position: list
            positions list

        Returns
        -------
        1st, 2nd and 3rd frequency approximations (torch.Tensor)

        """
        rate = self.size * (2.0 if self.dtype == self.cdtype else 1.0)
        f_min, f_max = (0.0, 0.5*rate) if f_range == (None, None) else f_range

        window = self.data.window.__class__(length, dtype=self.rdtype, device=self.device)
        data = self.data.from_data(window, self.data.work[:, :length])
        if normalize:
            data.normalize(window=True)

        data = data.make_signal(length, data.work)
        window = self.data.window.__class__(len(data), name, order, dtype=self.rdtype, device=self.device)
        data = self.data.from_data(window, data.reshape(1, -1))
        data.window_apply()

        if position is None:
            frequency = self.__class__(data)
            frequency('parabola', f_range=(f_min/rate, f_max/rate))
            f1, f2, f3 = frequency.fft_frequency, frequency.ffrft_frequency, frequency.parabola_frequency
            result = rate*torch.hstack([f1, f2, f3])
        else:
            data = data.work.detach().cpu().numpy().flatten()
            time = numpy.array([position + i for i in range(0, length)]).flatten()
            grid = 2.0*numpy.pi*numpy.linspace(f_min, f_max, len(time))
            spectrum = numpy.abs(nufft.nufft1d3(time, data, grid, **kwargs))
            index = numpy.argmax(spectrum)
            f1 = grid[index]/(2*numpy.pi)
            omega_min = grid[index - 1]
            omega_max = grid[index + 1]
            grid = numpy.linspace(omega_min, omega_max, len(grid))
            spectrum = numpy.abs(nufft.nufft1d3(time, data, grid))
            index = numpy.argmax(spectrum)
            f2 = grid[index]/(2*numpy.pi)
            y1, y2, y3 = spectrum[index - 1], spectrum[index], spectrum[index + 1]
            delta = (grid[index] - grid[index-1])/(2*numpy.pi)
            f3 = f2 + 0.5*delta*(y1 - y3)/(y1 - 2.0*y2 + y3)
            result = torch.tensor([f1, f2, f3], dtype=self.rdtype, device=self.device)

        return result


    def task_fit(self, *, size:int=32, mode:str='ols', std:torch.Tensor=None) -> torch.Tensor:
        """
        Estimate frequency and its uncertainty with OLS (or WLS) parabola fit.

        Note, data should be precomputed using task_ffrft() or task_parabola()
        Note, self.data.work container is used in computation, not altered

        For fit, DTFT amplitudes for points around the expected maximum location are computed
        Optionaly, error propagation can be used to estimate each amplitude standard error
        Points from FFRFT grid are used, i.e. have same separation as for FFRFT grid

        Note, size value (number of points around expected maximin) has strong effect on error estimation
        Frequency range (2*size/length**2) used for fitting and should not be too small
        A fraction of DFT bin size (1/length) can be used
        In general size should be increased for larger signal length
        Size selection is also effected by window, i.e. local peak shape

        OLS and WLS give similar results, while OLS is much faster to compute

        Parameters
        ----------
        size: int
            number of points around expected peak to use
        mode: str
            fit mode (ols or wls)
        std: torch.Tensor
            noise std for each signal for wls mode

        Returns
        -------
        fitted frequency value and error estimation for each signal (torch.Tensor)

        """
        try:
            from statsmodels.api import OLS, WLS
        except ModuleNotFoundError:
            raise Exception(f'FREQUENCY: statsmodels not found')

        length = 0 if std == None else len(std)

        if mode == 'wls' and length == 0:
            raise Exception(f'FREQUENCY: invalid std argument for WLS mode')

        value, error = [], []

        time = self.ffrft_range

        for idx, signal in enumerate(self.data.work):

            mbin = self.ffrft_bin[idx].to(torch.int32)
            grid = self.ffrft_get_grid(idx)
            grid = grid[mbin - size : mbin + size]

            if mode == 'wls':
                signal.requires_grad_(True)
                matrix = std[idx]**2 + torch.zeros(self.length, dtype=self.rdtype, device=self.device)
                matrix = torch.diag(matrix)

            X, y, w = [], [], []

            for frequency in grid:

                X.append([frequency.cpu().item()**2, frequency.cpu().item(), 1.0])

                c = 2.0/self.data.window.total*torch.sum(torch.cos(2.0*numpy.pi*frequency*time)*signal)
                s = 2.0/self.data.window.total*torch.sum(torch.sin(2.0*numpy.pi*frequency*time)*signal)
                a = torch.log10(torch.sqrt(c*c + s*s))
                y.append(a.cpu().item())

                if mode == 'wls':
                    a.backward()
                    grad = signal.grad
                    w.append(1/torch.dot(grad, torch.matmul(matrix, grad)).detach().cpu().item())
                    signal.grad = None

            out = OLS(y, X).fit() if mode == 'ols' else WLS(y, X, weights=numpy.array(w)).fit()
            c_x, c_y, c_z = out.params
            s_x, s_y, s_z = out.bse
            value.append(-c_y/(2.0*c_x))
            error.append(1.0/(2.0*c_x**2)*numpy.sqrt(c_y**2*s_x**2+c_x**2*s_y**2))

        value = torch.tensor(value, dtype=self.rdtype, device=self.device)
        error = torch.tensor(error, dtype=self.rdtype, device=self.device)

        return torch.stack([value, error]).T


    def task_shift(self, length:int, shift:int, *, task:str='parabola',
                   name:str='cosine_window', order:float=1.0,
                   f_range:tuple=(None, None), center:float=None, span:float=None) -> torch.Tensor:
        """
        Estimate frequency using shifted signals.

        Note, self.data.work container is used in computation, not altered

        Parameters
        ----------
        length: int
            shifted signal length
        shift: int
            shift step
        name: str
            window name
        order: float
            window order
        f_range: tuple
            frequency range in (0.0, 0.5) for real or in (0.0, 1.0) for complex data
        center: float
            center frequency
        span: float
            frequency span

        Returns
        -------
        frequencies for shifted signals grouped by signal (torch.Tensor)

        """
        data = torch.cat([self.data.__class__.make_matrix(length, shift, self.data.work[idx]) for idx in range(self.size)])
        size, length = data.shape
        step = size//self.size
        window = self.data.window.__class__(length, name, order, dtype=self.rdtype, device=self.device)
        data = self.data.__class__.from_data(window, data)
        data.window_apply()
        frequency = self.__class__(data)
        frequency(task, f_range=f_range, center=center, span=span)
        return frequency.frequency.reshape(self.size, -1)


    def __repr__(self) -> str:
        """
        String representation.

        """
        return f'{self.__class__.__name__}({self.data}, f_range={self.fft_min.item(), self.fft_max.item()})'


    @staticmethod
    def mod(x: float, y:float, d:float=0.0) -> float:
        """
        Return the remainder on division of x by y with offset d.

        """
        return x - ((x - d) - (x - d) % y)


    @classmethod
    def harmonics(cls, order:int, basis:list, *,
                  limit:float=1.0, offset:float=-0.5) -> dict:
        """
        Generate list of harmonics up to given order for list of given basis frequencies.

        Parameters
        ----------
        order: int
            harmonic order
        basis: list
            frequency basis
        limit: float
            mod parameter
        offset: float
            mod parameter

        Returns
        -------
        harmonics (dict)

        """
        table = {}
        from itertools import product
        for combo in product(range(-order, order + 1), repeat=len(basis)):
            if sum(map(abs, combo)) > order:
                continue
            first, *_ = combo
            if first < 0:
                continue
            frequency = numpy.sum(basis*numpy.array(combo))
            if first == 0 and frequency <= 0.0:
                continue
            table[combo] = abs(cls.mod(frequency, limit, offset))
        keys = sorted(table, key=lambda x: sum(map(abs, x)))
        return {key: table[key] for key in keys}


    @classmethod
    def identify(cls, order:int, basis:list, frequencies:list, *,
                 limit:float=1.0, offset:float=-0.5) -> dict:
        """
        Identify list of frequencies up to maximum order for given frequency basis.

        Parameters
        ----------
        order: int
            harmonic order
        basis: list
            frequency basis
        frequencies: list
            list of frequencies to identify
        limit: float
            mod parameter
        offset: float
            mod parameter

        Returns
        -------
        closest harmonics (dict)

        """
        table = cls.harmonics(order, basis, limit=limit, offset=offset)
        out = {}
        for frequency in frequencies:
            data = []
            for combo, harmonic in table.items():
                data.append((combo, harmonic, frequency, abs(frequency - harmonic)))
            key, *value = min(data, key=lambda x: x[-1])
            out[key] = value
        return out


    @staticmethod
    def autocorrelation(data:torch.Tensor) -> torch.Tensor:
        """
        Compute the autocorrelation for a given batch of signals.

        Note, real signals with zero mean are assumed

        Parameters
        ----------
        data: torch.Tensor
            batch of input signals

        Returns
        -------
        autocorrelation of input signals (torch.Tensor)


        """
        dtype = data.dtype
        device = data.device
        size, length = data.shape
        out = torch.view_as_real(torch.fft.rfft(data, n=2*length)).pow_(2).sum(-1)
        out = torch.fft.irfft(out, n=2*length)
        out = out[:, :length]
        out = out/torch.tensor(range(length, 0, -1), dtype=dtype, device=device)
        out = out/out[:, :1]
        return out


    @staticmethod
    def dht(data:torch.Tensor) -> torch.Tensor:
        """
        Compute discrete Hilbert transform for a given batch of signals.

        Note, real signals are assumed
        Envelope can be computed as out.abs()
        Instantaneous frequency can be computed as (out[:, :-1]*out[:, 1:].conj()).angle()/(2.0*numpy.pi)

        Parameters
        ----------
        data: torch.Tensor
            batch of input signals

        Returns
        -------
        DHT of input signals (torch.Tensor)

        """
        dtype = data.dtype
        device = data.device
        size, length = data.shape
        pad = torch.ones(length, dtype=dtype, device=device)
        length = length // 2
        pad[1:length] = 2.0
        pad[1 + length:] = 0.0
        out = data - 1j*torch.imag(torch.fft.ifft(torch.fft.fft(data)*pad))
        return out


def main():
    pass

if __name__ == '__main__':
    main()