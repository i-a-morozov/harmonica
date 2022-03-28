"""
Decomposition module.

"""

import numpy
import torch

from .util import mod, chain, generate_other, make_mark
from .statistics import weighted_mean, weighted_variance
from .statistics import median, biweight_midvariance
from .statistics import standardize
from .anomaly import threshold, dbscan
from .window import Window
from .data import Data
from .frequency import Frequency
from .filter import Filter


class Decomposition():
    """
    Returns
    ----------
    Decomposition class instance.

    Parameters
    ----------
    data: 'Data'
        Data instance

    Attributes
    ----------
    data: 'Data'
        data instance

    Methods
    ----------
    __init__(self, data:'Data') -> None
        Decomposition instance initialization.
    harmonic_sum(frequency:float, window:torch.Tensor, data:torch.Tensor, *, error:bool=False, sigma:torch.Tensor=None, sigma_frequency:float=None) -> tuple
        Estimate parameters (and corresponding errors) for given frequency and batch of signals using (weighted) Fourier sum and direct error propagation.
    harmonic_sum_batched(frequency:torch.Tensor, window:torch.Tensor, data:torch.Tensor, *, error:bool=False, sigma:torch.Tensor=None, sigma_frequency:torch.Tensor=None) -> tuple
        Estimate parameters (and corresponding errors) for given batch of frequencies and batch of signals using (weighted) Fourier sum and direct error propagation.
    harmonic_sum_automatic(frequency:float, window:torch.Tensor, data:torch.Tensor, *, error:bool=False, sigma:torch.Tensor=None, sigma_frequency:float=None) -> tuple
        Estimate parameters (and corresponding errors) for given frequency and batch of signals using (weighted) Fourier sum and automatic error propagation.
    harmonic_amplitude(self, frequency:float, length:int=128, *, order:float=1.0, window:str='cosine_window', error:bool=False, sigma_frequency:float=None, limit:int=16, cpu:bool=True, shift:bool=False, count:int=64, step:int=8, clean:bool=False, factor:float=5.0, method:str='none') -> tuple
        Estimate amplitude (and corresponding errors) for given frequency.
    harmonic_phase(self, frequency:float, length:int=256, *, order:float=0.0, window:str='cosine_window', error:bool=False, sigma_frequency:float=None, limit:int=16, cpu:bool=True, shift:bool=False, count:int=64, step:int=8, tolerance:float=1.0, clean:bool=False, factor:float=5.0, method:str='none') -> tuple
        Estimate phase (and corresponding errors) for given frequency.
    phase_adjust(probe:torch.Tensor, frequency:float, phase:torch.Tensor, *, error:bool=True, sigma_frequency:float=None, sigma_phase:torch.Tensor=None) -> tuple
        Adjust phase of given probe indices.
    phase_advance(cls, probe:torch.Tensor, other:torch.Tensor, frequency:float, phase:torch.Tensor, *, error:bool=True, sigma_frequency:torch.Tensor=None, sigma_phase:torch.Tensor=None, model:bool=False) -> tuple
        Compute phase advance mod 2*pi from probe to other indices for given measured or model phases.
    phase_adjacent(cls, frequency:float, phase:torch.Tensor, *, error:bool=True, sigma_frequency:torch.Tensor=None, sigma_phase:torch.Tensor=None, model:bool=False) -> tuple
        Compute phase advance mod 2*pi between adjacent locations.
    phase_check(cls, frequency:float, frequency_model:float, phase:torch.Tensor, phase_model:torch.Tensor, *, drop_endpoints:bool=True, trust_sequence_length:int=5, clean:bool=False, factor:float=5.0) -> tuple
        Perform synchronization check based on adjacent advance difference for measured and model values.
    phase_virtual(cls, probe:int, limit:int, flags:torch.Tensor, frequency:torch.Tensor, frequency_model:torch.Tensor, phase:torch.Tensor, phase_model:torch.Tensor, *, use_probe:bool=False, full:bool=True, clean:bool=False, factor:float=5.0, error:bool=True, sigma_freuency:torch.Tensor=None, sigma_frequency_model:torch.Tensor=None, sigma_phase:torch.Tensor=None, sigma_phase_model:torch.Tensor=None) -> dict
        Estimate phase at virtual or monitor location using other monitor locations and model phase data.
    __repr__(self) -> str
        String representation.

    """
    def __init__(self, data:'Data'=None) -> None:
        """
        Decomposition instance initialization.

        Parameters
        ----------
        data: 'Data'
            Data instance

        Returns
        -------
        None

        """
        self.data = data


    @staticmethod
    def harmonic_sum(frequency:float, window:torch.Tensor, data:torch.Tensor, *,
                     error:bool=False, sigma:torch.Tensor=None, sigma_frequency:float=None) -> tuple:
        """
        Estimate parameters (and corresponding errors) for given frequency and batch of signals using (weighted) Fourier sum and direct error propagation.

        Note, frequency is treated as an independent random variable in error propagation

        Parameters
        ----------
        frequency: float
            frequency
        window: torch.Tensor
            window
        data: torch.Tensor
            batch of input signals
        error: bool
            flag to estimate standard errors for parameters
        sigma: torch.Tensor
            noise sigma for each signal
        sigma_frequency: float
            frequency sigma

        Returns
        -------
        estimated parameters and standard errors for each signal (tuple)

        """
        dtype = window.dtype
        device = window.device

        size = len(data)
        length = len(window)

        total = window.sum()
        factor = 2.0/total

        pi = 2.0*torch.acos(torch.zeros(1, dtype=dtype, device=device))
        time = torch.linspace(0, length - 1, length, dtype=dtype, device=device)

        w_cos = window*torch.cos(2.0*pi*frequency*time)
        w_sin = window*torch.sin(2.0*pi*frequency*time)

        d_cos = w_cos*data[:, :length]
        d_sin = w_sin*data[:, :length]

        c = factor*torch.sum(d_cos, 1)
        s = factor*torch.sum(d_sin, 1)
        a = torch.sqrt(c*c + s*s)
        b = torch.atan2(-s, +c)

        param = torch.stack([c, s, a, b]).T

        if not error:
            return (param, None)

        sigma = sigma**2
        sigma_frequency = sigma_frequency**2 if sigma_frequency != None else None

        factor = factor**2
        c1a, s1a = c/a, s/a
        c2a, s2a = c1a/a, s1a/a

        dadx = torch.outer(c1a, w_cos) + torch.outer(s1a, w_sin)
        dbdx = torch.outer(s2a, w_cos) - torch.outer(c2a, w_sin)

        sigma_c = factor*sigma*torch.dot(w_cos, w_cos)
        sigma_s = factor*sigma*torch.dot(w_sin, w_sin)
        sigma_a = factor*sigma*torch.sum(dadx*dadx, 1)
        sigma_b = factor*sigma*torch.sum(dbdx*dbdx, 1)

        if sigma_frequency != None:

            dcdf = -torch.sum(2.0*pi*time*d_sin, 1)
            dsdf = +torch.sum(2.0*pi*time*d_cos, 1)

            sigma_c += factor*sigma_frequency*dcdf**2
            sigma_s += factor*sigma_frequency*dsdf**2
            sigma_a += factor*sigma_frequency*(c1a*dcdf + s1a*dsdf)**2
            sigma_b += factor*sigma_frequency*(c2a*dsdf - s2a*dcdf)**2

        sigma_c = torch.sqrt(sigma_c)
        sigma_s = torch.sqrt(sigma_s)
        sigma_a = torch.sqrt(sigma_a)
        sigma_b = torch.sqrt(sigma_b)

        sigma = torch.stack([sigma_c, sigma_s, sigma_a, sigma_b]).T

        return (param, sigma)


    @staticmethod
    def harmonic_sum_batched(frequency:torch.Tensor, window:torch.Tensor, data:torch.Tensor, *,
                             error:bool=False, sigma:torch.Tensor=None, sigma_frequency:torch.Tensor=None) -> tuple:
        """
        Estimate parameters (and corresponding errors) for given batch of frequencies and batch of signals using (weighted) Fourier sum and direct error propagation.

        Note, frequencies are treated as independent random variables in error propagation

        Parameters
        ----------
        frequency: torch.Tensor
            batch of frequencies
        window: torch.Tensor
            window
        data: torch.Tensor
            batch of input signals
        error: bool
            flag to estimate standard errors for parameters
        sigma: torch.Tensor
            noise sigma for each signal
        sigma_frequency: torch.Tensor
            sigma for each frequency

        Returns
        -------
        estimated parameters and standard errors for each signal and each frequency (tuple)

        """
        dtype = window.dtype
        device = window.device

        size = len(data)
        length = len(window)

        total = window.sum()
        factor = 2.0/total

        pi = 2.0*torch.acos(torch.zeros(1, dtype=dtype, device=device))
        time = torch.linspace(0, length - 1, length, dtype=dtype, device=device)

        w_cos = window*torch.cos(2.0*pi*frequency.reshape(-1, 1)*time)
        w_sin = window*torch.sin(2.0*pi*frequency.reshape(-1, 1)*time)

        d_cos = torch.unsqueeze(w_cos, 1)*data[:, :length]
        d_sin = torch.unsqueeze(w_sin, 1)*data[:, :length]

        c = factor*torch.sum(d_cos, -1)
        s = factor*torch.sum(d_sin, -1)
        a = torch.sqrt(c*c + s*s)
        b = torch.atan2(-s, +c)

        param = torch.stack([c.T, s.T, a.T, b.T]).swapaxes(0, -1)

        if not error:
            return (param, None)

        sigma = sigma**2
        sigma_frequency = sigma_frequency**2 if sigma_frequency != None else None

        factor = factor**2
        c1a, s1a = c/a, s/a
        c2a, s2a = c1a/a, s1a/a

        dadx = torch.transpose(torch.unsqueeze(c1a.T, -1)*w_cos + torch.unsqueeze(s1a.T, -1)*w_sin, 0, 1)
        dbdx = torch.transpose(torch.unsqueeze(s2a.T, -1)*w_cos - torch.unsqueeze(c2a.T, -1)*w_sin, 0, 1)

        sigma_c = factor*torch.outer(torch.sum(w_cos*w_cos, 1), sigma)
        sigma_s = factor*torch.outer(torch.sum(w_sin*w_sin, 1), sigma)
        sigma_a = factor*sigma*torch.sum(dadx*dadx, -1)
        sigma_b = factor*sigma*torch.sum(dbdx*dbdx, -1)

        if sigma_frequency != None:

            dcdf = -torch.sum(2.0*pi*time*d_sin, -1)
            dsdf = +torch.sum(2.0*pi*time*d_cos, -1)

            sigma_c += factor*torch.unsqueeze(sigma_frequency, 1)*dcdf**2
            sigma_s += factor*torch.unsqueeze(sigma_frequency, 1)*dsdf**2
            sigma_a += factor*torch.unsqueeze(sigma_frequency, 1)*(c1a*dcdf + s1a*dsdf)**2
            sigma_b += factor*torch.unsqueeze(sigma_frequency, 1)*(c2a*dsdf - s2a*dcdf)**2

        sigma_c = torch.sqrt(sigma_c)
        sigma_s = torch.sqrt(sigma_s)
        sigma_a = torch.sqrt(sigma_a)
        sigma_b = torch.sqrt(sigma_b)

        sigma = torch.stack([sigma_c.T, sigma_s.T, sigma_a.T, sigma_b.T]).swapaxes(0, -1)

        return (param, sigma)


    @staticmethod
    def harmonic_sum_automatic(frequency:float, window:torch.Tensor, data:torch.Tensor, *,
                               error:bool=False, sigma:torch.Tensor=None, sigma_frequency:float=None) -> tuple:
        """
        Estimate parameters (and corresponding errors) for given frequency and batch of signals using (weighted) Fourier sum and automatic error propagation.

        Note, frequency is treated as an independent random variable in error propagation

        Parameters
        ----------
        frequency: float
            frequency
        window: torch.Tensor
            window
        data: torch.Tensor
            batch of input signals
        error: bool
            flag to estimate standard errors for parameters
        sigma: torch.Tensor
            noise sigma for each signal
        sigma_frequency: float
            frequency sigma

        Returns
        -------
        estimated parameters and standard errors for each signal (tuple)

        """
        dtype = window.dtype
        device = window.device

        size = len(data)
        length = len(window)

        total = window.sum()
        factor = 2.0/total

        pi = 2.0*torch.acos(torch.zeros(1, dtype=dtype, device=device))
        time = torch.linspace(0, length - 1, length, dtype=dtype, device=device)

        if not error:
            c = factor*torch.sum(torch.cos(2.0*pi*frequency*time)*data[:, :length]*window, 1)
            s = factor*torch.sum(torch.sin(2.0*pi*frequency*time)*data[:, :length]*window, 1)
            a = torch.sqrt(c*c + s*s)
            b = torch.atan2(-s, +c)
            param = torch.stack([c, s, a, b]).T
            return (param, None)

        param = []
        error = []

        diagonal = torch.zeros(length + 1, dtype=dtype, device=device)
        diagonal[-1] = sigma_frequency**2 if sigma_frequency != None else 0.0

        frequency = torch.tensor(frequency, dtype=dtype, device=device)
        frequency.requires_grad_(True)

        w_cos = factor*window*torch.cos(2.0*pi*frequency*time)
        w_sin = factor*window*torch.sin(2.0*pi*frequency*time)

        for idx, signal in enumerate(data[:, :length]):
            diagonal[range(length)] = sigma[idx]**2
            signal.requires_grad_(True)

            c = torch.sum(w_cos*signal)
            s = torch.sum(w_sin*signal)
            a = torch.sqrt(c*c + s*s)
            b = torch.atan2(-s, +c)

            c.backward(retain_graph=True)
            grad = torch.hstack([signal.grad, frequency.grad])
            sigma_c = torch.sqrt(torch.sum(grad*diagonal*grad))
            signal.grad = None
            frequency.grad = None

            s.backward(retain_graph=True)
            grad = torch.hstack([signal.grad, frequency.grad])
            sigma_s = torch.sqrt(torch.sum(grad*diagonal*grad))
            signal.grad = None
            frequency.grad = None

            a.backward(retain_graph=True)
            grad = torch.hstack([signal.grad, frequency.grad])
            sigma_a = torch.sqrt(torch.sum(grad*diagonal*grad))
            signal.grad = None
            frequency.grad = None

            b.backward(retain_graph=True)
            grad = torch.hstack([signal.grad, frequency.grad])
            sigma_b = torch.sqrt(torch.sum(grad*diagonal*grad))
            signal.grad = None
            frequency.grad = None

            param.append(torch.stack([c, s, a, b]).detach())
            error.append(torch.stack([sigma_c, sigma_s, sigma_a, sigma_b]))

        param = torch.stack(param)
        error = torch.stack(error)
        return (param, error)


    def harmonic_amplitude(self, frequency:float, length:int=128, *,
                           order:float=1.0, window:str='cosine_window',
                           error:bool=False, sigma_frequency:float=None, limit:int=16, cpu:bool=True,
                           shift:bool=False, count:int=64, step:int=8,
                           clean:bool=False, factor:float=5.0,
                           method:str='none') -> tuple:
        """
        Estimate amplitude (and corresponding errors) for given frequency.

        Note, self.data.data is used for noise estimation, self.data.work is used for computation
        Filtered data can be stored in self.data.work

        Parameters
        ----------
        frequency: float
            frequency
        sigma_frequency: float
            frequency sigma, ignored if None (default)
        length: int
            length to use for amplitude estimation
        order: float
            window order
        window: str
            window name ('cosine_window' or 'kaiser_window')
        error: bool
            flag to estimate errors using error propagation, if True, noise is estimated for given length
        sigma_frequency:float
            frequency error
        limit: int
            number of columns to use for noise estimation
        cpu: bool
            flag to use CPU for SVD computation for noise estimation
        shift: bool
            flag to use shifted samples, generate shifted samples and perform computation for each sample
        count: int
            maximum number of samples to use
        step: int
            shift step size
        clean: bool
            flag to remove outliers for sampled case using robust threshold
        factor: float
            threshold factor
        method: str
            data processing method for sampled case
            'none'     - return mean & std over samples
            'noise'    - return weighted mean & std over samples (use noise for weighting)
            'error'    - return weighted mean & std over samples (use error for weighting)

        Returns
        -------
        (amplitude, [error], [sample data])

        """
        window = self.data.window.__class__(length, window, order, dtype=self.data.dtype, device=self.data.device)

        if shift == False and error == False:
            res, _ = self.harmonic_sum(frequency, window.window, self.data.work)
            *_, res, _ = res.T
            return (res, None, None)

        if shift == False and error == True:
            _, err = Filter.svd_optimal(Filter.make_matrix(self.data.data)[:, :length, :limit], cpu=cpu)
            res, err = self.harmonic_sum(frequency, window.window, self.data.work, error=True, sigma=err, sigma_frequency=sigma_frequency)
            *_, res, _ = res.T
            *_, err, _ = err.T
            return (res, err, None)

        size, _ = self.data.data.shape
        work = torch.stack([self.data.make_matrix(length, step, self.data.work[i])[:count, :] for i in range(size)])
        _, count, _ = work.shape
        work = work.reshape(-1, length)
        work = self.data.__class__.from_data(window, work)

        res, _ = self.harmonic_sum(frequency, window.window, work.work)
        *_, res, _ = res.T
        res = res.reshape(size, count)

        if clean:
            center = median(res)
            spread = biweight_midvariance(res).sqrt()
            min_value, max_value = center - factor*spread, center + factor*spread
            mask = threshold(res, min_value, max_value).to(window.dtype)
        else:
            mask = torch.ones_like(res)

        if method == 'none':
            center = weighted_mean(res, weight=mask)
            spread = weighted_variance(res, weight=mask, center=center).sqrt()
            return (center, spread, res)

        data = torch.stack([self.data.make_matrix(length, step, self.data.data[i])[:count, :] for i in range(size)])
        _, count, _ = data.shape
        data = data.reshape(-1, length)
        data = Data.from_data(window, data)

        _, err = Filter.svd_optimal(Filter.make_matrix(data.data)[:, :, :limit], cpu=cpu)
        err = err.reshape(size, count)

        if method == 'noise':
            weight = mask/err**2
            center = weighted_mean(res, weight=weight)
            spread = weighted_variance(res, weight=weight, center=center).sqrt()
            return (center, spread, res)

        if method != 'error':
            raise Exception(f'DECOMPOSITION: unknown method {method}')

        _, err = self.harmonic_sum(frequency, window.window, work.work, error=True, sigma=err.flatten(), sigma_frequency=sigma_frequency)
        *_, err, _ = err.T
        err = err.reshape(size, count)

        weight = mask/err**2
        center = weighted_mean(res, weight=weight)
        spread = weighted_variance(res, weight=weight, center=center).sqrt()
        return (center, spread, res)


    def harmonic_phase(self, frequency:float, length:int=256, *,
                           order:float=0.0, window:str='cosine_window',
                           error:bool=False, sigma_frequency:float=None, limit:int=16, cpu:bool=True,
                           shift:bool=False, count:int=64, step:int=8, tolerance:float=1.0,
                           clean:bool=False, factor:float=5.0,
                           method:str='none') -> tuple:
        """
        Estimate phase (and corresponding errors) for given frequency.

        Note, self.data.data is used for noise estimation, self.data.work is used for computation
        Filtered data can be stored in self.data.work

        Parameters
        ----------
        frequency: float
            frequency
        sigma_frequency: float
            frequency sigma, ignored if None (default)
        length: int
            length to use for amplitude estimation
        order: float
            window order
        window: str
            window name ('cosine_window' or 'kaiser_window')
        error: bool
            flag to estimate errors using error propagation, if True, noise is estimated for given length
        sigma_frequency:float
            frequency error
        limit: int
            number of columns to use for noise estimation
        cpu: bool
            flag to use CPU for SVD computation for noise estimation
        shift: bool
            flag to use shifted samples, generate shifted samples and perform computation for each sample
        count: int
            maximum number of samples to use
        step: int
            shift step size
        tolerance: float
            zero-crossing tolerance
        clean: bool
            flag to remove outliers for sampled case using robust threshold
        factor: float
            threshold factor
        method: str
            data processing method for sampled case
            'none'     - return mean & std over samples
            'noise'    - return weighted mean & std over samples (use noise for weighting)
            'error'    - return weighted mean & std over samples (use error for weighting)

        Returns
        -------
        (phase, [error], [sample data])

        """
        window = self.data.window.__class__(length, window, order, dtype=self.data.dtype, device=self.data.device)

        if shift == False and error == False:
            res, _ = self.harmonic_sum(frequency, window.window, self.data.work)
            *_, res = res.T
            return (res, None, None)

        if shift == False and error == True:
            _, err = Filter.svd_optimal(Filter.make_matrix(self.data.data)[:, :length, :limit], cpu=cpu)
            res, err = self.harmonic_sum(frequency, window.window, self.data.work, error=True, sigma=err, sigma_frequency=sigma_frequency)
            *_, res = res.T
            *_, err = err.T
            return (res, err, None)

        size, _ = self.data.data.shape
        work = torch.stack([self.data.make_matrix(length, step, self.data.work[i])[:count, :] for i in range(size)])
        _, count, _ = work.shape
        work = work.reshape(-1, length)
        work = self.data.__class__.from_data(window, work)

        res, _ = self.harmonic_sum(frequency, window.window, work.work)
        *_, res = res.T
        res = res.reshape(size, count)

        add = torch.linspace(0, (count - 1)*step*2.0*numpy.pi*frequency, count, dtype=self.data.dtype, device=self.data.device)
        res = mod(res - add, 2.0*numpy.pi, -numpy.pi)

        msk = res.sign()
        res[:, 1:] += numpy.pi*(msk[:, 0].reshape(-1, 1) - msk[:, 1:])

        msk = torch.abs(res[:, 0].reshape(-1, 1) - res[:, 1:]) > tolerance
        res[:, 1:][msk] = mod(res[:, 1:][msk], 2.0*numpy.pi, -numpy.pi)

        if clean:
            center = median(res)
            spread = biweight_midvariance(res).sqrt()
            min_value, max_value = center - factor*spread, center + factor*spread
            mask = threshold(res, min_value, max_value).to(window.dtype)
        else:
            mask = torch.ones_like(res)

        if method == 'none':
            center = weighted_mean(res, weight=mask)
            spread = weighted_variance(res, weight=mask, center=center).sqrt()
            return (center, spread, res)

        data = torch.stack([self.data.make_matrix(length, step, self.data.data[i])[:count, :] for i in range(size)])
        _, count, _ = data.shape
        data = data.reshape(-1, length)
        data = Data.from_data(window, data)

        _, err = Filter.svd_optimal(Filter.make_matrix(data.data)[:, :, :limit], cpu=cpu)
        err = err.reshape(size, count)

        if method == 'noise':
            weight = mask/err**2
            center = weighted_mean(res, weight=weight)
            spread = weighted_variance(res, weight=weight, center=center).sqrt()
            return (center, spread, res)

        if method != 'error':
            raise Exception(f'DECOMPOSITION: unknown method {method}')

        _, err = self.harmonic_sum(frequency, window.window, work.work, error=True, sigma=err.flatten(), sigma_frequency=sigma_frequency)
        *_, err, _ = err.T
        err = err.reshape(size, count)

        weight = mask/err**2
        center = weighted_mean(res, weight=weight)
        spread = weighted_variance(res, weight=weight, center=center).sqrt()
        return (center, spread, res)


    @staticmethod
    def phase_adjust(probe:torch.Tensor, frequency:float, phase:torch.Tensor, *,
                    error:bool=True, sigma_frequency:float=None, sigma_phase:torch.Tensor=None) -> tuple:
        """
        Adjust phase of given probe indices.

        Parameters
        ----------
        probe: torch.Tensor
            probe indices
        frequency: float
            frequency value (fractional part)
        phase: torch.Tensor
            phase data for all location
        error: bool
            flag to compute errors
        sigma_frequency: float
            frequency error
        sigma_phase: torch.Tensor
            phase error data for all locations

        Returns
        -------
        adjusted phase and error (tuple)

        """
        count, *_ = phase.shape

        dtype, device = phase.dtype, phase.device

        probe = probe.to(dtype).to(device)
        probe_index = mod(probe, count).to(torch.int64).to(device)

        shift = torch.div(probe - probe_index, count, rounding_mode='floor')
        phase = mod(phase[probe_index] + shift*2.0*numpy.pi*frequency, 2.0*numpy.pi, -numpy.pi)

        if not error:
            return (phase, torch.zeros_like(phase))

        if sigma_phase is None:
            return (phase, torch.zeros_like(phase))

        if sigma_frequency is None:
            return (phase, sigma_phase[probe_index])

        error = (sigma_phase[probe_index]**2 + (2.0*numpy.pi*shift)**2*sigma_frequency**2).sqrt()

        return (phase, error)


    @classmethod
    def phase_advance(cls, probe:torch.Tensor, other:torch.Tensor, frequency:float, phase:torch.Tensor, *,
                      error:bool=True, sigma_frequency:torch.Tensor=None, sigma_phase:torch.Tensor=None, model:bool=False) -> tuple:
        """
        Compute phase advance mod 2*pi from probe to other indices for given measured or model phases.

        Note, phase advance is computed from probe to other, where probe > other, phase advance is negative

        Parameters
        ----------
        probe: torch.Tensor
            probe indices
        other: torch.Tensor
            other indices
        frequency: float
            frequency (fractional part)
        phase: torch.Tensor
            phase data for all location
        error: bool
            flag to compute errors
        sigma_frequency: float
            frequency error
        sigma_phase: torch.Tensor
            phase error data for all locations
        model: bool
            input is for model flag

        Returns
        -------
        phase advance mod 2*pi and error (tuple)

        """
        count, *_ = phase.shape

        probe_phase, probe_sigma = cls.phase_adjust(probe, frequency, phase,
                                                    error=error, sigma_frequency=sigma_frequency, sigma_phase=sigma_phase)
        other_phase, other_sigma = cls.phase_adjust(other, frequency, phase,
                                                    error=error, sigma_frequency=sigma_frequency, sigma_phase=sigma_phase)

        mask = probe < other

        advance = (-1)**(mask.logical_not())*mod((-1)**mask*(probe_phase - other_phase), 2.0*numpy.pi)

        if not error:
            return (advance, torch.zeros_like(advance))

        if sigma_phase is None:
            return (advance, torch.zeros_like(advance))

        if not model:
            return (advance, (probe_sigma**2 + other_sigma**2).sqrt())

        error = []
        for i, j in zip(probe.flatten(), other.flatten()):
            index = mod(torch.tensor(range(*sorted((i.item(), j.item()))), dtype=torch.int64), count).to(torch.int64)
            error.append(torch.sqrt(torch.sum(sigma_phase[index]**2)))

        return (advance, torch.stack(error).reshape_as(advance))


    @classmethod
    def phase_adjacent(cls, frequency:float, phase:torch.Tensor, *,
                       error:bool=True, sigma_frequency:torch.Tensor=None, sigma_phase:torch.Tensor=None, model:bool=False) -> tuple:
        """
        Compute phase advance mod 2*pi between adjacent locations.

        Parameters
        ----------
        frequency: float
            frequency (fractional part)
        phase: torch.Tensor
            phase data for all location
        error: bool
            flag to compute errors
        sigma_frequency: float
            frequency error
        sigma_phase: torch.Tensor
            phase error data for all locations
        model: bool
            input is for model flag

        Returns
        -------
        adjacent phase advance mod 2*pi and error (tuple)

        """
        size, *_ = phase.shape
        index = torch.arange(0, size, dtype=torch.int64, device=phase.device)
        return cls.phase_advance(index, index + 1, frequency, phase,
                                 error=error, sigma_frequency=sigma_frequency, sigma_phase=sigma_phase, model=model)


    @classmethod
    def phase_check(cls, frequency:float, frequency_model:float, phase:torch.Tensor, phase_model:torch.Tensor, *,
                    drop_endpoints:bool=True, trust_sequence_length:int=5, factor:float=5.0) -> tuple:
        """
        Perform synchronization check based on adjacent advance difference for measured and model values.

        Only one turn error is checked
        Empty dict is returned if all locations pass

        Parameters
        ----------
        frequency: float
            measured tune (fractional part)
        frequency_model: float
            model tune
        phase: torch.Tensor
            measured phase data
        phase_model: torch.Tensor
            model phase data
        drop_endpoints: bool
            flag to drop endpoints
        trust_sequence_length: int
            minimun trust sequence length
        factor: float
            threshold factor

        Returns
        -------
        ({marked: (shift, phase)}, auxiliary)

        """
        select = {}
        result = {}

        advance_phase, _ = cls.phase_adjacent(frequency, phase, error=False)
        advance_model, _ = cls.phase_adjacent(frequency_model, phase_model, error=False)

        advance_error = (advance_phase - advance_model)/advance_model
        advance_error = standardize(advance_error, center_estimator=median, spread_estimator=biweight_midvariance)

        select['phase'] = advance_phase
        select['model'] = advance_model
        select['check'] = advance_phase

        pair = set()

        factor = torch.tensor(factor, dtype=advance_error.dtype, device=advance_error.device)
        mask = threshold(advance_error, -factor, +factor).squeeze(0)
        mark = make_mark(len(mask), mask.logical_not()).cpu().numpy()
        pair.update(set((i, i + 1) for i in mark))

        if len(pair) == 0:
            return result, select

        table = chain([list(pair) for  pair in pair])

        if drop_endpoints:
            table = [*map(lambda chain: chain[1:-1] if len(chain) > len([1, 1]) else chain, table)]

        marked = [j for i in table for j in i]
        passed = chain([i for i in range(len(phase)) if i not in marked])
        passed = numpy.array([element for sequence in passed for element in sequence if len(sequence) > trust_sequence_length])
        for i in range(len(advance_error)):
            if i not in passed:
                marked.append(i)
        marked = numpy.unique(numpy.array(marked))

        marked = torch.tensor(marked, dtype=torch.int64)
        passed = torch.tensor(passed, dtype=torch.int64)

        for index in marked:

            local_model, _ = torch.stack([torch.tensor(cls.phase_advance(index, other, frequency_model, phase_model, error=False, model=True)) for other in passed]).T

            local_phase = torch.clone(phase)
            phase_x = mod(local_phase[index] + 1.0*2.0*numpy.pi*frequency, 2*numpy.pi, -numpy.pi).cpu().item()
            local_phase[index] = phase_x
            local_phase, _ = torch.stack([torch.tensor(cls.phase_advance(index, other, frequency, local_phase, error=False, model=False)) for other in passed]).T
            error_x = torch.sum((local_model - local_phase)**2)

            local_phase = torch.clone(phase)
            phase_y = mod(local_phase[index] + 0.0*2.0*numpy.pi*frequency, 2*numpy.pi, -numpy.pi).cpu().item()
            local_phase[index] = phase_y
            local_phase, _ = torch.stack([torch.tensor(cls.phase_advance(index, other, frequency, local_phase, error=False, model=False)) for other in passed]).T
            error_y = torch.sum((local_model - local_phase)**2)

            local_phase = torch.clone(phase)
            phase_z = mod(local_phase[index] - 1.0*2.0*numpy.pi*frequency, 2*numpy.pi, -numpy.pi).cpu().item()
            local_phase[index] = phase_z
            local_phase, _ = torch.stack([torch.tensor(cls.phase_advance(index, other, frequency, local_phase, error=False, model=False)) for other in passed]).T
            error_z = torch.sum((local_model - local_phase)**2)

            error = torch.stack([error_x, error_y, error_z]).argmin().item()
            select[index.item()] = [phase_x, phase_y, phase_z][error]
            result[index.item()] = (error - 1, [phase_x, phase_y, phase_z][error])

        if result != {}:
            check = torch.clone(phase)
            for index in result:
                check[index] = select[index]
            advance_check, _ = cls.phase_adjacent(frequency, check)
            select['check'] = advance_check

        return result, select


    @classmethod
    def phase_virtual(cls, probe:int, limit:int, flags:torch.Tensor,
                    frequency:torch.Tensor, frequency_model:torch.Tensor, phase:torch.Tensor, phase_model:torch.Tensor, *,
                    use_probe:bool=False, full:bool=True, clean:bool=False, factor:float=5.0, error:bool=True,
                    sigma_frequency:torch.Tensor=None, sigma_frequency_model:torch.Tensor=None,
                    sigma_phase:torch.Tensor=None, sigma_phase_model:torch.Tensor=None) -> dict:
        """
        Estimate phase at virtual or monitor location using other monitor locations and model phase data.

        Note, sigma_phase contains error for measured phase, only errors for monitors are used
        Note, sigma_PHASE contains error from given location to the next location, should be defined for all locations

        Parameters
        ----------
        probe: int
            location index
        limit: int
            range limit around probe location
        flags: torch.Tensor
            virtual/monitor flags 0/1 for each location
        frequency: float
            measured tune (fractional part)
        frequency_model: float
            model tune
        phase: torch.Tensor
            measured phase data
        phase_model: torch.Tensor
            model phase data
        use_probe: bool
            flag to use probe phase
        full: bool
            flag to allow indices on different turns
        clean: bool
            flag to clean data (threashold)
        factor: float
            threshold factor
        error:
            flag to compute errors
        sigma_frequency: torch.Tensor
            measured tune error
        sigma_frequency_model: torch.Tensor
            model tune error
        sigma_phase: torch.Tensor
            measured phase error
        sigma_phase_model: torch.Tensor
            model phase error

        Returns
        -------
        virtual phase dictionary

        """
        result = {}

        count = len(phase)
        limit = min(limit, count // 2 - 1)

        index = {}
        other = generate_other(probe, limit, flags)
        if use_probe:
            other.append(probe)
        for i in other:
            if flags[int(mod(i, count))] == 1:
                index[i] = int(mod(i, count))

        if not full:
            index = {key: value for key, value in index.items() if key > 0 and key < count}

        result['model'] = None
        result['probe'] = probe
        result['limit'] = limit
        result['index'] = index
        result['clean'] = None

        other = torch.tensor(list(index.keys()), dtype=torch.int64, device=flags.device)
        probe = probe*torch.ones_like(other)

        correct, correct_error = cls.phase_adjust(other, frequency, phase,
                                                  error=error, sigma_frequency=sigma_frequency, sigma_phase=sigma_phase)
        advance, advance_error = cls.phase_advance(probe, other, frequency_model, phase_model,
                                                   error=error, sigma_frequency=sigma_frequency_model, sigma_phase=sigma_phase_model, model=True)
        virtual, virtual_error = mod(correct - advance, 2.0*numpy.pi, -numpy.pi), (correct_error**2 + advance_error**2).sqrt()

        result['phase'] = virtual
        result['error'] = virtual_error

        if clean:
            center = median(virtual)
            spread = biweight_midvariance(virtual).sqrt()
            min_value, max_value = center - factor*spread, center + factor*spread
            mask = threshold(virtual, min_value, max_value).squeeze(0)
        else:
            mask = torch.ones_like(virtual).to(torch.bool)

        result['clean'] = mask

        weight = mask.to(virtual_error.dtype) if torch.allclose(virtual_error, torch.zeros_like(virtual_error)) else mask/virtual_error**2
        center = weighted_mean(virtual, weight=weight)
        spread = weighted_variance(virtual, weight=weight, center=center).sqrt()
        result['model'] = torch.stack([center, spread])

        return result


    def __repr__(self) -> str:
        """
        String representation.

        """
        return f'{self.__class__.__name__}({self.data})'


def main():
    pass

if __name__ == '__main__':
    main()