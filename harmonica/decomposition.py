"""
Decomposition module.

"""

import epics
import numpy
import pandas
import torch
import nufft

from .util import LIMIT
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
    harmonic_amplitude(self, frequency:float, *, length:int=64, name:str='cosine_window', order:float=1.0, error:bool=False, limit:int=16, cpu:bool=True, sigma_frequency:float=None, shift:bool=False, count:int=64, step:int=1, fit:str='none') -> tuple
        Estimate amplitude (and corresponding errors) for given frequency.
    harmonic_phase(self, frequency:float, *, length:int=64, name:str='cosine_window', order:float=1.0, error:bool=False, limit:int=16, cpu:bool=True, sigma_frequency:float=None, shift:bool=False, count:int=64, step:int=1, fit:str='none') -> tuple
        Estimate phase (and corresponding errors) for given frequency.
    advance(cls, phase_x:torch.Tensor, phase_y:torch.Tensor, *, error_x:torch.Tensor=None, error_y:torch.Tensor=None) -> tuple
        Compute phase advance between two locations (pairwise).
    advance_adjacent(cls, frequency:float, phase:torch.Tensor, *, sigma_phase:torch.Tensor=None, sigma_frequency:torch.Tensor=None) -> tuple
        Compute phase advance between adjacent locations.
    advance_tune(cls, frequency:float, phase:torch.Tensor, *, sigma_phase:torch.Tensor=None, sigma_frequency:torch.Tensor=None) -> tuple
        Compute tune.

    """

    def __init__(self, data:'Data') -> None:
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

        param = torch.stack([c.T, s.T, a.T, b.T]).T

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

        sigma = torch.stack([sigma_c.T, sigma_s.T, sigma_a.T, sigma_b.T]).T

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


    def harmonic_amplitude(self, frequency:float, *,
                           length:int=64, name:str='cosine_window', order:float=1.0,
                           error:bool=False, limit:int=16, cpu:bool=True, sigma_frequency:float=None,
                           shift:bool=False, count:int=64, step:int=1,
                           fit:str='none') -> tuple:
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
        name: str
            window name ('cosine_window' or 'kaiser_window')
        order: float
            window order
        error: bool
            flag to estimate standard errors using error propagation, if True, noise is estimated for given length
        limit: int
            number of columns to use for noise estimation
        cpu: bool
            flag to use CPU for SVD computation for noise estimation
        shift: bool
            flag to use shifted samples, generate shifted samples and perform computation for each sample
        count: int
            maximum number of samples to use
        step: int
            shift step
        fit: str
            fit method for sampled case
            'none'     - mean & std over samples
            'noise'    - estimate noise for each sample and perform WLS fit using estimated noise as weights
            'average'  - estimate error for each sample and average result
            'fit'      - estimate error for each sample and perform WLS fit

            Note, 'none' and 'average' seem to give more realistic error estimates, while other options underestimate errors
            If both data and work are filtered errors are underestimated

        Returns
        -------
        estimated amplitude (and error) for each signal

        """
        window = self.data.window.__class__(length, name, order, dtype=self.data.dtype, device=self.data.device)

        if shift == False and error == False:
            out, std  = self.harmonic_sum(frequency, window.window, self.data.work)
            _, _, out, _ = out.T
            return (out, None)

        if shift == False and error == True:
            _, sigma = Filter.svd_optimal(Filter.make_matrix(self.data.data)[:, :, :limit], cpu=cpu)
            out, std = self.harmonic_sum(frequency, window.window, self.data.work,
                                         error=True, sigma=sigma, sigma_frequency=sigma_frequency)
            _, _, out, _ = out.T
            _, _, std, _ = std.T
            return (out, std)

        size, total = self.data.data.shape
        work = torch.stack([self.data.make_matrix(length, step, self.data.work[i])[:count, :] for i in range(size)])
        _, count, _ = work.shape
        work = work.reshape(-1, length)
        work = Data.from_data(window, work)

        out, std = self.harmonic_sum(frequency, window.window, work.work)
        _, _, out, _ = out.T
        out = out.reshape(size, count)

        if fit == 'none':
            return (out.mean(1), out.std(1))

        data = torch.stack([self.data.make_matrix(length, step, self.data.data[i])[:count, :] for i in range(size)])
        _, count, _ = data.shape
        data = data.reshape(-1, length)
        data = Data.from_data(window, data)
        _, sigma = Filter.svd_optimal(Filter.make_matrix(data.data)[:, :, :limit], cpu=cpu)
        sigma = sigma.reshape(size, count)

        if fit == 'noise':
            from statsmodels.api import WLS
            x = numpy.ones((count, 1))
            result = []
            for y, w in zip(out, sigma):
                y = y.cpu().numpy()
                w = (1/w**2).cpu().numpy()
                fit = WLS(y, x, w).fit()
                result.append([fit.params.item(), fit.bse.item()])
            result = tuple(torch.tensor(result, dtype=self.data.dtype, device=self.data.device).T)
            return result

        out, std = self.harmonic_sum(frequency, window.window, work.data.reshape(size*count, -1),
                                     error=True, sigma=sigma.flatten(), sigma_frequency=sigma_frequency)
        _, _, out, _ = out.T
        _, _, std, _ = std.T
        out = out.reshape(size, count)
        std = std.reshape(size, count)

        if fit == 'average':
            return (out.mean(1), std.mean(1))

        if fit == 'fit':
            from statsmodels.api import WLS
            x = numpy.ones((count, 1))
            result = []
            for y, w in zip(out, std):
                y = y.cpu().numpy()
                w = (1/w**2).cpu().numpy()
                fit = WLS(y, x, w).fit()
                result.append([fit.params.item(), fit.bse.item()])
            result = tuple(torch.tensor(result, dtype=self.data.dtype, device=self.data.device).T)
            return result


    def harmonic_phase(self, frequency:float, *,
                           length:int=64, name:str='cosine_window', order:float=1.0,
                           error:bool=False, limit:int=16, cpu:bool=True, sigma_frequency:float=None,
                           shift:bool=False, count:int=64, step:int=1,
                           fit:str='none') -> tuple:
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
        name: str
            window name ('cosine_window' or 'kaiser_window')
        order: float
            window order
        error: bool
            flag to estimate standard errors using error propagation, if True, noise is estimated for given length
        limit: int
            number of columns to use for noise estimation
        cpu: bool
            flag to use CPU for SVD computation for noise estimation
        shift: bool
            flag to use shifted samples, generate shifted samples and perform computation for each sample
        count: int
            maximum number of samples to use
        step: int
            shift step
        fit: str
            fit method for sampled case
            'none'     - mean & std over samples
            'noise'    - estimate noise for each sample and perform WLS fit using estimated noise as weights
            'average'  - estimate error for each sample and average result
            'fit'      - estimate error for each sample and perform WLS fit

            Note, 'none' and 'average' seem to give more realistic error estimates, while other options underestimate errors
            If both data and work are filtered errors are underestimated

        Returns
        -------
        estimated phase (and error) for each signal

        """
        window = self.data.window.__class__(length, name, order, dtype=self.data.dtype, device=self.data.device)

        if shift == False and error == False:
            out, std  = self.harmonic_sum(frequency, window.window, self.data.work)
            _, _, _, out = out.T
            return (out, None)

        if shift == False and error == True:
            _, sigma = Filter.svd_optimal(Filter.make_matrix(self.data.data)[:, :, :limit], cpu=cpu)
            out, std = self.harmonic_sum(frequency, window.window, self.data.work,
                                         error=True, sigma=sigma, sigma_frequency=sigma_frequency)
            _, _, _, out = out.T
            _, _, _, std = std.T
            return (out, std)

        size, total = self.data.data.shape
        work = torch.stack([self.data.make_matrix(length, step, self.data.work[i])[:count, :] for i in range(size)])
        _, count, _ = work.shape
        work = work.reshape(-1, length)
        work = Data.from_data(window, work)

        out, std = self.harmonic_sum(frequency, window.window, work.work)
        _, _, _, out = out.T
        out = out.reshape(size, count)

        add = torch.linspace(0, (count - 1)*step*2.0*numpy.pi*frequency, count, dtype=self.data.dtype, device=self.data.device)
        out = Frequency.mod(out - add, 2.0*numpy.pi, -numpy.pi)

        if fit == 'none':
            return (out.mean(1), out.std(1))

        data = torch.stack([self.data.make_matrix(length, step, self.data.data[i])[:count, :] for i in range(size)])
        _, count, _ = data.shape
        data = data.reshape(-1, length)
        data = Data.from_data(window, data)
        _, sigma = Filter.svd_optimal(Filter.make_matrix(data.data)[:, :, :limit], cpu=cpu)
        sigma = sigma.reshape(size, count)

        if fit == 'noise':
            from statsmodels.api import WLS
            x = numpy.ones((count, 1))
            result = []
            for y, w in zip(out, sigma):
                y = y.cpu().numpy()
                w = (1/w**2).cpu().numpy()
                fit = WLS(y, x, w).fit()
                result.append([fit.params.item(), fit.bse.item()])
            result = tuple(torch.tensor(result, dtype=self.data.dtype, device=self.data.device).T)
            return result

        out, std = self.harmonic_sum(frequency, window.window, work.data.reshape(size*count, -1),
                                     error=True, sigma=sigma.flatten(), sigma_frequency=sigma_frequency)
        _, _, _, out = out.T
        _, _, _, std = std.T
        out = out.reshape(size, count)
        std = std.reshape(size, count)
        out = Frequency.mod(out - add, 2.0*numpy.pi, -numpy.pi)

        if fit == 'average':
            return (out.mean(1), std.mean(1))

        if fit == 'fit':
            from statsmodels.api import WLS
            x = numpy.ones((count, 1))
            result = []
            for y, w in zip(out, std):
                y = y.cpu().numpy()
                w = (1/w**2).cpu().numpy()
                fit = WLS(y, x, w).fit()
                result.append([fit.params.item(), fit.bse.item()])
            result = tuple(torch.tensor(result, dtype=self.data.dtype, device=self.data.device).T)
            return result


    @classmethod
    def advance(cls, phase_x:torch.Tensor, phase_y:torch.Tensor, *,
                error_x:torch.Tensor=None, error_y:torch.Tensor=None) -> tuple:
        """
        Compute phase advance between two locations (pairwise).

        Parameters
        ----------
        phase_x: torch.Tensor
            phase at the first location
        phase_y: torch.Tensor
            phase at the second location
        error_x: torch.Tensor
            phase error at the first location
        error_y: torch.Tensor
            phase error at the second location

        Returns
        -------
        advance and standard errors for each pair (tuple)

        """
        phase = Frequency.mod(phase_y - phase_x, 2.0*numpy.pi, -numpy.pi)
        if error_x == None or error_y == None:
            return (phase, None)
        return (phase, torch.sqrt(error_x**2 + error_y**2))


    @classmethod
    def advance_adjacent(cls, frequency:float, phase:torch.Tensor, *,
                         sigma_phase:torch.Tensor=None, sigma_frequency:torch.Tensor=None) -> tuple:
        """
        Compute phase advance between adjacent locations.

        Parameters
        ----------
        frequency: float
            frequency
        phase: torch.Tensor
            phase at all locations (ordered from first to last)
        sigma_phase: torch.Tensor
            phase error at all locations (ordered from first to last)
        sigma_frequency: torch.Tensor
            frequency error

        Returns
        -------
        advance between adjacent locations and standard errors for each pair (tuple)

        """
        phase_x = phase
        phase_y = torch.cat([phase, phase[:1] + 2.0*numpy.pi*frequency])[1:]

        if sigma_phase != None:
            error_x = sigma_phase
            error_y = torch.cat([sigma_phase, sigma_phase[:1]])[1:]
            if sigma_frequency != None:
                error_y[-1] = torch.sqrt(error_y[-1]**2 + (2.0*numpy.pi)**2*sigma_frequency**2)
        else:
            error_x, error_y = None, None

        return cls.advance(phase_x, phase_y, error_x=error_x, error_y=error_y)


    @classmethod
    def advance_tune(cls, frequency:float, phase:torch.Tensor, *,
                         sigma_phase:torch.Tensor=None, sigma_frequency:torch.Tensor=None) -> tuple:
        """
        Compute tune.

        Parameters
        ----------
        frequency: float
            frequency
        phase: torch.Tensor
            phase at all locations (ordered from first to last)
        sigma_phase: torch.Tensor
            phase error at all locations (ordered from first to last)
        sigma_frequency: torch.Tensor
            frequency error

        Returns
        -------
        tune and tune error (tuple)

        """
        phase, error = cls.advance_adjacent(frequency, phase, sigma_phase=sigma_phase, sigma_frequency=sigma_frequency)
        tune = (phase.sum()/(2.0*numpy.pi)).cpu().item()
        error = ((error**2).sum().sqrt()/(2.0*numpy.pi)).cpu().item()
        return (tune, error)