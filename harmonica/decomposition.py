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
    def advance(cls, i:int, j:int, total:float, table:torch.Tensor) -> torch.Tensor:
        """
        Compute phase advance mod 2*pi between i and j locations.

        Note, phase advance is computed from i to j, if i > j, phase advance is negative

        Parameters
        ----------
        i: int
            probed location
        j: int
            second location
        total: float
            total phase advane for one turn (fractional part)
        table: torch.Tensor
            model phase advance data or measured phase data

        Returns
        -------
        phase advance mod 2*pi and optional error (torch.Tensor)

        """
        size, *_ = table.shape
        if i < j:
            i_index, j_index = int(Frequency.mod(i, size)), int(Frequency.mod(j, size))
            i_count, j_count = (i - i_index)//size, (j - j_index)//size
            i_phase, j_phase = table[i_index] + i_count*total, table[j_index] + j_count*total
            return Frequency.mod(j_phase - i_phase, 2.0*numpy.pi)
        return -cls.advance(j, i, total, table)


    @staticmethod
    def advance_error(i:int, j:int, error:torch.Tensor) -> tuple:
        """
        Compute phase advance error between i and j locations.

        Note, frequency error is ignored

        Parameters
        ----------
        i: int
            probed location
        j: int
            second location
        error: torch.Tensor
            error data

        Returns
        -------
        phase advance error (torch.Tensor)

        """
        size, *_ = error.shape
        i_index, j_index = int(Frequency.mod(i, size)), int(Frequency.mod(j, size))
        i_error, j_error = error[i_index], error[j_index]
        return torch.sqrt(i_error**2 + j_error**2)


    @classmethod
    def advance_adjacent(cls, total:float, table:torch.Tensor, error:torch.Tensor=None) -> tuple:
        """
        Compute phase advance mod 2*pi between adjacent locations.

        Parameters
        ----------
        total: float
            total phase advane for one turn (fractional part)
        table: torch.Tensor
            model phase advance data or measured phase data
        error: torch.Tensor
            error data

        Returns
        -------
        adjacent phase advance mod 2*pi and optional error (torch.Tensor, torch.Tensor)

        """
        size, *_ = table.shape
        data = torch.stack([cls.advance(i, i + 1, total, table) for i in range(size)])
        if error == None:
            return (data, None)
        return (data, torch.stack([cls.advance_error(i, i + 1, error) for i in range(size)]))

    @classmethod
    def advance_check(cls, q:float, Q:float, phase:torch.Tensor, PHASE:torch.Tensor, *,
                    method:str='quantile', factor:float=10.0, epsilon:float=0.5,
                    remove:bool=True, **kwargs) -> dict:
        """
        Perform synchronization check based on adjacent locations advance diffrence with model values.

        Only one turn error is checked
        Empty dict is returned if all locations pass

        Parameters
        ----------
        q: float
            measured tune
        Q: float
            model tune
        phase: torch.Tensor
            measured phase data
        PHASE: torch.Tensor
            model phase data
        method: str
            detection method ('quantile' or 'dbscan')
        factor: float
            factor for quantile method
        epsilon: float
            epsilon for dbscan method
        remove: bool
            flag to remove endpoints
        **kwargs:
            passed to DBSCAN

        Returns
        -------
        {marked: (shift, phase)}

        """

        advance_phase, _ = Decomposition.advance_adjacent(2.0*numpy.pi*q, phase)
        advance_model, _ = Decomposition.advance_adjacent(2.0*numpy.pi*Q, PHASE)
        advance_error = (advance_phase - advance_model)/advance_model

        if method == 'dbscan':
            from sklearn.cluster import DBSCAN
            from collections import Counter
            advance_error = advance_error.cpu().numpy()
            group = DBSCAN(eps=epsilon, **kwargs).fit(advance_error.reshape(-1, 1))
            label, *_ = Counter(group.labels_)
            pairs, *_ = numpy.in1d(advance_error, advance_error[group.labels_ != label]).nonzero()
            pairs = [[i, i + 1] for i in tuple(pairs)]

        if method == 'quantile':
            q_25 = torch.quantile(advance_error, 0.25).cpu().numpy()
            q_75 = torch.quantile(advance_error, 0.75).cpu().numpy()
            q_l = q_25 - factor*(q_75 - q_25)
            q_u = q_75 + factor*(q_75 - q_25)
            advance_error = advance_error.cpu().numpy()
            pairs_l, *_ = numpy.where(advance_error < q_l)
            pairs_u, *_ = numpy.where(advance_error > q_u)
            pairs = (*tuple(pairs_l), *tuple(pairs_u))
            pairs = [[i, i + 1] for i in pairs]

        if pairs == []:
            return {}

        table = []
        chain = []
        for i in numpy.unique(numpy.array(pairs).flatten()):
            if chain == []:
                chain.append(i)
                value = i
                continue
            if i == value + 1:
                chain.append(i)
                value = i
                continue
            table.append(chain)
            chain = []
            chain.append(i)
            value = i
        else:
            table.append(chain)

        if remove:
            table = [*map(lambda chain: chain[1:-1], table)]

        marked = [j for i in table for j in i]
        passed = [i for i in range(len(phase)) if i not in marked]

        result = {}

        for index in marked:

            local_model = torch.stack([Decomposition.advance(index, other, 2.0*numpy.pi*Q, PHASE) for other in passed])

            local_phase = torch.clone(phase)
            phase_x = Frequency.mod(local_phase[index] + 1.0*2.0*numpy.pi*q, 2*numpy.pi, -numpy.pi).cpu().item()
            local_phase[index] = phase_x
            local_phase = torch.stack([Decomposition.advance(index, other, 2.0*numpy.pi*q, local_phase) for other in passed])
            error_x = torch.sum((local_model - local_phase)**2)

            local_phase = torch.clone(phase)
            phase_y = Frequency.mod(local_phase[index] + 0.0*2.0*numpy.pi*q, 2*numpy.pi, -numpy.pi).cpu().item()
            local_phase[index] = phase_y
            local_phase = torch.stack([Decomposition.advance(index, other, 2.0*numpy.pi*q, local_phase) for other in passed])
            error_y = torch.sum((local_model - local_phase)**2)

            local_phase = torch.clone(phase)
            phase_z = Frequency.mod(local_phase[index] - 1.0*2.0*numpy.pi*q, 2*numpy.pi, -numpy.pi).cpu().item()
            local_phase[index] = phase_z
            local_phase = torch.stack([Decomposition.advance(index, other, 2.0*numpy.pi*q, local_phase) for other in passed])
            error_z = torch.sum((local_model - local_phase)**2)

            error = torch.stack([error_x, error_y, error_z]).argmin().item()
            result[index] = (error - 1, [phase_x, phase_y, phase_z][error])

        return result