"""
Twiss module.

"""

import numpy
import torch

from collections import Counter
from joblib import Parallel, delayed
from statsmodels.api import OLS, WLS
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from harmonica.util import mod, generate_pairs
from harmonica.model import Model
from harmonica.table import Table

class Twiss():

    job = 8

    def __init__(self, model:'Model', table:'Table') -> None:
        """
        Twiss instance initialization.

        Parameters
        ----------
        model: 'Model'
            Model instance
        table: 'Table'
            Table instance

        Returns
        -------
        None

        """
        self.model, self.table = model, table

        self.dtype, self.device = self.model.dtype, self.model.device

        self.decomposition = self.model.decomposition

        if self.model.monitor_count != self.table.size:
            raise Exception(f'error: expected {self.model.monitor_count} monitors in Model, got {self.table.size} in Table')

        if self.model.monitor_name != self.table.name:
            raise Exception(f'error: expected monitor names to match')

        self.flag = [flag if kind == self.model._monitor else 0 for flag, kind in zip(self.model.flag, self.model.kind)]

        self.fx = torch.zeros(self.model.size, dtype=self.dtype, device=self.device)
        self.fy = torch.zeros(self.model.size, dtype=self.dtype, device=self.device)

        self.fx[self.model.monitor_index] = self.table.fx
        self.fy[self.model.monitor_index] = self.table.fy

        self.sigma_fx, self.sigma_fy = None, None

        if self.table.sigma_fx != None:
            self.sigma_fx = torch.zeros(self.model.size, dtype=self.dtype, device=self.device)
            self.sigma_fx[self.model.monitor_index] = self.table.sigma_fx

        if self.table.sigma_fy != None:
            self.sigma_fy = torch.zeros(self.model.size, dtype=self.dtype, device=self.device)
            self.sigma_fy[self.model.monitor_index] = self.table.sigma_fy

        self.fx_correct, self.sigma_fx_correct = torch.clone(self.fx), torch.clone(self.sigma_fx)
        self.fy_correct, self.sigma_fy_correct = torch.clone(self.fy), torch.clone(self.sigma_fy)

        self.monitor_phase_x, self.monitor_sigma_x = torch.zeros_like(self.table.fx), torch.zeros_like(self.table.fx)
        self.monitor_phase_y, self.monitor_sigma_y = torch.zeros_like(self.table.fy), torch.zeros_like(self.table.fy)

        self.virtual_x, self.correct_x = {}, {}
        self.virtual_y, self.correct_y = {}, {}

        self.action = {}
        self.twiss_from_amplitude = {}
        self.twiss_from_phase = {}


    def count(self, probe:int, other:int) -> int:
        """
        Count number of locations between probed and other.

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


    def monitor_count(self, probe:int, other:int) -> int:
        """
        Count number of monitor locations between probed and other.

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
        index = range(probe, other + 1) if probe < other else range(other, probe + 1)
        index = mod(torch.tensor(index, dtype=torch.int64), self.model.size).to(torch.int64)
        count = torch.tensor(self.flag, dtype=torch.int64)[index]
        return count.sum().item()


    def virtual_count(self, probe:int, other:int) -> int:
        """
        Count number of virtual locations between probed and other.

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
        index = range(probe, other + 1) if probe < other else range(other, probe + 1)
        index = mod(torch.tensor(index, dtype=torch.int64), self.model.size).to(torch.int64)
        count = torch.tensor(self.flag, dtype=torch.int64)[index] - 1
        return count.abs().sum().item()


    def get_action(self, *, remove:bool=True, factor:float=3.0, fitted:bool=True) -> None:
        """
        Estimate actions with optional data cleaning and fitting.

        Note, data cleaning is performed with DBSCAN
        DBSCAN epsilon parameter is set as a multiple of individual actions std
        Fit is performed using WLS using (cleaned) individual actions

        Parameters
        ----------
        remove: bool
            flag to clean data (computed individual actions) using DBSCAN
        factor: float
            DBSCAN epsilon multiplication factor
        fitted: bool
            flag to fit action

        Returns
        -------
        None, update self.action dictionary

        """
        self.action = {}

        self.action['remove'] = remove
        self.action['factor'] = factor
        self.action['fitted'] = fitted

        jx = self.table.ax**2/(2.0*self.model.bx[self.model.monitor_index])
        jy = self.table.ay**2/(2.0*self.model.by[self.model.monitor_index])

        normal_x, normal_y = list(range(len(jx))), list(range(len(jy)))

        self.action['remove_x'], self.action['remove_y'] = [], []
        self.action['normal_x'], self.action['normal_y'] = normal_x, normal_y

        self.action['jx'], self.action['jy'] = jx, jy

        if remove:

            epsilon_x = factor*jx.cpu().numpy().std()
            epsilon_y = factor*jy.cpu().numpy().std()

            cluster_x = DBSCAN(eps=epsilon_x).fit(jx.cpu().numpy().reshape(-1, 1))
            cluster_y = DBSCAN(eps=epsilon_y).fit(jy.cpu().numpy().reshape(-1, 1))

            primary_x, primary_y = Counter(cluster_x.labels_), Counter(cluster_y.labels_)
            primary_x, primary_y = max(primary_x, key=primary_x.get), max(primary_y, key=primary_y.get)

            remove_x = [index for index, label in zip(normal_x, cluster_x.labels_) if label != primary_x]
            remove_y = [index for index, label in zip(normal_y, cluster_y.labels_) if label != primary_y]

            normal_x = [index for index in normal_x if index not in remove_x]
            normal_y = [index for index in normal_y if index not in remove_y]

            self.action['remove_x'], self.action['remove_y'] = remove_x, remove_y
            self.action['normal_x'], self.action['normal_y'] = normal_x, normal_y

        self.action['mean_jx'], self.action['std_jx'] = jx[normal_x].mean(), jx[normal_x].std()
        self.action['mean_jy'], self.action['std_jy'] = jy[normal_y].mean(), jy[normal_y].std()

        self.action['sigma_jx'] = None
        self.action['sigma_jy'] = None

        if self.table.sigma_ax != None:
            self.action['sigma_jx'] = self.table.ax/self.model.bx[self.model.monitor_index]*self.table.sigma_ax

        if self.table.sigma_ay != None:
            self.action['sigma_jy'] = self.table.ay/self.model.by[self.model.monitor_index]*self.table.sigma_ay

        self.action['value_jx'], self.action['error_jx'] = None, None
        self.action['value_jy'], self.action['error_jy'] = None, None

        if fitted:

            x = numpy.ones(len(normal_x))
            y = jx[normal_x].cpu().numpy()
            w = x if self.action['sigma_jx'] is None else (1/self.action['sigma_jx'][normal_x]**2).cpu().numpy()
            fit = WLS(y, x, w).fit()
            self.action['value_jx'], self.action['error_jx'] = torch.tensor([fit.params.item(), fit.bse.item()], dtype=self.dtype, device=self.device)

            x = numpy.ones(len(normal_y))
            y = jy[normal_y].cpu().numpy()
            w = x if self.action['sigma_jy'] is None else (1/self.action['sigma_jy'][normal_y]**2).cpu().numpy()
            fit = WLS(y, x, w).fit()
            self.action['value_jy'], self.action['error_jy'] = torch.tensor([fit.params.item(), fit.bse.item()], dtype=self.dtype, device=self.device)


    def get_twiss_from_amplitude(self, use_fitted:bool=True) -> None:
        """
        Estimate twiss from amplitude.

        Note, action data should be precomputed

        Parameters
        ----------
        use_fitted: bool
            flag to use fitted action

        Returns
        -------
        None, update self.twiss_from_amplitude dictionary

        """
        if self.action == {}:
            raise Exception('error: action dictionary is empty')

        self.twiss_from_amplitude = {}

        ax, ay = self.table.ax, self.table.ay

        if use_fitted:
            if self.action['value_jx'] is None or self.action['value_jy'] is None:
                raise Exception('error: action fit data not found')
            jx, sigma_jx = self.action['value_jx'], self.action['error_jx']
            jy, sigma_jy = self.action['value_jy'], self.action['error_jy']
        else:
            jx, sigma_jx = self.action['mean_jx'], self.action['std_jx']
            jy, sigma_jy = self.action['mean_jy'], self.action['std_jy']

        bx, by = ax**2/(2.0*jx), ay**2/(2.0*jy)

        sigma_ax, sigma_ay = self.table.sigma_ax, self.table.sigma_ay

        sigma_bx = torch.sqrt(ax**2/jx**2*sigma_ax**2 + 0.25*ax**4/jx**4*sigma_jx**2) if sigma_ax != None else None
        sigma_by = torch.sqrt(ay**2/jy**2*sigma_ay**2 + 0.25*ay**4/jy**4*sigma_jy**2) if sigma_ay != None else None

        self.twiss_from_amplitude['bx'], self.twiss_from_amplitude['sigma_bx'] = bx, sigma_bx
        self.twiss_from_amplitude['by'], self.twiss_from_amplitude['sigma_by'] = by, sigma_by

        bx_model, by_model = self.model.bx[self.model.monitor_index], self.model.by[self.model.monitor_index]

        self.twiss_from_amplitude['error_bx'] = 100.0*(bx_model - bx)/bx_model
        self.twiss_from_amplitude['error_by'] = 100.0*(by_model - by)/by_model


    def phase_virtual(self, *, limit:int=None, fitted:bool=True, **kwargs) -> None:
        """
        Estimate x & y phase for virtual locations.

        Parameters
        ----------
        limit: int
            range limit
        fitted: bool
            flag to fit data
        **kwargs:
            passed to phase_virtual Decomposition method

        Returns
        -------
        None, update self.virtual_x and self.virtual_y dictionaries

        """
        self.virtual_x, self.virtual_y = {}, {}

        limit = limit if limit else self.model.size - 1

        index = self.model.virtual_index

        nux, sigma_nux = self.table.nux, self.table.sigma_nux
        NUX, sigma_NUX = self.model.nux, self.model.sigma_nux

        nuy, sigma_nuy = self.table.nuy, self.table.sigma_nuy
        NUY, sigma_NUY = self.model.nuy, self.model.sigma_nuy

        fx, sigma_fx = self.fx, self.sigma_fx
        FX, sigma_FX = self.model.fx, self.model.sigma_fx

        fy, sigma_fy = self.fy, self.sigma_fy
        FY, sigma_FY = self.model.fy, self.model.sigma_fy

        def auxiliary_x(probe):
            return self.decomposition.phase_virtual(probe, limit, self.flag, nux, NUX, fx, FX,
                                                    sigma_q=sigma_nux, sigma_Q=sigma_NUX,
                                                    sigma_phase=sigma_fx, sigma_PHASE=sigma_FX,
                                                    fit=fitted, **kwargs)

        def auxiliary_y(probe):
            return self.decomposition.phase_virtual(probe, limit, self.flag, nuy, NUY, fy, FY,
                                                    sigma_q=sigma_nuy, sigma_Q=sigma_NUY,
                                                    sigma_phase=sigma_fy, sigma_PHASE=sigma_FY,
                                                    fit=fitted, **kwargs)

        data_x = Parallel(n_jobs=self.job)(delayed(auxiliary_x)(probe) for probe in index)
        data_y = Parallel(n_jobs=self.job)(delayed(auxiliary_y)(probe) for probe in index)

        for count, probe in enumerate(index):
            self.virtual_x[probe], self.virtual_y[probe] = data_x[count], data_y[count]
            if fitted:
                self.fx[probe], self.sigma_fx[probe] = self.virtual_x[probe].get('model')
                self.fy[probe], self.sigma_fy[probe] = self.virtual_y[probe].get('model')
            else:
                self.fx[probe], self.sigma_fx[probe] = self.virtual_x[probe].get('phase').mean(), self.virtual_x[probe].get('phase').std()
                self.fy[probe], self.sigma_fy[probe] = self.virtual_y[probe].get('phase').mean(), self.virtual_y[probe].get('phase').std()


    def phase_correct(self, *, limit:int=None, fitted:bool=True, **kwargs) -> None:
        """
        Correct x & y phase for monitor locations.

        Parameters
        ----------
        limit: int
            range limit
        fitted: bool
            flag to fit data
        **kwargs:
            passed to phase_virtual Decomposition method

        Returns
        -------
        None, update self.correct_x and self.correct_y dictionaries

        """
        self.correct_x, self.correct_y = {}, {}

        limit = limit if limit else self.model.size - 1

        index = self.model.monitor_index

        self.fx_correct, self.sigma_fx_correct = torch.clone(self.fx), torch.clone(self.sigma_fx)
        self.fy_correct, self.sigma_fy_correct = torch.clone(self.fy), torch.clone(self.sigma_fy)

        nux, sigma_nux = self.table.nux, self.table.sigma_nux
        NUX, sigma_NUX = self.model.nux, self.model.sigma_nux

        nuy, sigma_nuy = self.table.nuy, self.table.sigma_nuy
        NUY, sigma_NUY = self.model.nuy, self.model.sigma_nuy

        fx, sigma_fx = self.fx, self.sigma_fx
        FX, sigma_FX = self.model.fx, self.model.sigma_fx

        fy, sigma_fy = self.fy, self.sigma_fy
        FY, sigma_FY = self.model.fy, self.model.sigma_fy

        def auxiliary_x(probe):
            table = self.decomposition.phase_virtual(probe, limit, self.flag, nux, NUX, fx, FX,
                                                    sigma_q=sigma_nux, sigma_Q=sigma_NUX,
                                                    sigma_phase=sigma_fx, sigma_PHASE=sigma_FX,
                                                    fit=None, **kwargs)
            if fitted:
                phase = table['phase'].cpu().numpy()
                error = table['error'].cpu().numpy()
                phase = numpy.append(self.fx[probe].item(), phase)
                error = numpy.append(self.sigma_fx[probe].item(), error)
                x = numpy.ones((len(phase), 1))
                y = phase
                s = error
                w = 1/s**2
                fit = WLS(y, x, w).fit()
                value, error = [fit.params.item(), fit.bse.item()]
            else:
                value, error = table.get('phase').mean(), table.get('phase').std()

            return [table, torch.tensor([value, error])]

        def auxiliary_y(probe):
            table = self.decomposition.phase_virtual(probe, limit, self.flag, nuy, NUY, fy, FY,
                                                    sigma_q=sigma_nuy, sigma_Q=sigma_NUY,
                                                    sigma_phase=sigma_fy, sigma_PHASE=sigma_FY,
                                                    fit=None, **kwargs)
            if fitted:
                phase = table['phase'].cpu().numpy()
                error = table['error'].cpu().numpy()
                phase = numpy.append(self.fy[probe].item(), phase)
                error = numpy.append(self.sigma_fy[probe].item(), error)
                x = numpy.ones((len(phase), 1))
                y = phase
                s = error
                w = 1/s**2
                fit = WLS(y, x, w).fit()
                value, error = [fit.params.item(), fit.bse.item()]
            else:
                value, error = table.get('phase').mean(), table.get('phase').std()

            return [table, torch.tensor([value, error])]

        data_x = Parallel(n_jobs=self.job)(delayed(auxiliary_x)(probe) for probe in index)
        data_y = Parallel(n_jobs=self.job)(delayed(auxiliary_y)(probe) for probe in index)

        for count, probe in enumerate(index):
            self.correct_x[probe], self.correct_x[probe]['model'] = data_x[count]
            self.correct_y[probe], self.correct_y[probe]['model'] = data_y[count]
            self.fx_correct[probe], self.sigma_fx_correct[probe] = self.correct_x[probe]['model']
            self.fy_correct[probe], self.sigma_fy_correct[probe] = self.correct_y[probe]['model']


    def phase_adjacent(self, use_correct:bool=False, use_correct_sigma:bool=False) -> None:
        """
        Compute adjacent phase advance between monitors.

        Note, phase advance is computed for each monitor to the next monitor

        Parameters
        ----------
        use_correct: bool
            flag to use corrected phases
        use_correct_sigma: bool
            flag to use corrected phase errors

        Returns
        -------
        None, set self.monitor_phase_x, self.monitor_phase_y, monitor_sigma_x and monitor_sigma_y

        """

        fx = self.fx_correct if use_correct else self.fx
        fy = self.fy_correct if use_correct else self.fy

        sigma_fx = self.sigma_fx_correct if use_correct_sigma else self.sigma_fx
        sigma_fy = self.sigma_fy_correct if use_correct_sigma else self.sigma_fy

        for index, probe in enumerate(self.model.monitor_index):

            other = self.model.monitor_index.index(probe) + 1
            count = 1 if other >= self.model.monitor_count else 0
            other = self.model.monitor_index[int(mod(other, self.model.monitor_count))] + count*self.model.size

            self.monitor_phase_x[index], self.monitor_sigma_x[index] = self.decomposition.phase_advance(probe, other, self.table.nux, fx, False, self.table.sigma_nux, sigma_fx)
            self.monitor_phase_y[index], self.monitor_sigma_y[index] = self.decomposition.phase_advance(probe, other, self.table.nuy, fy, False, self.table.sigma_nuy, sigma_fx)

        self.monitor_phase_x = mod(self.monitor_phase_x, 2.0*numpy.pi)
        self.monitor_phase_y = mod(self.monitor_phase_y, 2.0*numpy.pi)


    @staticmethod
    def phase_alfa(a_m:torch.Tensor,
                   f_ij:torch.Tensor, f_m_ij:torch.Tensor,
                   f_ik:torch.Tensor, f_m_ik:torch.Tensor,
                   *,
                   error:bool=True,
                   sigma_a_m:torch.Tensor=0.0,
                   sigma_f_ij:torch.Tensor=0.0, sigma_f_m_ij:torch.Tensor=0.0,
                   sigma_f_ik:torch.Tensor=0.0, sigma_f_m_ik:torch.Tensor=0.0) -> tuple:
        """
        Estimate twiss alfa at location (i) from given triplet (i, j, k) phase data.

        Note, probed location (i), other locations (j) and (k)
        Phase advance is assumed to be from (i) to other locations, should be negative if (i) is ahead of the other location (timewise)

        Parameters
        ----------
        a_m: torch.Tensor
            model value
        f_ij: torch.Tensor
            phase advance between probed and 1st location
        f_m_ij: torch.Tensor
            model phase advance between probed and 1st location
        f_ik: torch.Tensor
            phase advance between probed and 2nd location
        f_m_ik: torch.Tensor
            model phase advance between probed and 2nd location
        error: bool
            flag to compute error
        sigma_a_m: torch.Tensor
            model value error
        sigma_f_ij: torch.Tensor
            phase advance error between probed and 1st location
        sigma_f_m_ij: torch.Tensor
            model phase advance error between probed and 1st location
        sigma_f_ik: torch.Tensor
            phase advance error between probed and 2nd location
        sigma_f_m_ik: torch.Tensor
            model phase advance error between probed and 2nd location

        Returns
        -------
        (a, None) or (a, sigma_a)

        """
        a = a_m*(1.0/torch.tan(f_ij)-1.0/torch.tan(f_ik))/(1.0/torch.tan(f_m_ij)-1.0/torch.tan(f_m_ik))-1.0/torch.tan(f_ij)*1.0/torch.sin(f_m_ij - f_m_ik)*torch.cos(f_m_ik)*torch.sin(f_m_ij) + 1.0/torch.tan(f_ik)*1.0/torch.sin(f_m_ij - f_m_ik)*torch.cos(f_m_ij)*torch.sin(f_m_ik)

        if not error:
            return (a, None)

        sigma_a  = sigma_a_m**2*((1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))**2/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**2)
        sigma_a += sigma_f_ij**2*(1.0/torch.sin(f_ij))**4*(1.0/torch.tan(f_m_ik) + a_m)**2/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**2
        sigma_a += sigma_f_ik**2*(1.0/torch.sin(f_ik))**4*(1.0/torch.tan(f_m_ij) + a_m)**2/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**2
        sigma_a += sigma_f_m_ik**2*(1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))**2*(1.0/torch.sin(f_m_ij - f_m_ik))**4*torch.sin(f_m_ij)**2*(torch.cos(f_m_ij) + a_m*torch.sin(f_m_ij))**2
        sigma_a += sigma_f_m_ij**2*(1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))**2*(1.0/torch.sin(f_m_ij - f_m_ik))**4*torch.sin(f_m_ik)**2*(torch.cos(f_m_ik) + a_m*torch.sin(f_m_ik))**2

        sigma_a.sqrt_()
        return (a, sigma_a)


    @staticmethod
    def phase_beta(b_m:torch.Tensor,
                   f_ij:torch.Tensor, f_m_ij:torch.Tensor,
                   f_ik:torch.Tensor, f_m_ik:torch.Tensor,
                   *,
                   error:bool=True,
                   sigma_b_m:torch.Tensor=0.0,
                   sigma_f_ij:torch.Tensor=0.0, sigma_f_m_ij:torch.Tensor=0.0,
                   sigma_f_ik:torch.Tensor=0.0, sigma_f_m_ik:torch.Tensor=0.0) -> tuple:
        """
        Estimate twiss beta at location (i) from given triplet (i, j, k) phase data.

        Note, probed location (i), other locations (j) and (k)
        Phase advance is assumed to be from (i) to other location, should be negative if (i) is ahead of the other location (timewise)

        Parameters
        ----------
        b_m: torch.Tensor
            model value
        f_ij: torch.Tensor
            phase advance between probed and 1st location
        f_m_ij: torch.Tensor
            model phase advance between probed and 1st location
        f_ik: torch.Tensor
            phase advance between probed and 2nd location
        f_m_ik: torch.Tensor
            model phase advance between probed and 2nd location
        error: bool
            flag to compute error
        sigma_b_m: torch.Tensor
            model value error
        sigma_f_ij: torch.Tensor
            phase advance error between probed and 1st location
        sigma_f_m_ij: torch.Tensor
            model phase advance error between probed and 1st location
        sigma_f_ik: torch.Tensor
            phase advance error between probed and 2nd location
        sigma_f_m_ik: torch.Tensor
            model phase advance error between probed and 2nd location

        Returns
        -------
        (b, None) or (b, sigma_b)

        """
        b = b_m*(1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))

        if not error:
            return (b, None)

        sigma_b  = sigma_b_m**2*(1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))**2/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**2
        sigma_b += sigma_f_ij**2*b_m**2*(1.0/torch.sin(f_ij))**4/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**2
        sigma_b += sigma_f_ik**2*b_m**2*(1.0/torch.sin(f_ik))**4/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**2
        sigma_b += sigma_f_m_ij**2*b_m**2*(1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))**2*(1.0/torch.sin(f_m_ij))**4/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**4
        sigma_b += sigma_f_m_ik**2*b_m**2*(1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))**2*(1.0/torch.sin(f_m_ik))**4/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**4

        sigma_b.sqrt_()
        return (b, sigma_b)


    def get_twiss_from_phase(self, limit:int=8,
                             use_virtual:bool=False, error:bool=True,
                             use_correct:bool=False, use_correct_sigma:bool=False) -> None:
        """
        Estimate twiss from phase data.

        Note, raw data is saved, no cleaning is performed
        Values (and errors) are computed for each triplet

        Parameters
        ----------
        use_correct: bool
            flag to use corrected phases
        use_correct_sigma: bool
            flag to use corrected phase errors
        limit: int
            range limit
        use_virtual: bool
            flag to use virtual locations in combinations
        error: bool
            flag to compute errors

        Returns
        -------
        None, update self.twiss_from_phase dictionary

        """
        self.twiss_from_phase = {}

        fx = self.fx_correct if use_correct else self.fx
        fy = self.fy_correct if use_correct else self.fy

        sigma_fx = self.sigma_fx_correct if use_correct_sigma else self.sigma_fx
        sigma_fy = self.sigma_fy_correct if use_correct_sigma else self.sigma_fy

        count, table = [], []

        for distance, triplet in generate_pairs(limit, len([1, 1])):
            count.append(distance)
            table.append(triplet)

        count = numpy.array(count)
        table = numpy.array(table)

        def auxiliary(probe):

            twiss = {}

            ax_m, bx_m = self.model.ax[probe], self.model.bx[probe]
            ay_m, by_m = self.model.ay[probe], self.model.by[probe]

            other, limit = [], []

            fx_ij, fx_ik, fx_m_ij, fx_m_ik = [], [], [], []
            fy_ij, fy_ik, fy_m_ij, fy_m_ik = [], [], [], []

            for distance, triplet in zip(count, table + probe):

                ij, ik = triplet

                i, j = ij
                i, k = ik

                if self.model.is_same(i, j) or self.model.is_same(i, k) or self.model.is_same(j, k):
                    continue

                if not use_virtual:
                    if not self.model.is_monitor(j) or not self.model.is_monitor(k):
                        continue

                limit.append(distance)
                other.append([j, k])

                fx_ij.append(self.decomposition.phase_advance(*ij, self.table.nux, fx, model=False, sigma_frequency=self.table.sigma_nux, sigma_phase=sigma_fx))
                fx_ik.append(self.decomposition.phase_advance(*ik, self.table.nux, fx, model=False, sigma_frequency=self.table.sigma_nux, sigma_phase=sigma_fx))
                fy_ij.append(self.decomposition.phase_advance(*ij, self.table.nuy, fy, model=False, sigma_frequency=self.table.sigma_nuy, sigma_phase=sigma_fy))
                fy_ik.append(self.decomposition.phase_advance(*ik, self.table.nuy, fy, model=False, sigma_frequency=self.table.sigma_nuy, sigma_phase=sigma_fy))

                fx_m_ij.append(self.decomposition.phase_advance(*ij, self.model.nux, self.model.fx, model=True, sigma_frequency=self.model.sigma_nux, sigma_phase=self.model.sigma_fx))
                fx_m_ik.append(self.decomposition.phase_advance(*ik, self.model.nux, self.model.fx, model=True, sigma_frequency=self.model.sigma_nux, sigma_phase=self.model.sigma_fx))
                fy_m_ij.append(self.decomposition.phase_advance(*ij, self.model.nuy, self.model.fy, model=True, sigma_frequency=self.model.sigma_nuy, sigma_phase=self.model.sigma_fy))
                fy_m_ik.append(self.decomposition.phase_advance(*ik, self.model.nuy, self.model.fy, model=True, sigma_frequency=self.model.sigma_nuy, sigma_phase=self.model.sigma_fy))

            twiss['limit'], twiss['other'] = limit, other

            twiss['ax_m'], twiss['bx_m'] = ax_m, bx_m
            twiss['ay_m'], twiss['by_m'] = ay_m, by_m

            if len(limit) > 0:

                fx_ij, sigma_fx_ij = torch.stack(fx_ij).T
                fx_ik, sigma_fx_ik = torch.stack(fx_ik).T

                fy_ij, sigma_fy_ij = torch.stack(fy_ij).T
                fy_ik, sigma_fy_ik = torch.stack(fy_ik).T

                fx_m_ij, sigma_fx_m_ij = torch.stack(fx_m_ij).T
                fx_m_ik, sigma_fx_m_ik = torch.stack(fx_m_ik).T

                fy_m_ij, sigma_fy_m_ij = torch.stack(fy_m_ij).T
                fy_m_ik, sigma_fy_m_ik = torch.stack(fy_m_ik).T

                ax, sigma_ax = self.phase_alfa(ax_m, fx_ij, fx_m_ij, fx_ik, fx_m_ik, error=error, sigma_a_m=0.0, sigma_f_ij=sigma_fx_ij, sigma_f_ik=sigma_fx_ik, sigma_f_m_ij=sigma_fx_m_ij, sigma_f_m_ik=sigma_fx_m_ik)
                bx, sigma_bx = self.phase_beta(bx_m, fx_ij, fx_m_ij, fx_ik, fx_m_ik, error=error, sigma_b_m=0.0, sigma_f_ij=sigma_fx_ij, sigma_f_ik=sigma_fx_ik, sigma_f_m_ij=sigma_fx_m_ij, sigma_f_m_ik=sigma_fx_m_ik)

                ay, sigma_ay = self.phase_alfa(ay_m, fy_ij, fy_m_ij, fy_ik, fy_m_ik, error=error, sigma_a_m=0.0, sigma_f_ij=sigma_fy_ij, sigma_f_ik=sigma_fy_ik, sigma_f_m_ij=sigma_fy_m_ij, sigma_f_m_ik=sigma_fy_m_ik)
                by, sigma_by = self.phase_beta(by_m, fy_ij, fy_m_ij, fy_ik, fy_m_ik, error=error, sigma_b_m=0.0, sigma_f_ij=sigma_fy_ij, sigma_f_ik=sigma_fy_ik, sigma_f_m_ij=sigma_fy_m_ij, sigma_f_m_ik=sigma_fy_m_ik)

                twiss['fx_ij'], twiss['sigma_fx_ij'], twiss['fx_m_ij'], twiss['sigma_fx_m_ij'] = fx_ij, sigma_fx_ij, fx_m_ij, sigma_fx_m_ij
                twiss['fx_ik'], twiss['sigma_fx_ik'], twiss['fx_m_ik'], twiss['sigma_fx_m_ik'] = fx_ik, sigma_fx_ik, fx_m_ik, sigma_fx_m_ik

                twiss['fy_ij'], twiss['sigma_fy_ij'], twiss['fy_m_ij'], twiss['sigma_fy_m_ij'] = fy_ij, sigma_fy_ij, fy_ij, sigma_fy_m_ij
                twiss['fy_ik'], twiss['sigma_fy_ik'], twiss['fy_m_ik'], twiss['sigma_fy_m_ik'] = fy_ik, sigma_fy_ik, fy_ik, sigma_fy_m_ik

                twiss['ax'], twiss['sigma_ax'], twiss['bx'], twiss['sigma_bx'] = ax, sigma_ax, bx, sigma_bx
                twiss['ay'], twiss['sigma_ay'], twiss['by'], twiss['sigma_by'] = ay, sigma_ay, by, sigma_by

            return twiss

        data = Parallel(n_jobs=self.job)(delayed(auxiliary)(probe) for probe in range(self.model.size))

        for index, probe in enumerate(range(self.model.size)):
            self.twiss_from_phase[probe] = data[index]


    def filter_twiss(self, probe:int, plane:str='x',
                     limit:dict={'use': True, 'limit': 16, 'flag': True},
                     flag:dict={'use': True},
                     phase:dict={'use': True, 'threshold': 5.00},
                     value:dict={'use': True, 'threshold': 0.50},
                     error:dict={'use': True, 'threshold': 0.25},
                     quantile:dict={'use': True, 'factor': 5.00},
                     dbscan:dict={'use': True, 'factor': 3.00},
                     lof:dict={'use': True, 'count': 25, 'contamination': 0.05},
                     iforest:dict={'use': True, 'count': 25, 'contamination': 0.05}) -> dict:
        """
        Filter twiss data for given location, data plane and cleaning methods.

        Cleaning is performed for a single location, since depending on location, different cleaning settings apply
        Note, cleaning is nested
        Note, if some method generates empty result, it is ignored

        Parameters
        ----------
        probe: int
            probe location
        plane: str
            data plane ('x' or 'y')
        limit: dict
            clean based on number of (monitor) location in a combination
            used if 'use' is True, maximum number of locations is set by 'limit', if 'flag' is True, only monitor locations are used
        flag: dict
            clean based on location flag
            used if 'use' is True, remove combinations with zero flag locations, probed location can have zero flag
        phase: dict
            clean based on phase data
            used if 'use' is True, remove combinations with cot of phase advance above threshold value
        value: dict
            clean based on model proximity
            used if 'use' is True, remove combinations with (x - x_model)/x_model > threshold value
        error: dict
            clean based on estimated error
            used if 'use' is True, remove combinations with x/sigma_x < 1/threshold value
        quantile: dict
            clean outliers outside scaled interquantile region
            used if 'use' is True
        dbscan: dict
            clean outliers with DBSCAN
            used if 'use' is True, eplilon = factor*std
        lof: dict
            clean outliers with LOF
            used if 'use' is True, always removes some data
        iforest:
            clean outliers with Isolation Forest
            used if 'use' is True, always removes some data

        Returns
        -------
        dictionary with normal indices

        """
        table = {}

        other = torch.tensor(self.twiss_from_phase[probe]['other'], dtype=torch.int64)

        index = torch.tensor([*range(len(other))], dtype=torch.int64)

        if limit['use']:
            if limit['flag']:
                mask = [self.monitor_count(probe, i) < limit['limit'] and self.monitor_count(probe, j) < limit['limit'] for i, j in other]
            else:
                mask = [self.count(probe, i) < limit['limit'] and self.count(probe, j) < limit['limit'] for i, j in other]
            table['limit'] = index[mask]
            if len(table['limit']) != 0:
                index = table['limit']

        if flag['use']:
            mask = [all(map(self.model.is_monitor, pair)) for pair in other[index]]
            table['flag'] = index[mask]
            if len(table['flag']) != 0:
                index = table['flag']

        if phase['use']:
            f_ij, f_m_ij = self.twiss_from_phase[probe][f'f{plane}_ij'][index], self.twiss_from_phase[probe][f'f{plane}_m_ij'][index]
            f_ik, f_m_ik = self.twiss_from_phase[probe][f'f{plane}_ik'][index], self.twiss_from_phase[probe][f'f{plane}_m_ik'][index]
            cot_ij, cot_m_ij = torch.abs(1.0/torch.tan(f_ij)), torch.abs(1.0/torch.tan(f_m_ij))
            cot_ik, cot_m_ik = torch.abs(1.0/torch.tan(f_ij)), torch.abs(1.0/torch.tan(f_m_ij))
            mask  = phase['threshold'] > cot_ij
            mask *= phase['threshold'] > cot_m_ij
            mask *= phase['threshold'] > cot_ik
            mask *= phase['threshold'] > cot_m_ik
            table['phase'] = index[mask]
            if len(table['phase']) != 0:
                index = table['phase']

        if value['use']:
            a, a_m = self.twiss_from_phase[probe][f'a{plane}'][index], self.twiss_from_phase[probe][f'a{plane}_m']
            b, b_m = self.twiss_from_phase[probe][f'b{plane}'][index], self.twiss_from_phase[probe][f'b{plane}_m']
            mask  = a*a_m > 0
            mask *= value['threshold'] > torch.abs((a - a_m)/a_m)
            mask *= value['threshold'] > torch.abs((b - b_m)/b_m)
            table['value'] = index[mask]
            if len(table['value']) != 0:
                index = table['value']

        if error['use']:
            f_ij, sigma_f_ij = self.twiss_from_phase[probe][f'f{plane}_ij'][index], self.twiss_from_phase[probe][f'sigma_f{plane}_ij'][index]
            f_ik, sigma_f_ik = self.twiss_from_phase[probe][f'f{plane}_ik'][index], self.twiss_from_phase[probe][f'sigma_f{plane}_ik'][index]
            a, sigma_a = self.twiss_from_phase[probe][f'a{plane}'][index], self.twiss_from_phase[probe][f'sigma_a{plane}'][index]
            b, sigma_b = self.twiss_from_phase[probe][f'b{plane}'][index], self.twiss_from_phase[probe][f'sigma_b{plane}'][index]
            mask  = 1/error['threshold'] < torch.abs(f_ij/sigma_f_ij)
            mask *= 1/error['threshold'] < torch.abs(f_ik/sigma_f_ik)
            mask *= 1/error['threshold'] < torch.abs(a/sigma_a)
            mask *= 1/error['threshold'] < torch.abs(b/sigma_b)
            table['error'] = index[mask]
            if len(table['error']) != 0:
                index = table['error']

        if quantile['use']:
            a = self.twiss_from_phase[probe][f'a{plane}'][index]
            b = self.twiss_from_phase[probe][f'b{plane}'][index]
            a_ql, a_qu = torch.quantile(a, 0.25), torch.quantile(a, 0.75)
            a_ql, a_qu = a_ql - quantile['factor']*(a_qu - a_ql), a_qu + quantile['factor']*(a_qu - a_ql)
            b_ql, b_qu = torch.quantile(b, 0.25), torch.quantile(b, 0.75)
            b_ql, b_qu = b_ql - quantile['factor']*(b_qu - b_ql), b_qu + quantile['factor']*(b_qu - b_ql)
            mask  = (a_ql < a)*(a < a_qu)
            mask *= (b_ql < b)*(b < b_qu)
            table['quantile'] = index[mask]
            if len(table['quantile']) != 0:
                index = table['quantile']

        if dbscan['use']:
            a = self.twiss_from_phase[probe][f'a{plane}'][index].cpu().numpy()
            b = self.twiss_from_phase[probe][f'b{plane}'][index].cpu().numpy()
            a_group = DBSCAN(eps=dbscan['factor']*a.std()).fit(a.reshape(-1, 1))
            b_group = DBSCAN(eps=dbscan['factor']*b.std()).fit(b.reshape(-1, 1))
            a_label = Counter(a_group.labels_)
            b_label = Counter(b_group.labels_)
            a_label = max(a_label, key=a_label.get)
            b_label = max(b_label, key=b_label.get)
            mask  = a_group.labels_ == a_label
            mask *= b_group.labels_ == b_label
            table['dbscan'] = index[mask]
            if len(table['dbscan']) != 0:
                index = table['dbscan']

        if lof['use']:
            indicator = LocalOutlierFactor(n_neighbors=lof['count'], contamination=lof['contamination'])
            a = self.twiss_from_phase[probe][f'a{plane}'][index].cpu().numpy()
            b = self.twiss_from_phase[probe][f'b{plane}'][index].cpu().numpy()
            a, b = (a - a.mean())/a.std(), (b - b.mean())/b.std()
            c = numpy.array([a, b]).T
            mask = indicator.fit_predict(c) > 0
            table['lof'] = index[mask]
            if len(table['lof']) != 0:
                index = table['lof']

        if iforest['use']:
            indicator = IsolationForest(n_estimators=iforest['count'], contamination=iforest['contamination'])
            a = self.twiss_from_phase[probe][f'a{plane}'][index].cpu().numpy()
            b = self.twiss_from_phase[probe][f'b{plane}'][index].cpu().numpy()
            a, b = (a - a.mean())/a.std(), (b - b.mean())/b.std()
            c = numpy.array([a, b]).T
            mask = indicator.fit(c).predict(c) > 0
            table['iforest'] = index[mask]
            if len(table['iforest']) != 0:
                index = table['iforest']

        table['normal'] = index
        return table


    def process_twiss(self, probe:int, plane:str='x', *,
                      index:torch.Tensor=None, fitted:bool=True) -> dict:
        """
        Process twiss data.

        Parameters
        ----------
        probe: int
            probe location
        plane: str
            data plane ('x' or 'y')
        index: torch.Tensor
            index list to use
        fitted: bool
            flag to fit data

        Returns
        -------
        processed twiss data (dict)

        """
        table = {}

        index = index if index != None else torch.tensor([*range(len(self.twiss_from_phase[probe]['other']))], dtype=torch.int64)

        table['probe'] = probe
        table['name'] = self.model.name[probe]

        a_m, b_m = self.twiss_from_phase[probe][f'a{plane}_m'], self.twiss_from_phase[probe][f'b{plane}_m']
        table[f'a{plane}_m'], table[f'b{plane}_m'] = a_m, b_m

        a, b = self.twiss_from_phase[probe][f'a{plane}'][index], self.twiss_from_phase[probe][f'b{plane}'][index]

        table[f'mean_a{plane}'], table[f'mean_b{plane}'] = a.mean(), b.mean()
        table[f'std_a{plane}'], table[f'std_b{plane}'] = a.std(), b.std()

        if not fitted:
            return table

        sigma_a, sigma_b = self.twiss_from_phase[probe][f'sigma_a{plane}'][index], self.twiss_from_phase[probe][f'sigma_b{plane}'][index]

        x = numpy.ones((len(index), 1))
        y = a.cpu().numpy()
        s = sigma_a.cpu().numpy()
        w = 1/s**2 if sigma_a != None else x
        fit = WLS(y, x, w).fit()
        value, error = fit.params.item(), fit.bse.item()
        table[f'value_a{plane}'], table[f'error_a{plane}'] = torch.tensor([value, error], dtype=self.dtype, device=self.device)

        x = numpy.ones((len(index), 1))
        y = b.cpu().numpy()
        s = sigma_b.cpu().numpy()
        w = 1/s**2 if sigma_b != None else x
        fit = WLS(y, x, w).fit()
        value, error = fit.params.item(), fit.bse.item()
        table[f'value_b{plane}'], table[f'error_b{plane}'] = torch.tensor([value, error], dtype=self.dtype, device=self.device)

        return table