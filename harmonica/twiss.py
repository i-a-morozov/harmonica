"""
Twiss module.
Compute twiss parameters from amplitude & phase data.
Twiss filtering & processing.

"""
from __future__ import annotations

import numpy
import torch
import pandas

from typing import Callable
from scipy.optimize import leastsq, minimize
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

from .util import mod, generate_pairs, generate_other
from .statistics import mean, variance
from .statistics import weighted_mean, weighted_variance
from .statistics import median, biweight_midvariance, standardize
from .anomaly import threshold, dbscan, local_outlier_factor, isolation_forest
from .decomposition import Decomposition
from .model import Model
from .table import Table
from .parameterization import matrix_uncoupled, matrix_coupled, matrix_rotation
from .parameterization import cs_normal, lb_normal, parametric_normal
from .parameterization import wolski_to_cs, cs_to_wolski
from .parameterization import wolski_to_lb, lb_to_wolski
from .parameterization import wolski_to_normal, normal_to_wolski
from .parameterization import invariant, momenta
from .parameterization import to_symplectic, twiss_compute, twiss_propagate


class Twiss():
    """
    Returns
    ----------
    Twiss class instance.

    Parameters
    ----------
    model: Model
        Model instance
    table: Table
        Table instance
    limit: int | tuple[int, int]
        range limit to use, (min, max), 1 <= min <= max, min is excluded, for full range min == max
    flag: torch.Tensor
        external flags for each model location
    use_model: bool
        flag to use precomputed model data

    Attributes
    ----------
    model: Model
        Model instance
    table: Table
        Table instance
    size: int
        total numner of locations
    dtype: torch.dtype
        data type
    device: torch.device
        data device
    limit: tuple[int, int]
        range limit
    flag: torch.Tensor
        location flags
    use_model: bool
        flag to use precomputed model data
    count: torch.Tensor
        number of triplet combinations grouped by range
    combo: torch.Tensor
        triplet combinations for each probe location sorted by range
    index: torch.Tensor
        triplet combination indices for each probe location sorted by range
    distance: torch.Tensor
        range distance for each combination
    shape: torch.Size
        index shape
    fx, fy: torch.Tensor
        location x & y phase
    sigma_fx, sigma_fy: torch.Tensor
        location x & y phase error
    fx_corrected, fy_corrected: torch.Tensor
        location x & y corrected phase
    sigma_fx_corrected, sigma_fy_corrected: torch.Tensor
        location x & y corrected phase error
    data_corrected: dict
        corrected phase data
    data_virtual: dict
        virtual phase data
    data_action: dict
        action data
    data_amplitude: dict
        twiss from amplitude data
    data_phase: dict
        twiss from phase data
    ax, bx, ay, by: torch.Tensor
        cs twiss parameters
    sigma_ax, sigma_bx, sigma_ay, sigma_by: torch.Tensor
        cs twiss parameters errors
    normal, sigma_normal: torch.Tensor
        normalization matrices and corresponding errors
    fx_ij, sigma_fx_ij: torch.Tensor
        (use_model) model phase advance and errors
    fx_ik, sigma_fx_ik: torch.Tensor
        (use_model) model phase advance and errors
    fy_ij, sigma_fy_ij: torch.Tensor
        (use_model) model phase advance and errors
    fy_ik, sigma_fy_ik: torch.Tensor
        (use_model) model phase advance and errors
    mask: torch.Tensor
        (use_model) mask

    Methods
    ----------
    __init__(self, model:Model, table:Table, *, limit:int=4, flag:torch.Tensor=None, use_model:bool=False) -> None
        Twiss instance initialization.
    get_action(self, *, dict_threshold:dict={'use': True, 'factor': 5.0}, dict_dbscan:dict={'use': False, 'factor': 2.5}, dict_local_outlier_factor:dict={'use': False, 'contamination': 0.01}, dict_isolation_forest:dict={'use': False, 'contamination': 0.01}, bx:torch.Tensor=None, by:torch.Tensor=None, sigma_bx:torch.Tensor=None, sigma_by:torch.Tensor=None, jx:torch.Tensor=None, jy:torch.Tensor=None, sigma_jx:torch.Tensor=None, sigma_jy:torch.Tensor=None) -> None
        Estimate uncoupled actions at each monitor location from amplitude data with optional cleaning and estimate overall action center and spread.
    get_twiss_from_amplitude(self) -> None
        Estimate cs twiss from amplitude.
    phase_correct(self, *, limit:int=None, **kwargs) -> None
        Use measured phases and model phases advances to correct x & y phase at monitor locations.
    phase_virtual(self, *, limit:int=None, exclude:list=None, method:str='model', inverse:bool=True, forward:bool=True, nearest:bool=False, **kwargs) -> None
        Estimate x & y phase at virtual locations.
    phase_alfa(a_m:torch.Tensor, f_ij:torch.Tensor, f_m_ij:torch.Tensor, f_ik:torch.Tensor, f_m_ik:torch.Tensor, *, error:bool=True, model:bool=True, sigma_a_m:torch.Tensor=0.0, sigma_f_ij:torch.Tensor=0.0, sigma_f_m_ij:torch.Tensor=0.0, sigma_f_ik:torch.Tensor=0.0, sigma_f_m_ik:torch.Tensor=0.0) -> tuple:
        Estimate uncoupled twiss alfa at index (i) from given triplet (i, j, k) phase data.
    phase_beta(b_m:torch.Tensor, f_ij:torch.Tensor, f_m_ij:torch.Tensor, f_ik:torch.Tensor, f_m_ik:torch.Tensor, *, error:bool=True, model:bool=True, sigma_b_m:torch.Tensor=0.0, sigma_f_ij:torch.Tensor=0.0, sigma_f_m_ij:torch.Tensor=0.0, sigma_f_ik:torch.Tensor=0.0, sigma_f_m_ik:torch.Tensor=0.0) -> tuple
        Estimate uncoupled twiss beta at index (i) from given triplet (i, j, k) phase data.
    get_twiss_from_phase(self, *, error:bool=True, model:bool=False, use_correct:bool=False, use_correct_sigma:bool=False, use_model:bool=False) -> None
        Estimate cs twiss from phase data.
    filter_twiss(self, plane:str = 'x', *, phase:dict={'use': True, 'threshold': 10.00}, model:dict={'use': True, 'threshold': 00.50}, value:dict={'use': True, 'threshold': 00.25}, sigma:dict={'use': True, 'threshold': 00.25}, limit:dict={'use': True, 'threshold': 05.00}, error:dict={'use': True, 'threshold': 05.00}) -> dict
        Filter twiss for given data plane and cleaning options.
    mask_range(self, limit:tuple) -> torch.Tensor
        Generate weight mask based on given range limit.
    mask_location(self, table:list) -> torch.Tensor
        Generate weight mask based on given range limit.
    mask_distance(self, function) -> torch.Tensor
        Generate weight mask based on given range limit.
    process_twiss(self, plane:str='x', *, weight:bool=True, mask:torch.Tensor=None) -> dict
        Process uncoupled twiss data.
    bootstrap_twiss(self, plane:str='x', *, weight:bool=True, mask:torch.Tensor=None, fraction:float=0.75, count:int=512) -> dict
        Bootstrap uncoupled twiss data.
    matrix(self, probe:torch.Tensor, other:torch.Tensor) -> torch.Tensor
        Generate transport matrices between given probe and other locations using measured twiss.
    matrix_virtual(self, probe:int, other:int, *, close:str='nearest') -> torch.Tensor
        Compute virtual transport matrix between probe and other.
    phase_advance(self, probe:torch.Tensor, other:torch.Tensor, **kwargs) -> torch.Tensor
        Compute x & y phase advance between probe and other.
    get_momenta(self, start:int, count:int, probe:int, other:int, matrix:Callable[[int, int], torch.Tensor]) -> torch.Tensor
        Compute x & y momenta at the probe monitor location using single other monitor location.
    get_momenta_range(self, start:int, count:int, probe:int, limit:int, matrix:Callable[[int, int], torch.Tensor]) -> torch.Tensor
        Compute x & y momenta at the probe monitor location using range of monitor locations around probed monitor location (average momenta).
    get_momenta_lstsq(self, start:int, count:int, probe:int, limit:int, matrix:Callable[[int, int], torch.Tensor], *, phony:bool=False, forward:bool=True, inverse:bool=True) -> torch.Tensor
        Compute x & y coordinates and momenta at the probe monitor location using range of monitor locations around probed monitor location (lstsq fit).
    invariant_objective(beta:torch.Tensor, X:torch.Tensor, normalization:Callable[[torch.Tensor], torch.Tensor], product:bool) -> torch.Tensor
        Evaluate invariant objective.
    invariant_objective_fixed(beta:torch.Tensor, X:torch.Tensor, normalization:Callable[[torch.Tensor], torch.Tensor], product:bool) -> torch.Tensor
        Evaluate invariant objective (fixed invariants).
    fit_objective(self, length:int, twiss:torch.Tensor, qx:torch.Tensor, px:torch.Tensor, qy:torch.Tensor, py:torch.Tensor, normalization:Callable[[torch.Tensor], torch.Tensor], *, product:bool=True, jacobian:bool=False, ix:float=None, iy:float=None, count:int=512, fraction:float=0.75, sigma:float=0.0, n_jobs:int=6, **kwargs) -> tuple
        Fit invariant objective.
    fit_objective_fixed(self, length:int, twiss:torch.Tensor, ix:torch.Tensor, iy:torch.Tensor, qx:torch.Tensor, px:torch.Tensor, qy:torch.Tensor, py:torch.Tensor, normalization:Callable[[torch.Tensor], torch.Tensor], *, product:bool=True, jacobian:bool=False, count:int=512, fraction:float=0.75, sigma:float=0.0, n_jobs:int=6, **kwargs) -> tuple
        Fit invariant objective (fixed invariants).
    get_twiss_from_data(self, start:int, length:int, normalization:Callable[[torch.Tensor], torch.Tensor], matrix:Callable[[int, int], torch.Tensor], *, twiss:torch.Tensor=None, method:str='pair', limit:int=1, index:list[int]=None, phony:bool=False, inverse:bool=True, forward:bool=True, product:bool=True, jacobian:bool=False, count:int=256, fraction:float=0.75, ix:float=None, iy:float=None, sigma:float=0.0, n_jobs:int=6, verbose:bool=False, **kwargs) -> torch.Tensor
        Estimate twiss from signals (fit linear invariants).
    get_invariant(self, ix:torch.Tensor, iy:torch.Tensor, sx:torch.Tensor=None, sy:torch.Tensor=None, *, cut:float=5.0, use_error:bool=True, center_estimator:Callable[[torch.Tensor], torch.Tensor]=median, spread_estimator:Callable[[torch.Tensor], torch.Tensor]=biweight_midvariance) -> dict
        Compute invariants from get_twiss_from_data output.
    ratio_objective(beta:torch.Tensor, X:torch.Tensor, window:torch.Tensor, nux:torch.Tensor, nuy:torch.Tensor, normalization:Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor
        Evaluate ratio objective.
    get_twiss_from_ratio(self, start:int, length:int, window:torch.Tensor, nux:torch.Tensor, nuy:torch.Tensor, normalization:Callable[[torch.Tensor], torch.Tensor], matrix:Callable[[int, int], torch.Tensor], *, step:int=1, method:str='pair', limit:int=1, index:list[int]=None, phony:bool=False, inverse:bool=True, forward:bool=True, twiss:torch.Tensor=None, n_jobs:int=6, verbose:bool=False, **kwargs) -> torch.Tensor
        Estimate twiss from ratio.
    get_twiss_from_matrix(self, start:int, length:int, matrix:Callable[[int, int], torch.Tensor], *, power:int=1, method:str='pair', limit:int=1, index:list[int]=None, phony:bool=False, inverse:bool=True, forward:bool=True, count:int=256, fraction:float=0.75, verbose:bool=False) -> torch.tensor
        Estimate twiss from n-turn matrix.
    get_twiss_virtual_uncoupled(self, probe:int, *, limit:int=1, inverse:bool=True, forward:bool=True, use_phase:bool=False, bootstrap:bool=True, count:int=256, ax:torch.Tensor=None, bx:torch.Tensor=None, ay:torch.Tensor=None, by:torch.Tensor=None, sigma_ax:torch.Tensor=None, sigma_bx:torch.Tensor=None, sigma_ay:torch.Tensor=None, sigma_by:torch.Tensor=None) -> tuple
        Estimate CS twiss at (virtual) location.
    get_twiss_virtual_coupled(self, probe:int, *, limit:int=1, inverse:bool=True, forward:bool=True, use_phase:bool=False, bootstrap:bool=True, count:int=256, n11:torch.Tensor=None, n33:torch.Tensor=None, n21:torch.Tensor=None, n43:torch.Tensor=None, n13:torch.Tensor=None, n31:torch.Tensor=None, n14:torch.Tensor=None, n41:torch.Tensor=None, sigma_n11:torch.Tensor=None, sigma_n33:torch.Tensor=None, sigma_n21:torch.Tensor=None, sigma_n43:torch.Tensor=None, sigma_n13:torch.Tensor=None, sigma_n31:torch.Tensor=None, sigma_n14:torch.Tensor=None, sigma_n41:torch.Tensor=None) -> tuple
        Estimate free normalization matrix elements at (virtual) location.
    phase_objective(beta:torch.Tensor, mux:torch.Tensor, muy:torch.Tensor, matrix:torch.Tensor) -> torch.Tensor
        Evaluate phase objective.
    get_twiss_from_phase_fit(self, *, limit:int=10, count:int=64, model:boot=True, verbose:bool=False, **kwargs) -> torch.Tensor
        Estimate twiss from phase fit.
    process(self, value:torch.Tensor, error:torch.Tensor, *, mask:torch.Tensor=None, cut:float=5.0, use_error:bool=True, center_estimator:Callable[[torch.Tensor], torch.Tensor]=median, spread_estimator:Callable[[torch.Tensor], torch.Tensor]=biweight_midvariance) -> tuple
        Process data for single parameter for all locations with optional mask.
    save_model(self, file:str, **kwargs) -> None:
        Save measured twiss as model.
    get_ax(self, index:int) -> torch.Tensor
        Get ax value and error at given index.
    get_bx(self, index:int) -> torch.Tensor
        Get bx value and error at given index.
    get_fx(self, index:int) -> torch.Tensor
        Get fx value and error at given index.
    get_ay(self, index:int) -> torch.Tensor
        Get ay value and error at given index.
    get_by(self, index:int) -> torch.Tensor
        Get by value and error at given index.
    get_fy(self, index:int) -> torch.Tensor
        Get fy value and error at given index.
    get_twiss(self, index:int) -> dict
        Return twiss data at given index.
    get_table(self) -> pandas.DataFrame
        Return twiss data at all locations as dataframe.
    __repr__(self) -> str
        String representation.
    __len__(self) -> int:
        Number of locations.
    __call__(self, limit:int=None) -> pandas.DataFrame
        Perform twiss loop with default parameters.

    """
    def __init__(self,
                 model:Model,
                 table:Table,
                 *,
                 limit:int=4,
                 flag:torch.Tensor=None,
                 use_model:bool=False) -> None:
        """
        Twiss instance initialization.

        Parameters
        ----------
        model: Model
            Model instance
        table: Table
            Table instance
        limit: int
            range limit to use, int or tuple (min, max), 1 <= min <= max, min is excluded, for full range set min == max
        flag: torch.Tensor
            external flags for each locations
        use_model: bool
            flag to use precomputed model data

        Returns
        -------
        None

        """
        self.model, self.table = model, table

        if self.model.monitor_count != self.table.size:
            raise ValueError(f'TWISS: expected {self.model.monitor_count} monitors in Table, got {self.table.size}')

        self.size = self.model.size
        self.dtype, self.device = self.model.dtype, self.model.device

        self.limit = limit if isinstance(limit, tuple) else (limit, limit)

        if flag is None:
            self.flag = [flag if kind == self.model._monitor else 0 for flag, kind in zip(self.model.flag, self.model.kind)]
            self.flag = torch.tensor(self.flag, dtype=torch.int64, device=self.device)
        else:
            if len(flag) != self.size:
                raise ValueError(f'TWISS: external flag length {len(flag)}, expected length {self.size}')
            self.flag = flag.to(torch.int64).to(self.device)

        self.use_model = use_model

        if self.use_model:
            if self.model.limit is None:
                raise ValueError(f'TWISS: model limit is None')
            if self.model.limit < max(self.limit):
                raise ValueError(f'TWISS: requested limit={self.limit} should be less or equal to avaliable model limit={self.model.limit}')

        if self.use_model:
            self.count = self.model.count
            self.combo = self.model.combo
            self.index = self.model.index
        else:
            self.count = torch.tensor([limit*(2*limit - 1) for limit in range(1, max(self.limit) + 1)], dtype=torch.int64, device=self.device)
            self.combo = [generate_other(probe, max(self.limit), self.flag) for probe in range(self.size)]
            self.combo = torch.stack([generate_pairs(max(self.limit), 1 + 1, probe=probe, table=table, dtype=torch.int64, device=self.device) for probe, table in enumerate(self.combo)])
            self.index = mod(self.combo, self.size).to(torch.int64)

        self.distance = torch.ones(max(self.limit)*(2*max(self.limit) - 1), dtype=self.dtype, device=self.device)
        for index in self.count:
            self.distance[index:] += 1.0

        limit_min, limit_max = self.limit

        if limit_min == limit_max:
            self.count = self.count[:limit_max]
            *_, count_max = self.count
            self.combo = self.combo[:, :count_max]
            self.index = self.index[:, :count_max]
            self.distance = self.distance[:count_max]

        if limit_min < limit_max:
            self.count = self.count[limit_min - 1:limit_max]
            count_min, *_, count_max = self.count
            self.combo = self.combo[:, count_min:count_max]
            self.index = self.index[:, count_min:count_max]
            self.distance = self.distance[count_min:count_max]

        if limit_min > limit_max:
            raise ValueError(f'TWISS: invalid limit={self.limit}')

        self.shape = self.index.shape

        self.fx = torch.zeros_like(self.model.fx)
        self.fy = torch.zeros_like(self.model.fy)

        self.fx[self.model.monitor_index] = self.table.fx
        self.fy[self.model.monitor_index] = self.table.fy

        self.sigma_fx = torch.zeros_like(self.model.sigma_fx)
        self.sigma_fy = torch.zeros_like(self.model.sigma_fy)

        self.sigma_fx[self.model.monitor_index] = self.table.sigma_fx
        self.sigma_fy[self.model.monitor_index] = self.table.sigma_fy

        self.fx_corrected, self.sigma_fx_corrected = torch.clone(self.fx), torch.clone(self.sigma_fx)
        self.fy_corrected, self.sigma_fy_corrected = torch.clone(self.fy), torch.clone(self.sigma_fy)

        self.data_corrected, self.data_virtual = {}, {}
        self.data_action, self.data_amplitude, self.data_phase = {}, {}, {}

        self.ax, self.sigma_ax = torch.zeros_like(self.model.ax), torch.zeros_like(self.model.sigma_ax)
        self.bx, self.sigma_bx = torch.zeros_like(self.model.bx), torch.zeros_like(self.model.sigma_bx)

        self.ay, self.sigma_ay = torch.zeros_like(self.model.ay), torch.zeros_like(self.model.sigma_ay)
        self.by, self.sigma_by = torch.zeros_like(self.model.by), torch.zeros_like(self.model.sigma_by)

        self.normal = torch.zeros((self.size, 4, 4), dtype=self.dtype, device=self.device)
        self.sigma_normal = torch.zeros_like(self.normal)

        if self.use_model:
            self.fx_ij, self.sigma_fx_ij = self.model.fx_ij.to(self.dtype).to(self.device), self.model.sigma_fx_ij.to(self.dtype).to(self.device)
            self.fx_ik, self.sigma_fx_ik = self.model.fx_ik.to(self.dtype).to(self.device), self.model.sigma_fx_ik.to(self.dtype).to(self.device)
            self.fy_ij, self.sigma_fy_ij = self.model.fy_ij.to(self.dtype).to(self.device), self.model.sigma_fy_ij.to(self.dtype).to(self.device)
            self.fy_ik, self.sigma_fy_ik = self.model.fy_ik.to(self.dtype).to(self.device), self.model.sigma_fy_ik.to(self.dtype).to(self.device)

        if self.use_model and flag != None:
            size, length, *_ = self.index.shape
            self.mask = torch.ones((size, length)).to(torch.bool).to(self.device)
            for location, flag in enumerate(self.flag):
                if not flag and self.model.flag[location] != 0:
                    _, other = self.index.swapaxes(0, -1)
                    other = torch.mul(*(other != location).swapaxes(0, 1)).T
                    self.mask = (self.mask == other)


    def get_action(self,
                   *,
                   dict_threshold:dict={'use': True, 'factor': 5.0},
                   dict_dbscan:dict={'use': False, 'factor': 2.5},
                   dict_local_outlier_factor:dict={'use': False, 'contamination': 0.01},
                   dict_isolation_forest:dict={'use': False, 'contamination': 0.01},
                   bx:torch.Tensor=None,
                   by:torch.Tensor=None,
                   sigma_bx:torch.Tensor=None,
                   sigma_by:torch.Tensor=None,
                   jx:torch.Tensor=None,
                   jy:torch.Tensor=None,
                   sigma_jx:torch.Tensor=None,
                   sigma_jy:torch.Tensor=None) -> None:
        """
        Estimate uncoupled actions at each monitor location from amplitude data with optional cleaning and estimate overall action center and spread.

        Set self.action dictionary
        ['jx', 'sigma_jx', 'center_jx', 'spread_jx', 'jy', 'sigma_jy', 'center_jy', 'spread_jy', 'mask']

        Note, beta values at monitors can be passed, e.g. measured values
        Note, action values at monitors can be passed, in this case only filtering and center/spread estimation is performed

        Parameters
        ----------
        dict_threshold: dict
            parameters for threshold filter
        dict_dbscan: dict
            parameters for dbscan filter
        dict_local_outlier_factor: dict
            parameters for local outlier factor filter
        dict_isolation_forest: dict
            parameters for isolation forest filter
        bx: torch.Tensor
            bx values at monitor locations
        by: torch.Tensor
            by values at monitor locations
        sigma_bx: torch.Tensor
            bx errors at monitor locations
        sigma_by: torch.Tensor
            by errors at monitor locations
        jx: torch.Tensor
            jx values at monitor locations
        jy: torch.Tensor
            jy values at monitor locations
        sigma_jx: torch.Tensor
            jx errors at monitor locations
        sigma_jy: torch.Tensor
            jy errors at monitor locations

        Returns
        -------
        None

        """
        self.action = {}

        index = self.model.monitor_index

        bx = bx if bx is not None else self.model.bx[index]
        by = by if by is not None else self.model.by[index]

        sigma_bx = sigma_bx if sigma_bx is not None else self.model.sigma_bx[index]
        sigma_by = sigma_by if sigma_by is not None else self.model.sigma_by[index]

        if jx is None:
            jx = self.table.ax**2/(2.0*bx)

        if sigma_jx is None:
            sigma_jx  = self.table.ax**2/bx**2*self.table.sigma_ax**2
            sigma_jx += self.table.ax**4/bx**4/4*sigma_bx**2
            sigma_jx.sqrt_()

        if jy is None:
            jy = self.table.ay**2/(2.0*by)

        if sigma_jy is None:
            sigma_jy  = self.table.ay**2/by**2*self.table.sigma_ay**2
            sigma_jy += self.table.ay**4/by**4/4*sigma_by**2
            sigma_jy.sqrt_()

        mask = torch.clone(self.flag[index])
        mask = torch.stack([mask, mask]).to(torch.bool)

        data = standardize(torch.stack([jx, jy]), center_estimator=median, spread_estimator=biweight_midvariance)

        if dict_threshold['use']:
            factor = dict_threshold['factor']
            center = median(data)
            spread = biweight_midvariance(data).sqrt()
            min_value, max_value = center - factor*spread, center + factor*spread
            mask *= threshold(data, min_value, max_value)

        if dict_dbscan['use']:
            factor = dict_dbscan['factor']
            for i, case in enumerate(data):
                mask[i] *= dbscan(case.reshape(-1, 1), epsilon=factor)

        if dict_local_outlier_factor['use']:
            for i, case in enumerate(data):
                mask[i] *= local_outlier_factor(case.reshape(-1, 1), contamination=dict_local_outlier_factor['contamination'])

        if dict_isolation_forest['use']:
            for i, case in enumerate(data):
                mask[i] *= isolation_forest(case.reshape(-1, 1), contamination=dict_isolation_forest['contamination'])

        mask_jx, mask_jy = mask
        mask_jx, mask_jy = mask_jx/sigma_jx**2, mask_jy/sigma_jy**2

        mask_jx /= mask_jx.sum()
        mask_jy /= mask_jy.sum()

        center_jx = weighted_mean(jx, weight=mask_jx)
        spread_jx = weighted_variance(jx, weight=mask_jx, center=center_jx).sqrt()

        center_jy = weighted_mean(jy, weight=mask_jy)
        spread_jy = weighted_variance(jy, weight=mask_jy, center=center_jy).sqrt()

        self.action['jx'], self.action['sigma_jx'] = jx, sigma_jx
        self.action['center_jx'], self.action['spread_jx'] = center_jx, spread_jx

        self.action['jy'], self.action['sigma_jy'] = jy, sigma_jy
        self.action['center_jy'], self.action['spread_jy'] = center_jy, spread_jy

        self.action['mask'] = mask


    def get_twiss_from_amplitude(self) -> None:
        """
        Estimate CS twiss from amplitude.

        Set self.twiss_from_amplitude dictionary
        ['bx', 'sigma_bx', 'by', 'sigma_by']

        Note, action dictionary should be precomputed

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if self.action == {}:
            raise Exception('TWISS: action dictionary is empty')

        self.data_amplitude = {}

        ax, sigma_ax = self.table.ax, self.table.sigma_ax
        ay, sigma_ay = self.table.ay, self.table.sigma_ay

        jx, sigma_jx = self.action['center_jx'], self.action['spread_jx']
        jy, sigma_jy = self.action['center_jy'], self.action['spread_jy']

        bx, by = ax**2/(2.0*jx), ay**2/(2.0*jy)

        sigma_bx = torch.sqrt(ax**2/jx**2*sigma_ax**2 + 0.25*ax**4/jx**4*sigma_jx**2)
        sigma_by = torch.sqrt(ay**2/jy**2*sigma_ay**2 + 0.25*ay**4/jy**4*sigma_jy**2)

        self.data_amplitude['bx'], self.data_amplitude['sigma_bx'] = bx, sigma_bx
        self.data_amplitude['by'], self.data_amplitude['sigma_by'] = by, sigma_by


    def phase_correct(self,
                      *,
                      limit:int=None,
                      **kwargs) -> None:
        """
        Use measured phases and model phases advances to correct x & y phase at monitor locations.

        Set self.corrected_x and self.corrected_y dictionaries with corrected phase values
        For each monitor location (key) the corresponding value is a ditionary
        ['model', 'probe', 'limit', 'index', 'clean', 'phase', 'error']

        Set self.fx_corrected, self.fy_corrected, self.sigma_fx_corrected, self.sigma_fy_corrected values

        Note, correction introduce bias towards model

        Parameters
        ----------
        limit: int
            range limit
        **kwargs:
            passed to Decomposition.phase_virtual

        Returns
        -------
        None

        """
        self.corrected_x, self.corrected_y = {}, {}

        limit = max(self.limit) if limit is None else limit
        if limit <= 0:
            raise ValueError(f'TWISS: expected limit > 0')
        index = self.model.monitor_index

        self.fx_corrected, self.sigma_fx_corrected = torch.clone(self.fx), torch.clone(self.sigma_fx)
        self.fy_corrected, self.sigma_fy_corrected = torch.clone(self.fy), torch.clone(self.sigma_fy)

        nux, sigma_nux = self.table.nux, self.table.sigma_nux
        NUX, sigma_NUX = self.model.nux, self.model.sigma_nux

        nuy, sigma_nuy = self.table.nuy, self.table.sigma_nuy
        NUY, sigma_NUY = self.model.nuy, self.model.sigma_nuy

        fx, sigma_fx = self.fx, self.sigma_fx
        FX, sigma_FX = self.model.fx, self.model.sigma_fx

        fy, sigma_fy = self.fy, self.sigma_fy
        FY, sigma_FY = self.model.fy, self.model.sigma_fy

        def auxiliary_x(probe):
            return Decomposition.phase_virtual(probe, limit, self.flag, nux, NUX, fx, FX,
                                               sigma_frequency=sigma_nux, sigma_frequency_model=sigma_NUX,
                                               sigma_phase=sigma_fx, sigma_phase_model=sigma_FX,
                                               use_probe=True,
                                               **kwargs)

        def auxiliary_y(probe):
            return Decomposition.phase_virtual(probe, limit, self.flag, nuy, NUY, fy, FY,
                                               sigma_frequency=sigma_nuy, sigma_frequency_model=sigma_NUY,
                                               sigma_phase=sigma_fy, sigma_phase_model=sigma_FY,
                                               use_probe=True,
                                               **kwargs)

        data_x = [auxiliary_x(probe) for probe in index]
        data_y = [auxiliary_y(probe) for probe in index]

        for count, probe in enumerate(index):
            self.corrected_x[probe], self.corrected_y[probe] = data_x[count], data_y[count]
            self.fx_corrected[probe], self.sigma_fx_corrected[probe] = self.corrected_x[probe].get('model')
            self.fy_corrected[probe], self.sigma_fy_corrected[probe] = self.corrected_y[probe].get('model')


    def phase_virtual(self,
                      *,
                      limit:int=None,
                      exclude:list=None,
                      method:str='model',
                      inverse:bool=True,
                      forward:bool=True,
                      nearest:bool=False,
                      **kwargs) -> None:
        """
        Estimate x & y phase at virtual locations.

        Set self.virtual_x and self.virtual_y dictionaries
        For each virtual location (key) the corresponding value is a ditionary
        ['model', 'probe', 'limit', 'index', 'clean', 'phase', 'error']

        Update self.fx, self.fy, self.sigma_fx, self.sigma_fy values for virtual locations (*_corrected)

        Parameters
        ----------
        limit: int
            range limit to use
        exclude: list
            list of virtual location to exclude
        method: str
            'model' or 'interpolate'
        inverse: bool
            flag to use other only in inverse direction
        forward: bool
            flag to use other only in forward direction
        nearest: bool
            flag to use nearest other
        **kwargs:
            passed to Decomposition.phase_virtual

        Returns
        -------
        None

        """
        if method == 'model':

            self.virtual_x, self.virtual_y = {}, {}

            limit = max(self.limit) if limit is None else limit
            if limit <= 0:
                raise ValueError(f'TWISS: expected limit > 0')
            exclude = [] if exclude is None else exclude
            index = [index for index in self.model.virtual_index if index not in exclude]

            nux, sigma_nux = self.table.nux, self.table.sigma_nux
            NUX, sigma_NUX = self.model.nux, self.model.sigma_nux

            nuy, sigma_nuy = self.table.nuy, self.table.sigma_nuy
            NUY, sigma_NUY = self.model.nuy, self.model.sigma_nuy

            fx, sigma_fx = self.fx, self.sigma_fx
            FX, sigma_FX = self.model.fx, self.model.sigma_fx

            fy, sigma_fy = self.fy, self.sigma_fy
            FY, sigma_FY = self.model.fy, self.model.sigma_fy

            def auxiliary_x(probe):
                return Decomposition.phase_virtual(probe, limit, self.flag, nux, NUX, fx, FX,
                                                sigma_frequency=sigma_nux, sigma_frequency_model=sigma_NUX,
                                                sigma_phase=sigma_fx, sigma_phase_model=sigma_FX,
                                                use_probe=False, inverse=inverse, forward=forward, nearest=nearest,
                                                **kwargs)

            def auxiliary_y(probe):
                return Decomposition.phase_virtual(probe, limit, self.flag, nuy, NUY, fy, FY,
                                                sigma_frequency=sigma_nuy, sigma_frequency_model=sigma_NUY,
                                                sigma_phase=sigma_fy, sigma_phase_model=sigma_FY,
                                                use_probe=False, inverse=inverse, forward=forward, nearest=nearest,
                                                **kwargs)

            data_x = [auxiliary_x(probe) for probe in index]
            data_y = [auxiliary_y(probe) for probe in index]

            for count, probe in enumerate(index):
                self.virtual_x[probe], self.virtual_y[probe] = data_x[count], data_y[count]
                self.fx[probe], self.sigma_fx[probe] = self.virtual_x[probe].get('model')
                self.fy[probe], self.sigma_fy[probe] = self.virtual_y[probe].get('model')
                self.fx_corrected[probe], self.sigma_fx_corrected[probe] = self.virtual_x[probe].get('model')
                self.fy_corrected[probe], self.sigma_fy_corrected[probe] = self.virtual_y[probe].get('model')

        if method == 'interpolate':

            xr, *xs, xl = self.model.fx[self.model.monitor_index]
            x = torch.stack([xl - self.model.mux, xr, *xs, xl, xr + self.model.mux]).cpu().numpy()

            yr, *ys, yl = self.fx[self.model.monitor_index]
            y1 = yl - 2.0*numpy.pi*self.table.nux
            y2 = yr + 2.0*numpy.pi*self.table.nux
            while y1 > yr: y1 -= 2.0*numpy.pi
            while y2 < yl: y2 += 2.0*numpy.pi
            y = torch.stack([y1, yr, *ys, yl, y2]).cpu().numpy()

            f = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate', assume_sorted=True)
            x = self.model.fx[self.model.virtual_index].cpu().numpy()
            self.fx[self.model.virtual_index] = torch.tensor(f(x), dtype=self.dtype, device=self.device)

            xr, *xs, xl = self.model.fy[self.model.monitor_index]
            x = torch.stack([xl - self.model.muy, xr, *xs, xl, xr + self.model.muy]).cpu().numpy()

            yr, *ys, yl = self.fy[self.model.monitor_index]
            y1 = yl - 2.0*numpy.pi*self.table.nuy
            y2 = yr + 2.0*numpy.pi*self.table.nuy
            while y1 > yr: y1 -= 2.0*numpy.pi
            while y2 < yl: y2 += 2.0*numpy.pi
            y = torch.stack([y1, yr, *ys, yl, y2]).cpu().numpy()

            f = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate', assume_sorted=True)
            x = self.model.fy[self.model.virtual_index].cpu().numpy()
            self.fy[self.model.virtual_index] = torch.tensor(f(x), dtype=self.dtype, device=self.device)


    @staticmethod
    def phase_alfa(a_m:torch.Tensor,
                   f_ij:torch.Tensor,
                   f_m_ij:torch.Tensor,
                   f_ik:torch.Tensor,
                   f_m_ik:torch.Tensor,
                   *,
                   error:bool=True,
                   model:bool=True,
                   sigma_a_m:torch.Tensor=0.0,
                   sigma_f_ij:torch.Tensor=0.0,
                   sigma_f_m_ij:torch.Tensor=0.0,
                   sigma_f_ik:torch.Tensor=0.0,
                   sigma_f_m_ik:torch.Tensor=0.0) -> tuple:
        """
        Estimate uncoupled twiss alfa at index (i) from given triplet (i, j, k) phase data.

        Note, probed index (i), other indices (j) and (k), pairs (i, j) and (i, k)
        Phase advance is assumed to be from (i) to other indices, should be negative if (i) is ahead of the other index (timewise)

        Parameters
        ----------
        a_m: torch.Tensor
            model value
        f_ij: torch.Tensor
            phase advance between probed and the 1st index (j)
        f_m_ij: torch.Tensor
            model phase advance between probed and the 1st index (j)
        f_ik: torch.Tensor
            phase advance between probed and the 2nd index (k)
        f_m_ik: torch.Tensor
            model phase advance between probed and 2nd index (k)
        error: bool
            flag to compute error
        model: bool
            flag to include model error
        sigma_a_m: torch.Tensor
            model value error
        sigma_f_ij: torch.Tensor
            phase advance error between probed and the 1st index (j)
        sigma_f_m_ij: torch.Tensor
            model phase advance error between probed and the 1st index (j)
        sigma_f_ik: torch.Tensor
            phase advance error between probed and the 2nd index (k)
        sigma_f_m_ik: torch.Tensor
            model phase advance error between probed and the 2nd index (k)

        Returns
        -------
        (a, sigma_a)

        """
        a = a_m*(1.0/torch.tan(f_ij)-1.0/torch.tan(f_ik))/(1.0/torch.tan(f_m_ij)-1.0/torch.tan(f_m_ik))-1.0/torch.tan(f_ij)*1.0/torch.sin(f_m_ij - f_m_ik)*torch.cos(f_m_ik)*torch.sin(f_m_ij) + 1.0/torch.tan(f_ik)*1.0/torch.sin(f_m_ij - f_m_ik)*torch.cos(f_m_ij)*torch.sin(f_m_ik)

        if not error:
            return (a, torch.zeros_like(a))

        sigma_a  = sigma_f_ij**2*(1.0/torch.sin(f_ij))**4*(1.0/torch.tan(f_m_ik) + a_m)**2/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**2
        sigma_a += sigma_f_ik**2*(1.0/torch.sin(f_ik))**4*(1.0/torch.tan(f_m_ij) + a_m)**2/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**2

        if model:
            sigma_a += sigma_a_m**2*((1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))**2/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**2)
            sigma_a += sigma_f_m_ik**2*(1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))**2*(1.0/torch.sin(f_m_ij - f_m_ik))**4*torch.sin(f_m_ij)**2*(torch.cos(f_m_ij) + a_m*torch.sin(f_m_ij))**2
            sigma_a += sigma_f_m_ij**2*(1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))**2*(1.0/torch.sin(f_m_ij - f_m_ik))**4*torch.sin(f_m_ik)**2*(torch.cos(f_m_ik) + a_m*torch.sin(f_m_ik))**2

        sigma_a.sqrt_()
        return (a, sigma_a)


    @staticmethod
    def phase_beta(b_m:torch.Tensor,
                   f_ij:torch.Tensor,
                   f_m_ij:torch.Tensor,
                   f_ik:torch.Tensor,
                   f_m_ik:torch.Tensor,
                   *,
                   error:bool=True,
                   model:bool=True,
                   sigma_b_m:torch.Tensor=0.0,
                   sigma_f_ij:torch.Tensor=0.0,
                   sigma_f_m_ij:torch.Tensor=0.0,
                   sigma_f_ik:torch.Tensor=0.0,
                   sigma_f_m_ik:torch.Tensor=0.0) -> tuple:
        """
        Estimate uncoupled twiss beta at index (i) from given triplet (i, j, k) phase data.

        Note, probed index (i), other indices (j) and (k), pairs (i, j) and (i, k)
        Phase advance is assumed to be from (i) to other indices, should be negative if (i) is ahead of the other index (timewise)

        Parameters
        ----------
        b_m: torch.Tensor
            model value
        f_ij: torch.Tensor
            phase advance between probed and the 1st index (j)
        f_m_ij: torch.Tensor
            model phase advance between probed and the 1st index (j)
        f_ik: torch.Tensor
            phase advance between probed and the 2nd index (k)
        f_m_ik: torch.Tensor
            model phase advance between probed and 2nd index (k)
        error: bool
            flag to compute error
        model: bool
            flag to include model error
        sigma_b_m: torch.Tensor
            model value error
        sigma_f_ij: torch.Tensor
            phase advance error between probed and the 1st index (j)
        sigma_f_m_ij: torch.Tensor
            model phase advance error between probed and the 1st index (j)
        sigma_f_ik: torch.Tensor
            phase advance error between probed and the 2nd index (k)
        sigma_f_m_ik: torch.Tensor
            model phase advance error between probed and the 2nd index (k)

        Returns
        -------
        (b, sigma_b)

        """
        b = b_m*(1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))

        if not error:
            return (b, torch.zeros_like(b))

        sigma_b  = sigma_f_ij**2*b_m**2*(1.0/torch.sin(f_ij))**4/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**2
        sigma_b += sigma_f_ik**2*b_m**2*(1.0/torch.sin(f_ik))**4/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**2

        if model:
            sigma_b += sigma_b_m**2*(1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))**2/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**2
            sigma_b += sigma_f_m_ij**2*b_m**2*(1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))**2*(1.0/torch.sin(f_m_ij))**4/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**4
            sigma_b += sigma_f_m_ik**2*b_m**2*(1.0/torch.tan(f_ij) - 1.0/torch.tan(f_ik))**2*(1.0/torch.sin(f_m_ik))**4/(1.0/torch.tan(f_m_ij) - 1.0/torch.tan(f_m_ik))**4

        sigma_b.sqrt_()
        return (b, sigma_b)


    def get_twiss_from_phase(self,
                             *,
                             error:bool=True,
                             model:bool=False,
                             use_correct:bool=False,
                             use_correct_sigma:bool=False,
                             use_model:bool=False) -> None:
        """
        Estimate CS twiss from phase data.

        Set self.data_phase dictionary
        ['fx_ij', 'sigma_fx_ij', 'fx_m_ij', 'sigma_fx_m_ij', 'fx_ik', 'sigma_fx_ik', 'fx_m_ik', 'sigma_fx_m_ik', 'fy_ij', 'sigma_fy_ij', 'fy_m_ij', 'sigma_fy_m_ij', 'fy_ik', 'sigma_fy_ik', 'fy_m_ik', 'sigma_fy_m_ik', 'ax', 'sigma_ax', 'bx', 'sigma_bx', 'ay', 'sigma_ay', 'by', 'sigma_by']

        Note, raw data is saved, no cleaning is performed
        Values (and errors) are computed for each triplet

        Parameters
        ----------
        error: bool
            flag to compute twiss errors
        model: bool
            flag to include model errors
        use_correct: bool
            flag to use corrected phases
        use_correct_sigma: bool
            flag to use corrected phase errors
        use_model: bool
            flag to use precomputed model data

        Returns
        -------
        None

        """
        self.data_phase = {}

        fx = self.fx_corrected if use_correct else self.fx
        fy = self.fy_corrected if use_correct else self.fy

        sigma_fx = self.sigma_fx_corrected if use_correct_sigma else self.sigma_fx
        sigma_fy = self.sigma_fy_corrected if use_correct_sigma else self.sigma_fy

        ax_m, bx_m = self.model.ax, self.model.bx
        ay_m, by_m = self.model.ay, self.model.by

        index = self.combo.swapaxes(0, -1)

        value, sigma = Decomposition.phase_advance(*index, self.table.nux, fx,
                                                   error=error, model=False, sigma_frequency=self.table.sigma_nux, sigma_phase=sigma_fx)
        fx_ij, fx_ik = value.swapaxes(0, 1)
        sx_ij, sx_ik = sigma.swapaxes(0, 1)

        value, sigma = Decomposition.phase_advance(*index, self.table.nuy, fy,
                                                   error=error, model=False, sigma_frequency=self.table.sigma_nuy, sigma_phase=sigma_fy)
        fy_ij, fy_ik = value.swapaxes(0, 1)
        sy_ij, sy_ik = sigma.swapaxes(0, 1)

        if use_model:

            fx_m_ij, fx_m_ik = self.fx_ij, self.fx_ik
            sx_m_ij, sx_m_ik = self.sigma_fx_ij, self.sigma_fx_ik

            fy_m_ij, fy_m_ik = self.fy_ij, self.fy_ik
            sy_m_ij, sy_m_ik = self.sigma_fy_ij, self.sigma_fy_ik

        else:

            value, sigma = Decomposition.phase_advance(*index, self.model.nux, self.model.fx, error=error*model, model=True, sigma_frequency=self.model.sigma_nux, sigma_phase=self.model.sigma_fx)
            fx_m_ij, fx_m_ik = value.swapaxes(0, 1)
            sx_m_ij, sx_m_ik = sigma.swapaxes(0, 1)

            value, sigma = Decomposition.phase_advance(*index, self.model.nuy, self.model.fy, error=error*model, model=True, sigma_frequency=self.model.sigma_nuy, sigma_phase=self.model.sigma_fy)
            fy_m_ij, fy_m_ik = value.swapaxes(0, 1)
            sy_m_ij, sy_m_ik = sigma.swapaxes(0, 1)

        ax, sigma_ax = self.phase_alfa(ax_m, fx_ij, fx_m_ij, fx_ik, fx_m_ik, error=error, model=model, sigma_a_m=self.model.sigma_ax, sigma_f_ij=sx_ij, sigma_f_ik=sx_ik, sigma_f_m_ij=sx_m_ij, sigma_f_m_ik=sx_m_ik)
        bx, sigma_bx = self.phase_beta(bx_m, fx_ij, fx_m_ij, fx_ik, fx_m_ik, error=error, model=model, sigma_b_m=self.model.sigma_bx, sigma_f_ij=sx_ij, sigma_f_ik=sx_ik, sigma_f_m_ij=sx_m_ij, sigma_f_m_ik=sx_m_ik)

        ay, sigma_ay = self.phase_alfa(ay_m, fy_ij, fy_m_ij, fy_ik, fy_m_ik, error=error, model=model, sigma_a_m=self.model.sigma_ay, sigma_f_ij=sy_ij, sigma_f_ik=sy_ik, sigma_f_m_ij=sy_m_ij, sigma_f_m_ik=sy_m_ik)
        by, sigma_by = self.phase_beta(by_m, fy_ij, fy_m_ij, fy_ik, fy_m_ik, error=error, model=model, sigma_b_m=self.model.sigma_by, sigma_f_ij=sy_ij, sigma_f_ik=sy_ik, sigma_f_m_ij=sy_m_ij, sigma_f_m_ik=sy_m_ik)

        self.data_phase['fx_ij'], self.data_phase['sigma_fx_ij'], self.data_phase['fx_m_ij'], self.data_phase['sigma_fx_m_ij'] = fx_ij.T, sx_ij.T, fx_m_ij.T, sx_m_ij.T
        self.data_phase['fx_ik'], self.data_phase['sigma_fx_ik'], self.data_phase['fx_m_ik'], self.data_phase['sigma_fx_m_ik'] = fx_ik.T, sx_ik.T, fx_m_ik.T, sx_m_ik.T

        self.data_phase['fy_ij'], self.data_phase['sigma_fy_ij'], self.data_phase['fy_m_ij'], self.data_phase['sigma_fy_m_ij'] = fy_ij.T, sy_ij.T, fy_ij.T, sy_m_ij.T
        self.data_phase['fy_ik'], self.data_phase['sigma_fy_ik'], self.data_phase['fy_m_ik'], self.data_phase['sigma_fy_m_ik'] = fy_ik.T, sy_ik.T, fy_ik.T, sy_m_ik.T

        self.data_phase['ax'], self.data_phase['sigma_ax'] = ax.T.nan_to_num(), sigma_ax.T.nan_to_num()
        self.data_phase['bx'], self.data_phase['sigma_bx'] = bx.T.nan_to_num(), sigma_bx.T.nan_to_num()
        self.data_phase['ay'], self.data_phase['sigma_ay'] = ay.T.nan_to_num(), sigma_ay.T.nan_to_num()
        self.data_phase['by'], self.data_phase['sigma_by'] = by.T.nan_to_num(), sigma_by.T.nan_to_num()


    def filter_twiss(self,
                     plane:str = 'x', *,
                     phase:dict={'use': True, 'threshold': 10.00},
                     model:dict={'use': True, 'threshold': 00.50},
                     value:dict={'use': True, 'threshold': 00.25},
                     sigma:dict={'use': True, 'threshold': 00.25},
                     limit:dict={'use': True, 'threshold': 05.00}, 
                     error:dict={'use': True, 'threshold': 05.00}) -> dict:
        """
        Filter twiss for given data plane and cleaning options.

        Note, x & y planes are processed separately

        Parameters
        ----------
        plane: str
            data plane ('x' or 'y')
        phase: dict
            clean based on advance phase data
            used if 'use' is True, remove combinations with absolute value of phase advance cotangents above threshold value
        model: dict
            clean based on phase advance proximity to model
            used if 'use' is True, remove combinations with (x - x_model)/x_model > threshold value
        value: dict
            clean based on estimated twiss beta error value
            used if 'use' is True, remove combinations with x/sigma_x < 1/threshold value
        sigma: dict
            clean based on estimated phase advance error value
            used if 'use' is True, remove combinations with x/sigma_x < 1/threshold value
        limit: dict
            clean outliers outside scaled interval based on estimated values
            used if 'use' is True
        error: dict
            clean outliers outside scaled interval based on estimated errors
            used if 'use' is True

        Returns
        -------
        filter mask (torch.Tensor)

        """
        size, length, *_ = self.index.shape
        mask = torch.ones((size, length), device=self.device).to(torch.bool)

        if max(self.limit) == 1:
            return mask

        if plane == 'x':
            a_m, b_m = self.model.ax.reshape(-1, 1), self.model.bx.reshape(-1, 1)
            a, b, sigma_a, sigma_b = self.data_phase['ax'], self.data_phase['bx'], self.data_phase['sigma_ax'], self.data_phase['sigma_bx']
            f_ij, sigma_f_ij, f_m_ij, sigma_f_m_ij = self.data_phase['fx_ij'], self.data_phase['sigma_fx_ij'], self.data_phase['fx_m_ij'], self.data_phase['sigma_fx_m_ij']
            f_ik, sigma_f_ik, f_m_ik, sigma_f_m_ik = self.data_phase['fx_ik'], self.data_phase['sigma_fx_ik'], self.data_phase['fx_m_ik'], self.data_phase['sigma_fx_m_ik']

        if plane == 'y':
            a_m, b_m = self.model.ay.reshape(-1, 1), self.model.by.reshape(-1, 1)
            a, b, sigma_a, sigma_b = self.data_phase['ay'], self.data_phase['by'], self.data_phase['sigma_ay'], self.data_phase['sigma_by']
            f_ij, sigma_f_ij, f_m_ij, sigma_f_m_ij = self.data_phase['fy_ij'], self.data_phase['sigma_fy_ij'], self.data_phase['fy_m_ij'], self.data_phase['sigma_fy_m_ij']
            f_ik, sigma_f_ik, f_m_ik, sigma_f_m_ik = self.data_phase['fy_ik'], self.data_phase['sigma_fy_ik'], self.data_phase['fy_m_ik'], self.data_phase['sigma_fy_m_ik']
            
        mask *= b > 0.0

        if phase['use']:
            cot_ij, cot_m_ij = torch.abs(1.0/torch.tan(f_ij)), torch.abs(1.0/torch.tan(f_m_ij))
            cot_ik, cot_m_ik = torch.abs(1.0/torch.tan(f_ij)), torch.abs(1.0/torch.tan(f_m_ij))
            mask *= phase['threshold'] > cot_ij
            mask *= phase['threshold'] > cot_m_ij
            mask *= phase['threshold'] > cot_ik
            mask *= phase['threshold'] > cot_m_ik

        if model['use']:
            mask *= model['threshold'] > torch.abs((f_ij - f_m_ij)/f_m_ij)
            mask *= model['threshold'] > torch.abs((f_ik - f_m_ik)/f_m_ik)

        if value['use']:
            mask *= value['threshold'] > torch.abs((b - b_m)/b_m)

        if sigma['use']:
            mask *= 1/sigma['threshold'] < torch.abs(f_ij/sigma_f_ij)
            mask *= 1/sigma['threshold'] < torch.abs(f_ik/sigma_f_ik)

        if limit['use']:
            factor = torch.tensor(limit['threshold'], dtype=self.dtype, device=self.device)
            mask *= threshold(standardize(a, center_estimator=median, spread_estimator=biweight_midvariance), -factor, +factor)
            mask *= threshold(standardize(b, center_estimator=median, spread_estimator=biweight_midvariance), -factor, +factor)

        if error['use']:
            factor = torch.tensor(error['threshold'], dtype=self.dtype, device=self.device)
            mask *= threshold(standardize(sigma_a, center_estimator=median, spread_estimator=biweight_midvariance), -factor, +factor)
            mask *= threshold(standardize(sigma_b, center_estimator=median, spread_estimator=biweight_midvariance), -factor, +factor)

        return mask


    def mask_range(self,
                   limit:tuple) -> torch.Tensor:
        """
        Generate weight mask based on given range limit.

        Parameters
        ----------
        limit: tuple
            range limit to use, (min, max), 1 <= min <= max, mim is excluded, for full range min==max

        Returns
        -------
        weight mask (torch.Tensor)

        """
        size, length, *_ = self.shape
        mask = torch.zeros((size, length), dtype=torch.int64, device=self.device)

        count = torch.tensor([limit*(2*limit - 1) for limit in range(1, max(self.limit) + 1)], dtype=torch.int64, device=self.device)
        limit_min, limit_max = limit

        if limit_min == limit_max:
            count = count[:limit_max]
            *_, count_max = count
            mask[:, :count_max] = 1

        if limit_min < limit_max:
            count = count[limit_min - 1:limit_max]
            count_min, *_, count_max = count
            mask[:, count_min:count_max] = 1

        count = torch.tensor([limit*(2*limit - 1) for limit in range(1, max(self.limit) + 1)], dtype=torch.int64, device=self.device)

        limit_min, limit_max = self.limit

        if limit_min == limit_max:
            count = count[:limit_max]
            *_, count_max = count
            mask = mask[:, :count_max]

        if limit_min < limit_max:
            count = count[limit_min - 1:limit_max]
            count_min, *_, count_max = count
            mask = mask[:, count_min:count_max]

        return mask


    def mask_location(self,
                      table:list) -> torch.Tensor:
        """
        Generate weight mask based on given range limit.

        Parameters
        ----------
        table: list
            list of locations to remove

        Returns
        -------
        weight mask (torch.Tensor)

        """
        size, length, *_ = self.combo.shape
        mask = torch.zeros((size, length), dtype=torch.int64, device=self.device)

        for location in table:
            _, other = self.index.swapaxes(0, -1)
            other = torch.mul(*(other != location).swapaxes(0, 1)).T
            mask = (mask == other)

        return mask.logical_not()


    def mask_distance(self,
                      function:Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Generate weight mask based on given range limit.

        Parameters
        ----------
        function: Callable[[torch.Tensor], torch.Tensor]
            function to apply to distance data

        Returns
        -------
        weight mask (torch.Tensor)

        """
        mask = torch.stack([function(distance) for distance in self.distance])
        mask = torch.stack([mask for _ in range(self.size)])
        return mask


    def process_twiss(self,
                      plane:str='x',
                      *,
                      weight:bool=True,
                      mask:torch.Tensor=None) -> dict:
        """
        Process uncoupled twiss data.

        Set self.ax, self.bx, self.ay, self.by and corresponding errors

        Parameters
        ----------
        plane: str
            data plane ('x' or 'y')
        weight: bool
            flag to use weights
        mask: torch.Tensor
            mask

        Returns
        -------
        twiss data (dict)
        dict_keys(['value_a', 'sigma_a', 'error_a', 'value_b', 'sigma_b', 'error_b'])

        """
        result = {}

        if mask == None:
            size, length, *_ = self.index.shape
            mask = torch.ones((size, length), device=self.device).to(torch.bool)

        if plane == 'x':
            a, sigma_a, a_m = self.data_phase['ax'], self.data_phase['sigma_ax'], self.model.ax
            b, sigma_b, b_m = self.data_phase['bx'], self.data_phase['sigma_bx'], self.model.bx

        if plane == 'y':
            a, sigma_a, a_m = self.data_phase['ay'], self.data_phase['sigma_ay'], self.model.ay
            b, sigma_b, b_m = self.data_phase['by'], self.data_phase['sigma_by'], self.model.by

        if max(self.limit) == 1:

            result['value_a'] = a.flatten()
            result['sigma_a'] = sigma_a.flatten()
            result['error_a'] = (a.flatten() - a_m)/a_m

            result['value_b'] = b.flatten()
            result['sigma_b'] = sigma_b.flatten()
            result['error_b'] = (b.flatten() - b_m)/b_m

            if plane == 'x':
                self.ax, self.sigma_ax = result['value_a'], result['sigma_a']
                self.bx, self.sigma_bx = result['value_b'], result['sigma_b']

            if plane == 'y':
                self.ay, self.sigma_ay = result['value_a'], result['sigma_a']
                self.by, self.sigma_by = result['value_b'], result['sigma_b']

            return result

        if not weight:

            center = weighted_mean(a, weight=mask)
            spread = weighted_variance(a, weight=mask, center=center).sqrt()
            result['value_a'] = center
            result['sigma_a'] = spread
            result['error_a'] = (center - a_m)/a_m

            center = weighted_mean(b, weight=mask)
            spread = weighted_variance(b, weight=mask, center=center).sqrt()
            result['value_b'] = center
            result['sigma_b'] = spread
            result['error_b'] = (center - b_m)/b_m

        else:

            weight = (mask.to(self.dtype)/sigma_a**2).nan_to_num(posinf=1.0)
            weight /= weight.sum()
            center = weighted_mean(a, weight=weight)
            spread = weighted_variance(a, weight=weight, center=center).sqrt()
            result['value_a'] = center
            result['sigma_a'] = spread
            result['error_a'] = (center - a_m)/a_m

            weight = (mask.to(self.dtype)/sigma_b**2).nan_to_num(posinf=1.0)
            weight /= weight.sum()
            center = weighted_mean(b, weight=weight)
            spread = weighted_variance(b, weight=weight, center=center).sqrt()
            result['value_b'] = center
            result['sigma_b'] = spread
            result['error_b'] = (center - b_m)/b_m

        if plane == 'x':
            self.ax, self.sigma_ax = result['value_a'], result['sigma_a']
            self.bx, self.sigma_bx = result['value_b'], result['sigma_b']

        if plane == 'y':
            self.ay, self.sigma_ay = result['value_a'], result['sigma_a']
            self.by, self.sigma_by = result['value_b'], result['sigma_b']

        return result


    def bootstrap_twiss(self,
                        plane:str='x',
                        *,
                        weight:bool=True,
                        mask:torch.Tensor=None,
                        fraction:float=0.75,
                        count:int=512) -> dict:
        """
        Bootstrap uncoupled twiss data.

        Set self.ax, self.bx, self.ay, self.by and corresponding errors

        Parameters
        ----------
        plane: str
            data plane ('x' or 'y')
        weight: bool
            flag to use weights
        mask: torch.Tensor
            mask
        fraction: float
            fraction of data to use for sample generation
        count: int
            total number of samples

        Returns
        -------
        twiss data (dict)
        dict_keys(['value_a', 'sigma_a', 'error_a', 'value_b', 'sigma_b', 'error_b'])

        """
        result = {}

        if mask == None:
                size, length, *_ = self.index.shape
                mask = torch.ones((size, length), device=self.device).to(torch.bool)

        if plane == 'x':
            a, sigma_a, a_m = self.data_phase['ax'], self.data_phase['sigma_ax'], self.model.ax
            b, sigma_b, b_m = self.data_phase['bx'], self.data_phase['sigma_bx'], self.model.bx

        if plane == 'y':
            a, sigma_a, a_m = self.data_phase['ay'], self.data_phase['sigma_ay'], self.model.ay
            b, sigma_b, b_m = self.data_phase['by'], self.data_phase['sigma_by'], self.model.by

        if max(self.limit) == 1:

            result['value_a'] = a.flatten()
            result['sigma_a'] = sigma_a.flatten()
            result['error_a'] = (a.flatten() - a_m)/a_m

            result['value_b'] = b.flatten()
            result['sigma_b'] = sigma_b.flatten()
            result['error_b'] = (b.flatten() - b_m)/b_m

            if plane == 'x':
                self.ax, self.sigma_ax = result['value_a'], result['sigma_a']
                self.bx, self.sigma_bx = result['value_b'], result['sigma_b']

            if plane == 'y':
                self.ay, self.sigma_ay = result['value_a'], result['sigma_a']
                self.by, self.sigma_by = result['value_b'], result['sigma_b']

            return result

        if weight:
            weight_a = (mask.to(self.dtype)/sigma_a**2).nan_to_num(posinf=0.0, neginf=0.0)
            weight_b = (mask.to(self.dtype)/sigma_b**2).nan_to_num(posinf=0.0, neginf=0.0)
        else:
            weight_a = mask.to(self.dtype)
            weight_b = mask.to(self.dtype)

        weight_a /= weight_a.sum()
        weight_b /= weight_b.sum()

        _, length = mask.shape
        size = int(fraction*length)

        center_a = torch.zeros(count, dtype=self.dtype, device=self.device)
        spread_a = torch.zeros(count, dtype=self.dtype, device=self.device)
        center_b = torch.zeros(count, dtype=self.dtype, device=self.device)
        spread_b = torch.zeros(count, dtype=self.dtype, device=self.device)

        data_a = torch.zeros(self.size, dtype=self.dtype, device=self.device)
        data_b = torch.zeros(self.size, dtype=self.dtype, device=self.device)
        data_sigma_a = torch.zeros(self.size, dtype=self.dtype, device=self.device)
        data_sigma_b = torch.zeros(self.size, dtype=self.dtype, device=self.device)

        for location in range(self.size):

            index = torch.randint(length, (count, size), dtype=torch.int64, device=self.device)

            center_a = weighted_mean(a[location][index], weight_a[location][index])
            spread_a = weighted_variance(a[location][index], weight_a[location][index]).sqrt()
            data_a[location] = mean(center_a)
            data_sigma_a[location] = mean(spread_a)

            center_b = weighted_mean(b[location][index], weight_b[location][index])
            spread_b = weighted_variance(b[location][index], weight_b[location][index]).sqrt()
            data_b[location] = mean(center_b)
            data_sigma_b[location] = mean(spread_b)

        result['value_a'] = data_a
        result['sigma_a'] = data_sigma_a
        result['error_a'] = (data_a - a_m)/a_m

        result['value_b'] = data_b
        result['sigma_b'] = data_sigma_b
        result['error_b'] = (data_b - b_m)/b_m

        if plane == 'x':
            self.ax, self.sigma_ax = result['value_a'], result['sigma_a']
            self.bx, self.sigma_bx = result['value_b'], result['sigma_b']

        if plane == 'y':
            self.ay, self.sigma_ay = result['value_a'], result['sigma_a']
            self.by, self.sigma_by = result['value_b'], result['sigma_b']

        return result


    def matrix(self,
               probe:torch.Tensor,
               other:torch.Tensor) -> torch.Tensor:
        """
        Generate transport matrices between given probe and other locations using measured twiss.

        Matrices are generated from probe to other
        Identity matrices are generated if probe == other
        One-turn matrices can be generated with other = probe + self.size
        Input parameters should be 1D tensors with matching length
        Additionaly probe and/or other input parameter can be an int or str in self.model.name (not checked)

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
            probe = torch.tensor([self.model.name.index(probe)], dtype=torch.int64, device=self.device)

        if isinstance(other, int):
            other = torch.tensor([other], dtype=torch.int64, device=self.device)

        if isinstance(other, str):
            other = torch.tensor([self.model.name.index(other)], dtype=torch.int64, device=self.device)

        fx, _ = Decomposition.phase_advance(probe, other, self.table.nux, self.fx, error=False)
        fy, _ = Decomposition.phase_advance(probe, other, self.table.nuy, self.fy, error=False)

        probe = mod(probe, self.size).to(torch.int64)
        other = mod(other, self.size).to(torch.int64)

        if self.model.model == 'coupled':
            matrix = matrix_coupled(self.normal[probe], self.normal[other], torch.stack([fx, fy]).T)
        else:
            matrix = matrix_uncoupled(self.ax[probe], self.bx[probe], self.ax[other], self.bx[other], fx,
                                      self.ay[probe], self.by[probe], self.ay[other], self.by[other], fy)

        return matrix.squeeze()


    def matrix_virtual(self,
                    probe:int,
                    other:int, *,
                    close:str='nearest') -> torch.Tensor:
        """
        Compute virtual transport matrix between probe and other.

        Note, kind (virtual or monitor) is defined by self.flag
        Note, both probe and other are expected to be on the first turn

        (monitor -> monitor)
        M(m, m') = M(m, m')

        (monitor -> virtual)
        M(m, v)  = M(m', v) M(m, m')
        m  - monitor
        v  - virtual
        m' - monitor close to v

        (virtual -> monitor)
        M(v, m)  = M(v, m') M(m', m)
        m  - monitor
        v  - virtual
        m' - monitor close to v

        (virtual -> virtual)
        M(v, v') = M(m', v') M(m, m') M(v, m)
        v  - virtual
        v' - virtual
        m  - monitor close to v
        m' - monitor close to v'

        Parameters
        ----------
        probe: int
            probe locations
        other: int
            other locations
        close: str
            'forward', 'inverse' or 'nearest'

        Returns
        -------
        virtual transport matrix (torch.Tensor)

        """
        if isinstance(probe, str):
            probe = self.model.name.index(probe)

        if isinstance(other, str):
            other = self.model.name.index(other)

        is_probe = int(self.flag[probe])
        is_other = int(self.flag[other])

        match (is_probe, is_other):

            case (1, 1):

                return self.matrix(probe, other)

            case (1, 0):

                forward = int(mod(other + 1, self.size))
                while not self.flag[forward]:
                    forward = int(mod(forward + 1, self.size))
                time_forward = self.model.time[forward]
                if forward < other: forward += self.model.size

                inverse = int(mod(other - 1, self.size))
                while not self.flag[inverse]:
                    inverse = int(mod(inverse - 1, self.size))
                time_inverse = self.model.time[inverse]
                if inverse > other: inverse -= self.model.size

                if close == 'forward':
                    return self.model.matrix(forward, other) @ self.matrix(probe, forward)

                if close == 'inverse':
                    return self.model.matrix(inverse, other) @ self.matrix(probe, inverse)

                if close == 'nearest':

                    delta_forward = abs(time_forward - self.model.time[other])
                    delta_inverse = abs(time_inverse - self.model.time[other])
                    nearest = forward if delta_forward <= delta_inverse else inverse

                    return self.model.matrix(nearest, other) @ self.matrix(probe, nearest)

            case (0, 1):

                forward = int(mod(probe + 1, self.size))
                while not self.flag[forward]:
                    forward = int(mod(forward + 1, self.size))
                time_forward = self.model.time[forward]
                if forward < probe: forward += self.model.size

                inverse = int(mod(probe - 1, self.size))
                while not self.flag[inverse]:
                    inverse = int(mod(inverse - 1, self.size))
                time_inverse = self.model.time[inverse]
                if inverse > probe: inverse -= self.model.size

                if close == 'forward':
                    return self.matrix(forward, other) @ self.model.matrix(probe, forward)

                if close == 'inverse':
                    return self.matrix(inverse, other) @ self.model.matrix(probe, inverse)

                if close == 'nearest':

                    delta_forward = abs(time_forward - self.model.time[other])
                    delta_inverse = abs(time_inverse - self.model.time[other])
                    nearest = forward if delta_forward <= delta_inverse else inverse

                    return self.matrix(nearest, other) @ self.model.matrix(probe, nearest)

            case (0, 0):

                forward_probe = int(mod(probe + 1, self.size))
                while not self.flag[forward_probe]:
                    forward_probe = int(mod(forward_probe + 1, self.size))
                time_forward_probe = self.model.time[forward_probe]
                if forward_probe < probe: forward_probe += self.model.size

                inverse_probe = int(mod(probe - 1, self.size))
                while not self.flag[inverse_probe]:
                    inverse_probe = int(mod(inverse_probe - 1, self.size))
                time_inverse_probe = self.model.time[inverse_probe]
                if inverse_probe > probe: inverse_probe -= self.model.size

                forward_other = int(mod(other + 1, self.size))
                while not self.flag[forward_other]:
                    forward_other = int(mod(forward_other + 1, self.size))
                time_forward_other = self.model.time[forward_other]
                if forward_other < other: forward_other += self.model.size

                inverse_other = int(mod(other - 1, self.size))
                while not self.flag[inverse_other]:
                    inverse_other = int(mod(inverse_other - 1, self.size))
                time_inverse_other = self.model.time[inverse_other]
                if inverse_other > other: inverse_other -= self.model.size

                if close == 'forward':
                    return self.model.matrix(forward_other, other) @ self.matrix(forward_probe, forward_other) @ self.model.matrix(probe, forward_probe)

                if close == 'inverse':
                    return self.model.matrix(inverse_other, other) @ self.matrix(inverse_probe, inverse_other) @ self.model.matrix(probe, inverse_probe)

                if close == 'nearest':

                    delta_forward_probe = abs(time_forward_probe - self.model.time[probe])
                    delta_inverse_probe = abs(time_inverse_probe - self.model.time[probe])
                    nearest_probe = forward_probe if delta_forward_probe <= delta_inverse_probe else inverse_probe

                    delta_forward_other = abs(time_forward_other - self.model.time[other])
                    delta_inverse_other = abs(time_inverse_other - self.model.time[other])
                    nearest_other = forward_other if delta_forward_other <= delta_inverse_other else inverse_other

                    return self.model.matrix(nearest_other, other) @ self.matrix(nearest_probe, nearest_other) @ self.model.matrix(probe, nearest_probe)


    def phase_advance(self,
                      probe:torch.Tensor,
                      other:torch.Tensor,
                      **kwargs) -> torch.Tensor:
        """
        Compute x & y phase advance between probe and other.

        Parameters
        ----------
        probe: torch.Tensor
            probe locations
        other: torch.Tensor
            other locations
        kwargs:
            passed to Decomposition.phase_advance

        Returns
        -------
        values & errors (torch.Tensor)

        """
        if isinstance(probe, int):
            probe = torch.tensor([probe], dtype=torch.int64, device=self.device)

        if isinstance(probe, str):
            probe = torch.tensor([self.model.name.index(probe)], dtype=torch.int64, device=self.device)

        if isinstance(other, int):
            other = torch.tensor([other], dtype=torch.int64, device=self.device)

        if isinstance(other, str):
            other = torch.tensor([self.model.name.index(other)], dtype=torch.int64, device=self.device)

        mux, sigma_mux = Decomposition.phase_advance(probe, other, self.table.nux, self.fx,
                                                     sigma_frequency=self.table.sigma_nux, sigma_phase=self.sigma_fx, **kwargs)

        muy, sigma_muy = Decomposition.phase_advance(probe, other, self.table.nuy, self.fy,
                                                     sigma_frequency=self.table.sigma_nuy, sigma_phase=self.sigma_fy, **kwargs)

        return torch.stack([torch.stack([mux, sigma_mux]), torch.stack([muy, sigma_muy])])


    def get_momenta(self,
                    start:int,
                    count:int,
                    probe:int,
                    other:int,
                    matrix:Callable[[int, int], torch.Tensor]) -> torch.Tensor:
        """
        Compute x & y momenta at the probe monitor location using single other monitor location.

        Note, self.table is expected to have x & y attributes

        Parameters
        ----------
        start: int
            start index
        count: int
            count length
        probe: int
            probe monitor location index
        other: int
            other monitor location index
        matrix: Callable[[int, int], torch.Tensor]
            transport matrix generator between locations
            self.model.matrix
            self.matrix
            self.matrix_virtual

        Returns
        -------
        (qx, px, qy, py) at the probe from start to start + count (torch.Tensor)

        """
        shift = (other - int(mod(other, self.model.monitor_count))) // self.model.monitor_count
        other = int(mod(other, self.model.monitor_count))

        start_probe = start
        start_other = start + shift

        qx1 = self.table.x[probe, start_probe:start_probe + count]
        qx2 = self.table.x[other, start_other:start_other + count]
        qy1 = self.table.y[probe, start_probe:start_probe + count]
        qy2 = self.table.y[other, start_other:start_other + count]

        index_probe = self.model.monitor_index[probe]
        index_other = self.model.monitor_index[other] + shift*self.model.size

        matrix = matrix(index_probe, index_other)

        px1, py1 = momenta(matrix, qx1, qx2, qy1, qy2)

        return torch.stack([qx1, px1, qy1, py1])


    def get_momenta_range(self,
                    start:int,
                    count:int,
                    probe:int,
                    limit:int,
                    matrix:Callable[[int, int], torch.Tensor]) -> torch.Tensor:
        """
        Compute x & y momenta at the probe monitor location using range of monitor locations around probed monitor location (average momenta).

        Note, self.table is expected to have x & y attributes

        Parameters
        ----------
        start: int
            start index
        count: int
            count length
        probe: int
            probe monitor location index
        limit: int
            range limit
        matrix: Callable[[int, int], torch.Tensor]
            transport matrix generator between locations
            self.model.matrix
            self.matrix
            self.matrix_virtual

        Returns
        -------
        (qx, px, qy, py) at the probe from start to start + count (torch.Tensor)

        """
        result = []
        others = [probe + index for index in range(-limit, limit + 1) if index != 0]
        for other in others:
            result.append(self.get_momenta(start, count, probe, other, matrix))
        return torch.stack(result).mean(0)


    def get_momenta_lstsq(self,
                          start:int,
                          count:int,
                          probe:int,
                          limit:int,
                          matrix:Callable[[int, int], torch.Tensor],
                          *,
                          phony:bool=False,
                          forward:bool=True,
                          inverse:bool=True) -> torch.Tensor:
        """
        Compute x & y coordinates and momenta at the probe monitor location using range of monitor locations around probed monitor location (lstsq fit).

        Note, self.table is expected to have x & y attributes

        Note, probe is treated as location index if phony=True
        In this case data at monitors in range around probe is used to fit coordinates and momenta
        Data at location is not used even if location index is a monitor

        Parameters
        ----------
        start: int
            start index
        count: int
            count length
        probe: int
            probe monitor location index
        limit: int
            range limit
        matrix: Callable[[int, int], torch.Tensor]
            transport matrix generator between locations
            self.model.matrix
            self.matrix
            self.matrix_virtual
        phony: bool
            flag to treat probe as virtual
        inverse: bool
            flag to move in the inverse direction
        forward: bool
            flag to move in the forward direction

        Returns
        -------
        (qx, px, qy, py) at the probe from start to start + count (torch.Tensor)

        """
        if limit <= 0:
            raise ValueError(f'TWISS: expected limit >= 1')

        if not phony:
            others = [probe + index for index in range(-limit*inverse, limit*forward + 1)]
            shifts = [(other - int(mod(other, self.model.monitor_count))) // self.model.monitor_count for other in others]
            others = [int(mod(other, self.model.monitor_count)) for other in others]
        else:
            others = [other for other in generate_other(probe, limit, self.flag, forward=forward, inverse=inverse)]
            shifts = [(other - int(mod(other, self.model.size))) // self.model.size for other in others]
            others = [int(mod(other, self.model.size)) for other in others]
            others = [other - sum(1 for virtual in self.model.virtual_index if virtual <= other) for other in others]

        A = []
        for other, shift in zip(others, shifts):
            index_probe = self.model.monitor_index[probe] if not phony else probe
            index_other = self.model.monitor_index[other] + shift*self.model.size
            transport = matrix(index_probe, index_other)
            RX, _, RY, _ = transport
            A.append(RX)
            A.append(RY)
        A = torch.stack(A)

        B = []
        for other, shift in zip(others, shifts):
            B.append(self.table.x[other, start + shift:start + shift + count])
            B.append(self.table.y[other, start + shift:start + shift + count])
        B = torch.stack(B).T

        return torch.linalg.lstsq(A, B.T).solution


    @staticmethod
    def invariant_objective(beta:torch.Tensor,
                            X:torch.Tensor,
                            normalization:Callable[[torch.Tensor], torch.Tensor],
                            product:bool) -> torch.Tensor:
        """
        Evaluate invariant objective.

        Parameters
        ----------
        beta: torch.Tensor
            [ix, iy, n11, n33, n21, n43, n13, n31, n14, n41]
            [ix, iy, ax, bx, ay, by]
            [ix, iy, a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2]
        X: torch.Tensor
            [[..., qx_i, ...], [..., px_i, ...], [..., qy_i, ...], [..., py_i, ...]]
        normalization: Callable[[torch.Tensor], torch.Tensor]
            parametric_normal
            cs_normal
            lb_normal
        product: bool
            flag to use product instead of sum

        Returns
        -------
        objective value for each qx_i, px_i, qy_i, py_i (torch.Tensor)

        """
        ix, iy, *twiss = beta
        qx, px, qy, py = X
        normal = normalization(*twiss, dtype=beta.dtype, device=beta.device)
        jx, jy = invariant(normal, torch.stack([qx, px, qy, py]).T)
        objective = ((jx - ix)**2 * (jy - iy)**2).sqrt() if product else ((jx - ix)**2 + (jy - iy)**2).sqrt()
        return objective


    @staticmethod
    def invariant_objective_fixed(beta:torch.Tensor,
                                  X:torch.Tensor,
                                  normalization:Callable[[torch.Tensor], torch.Tensor],
                                  product:bool) -> torch.Tensor:
        """
        Evaluate invariant objective (fixed invariants).

        Parameters
        ----------
        beta: torch.Tensor
            [n11, n33, n21, n43, n13, n31, n14, n41]
            [ax, bx, ay, by]
            [a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2]
        X: torch.Tensor
            [[..., ix_i, ...], [..., iy_i, ...], [..., qx_i, ...], [..., px_i, ...], [..., qy_i, ...], [..., py_i, ...]]
        normalization: Callable[[torch.Tensor], torch.Tensor]
            parametric_normal
            cs_normal
            lb_normal
        product: bool
            flag to use product instead of sum

        Returns
        -------
        objective value for each ix_i, iy_i, qx_i, px_i, qy_i, py_i (torch.Tensor)

        """
        normal = normalization(*beta, dtype=beta.dtype, device=beta.device)
        ix, iy, qx, px, qy, py = X
        jx, jy = invariant(normal, torch.stack([qx, px, qy, py]).T)
        objective = ((jx - ix)**2 * (jy - iy)**2).sqrt() if product else ((jx - ix)**2 + (jy - iy)**2).sqrt()
        return objective


    def fit_objective(self,
                      length:int,
                      twiss:torch.Tensor,
                      qx:torch.Tensor,
                      px:torch.Tensor,
                      qy:torch.Tensor,
                      py:torch.Tensor,
                      normalization:Callable[[torch.Tensor], torch.Tensor],
                      *,
                      product:bool=True,
                      jacobian:bool=False,
                      ix:float=None,
                      iy:float=None,
                      count:int=512,
                      fraction:float=0.75,
                      sigma:float=0.0,
                      n_jobs:int=6,
                      **kwargs) -> tuple:
        """
        Fit invariant objective.

        Parameters
        ----------
        length: int
            maximum sample length to use
        twiss: torch.Tensor
            [n11, n33, n21, n43, n13, n31, n14, n41]
            [ax, bx, ay, by]
            [a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2]
        qx, px, qy, py: torch.Tensor
            qx, px, qy, py
        normalization: Callable[[torch.Tensor], torch.Tensor]
            parametric_normal
            cs_normal
            lb_normal
        product: bool
            flag to use product instead of sum
        jacobian: bool
            flag to compute Jacobian
        ix, iy: float
            initial x & y invariant values
        count: int
            number of samples
        fraction: float
            sample length fraction
        sigma: float
            twiss perturbation sigma
        n_jobs: int
            number of jobs
        **kwargs:
            passed to leastsq

        Returns
        -------
        parameters and errors for each sample (tuple)

        """
        size = int(fraction*length)

        def objective(beta, X, normalization, product):
            beta = torch.tensor(beta, dtype=self.dtype, device=self.device)
            return self.invariant_objective(beta, X, normalization, product).cpu().numpy()

        def derivative(beta, X, normalization, product):
            beta = torch.tensor(beta, dtype=self.dtype, device=self.device)
            return (torch.func.jacrev(self.invariant_objective)(beta, X, normalization, product)).cpu().numpy()

        def task(sigma):

            index = torch.randint(0, length, (size, ), dtype=torch.int64, device=self.device)

            QX = qx[index]
            PX = px[index]
            QY = qy[index]
            PY = py[index]

            normal = normalization(*twiss, dtype=self.dtype, device=self.device)

            jx, jy = invariant(normal, torch.stack([QX, PX, QY, PY]).T)
            jx = jx.mean().item()
            jy = jy.mean().item()

            if ix is not None:
                jx = ix

            if iy is not None:
                jy = iy

            beta = numpy.array([jx, jy, *(twiss + sigma*torch.randn_like(twiss)).cpu().numpy()])
            X = torch.stack([QX, PX, QY, PY])

            if jacobian:
                fit, cov, *_ = leastsq(objective, beta, args=(X, normalization, product), Dfun=derivative, full_output=1, **kwargs)
            else:
                fit, cov, *_ = leastsq(objective, beta, args=(X, normalization, product), full_output=1, **kwargs)

            res = (objective(fit, X, normalization, product)**2).sum()/(size - len(fit))
            err = numpy.zeros_like(fit)

            if cov is not None:
                cov = cov*res
                err = numpy.sqrt(numpy.abs(numpy.diag(cov)))

            return fit, err

        result = Parallel(n_jobs=n_jobs)(delayed(task)(sigma) for _ in range(count))

        value, error = numpy.array(result).swapaxes(0, 1)
        value = torch.tensor(value.T, dtype=self.dtype, device=self.device)
        error = torch.tensor(error.T, dtype=self.dtype, device=self.device)

        return value, error


    def fit_objective_fixed(self,
                            length:int,
                            twiss:torch.Tensor,
                            ix:torch.Tensor,
                            iy:torch.Tensor,
                            qx:torch.Tensor,
                            px:torch.Tensor,
                            qy:torch.Tensor,
                            py:torch.Tensor,
                            normalization:Callable[[torch.Tensor], torch.Tensor],
                            *,
                            product:bool=True,
                            jacobian:bool=False,
                            count:int=512,
                            fraction:float=0.75,
                            sigma:float=0.0,
                            n_jobs:int=6,
                            **kwargs) -> tuple:
        """
        Fit invariant objective (fixed invariants).

        Parameters
        ----------
        length: int
            maximum sample length to use
        twiss: torch.Tensor
            [n11, n33, n21, n43, n13, n31, n14, n41]
            [ax, bx, ay, by]
            [a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2]
        ix, iy: torch.Tensor
            fixed invariant values
        qx, px, qy, py: torch.Tensor
            qx, px, qy, py
        normalization: Callable[[torch.Tensor], torch.Tensor]
            parametric_normal
            cs_normal
            lb_normal
        product: bool
            flag to use product instead of sum
        jacobian: bool
            flag to compute Jacobian
        count: int
            number of samples
        fraction: float
            sample length fraction
        sigma: float
            twiss perturbation sigma
        n_jobs: int
            number of jobs
        **kwargs:
            passed to leastsq

        Returns
        -------
        parameters and errors for each sample (tuple)

        """
        size = int(fraction*length)

        def objective(beta, X, normalization, product):
            beta = torch.tensor(beta, dtype=self.dtype, device=self.device)
            return self.invariant_objective_fixed(beta, X, normalization, product).cpu().numpy()

        def derivative(beta, X, normalization, product):
            beta = torch.tensor(beta, dtype=self.dtype, device=self.device)
            return (torch.func.jacrev(self.invariant_objective_fixed)(beta, X, normalization, product)).cpu().numpy()

        def task(sigma):

            index = torch.randint(0, length, (size, ), dtype=torch.int64, device=self.device)

            IX = ix[index]
            IY = iy[index]
            QX = qx[index]
            PX = px[index]
            QY = qy[index]
            PY = py[index]

            beta = numpy.array([*(twiss + sigma*torch.randn_like(twiss)).cpu().numpy()])
            X = torch.stack([IX, IY, QX, PX, QY, PY])

            if jacobian:
                fit, cov, *_ = leastsq(objective, beta, args=(X, normalization, product), Dfun=derivative, full_output=1, **kwargs)
            else:
                fit, cov, *_ = leastsq(objective, beta, args=(X, normalization, product), full_output=1, **kwargs)

            res = (objective(fit, X, normalization, product)**2).sum()/(size - len(fit))
            err = numpy.zeros_like(fit)

            if cov is not None:
                cov = cov*res
                err = numpy.sqrt(numpy.abs(numpy.diag(cov)))

            return fit, err

        result = Parallel(n_jobs=n_jobs)(delayed(task)(sigma) for _ in range(count))

        value, error = numpy.array(result).swapaxes(0, 1)
        value = torch.tensor(value.T, dtype=self.dtype, device=self.device)
        error = torch.tensor(error.T, dtype=self.dtype, device=self.device)

        return value, error


    def get_twiss_from_data(self,
                            start:int,
                            length:int,
                            normalization:Callable[[torch.Tensor], torch.Tensor],
                            matrix:Callable[[int, int], torch.Tensor],
                            *,
                            twiss:torch.Tensor=None,
                            method:str='pair',
                            limit:int=1,
                            index:list[int]=None,
                            phony:bool=False,
                            inverse:bool=True,
                            forward:bool=True,
                            product:bool=True,
                            jacobian:bool=False,
                            count:int=256,
                            fraction:float=0.75,
                            ix:float=None,
                            iy:float=None,
                            sigma:float=0.0,
                            n_jobs:int=6,
                            verbose:bool=False,
                            **kwargs) -> torch.Tensor:
        """
        Estimate twiss from signals (fit linear invariants).

        Note, invariant values can be fixed

        Note, 'lstsq' method can be used to estimate orbits at virtual locations
        If phony = True, twiss is only estimated at virtual locations

        Parameters
        ----------
        start: int
            first turn index
        length: int
            maximum sample length to use
        normalization: Callable[[torch.Tensor], torch.Tensor]
            parametric_normal
            cs_normal
            lb_normal
        matrix: Callable[[int, int], torch.Tensor]
            transport matrix generator between locations
            self.model.matrix
            self.matrix
            self.matrix_virtual
        twiss: torch.Tensor
            [..., [n11, n33, n21, n43, n13, n31, n14, n41], ...]
            [..., [ax, bx, ay, by], ...]
            [..., [a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2], ...]
            initial guess for twiss parameters for each location
            if None, model values are used
        method: str
            momenta computation method 'pair' or 'range' or 'lstsq'
        limit: int
            -1 or 1 (or relative shift value) for 'pair', >= 1 for 'range' or 'lstsq'
        index: list[int]
            list of location indices
        phony: bool
            (lstsq) flag to treat probe as virtual
        inverse: bool
            (lstsq) flag to move in the inverse direction
        forward: bool
            (lstsq) flag to move in the forward direction
        product: bool
            flag to use product instead of sum
        jacobian: bool
            flag to compute Jacobian
        count: int
            number of samples
        fraction: float
            sample length fraction
        ix & iy: float
            fixed invariant values
        sigma: float
            twiss perturbation sigma
        n_jobs: int
            number of jobs
        verbose: bool
            verbose flag
        **kwargs:
            passed to leastsq

        Returns
        -------
        estimated parameters and errors (torch.Tensor)

        """
        if phony == True and method != 'lstsq':
            raise Exception(f'TWISS: phony = True only works with method = "lstsq"')

        result = []

        locations = range(self.model.monitor_count) if not phony else self.model.virtual_index
        if index != None:
            locations = index
        for location in locations:

            if verbose:
                print(f'{location + 1}/{self.model.monitor_count}')

            probe = location
            index_probe = self.model.monitor_index[probe] if not phony else probe

            if twiss != None:
                guess = twiss[probe]
            else:
                guess = self.model.normal[index_probe]
                if normalization == parametric_normal:
                    guess = guess[[0, 2, 1, 3, 0, 2, 0, 3], [0, 2, 0, 2, 2, 0, 3, 0]]
                elif normalization == cs_normal:
                    guess = wolski_to_cs(normal_to_wolski(guess.unsqueeze(0)).squeeze())
                elif normalization == lb_normal:
                    guess = wolski_to_lb(normal_to_wolski(guess.unsqueeze(0)).squeeze())

            if method == 'pair':
                qx, px, qy, py = self.get_momenta(start, length, probe, probe + limit, matrix)

            if method == 'range':
                qx, px, qy, py = self.get_momenta_range(start, length, probe, limit, matrix)

            if method == 'lstsq':
                qx, px, qy, py = self.get_momenta_lstsq(start, length, probe, limit, matrix, phony=phony, inverse=inverse, forward=forward)

            if ix != None and iy != None:
                ix = ix*torch.ones_like(qx)
                iy = iy*torch.ones_like(qy)
                fit = self.fit_objective_fixed(length, guess, ix, iy, qx, px, qy, py, normalization, product=product, jacobian=jacobian, count=count, fraction=fraction, sigma=sigma, n_jobs=n_jobs, **kwargs)
            else:
                fit = self.fit_objective(length, guess, qx, px, qy, py, normalization, product=product, jacobian=jacobian, count=count, fraction=fraction, sigma=sigma, n_jobs=n_jobs, **kwargs)

            result.append(torch.stack(fit))

        return torch.stack(result)


    def get_invariant(self,
                      ix:torch.Tensor,
                      iy:torch.Tensor,
                      sx:torch.Tensor,
                      sy:torch.Tensor,
                      *,
                      cut:float=5.0,
                      use_error:bool=True,
                      center_estimator:Callable[[torch.Tensor], torch.Tensor]=median,
                      spread_estimator:Callable[[torch.Tensor], torch.Tensor]=biweight_midvariance) -> dict:
        """
        Compute invariants from get_twiss_from_data output.

        First, for each location threshold filtering is performed over samples and weighted center and spread are estimated
        Next, procedure is repeated over locations

        Note, if thresholding is not required, set cut to a large value

        Parameters
        ----------
        ix & iy: torch.Tensor
            invariant values for each sample at each location (nlocations, nsamples)
        sx & sy: torch.Tensor
            invariant errors for each sample at each location (nlocations, nsamples)
        cut: float
            threshold cut value for data cleaning
        use_error: bool
            flag to threshold using error values
        center_estimator: Callable[[torch.Tensor], torch.Tensor]
            (robust) center estimator
        spread_estimator:Callable[[torch.Tensor], torch.Tensor]
            (robust) spread/variance estimator

        Returns
        -------
        dict_keys(['ix_mask', 'iy_mask', 'sx_mask', 'sy_mask', 'ix_weight', 'ix_center', 'ix_spread', 'iy_weight', 'iy_center', 'iy_spread', 'ix_cut', 'sx_cut', 'ix_value', 'ix_error', 'iy_cut', 'sy_cut', 'iy_value', 'iy_error'])

        """
        result = {}

        cut = torch.tensor(cut, dtype=self.dtype, device=self.device)

        data = standardize(ix, center_estimator=center_estimator, spread_estimator=spread_estimator)
        ix_mask = threshold(data, -cut, +cut)
        result['ix_mask'] = ix_mask

        data = standardize(iy, center_estimator=center_estimator, spread_estimator=spread_estimator)
        iy_mask = threshold(data, -cut, +cut)
        result['iy_mask'] = iy_mask

        mask = ix_mask*iy_mask

        if use_error:

            data = standardize(sx, center_estimator=center_estimator, spread_estimator=spread_estimator)
            sx_mask = threshold(data, -cut, +cut)
            result['sx_mask'] = sx_mask

            data = standardize(sy, center_estimator=center_estimator, spread_estimator=spread_estimator)
            sy_mask = threshold(data, -cut, +cut)
            result['sy_mask'] = sy_mask

            mask *= sx_mask*sy_mask

        ix_weight = (1/sx**2).nan_to_num(posinf=0.0)
        ix_weight /= ix_weight.sum(-1, keepdims=True)
        ix_center = weighted_mean(ix, mask*ix_weight)
        ix_spread = weighted_variance(ix, mask*ix_weight).sqrt()
        result['ix_weight'] = ix_weight
        result['ix_center'] = ix_center
        result['ix_spread'] = ix_spread

        iy_weight = (1/sy**2).nan_to_num(posinf=0.0)
        iy_weight /= iy_weight.sum(-1, keepdims=True)
        iy_center = weighted_mean(iy, mask*iy_weight)
        iy_spread = weighted_variance(iy, mask*iy_weight).sqrt()
        result['iy_weight'] = iy_weight
        result['iy_center'] = iy_center
        result['iy_spread'] = iy_spread

        data = standardize(ix_center, center_estimator=center_estimator, spread_estimator=spread_estimator)
        ix_cut = threshold(data, -cut, +cut)
        data = standardize(ix_spread, center_estimator=center_estimator, spread_estimator=spread_estimator)
        sx_cut = threshold(data, -cut, +cut)
        ix_value = weighted_mean(ix_center, ix_cut*sx_cut*1/ix_spread**2)
        ix_error = weighted_variance(ix_center, ix_cut*sx_cut*1/ix_spread**2).sqrt()
        result['ix_cut'] = ix_cut
        result['sx_cut'] = sx_cut
        result['ix_value'] = ix_value.squeeze()
        result['ix_error'] = ix_error.squeeze()

        data = standardize(iy_center, center_estimator=center_estimator, spread_estimator=spread_estimator)
        iy_cut = threshold(data, -cut, +cut)
        data = standardize(iy_spread, center_estimator=center_estimator, spread_estimator=spread_estimator)
        sy_cut = threshold(data, -cut, +cut)
        iy_value = weighted_mean(iy_center, iy_cut*sy_cut*1/iy_spread**2)
        iy_error = weighted_variance(iy_center, iy_cut*sy_cut*1/iy_spread**2).sqrt()
        result['iy_cut'] = iy_cut
        result['sy_cut'] = sy_cut
        result['iy_value'] = iy_value.squeeze()
        result['iy_error'] = iy_error.squeeze()

        return result


    @staticmethod
    def ratio_objective(beta:torch.Tensor,
                        X:torch.Tensor,
                        window:torch.Tensor,
                        nux:torch.Tensor,
                        nuy:torch.Tensor,
                        normalization:Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Evaluate ratio objective.

        Parameters
        ----------
        beta: torch.Tensor
            [n11, n33, n21, n43, n13, n31, n14, n41]
            [ax, bx, ay, by]
            [a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2]
        X: torch.Tensor
            [[..., qx_i, ...], [..., px_i, ...], [..., qy_i, ...], [..., py_i, ...]]
        window: torch.Tensor
            window to apply
        nux & nuy: torch.Tensor
            fractional x & y tune values
        normalization: Callable[[torch.Tensor], torch.Tensor]
            parametric_normal
            cs_normal
            lb_normal

        Returns
        -------
        objective value (torch.Tensor)

        """
        normal = normalization(*beta, dtype=beta.dtype, device=beta.device)

        qx, px, qy, py = X
        qx, px, qy, py = normal.inverse() @ torch.stack([qx, px, qy, py])

        wx = window*(qx + 1j*px)
        wy = window*(qy + 1j*py)

        size = len(window)
        time = 1j*2.0*numpy.pi*torch.linspace(0, size - 1, size, dtype=beta.dtype, device=beta.device)

        NUX = 1.0 - nux
        NUY = 1.0 - nuy

        axx = (wx*torch.exp(nux*time)).sum().abs()
        bxx = (wx*torch.exp(NUX*time)).sum().abs()
        axy = (wx*torch.exp(nuy*time)).sum().abs()
        bxy = (wx*torch.exp(NUY*time)).sum().abs()

        ayx = (wy*torch.exp(nux*time)).sum().abs()
        byx = (wy*torch.exp(NUX*time)).sum().abs()
        ayy = (wy*torch.exp(nuy*time)).sum().abs()
        byy = (wy*torch.exp(NUY*time)).sum().abs()

        return (bxx + axy + bxy)/axx + (byy + ayx + byx)/ayy


    def get_twiss_from_ratio(self,
                             start:int,
                             length:int,
                             window:torch.Tensor,
                             nux:torch.Tensor,
                             nuy:torch.Tensor,
                             normalization:Callable[[torch.Tensor], torch.Tensor],
                             matrix:Callable[[int, int], torch.Tensor],
                             *,
                             step:int=1,
                             method:str='pair',
                             limit:int=1,
                             index:list[int]=None,
                             phony:bool=False,
                             inverse:bool=True,
                             forward:bool=True,
                             twiss:torch.Tensor=None,
                             n_jobs:int=6,
                             verbose:bool=False,
                             **kwargs) -> torch.Tensor:
        """
        Estimate twiss from ratio.

        Note, minimize method is fixed to 'BFGS'

        Note, 'lstsq' method can be used to estimate orbits at virtual locations
        If phony = True, twiss is only estimated at virtual locations

        Parameters
        ----------
        start: int
            first turn index
        length: int
            maximum sample length to use
        window: torch.Tensor
            window to apply (sample size if defined by window length)
        nux & nuy: torch.Tensor
            fractional x & y tune values
        normalization: Callable[[torch.Tensor], torch.Tensor]
            parametric_normal
            cs_normal
            lb_normal
        matrix: Callable[[int, int], torch.Tensor]
            transport matrix generator between locations
            self.model.matrix
            self.matrix
            self.matrix_virtual
        step: int
            shift step
        method: str
            momenta computation method 'pair' or 'range' or 'lstsq'
        limit: int
            -1 or 1 (or relative shift value) for 'pair', >= 1 for 'range' or 'lstsq'
        index: list[int]
            list of location indices
        phony: bool
            (lstsq) flag to treat probe as virtual
        inverse: bool
            (lstsq) flag to move in the inverse direction
        forward: bool
            (lstsq) flag to move in the forward direction
        twiss: torch.Tensor
            [..., [n11, n33, n21, n43, n13, n31, n14, n41], ...]
            [..., [ax, bx, ay, by], ...]
            [..., [a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2], ...]
            initial guess for twiss parameters for each location
            if None, model values are used
        n_jobs: int
            number of jobs
        verbose: bool
            verbose flag
        **kwargs:
            passed to minimize

        Returns
        -------
        estimated parameters and errors (torch.Tensor)

        """
        if phony == True and method != 'lstsq':
            raise Exception(f'TWISS: phony = True only works with method = "lstsq"')

        result = []

        size = 1 + (length - len(window))//step

        def objective(beta, X, window, nux, nuy, normalization):
            beta = torch.tensor(beta, dtype=window.dtype, device=window.device)
            return self.ratio_objective(beta, X, window, nux, nuy, normalization).cpu().numpy()

        locations = range(self.model.monitor_count) if not phony else self.model.virtual_index
        if index != None:
            locations = index
        for location in locations:

            if verbose:
                print(f'{location + 1}/{self.model.monitor_count}')

            probe = location
            index_probe = self.model.monitor_index[probe] if not phony else probe

            if twiss != None:
                guess = twiss[probe]
            else:
                guess = self.model.normal[index_probe]
                if normalization == parametric_normal:
                    guess = guess[[0, 2, 1, 3, 0, 2, 0, 3], [0, 2, 0, 2, 2, 0, 3, 0]]
                elif normalization == cs_normal:
                    guess = wolski_to_cs(normal_to_wolski(guess.unsqueeze(0)).squeeze())
                elif normalization == lb_normal:
                    guess = wolski_to_lb(normal_to_wolski(guess.unsqueeze(0)).squeeze())

            guess = guess.cpu().numpy()

            if method == 'pair':
                qx, px, qy, py = self.get_momenta(start, length, probe, probe + limit, matrix)

            if method == 'range':
                qx, px, qy, py = self.get_momenta_range(start, length, probe, limit, matrix)

            if method == 'lstsq':
                qx, px, qy, py = self.get_momenta_lstsq(start, length, probe, limit, matrix, phony=phony, inverse=inverse, forward=forward)

            X = torch.stack([qx, px, qy, py])

            def task(index):
                out = minimize(objective, guess, args=(X[:, index*step : index*step + len(window)], window, nux, nuy, normalization), method='BFGS', **kwargs)
                value = torch.tensor(out.x, dtype=self.dtype, device=self.device)
                error = torch.tensor(numpy.sqrt(numpy.diag(out.hess_inv)), dtype=self.dtype, device=self.device)
                return torch.stack([value, error]).T

            result.append(torch.stack(Parallel(n_jobs=n_jobs)(delayed(task)(i) for i in range(size))))

        return torch.stack(result).swapaxes(1, -1)


    def get_twiss_from_matrix(self,
                              start:int,
                              length:int,
                              matrix:Callable[[int, int], torch.Tensor],
                              *,
                              power:int=1,
                              method:str='pair',
                              limit:int=1,
                              index:list[int]=None,
                              phony:bool=False,
                              inverse:bool=True,
                              forward:bool=True,
                              count:int=256,
                              fraction:float=0.75,
                              verbose:bool=False) -> torch.tensor:
        """
        Estimate twiss from n-turn matrix.

        Note, return estimated tunes and free elements of normalization matrix for each location and each sample

        Note, 'lstsq' method can be used to estimate orbits at virtual locations
        If phony = True, twiss is only estimated at virtual locations

        Parameters
        ----------
        start: int
            first turn index
        length: int
            maximum sample length to use
        matrix: Callable[[int, int], torch.Tensor]
            transport matrix generator between locations
            self.model.matrix
            self.matrix
            self.matrix_virtual
        power: int
            matrix power
        method: str
            momenta computation method 'pair' or 'range' or 'lstsq'
        limit: int
            -1 or 1 (or relative shift value) for 'pair', >= 1 for 'range' or 'lstsq'
        index: list[int]
            list of location indices
        phony: bool
            (lstsq) flag to treat probe as virtual
        inverse: bool
            (lstsq) flag to move in the inverse direction
        forward: bool
            (lstsq) flag to move in the forward direction
        count: int
            number of samples
        fraction: float
            sample length fraction
        verbose: bool
            verbose flag

        Returns
        -------
        estimated parameters (torch.Tensor)

        """
        if phony == True and method != 'lstsq':
            raise Exception(f'TWISS: phony = True only works with method = "lstsq"')

        result = []

        size = int(fraction*length)

        empty = torch.zeros(10, dtype=self.dtype, device=self.device)

        locations = range(self.model.monitor_count) if not phony else self.model.virtual_index
        if index != None:
            locations = index
        for location in locations:

            if verbose:
                print(f'{location + 1}/{self.model.monitor_count}')

            probe = location

            if method == 'pair':
                qx, px, qy, py = self.get_momenta(start, length, probe, probe + limit, matrix)

            if method == 'range':
                qx, px, qy, py = self.get_momenta_range(start, length, probe, limit, matrix)

            if method == 'lstsq':
                qx, px, qy, py = self.get_momenta_lstsq(start, length, probe, limit, matrix, phony=phony, inverse=inverse, forward=forward)

            table = torch.randint(size - power, (count, size), dtype=torch.int64, device=self.device)

            box = []

            for index in table:

                index_n = index
                index_m = index_n + power

                qx_n, px_n, qy_n, py_n = qx[index_n], px[index_n], qy[index_n], py[index_n]
                qx_m, px_m, qy_m, py_m = qx[index_m], px[index_m], qy[index_m], py[index_m]

                A = torch.stack([qx_n, px_n, qy_n, py_n]).T
                B = torch.stack([qx_m, px_m, qy_m, py_m]).T

                transport = to_symplectic(torch.linalg.lstsq(A, B).solution.T)
                tune, normal, _ = twiss_compute(transport)
                if tune is not None:
                    box.append(torch.cat([tune, normal[[0, 2, 1, 3, 0, 2, 0, 3], [0, 2, 0, 2, 2, 0, 3, 0]]]))
                else:
                    box.append(empty)

            result.append(torch.stack(box).T)

        return torch.stack(result)


    def get_twiss_virtual_uncoupled(self,
                                    probe:int,
                                    *,
                                    limit:int=1,
                                    inverse:bool=True,
                                    forward:bool=True,
                                    use_phase:bool=False,
                                    bootstrap:bool=True,
                                    count:int=256,
                                    ax:torch.Tensor=None,
                                    bx:torch.Tensor=None,
                                    ay:torch.Tensor=None,
                                    by:torch.Tensor=None,
                                    sigma_ax:torch.Tensor=None,
                                    sigma_bx:torch.Tensor=None,
                                    sigma_ay:torch.Tensor=None,
                                    sigma_by:torch.Tensor=None) -> tuple:
        """
        Estimate CS twiss at (virtual) location.

        Note, probe location can be virtual or monitor
        Note, when use_phase flag is True, virtual phase is used

        Parameters
        ----------
        probe: int
            (virtual) location index
        limit: int
            range limit
        inverse: bool
            flag to use other only in inverse direction
        forward: bool
            flag to use other only in forward direction
        use_phase: bool
            flag to use (virtual) phase
        bootstrap: bool
            flag to bootstrap other
        count: int
            number of bootstrap samples
        ax, bx, ay, by: torch.Tensor
            CS twiss
        sigma_ax, sigma_bx, sigma_ay, sigma_by: torch.Tensor
            CS twiss errors

        Returns
        -------
        CS twiss and errors at probe (tuple)

        """
        others = [other for other in generate_other(probe, limit, self.flag, inverse=inverse, forward=forward)]

        ax = self.ax if ax == None else ax
        bx = self.bx if bx == None else bx
        ay = self.ay if ay == None else ay
        by = self.by if by == None else by

        sigma_ax = self.sigma_ax if sigma_ax == None else sigma_ax
        sigma_bx = self.sigma_bx if sigma_bx == None else sigma_bx
        sigma_ay = self.sigma_ay if sigma_ay == None else sigma_ay
        sigma_by = self.sigma_by if sigma_by == None else sigma_by

        value, error = [], []

        for other in others:

            index = int(mod(other, self.size))

            if use_phase:
                (mux, sigma_mux), (muy, sigma_muy) = self.phase_advance(other, probe).squeeze()

            if bootstrap:
                table_ax = ax[index] + sigma_ax[index]*torch.randn(count, dtype=self.dtype, device=self.device)
                table_bx = bx[index] + sigma_bx[index]*torch.randn(count, dtype=self.dtype, device=self.device)
                table_ay = ay[index] + sigma_ay[index]*torch.randn(count, dtype=self.dtype, device=self.device)
                table_by = by[index] + sigma_by[index]*torch.randn(count, dtype=self.dtype, device=self.device)
                if use_phase:
                    table_mux = mux + sigma_mux*torch.randn(count, dtype=self.dtype, device=self.device)
                    table_muy = muy + sigma_muy*torch.randn(count, dtype=self.dtype, device=self.device)
            else:
                table_ax = ax[index].unsqueeze(0)
                table_bx = bx[index].unsqueeze(0)
                table_ay = ay[index].unsqueeze(0)
                table_by = by[index].unsqueeze(0)
                if use_phase:
                    table_mux = mux.unsqueeze(0)
                    table_muy = muy.unsqueeze(0)

            matrix = self.model.matrix(other, probe)

            table = []
            if use_phase:
                for value_ax, value_bx, value_ay, value_by, value_mux, value_muy in zip(table_ax, table_bx, table_ay, table_by, table_mux, table_muy):
                    normal = cs_normal(value_ax, value_bx, value_ay, value_by, dtype=self.dtype, device=self.device)
                    normal = matrix @ normal @ matrix_rotation(torch.stack([value_mux, value_muy]).unsqueeze(0)).squeeze().inverse()
                    normal = parametric_normal(*normal[[0, 2, 1, 3, 0, 2, 0, 3], [0, 2, 0, 2, 2, 0, 3, 0]], dtype=self.dtype, device=self.device)
                    wolski = normal_to_wolski(normal.unsqueeze(0)).squeeze()
                    table.append(wolski_to_cs(wolski))
            else:
                for value_ax, value_bx, value_ay, value_by in zip(table_ax, table_bx, table_ay, table_by):
                    wolski = cs_to_wolski(value_ax, value_bx, value_ay, value_by, dtype=self.dtype, device=self.device)
                    wolski = twiss_propagate(wolski, matrix.unsqueeze(0)).squeeze()
                    table.append(wolski_to_cs(wolski))

            table = torch.stack(table)
            if bootstrap:
                table_value = mean(table.T)
                table_error = variance(table.T).sqrt()
            else:
                table_value = table.squeeze()
                table_error = torch.stack([sigma_ax[index], sigma_bx[index], sigma_ay[index], sigma_bx[index]])

            value.append(table_value)
            error.append(table_error)

        value = torch.stack(value).T
        error = torch.stack(error).T
        value, error = weighted_mean(value, weight=(1.0/error**2).nan_to_num(posinf=1)), weighted_variance(value, weight=(1.0/error**2).nan_to_num(posinf=1)).sqrt().nan_to_num()

        return value, error


    def get_twiss_virtual_coupled(self,
                                  probe:int,
                                  *,
                                  limit:int=1,
                                  inverse:bool=True,
                                  forward:bool=True,
                                  use_phase:bool=False,
                                  bootstrap:bool=True,
                                  count:int=256,
                                  n11:torch.Tensor=None,
                                  n33:torch.Tensor=None,
                                  n21:torch.Tensor=None,
                                  n43:torch.Tensor=None,
                                  n13:torch.Tensor=None,
                                  n31:torch.Tensor=None,
                                  n14:torch.Tensor=None,
                                  n41:torch.Tensor=None,
                                  sigma_n11:torch.Tensor=None,
                                  sigma_n33:torch.Tensor=None,
                                  sigma_n21:torch.Tensor=None,
                                  sigma_n43:torch.Tensor=None,
                                  sigma_n13:torch.Tensor=None,
                                  sigma_n31:torch.Tensor=None,
                                  sigma_n14:torch.Tensor=None,
                                  sigma_n41:torch.Tensor=None) -> tuple:
        """
        Estimate free normalization matrix elements at (virtual) location.

        Note, probe location can be virtual or monitor
        Note, when use_phase flag is True, virtual phase is used

        Parameters
        ----------
        probe: int
            (virtual) location index
        limit: int
            range limit
        inverse: bool
            flag to use other only in inverse direction
        forward: bool
            flag to use other only in forward direction
        use_phase: bool
            flag to use (virtual) phase
        bootstrap: bool
            flag to bootstrap other
        count: int
            number of bootstrap samples
        n11, n33, n21, n43, n13, n31, n14, n41: torch.Tensor
            free normalization matrix elements
        sigma_n11, sigma_n33, sigma_n21, sigma_n43, sigma_n13, sigma_n31, sigma_n14, sigma_n41: torch.Tensor
            free normalization matrix elements errors

        Returns
        -------
        normalization matrix elements and errors at probe (tuple)

        """
        others = [other for other in generate_other(probe, limit, self.flag, inverse=inverse, forward=forward)]

        n11 = self.normal[:, 0, 0] if n11 == None else n11
        n33 = self.normal[:, 2, 2] if n33 == None else n33
        n21 = self.normal[:, 1, 0] if n21 == None else n21
        n43 = self.normal[:, 3, 2] if n43 == None else n43
        n13 = self.normal[:, 0, 2] if n13 == None else n13
        n31 = self.normal[:, 2, 0] if n31 == None else n31
        n14 = self.normal[:, 0, 3] if n14 == None else n14
        n41 = self.normal[:, 3, 0] if n41 == None else n41

        sigma_n11 = self.sigma_normal[:, 0, 0] if sigma_n11 == None else sigma_n11
        sigma_n33 = self.sigma_normal[:, 2, 2] if sigma_n33 == None else sigma_n33
        sigma_n21 = self.sigma_normal[:, 1, 0] if sigma_n21 == None else sigma_n21
        sigma_n43 = self.sigma_normal[:, 3, 2] if sigma_n43 == None else sigma_n43
        sigma_n13 = self.sigma_normal[:, 0, 2] if sigma_n13 == None else sigma_n13
        sigma_n31 = self.sigma_normal[:, 2, 0] if sigma_n31 == None else sigma_n31
        sigma_n14 = self.sigma_normal[:, 0, 3] if sigma_n14 == None else sigma_n14
        sigma_n41 = self.sigma_normal[:, 3, 0] if sigma_n41 == None else sigma_n41

        value, error = [], []

        for other in others:

            index = int(mod(other, self.size))

            if use_phase:
                (mux, sigma_mux), (muy, sigma_muy) = self.phase_advance(other, probe).squeeze()

            if bootstrap:
                table_n11 = n11[index] + sigma_n11[index]*torch.randn(count, dtype=self.dtype, device=self.device)
                table_n33 = n33[index] + sigma_n33[index]*torch.randn(count, dtype=self.dtype, device=self.device)
                table_n21 = n21[index] + sigma_n21[index]*torch.randn(count, dtype=self.dtype, device=self.device)
                table_n43 = n43[index] + sigma_n43[index]*torch.randn(count, dtype=self.dtype, device=self.device)
                table_n13 = n13[index] + sigma_n13[index]*torch.randn(count, dtype=self.dtype, device=self.device)
                table_n31 = n31[index] + sigma_n31[index]*torch.randn(count, dtype=self.dtype, device=self.device)
                table_n14 = n14[index] + sigma_n14[index]*torch.randn(count, dtype=self.dtype, device=self.device)
                table_n41 = n41[index] + sigma_n41[index]*torch.randn(count, dtype=self.dtype, device=self.device)
                if use_phase:
                    table_mux = mux + sigma_mux*torch.randn(count, dtype=self.dtype, device=self.device)
                    table_muy = muy + sigma_muy*torch.randn(count, dtype=self.dtype, device=self.device)
            else:
                table_n11 = n11[index].unsqueeze(0)
                table_n33 = n33[index].unsqueeze(0)
                table_n21 = n21[index].unsqueeze(0)
                table_n43 = n43[index].unsqueeze(0)
                table_n13 = n13[index].unsqueeze(0)
                table_n31 = n31[index].unsqueeze(0)
                table_n14 = n14[index].unsqueeze(0)
                table_n41 = n41[index].unsqueeze(0)
                if use_phase:
                    table_mux = mux.unsqueeze(0)
                    table_muy = muy.unsqueeze(0)

            matrix = self.model.matrix(other, probe)

            table = []
            if use_phase:
                for value_n11, value_n33, value_n21, value_n43, value_n13, value_n31, value_n14, value_n41, value_mux, value_muy in zip(table_n11, table_n33, table_n21, table_n43, table_n13, table_n31, table_n14, table_n41, table_mux, table_muy):
                    normal = parametric_normal(value_n11, value_n33, value_n21, value_n43, value_n13, value_n31, value_n14, value_n41, dtype=self.dtype, device=self.device)
                    normal = matrix @ normal @ matrix_rotation(torch.stack([value_mux, value_muy]).unsqueeze(0)).squeeze().inverse()
                    normal = parametric_normal(*normal[[0, 2, 1, 3, 0, 2, 0, 3], [0, 2, 0, 2, 2, 0, 3, 0]])
                    table.append(normal[[0, 2, 1, 3, 0, 2, 0, 3], [0, 2, 0, 2, 2, 0, 3, 0]])
            else:
                for value_n11, value_n33, value_n21, value_n43, value_n13, value_n31, value_n14, value_n41 in zip(table_n11, table_n33, table_n21, table_n43, table_n13, table_n31, table_n14, table_n41):
                    wolski = normal_to_wolski(parametric_normal(value_n11, value_n33, value_n21, value_n43, value_n13, value_n31, value_n14, value_n41, dtype=self.dtype, device=self.device).unsqueeze(0)).squeeze()
                    wolski = twiss_propagate(wolski, matrix.unsqueeze(0))
                    table.append(wolski_to_normal(wolski).squeeze()[[0, 2, 1, 3, 0, 2, 0, 3], [0, 2, 0, 2, 2, 0, 3, 0]])

            table = torch.stack(table)
            if bootstrap:
                table_value = mean(table.T)
                table_error = variance(table.T).sqrt()
            else:
                table_value = table.squeeze()
                table_error = torch.stack([sigma_n11[index], sigma_n33[index], sigma_n21[index], sigma_n43[index], sigma_n13[index], sigma_n31[index], sigma_n14[index], sigma_n41[index]])

            value.append(table_value)
            error.append(table_error)

        value = torch.stack(value).T
        error = torch.stack(error).T
        value, error = weighted_mean(value, weight=(1.0/error**2).nan_to_num(posinf=1)), weighted_variance(value, weight=(1.0/error**2).nan_to_num(posinf=1)).sqrt().nan_to_num()

        return value, error


    @staticmethod
    def phase_objective(beta:torch.Tensor,
                        mux:torch.Tensor,
                        muy:torch.Tensor,
                        matrix:torch.Tensor) -> torch.Tensor:
        """
        Evaluate phase objective.

        Parameters
        ----------
        beta: torch.Tensor
            n11, n33, n21, n43, n13, n31, n14, n41
        mux & muy: torch.Tensor
            x & y phase advances
        matrix: torch.Tensor
            transport matrices

        Returns
        -------
        objective value (torch.Tensor)

        """
        n11, n33, n21, n43, n13, n31, n14, n41 = beta

        m11, m12, m13, m14 = matrix[:, 0].T
        m21, m22, m23, m24 = matrix[:, 1].T
        m31, m32, m33, m34 = matrix[:, 2].T
        m41, m42, m43, m44 = matrix[:, 3].T

        x_sin  = - m11*n11
        x_sin += - m12*n21
        x_sin += - m13*n31
        x_sin += - m14*n41
        x_sin *= torch.sin(mux)

        x_cos  = m12*n33*(n11 + n14*(n33*n41 - n31*n43))/(n11*(n11*n33 - n13*n31))
        x_cos += m13*n14*n33/n11
        x_cos += m14*(n13*n14*n33*n41 + n11*(n13 - n14*n33*n43))/(n11*(n13*n31 - n11*n33))
        x_cos *= torch.cos(mux)

        y_sin  = - m31*n13
        y_sin += - m32*(n13*n21 + n33*n41 - n31*n43)/n11
        y_sin += - m33*n33
        y_sin += - m34*n43
        y_sin *= torch.sin(muy)

        y_cos  = m31*n14
        y_cos += m32*(-n31 + n14*n21*n33 + n14*n31*(-n13*n21 - n33*n41 + n31*n43)/n11)/(n11*n33 - n13*n31)
        y_cos += m34*(n11 + n14*(n33*n41 - n31*n43))/(n11*n33 - n13*n31)
        y_cos *= torch.cos(muy)

        x_sum = x_cos + x_sin
        y_sum = y_cos + y_sin

        return (x_sum**2 * y_sum**2).sqrt()


    def get_twiss_from_phase_fit(self,
                                *,
                                limit:int=10,
                                count:int=64,
                                model:bool=True,
                                verbose:bool=False,
                                **kwargs) -> torch.Tensor:
        """
        Estimate twiss from phase fit.

        Note, if model is True, model transport between locations is used

        Parameters
        ----------
        limit: int
            range limit
        count: int
            number of samples
        model: bool
            frag to use model transport
        verbose: bool
            verbose flag
        **kwargs:
            passed to leastsq

        Returns
        -------
        estimated parameters (torch.Tensor)

        """
        def objective(beta, mux, muy, matrix):
            beta = torch.tensor(beta, dtype=self.dtype, device=self.device)
            return self.phase_objective(beta, mux, muy, matrix).cpu().numpy()

        result = []
        for location in range(self.model.monitor_count):

            if verbose:
                print(f'{location + 1}/{self.model.monitor_count}')

            probe = location

            others = [probe + index for index in range(-limit, limit + 1) if index != 0]
            shifts = [(other - int(mod(other, self.model.monitor_count))) // self.model.monitor_count for other in others]
            others = [int(mod(other, self.model.monitor_count)) for other in others]

            index_probe = self.model.monitor_index[probe]
            index_other = [self.model.monitor_index[other] + shift*self.model.size for other, shift in zip(others, shifts)]

            beta = self.model.normal[index_probe]
            beta = (beta[[0, 2, 1, 3, 0, 2, 0, 3], [0, 2, 0, 2, 2, 0, 3, 0]])
            beta = beta.cpu().numpy()

            index_other = torch.tensor(index_other, dtype=torch.int64, device=self.device)
            index_probe = index_probe*torch.ones_like(index_other)

            matrix = self.matrix(index_probe, index_other) if not model else self.model.matrix(index_probe, index_other)

            mux, *_ = Decomposition.phase_advance(index_probe, index_other, self.table.nux, self.fx)
            muy, *_ = Decomposition.phase_advance(index_probe, index_other, self.table.nuy, self.fy)

            index = torch.randint(2*limit, (count, 2*limit), dtype=torch.int64, device=self.device)
            table = []
            for i in index:
                fit, *_ = leastsq(objective, beta, args=(mux[i], muy[i], matrix[i]), full_output=1, **kwargs)
                table.append(fit)
            table = torch.tensor(numpy.array(table), dtype=self.dtype, device=self.device).T
            result.append(table)

        return torch.stack(result)


    def process(self,
                value:torch.Tensor,
                error:torch.Tensor,
                *,
                mask:torch.Tensor=None,
                cut:float=5.0,
                use_error:bool=True,
                center_estimator:Callable[[torch.Tensor], torch.Tensor]=median,
                spread_estimator:Callable[[torch.Tensor], torch.Tensor]=biweight_midvariance) -> tuple:
        """
        Process data for single parameter for all locations with optional mask.

        Parameters
        ----------
        value: torch.Tensor
            parameter values at each location for each sample
        error: torch.Tensor
            parameter errors at each location for each sample
        mask: torch.Tensor
            parameter mask at each loacation for each sample
        cut: float
            threshold cut value for data cleaning
        use_error: bool
            flag to threshold using error values
        center_estimator: Callable[[torch.Tensor], torch.Tensor]
            (robust) center estimator
        spread_estimator:Callable[[torch.Tensor], torch.Tensor]
            (robust) spread/variance estimator

        Returns
        -------
        value & error for each location (tuple)

        """
        cut = torch.tensor(cut, dtype=self.dtype, device=self.device)

        data = standardize(value, center_estimator=center_estimator, spread_estimator=spread_estimator)
        mask = mask*threshold(data, -cut, +cut) if mask != None else threshold(data, -cut, +cut)

        if use_error:
            data = standardize(error, center_estimator=center_estimator, spread_estimator=spread_estimator)
            mask *= threshold(data, -cut, +cut)

        weight = (1/error**2).nan_to_num(posinf=0.0)
        weight /= weight.sum(-1, keepdims=True)

        return weighted_mean(value, mask*weight), weighted_variance(value, mask*weight).sqrt()


    def save_model(self,
                   file:str,
                   **kwargs) -> None:
        """
        Save measured twiss as model.

        Parameters
        ----------
        file: str
            output file name

        Returns
        -------
        None

        """
        if self.model.model == 'uncoupled':
            self.model.config_uncoupled(file,
                                        ax=self.ax, bx=self.bx, fx=self.fx,
                                        ay=self.ay, by=self.by, fy=self.fy,
                                        sigma_ax=self.sigma_ax, sigma_bx=self.sigma_bx, sigma_fx=self.sigma_fx,
                                        sigma_ay=self.sigma_ay, sigma_by=self.sigma_by, sigma_fy=self.sigma_fy,
                                        **kwargs)

        if self.model.model == 'coupled':
            self.model.config_coupled(file,
                                      normal=self.normal, fx=self.fx, fy=self.fy,
                                      sigma_normal=self.sigma_normal, sigma_fx=self.sigma_fx, sigma_fy=self.sigma_fy,
                                      **kwargs)


    def get_ax(self,
               index:int) -> torch.Tensor:
        """
        Get ax value and error at given index.

        Parameters
        ----------
        index: int
            index or location name

        Returns
        -------
        [ax, sigma_ax] (torch.Tensor)

        """
        if isinstance(index, str) and index in self.model.name:
            return self.get_ax(self.model.get_index(index))

        index = int(mod(index, self.size))
        return torch.stack([self.ax[index], self.sigma_ax[index]])


    def get_bx(self,
               index:int) -> torch.Tensor:
        """
        Get bx value and error at given index.

        Parameters
        ----------
        index: int
            index or location name

        Returns
        -------
        [bx, sigma_bx] (torch.Tensor)

        """
        if isinstance(index, str) and index in self.model.name:
            return self.get_bx(self.model.get_index(index))

        index = int(mod(index, self.size))
        return torch.stack([self.bx[index], self.sigma_bx[index]])


    def get_fx(self,
               index:int) -> torch.Tensor:
        """
        Get fx value and error at given index.

        Parameters
        ----------
        index: int
            index or location name

        Returns
        -------
        [fx, sigma_fx] (torch.Tensor)

        """
        if isinstance(index, str) and index in self.model.name:
            return self.get_fx(self.model.get_index(index))

        index = int(mod(index, self.size))
        return torch.stack([self.fx[index], self.sigma_fx[index]])


    def get_ay(self,
               index:int) -> torch.Tensor:
        """
        Get ay value and error at given index.

        Parameters
        ----------
        index: int
            index or location name

        Returns
        -------
        [ay, sigma_ay] (torch.Tensor)

        """
        if isinstance(index, str) and index in self.model.name:
            return self.get_ay(self.model.get_index(index))

        index = int(mod(index, self.size))
        return torch.stack([self.ay[index], self.sigma_ay[index]])


    def get_by(self,
               index:int) -> torch.Tensor:
        """
        Get by value and error at given index.

        Parameters
        ----------
        index: int
            index or location name

        Returns
        -------
        [by, sigma_by] (torch.Tensor)

        """
        if isinstance(index, str) and index in self.model.name:
            return self.get_by(self.model.get_index(index))

        index = int(mod(index, self.size))
        return torch.stack([self.by[index], self.sigma_by[index]])


    def get_fy(self,
               index:int) -> torch.Tensor:
        """
        Get fy value and error at given index.

        Parameters
        ----------
        index: int
            index or location name

        Returns
        -------
        [fy, sigma_fy] (torch.Tensor)

        """
        if isinstance(index, str) and index in self.model.name:
            return self.get_fy(self.model.get_index(index))

        index = int(mod(index, self.size))
        return torch.stack([self.fy[index], self.sigma_fy[index]])


    def get_twiss(self,
                  index:int) -> dict:
        """
        Return twiss data at given index.

        Parameters
        ----------
        index: int
            index or location name

        Returns
        -------
        twiss data (dict)

        """
        if isinstance(index, str) and index in self.model.name:
            return self.get_twiss(self.model.get_index(index))

        table = {}
        table['ax'], table['sigma_ax'] = self.get_ax(index)
        table['bx'], table['sigma_bx'] = self.get_bx(index)
        table['fx'], table['sigma_fx'] = self.get_fx(index)
        table['ay'], table['sigma_ay'] = self.get_ay(index)
        table['by'], table['sigma_by'] = self.get_by(index)
        table['fy'], table['sigma_fy'] = self.get_fy(index)

        return table


    def get_table(self) -> pandas.DataFrame:
        """
        Return twiss data at all locations as dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        twiss data (pandas.DataFrame)

        """
        df = pandas.DataFrame()

        df['name'] = self.model.name
        df['kind'] = self.model.kind

        df['flag'] = self.flag.cpu().numpy()
        df['time'] = self.model.time.cpu().numpy()

        df['ax'], df['sigma_ax'] = self.ax.cpu().numpy(), self.sigma_ax.cpu().numpy()
        df['bx'], df['sigma_bx'] = self.bx.cpu().numpy(), self.sigma_bx.cpu().numpy()
        df['fx'], df['sigma_fx'] = self.fx.cpu().numpy(), self.sigma_fx.cpu().numpy()

        df['ay'], df['sigma_ay'] = self.ay.cpu().numpy(), self.sigma_ay.cpu().numpy()
        df['by'], df['sigma_by'] = self.by.cpu().numpy(), self.sigma_by.cpu().numpy()
        df['fy'], df['sigma_fy'] = self.fy.cpu().numpy(), self.sigma_fy.cpu().numpy()

        return df


    def __repr__(self) -> str:
        """
        String representation.

        """
        return f'{self.__class__.__name__}({self.model}, {self.table}, {self.limit})'


    def __len__(self) -> int:
        """
        Number of locations.

        """
        return self.size


    def __call__(self,
                 limit:int=None) -> pandas.DataFrame:
        """
        Perform twiss loop with default parameters.

        Parameters
        ----------
        limit: int
            range limit for virtual phase computation

        Returns
        -------
        twiss table (pandas.DataFrame)

        """
        limit = max(self.limit) if limit is None else limit

        self.get_action()
        self.get_twiss_from_amplitude()
        self.phase_virtual(limit=limit)
        self.get_twiss_from_phase()

        select = {
            'phase': {'use': True, 'threshold': 10.00},
            'model': {'use': False, 'threshold': 00.50},
            'value': {'use': False, 'threshold': 00.50},
            'sigma': {'use': False, 'threshold': 00.25},
            'limit': {'use': True, 'threshold': 05.00}
        }

        mask_x = self.filter_twiss(plane='x', **select)
        mask_y = self.filter_twiss(plane='y', **select)

        _ = self.process_twiss(plane='x', mask=mask_x, weight=True)
        _ = self.process_twiss(plane='y', mask=mask_y, weight=True)

        return self.get_table()


def main():
    pass

if __name__ == '__main__':
    main()