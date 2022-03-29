"""
Twiss module.
Compute twiss parameters from amplitude & phase data.
Twiss filtering & processing.

"""

import numpy
import torch
import pandas

from .util import mod, generate_pairs, generate_other
from .statistics import weighted_mean, weighted_variance
from .statistics import median, biweight_midvariance, standardize
from .anomaly import threshold, dbscan, local_outlier_factor, isolation_forest
from .decomposition import Decomposition
from .model import Model
from .table import Table


class Twiss():
    """
    Returns
    ----------
    Twiss class instance.

    Parameters
    ----------
    model: 'Model'
        Model instance
    table: 'Table'
        Table instance
    flag: torch.Tensor
            external flags for each model location
    limit: int | tuple
        range limit to use, (min, max), 1 <= min <= max, mim is excluded, for full range min==max
    use_model: bool
        flag to use precomputed model data

    Attributes
    ----------
    model: 'Model'
        Model instance
    table: 'Table'
        Table instance
    limit: int | tuple
        range limit to use, (min, max), 1 <= min <= max, mim is excluded, for full range min==max
    use_model: bool
        flag to use precomputed model data
    dtype: torch.dtype
        data type (from model)
    device: torch.device
        data device (from model)
    flag: torch.Tensor
        location flags
    count: torch.Tensor
        (uncoupled) range limit endpoints [1, 6, 15, 28, 45, 66, 91, 120, ...]
    combo: torch.Tensor
        (uncoupled) index combinations [..., [..., [[i, j], [i, k]], ...], ...]
    shape: torch.Size
        initial shape of combo
    distance: torch.Tensor
        (uncoupled) distance
    fx: torch.Tensor
        x phase for each location
    fy: torch.Tensor
        y phase  for each location
    sigma_fx: torch.Tensor
        x phase error for each location
    sigma_fy: torch.Tensor
        y phase error for each location
    fx_correct: torch.Tensor
        corrected x phase for each location
    fy_correct: torch.Tensor
        corrected y phase  for each location
    sigma_fx_correct: torch.Tensor
        corrected x phase error for each location
    sigma_fy_correct: torch.Tensor
        corrected y phase error for each location
    virtual_x: dict
        x plane virtual phase data
    virtual_y: dict
        y plane virtual phase data
    correct_x: dict
        x plane corrected phase data
    correct_y: dict
        y plane corrected phase data
    action: dict
        action data
        dict_keys(['jx', 'sigma_jx', 'center_jx', 'spread_jx', 'jy', 'sigma_jy', 'center_jy', 'spread_jy', 'mask'])
    data_amplitude: dict
        twiss from amplitude data
        dict_keys(['bx', 'sigma_bx', 'by', 'sigma_by'])
    data_phase: dict
        twiss from phase data
        dict_keys(['fx_ij', 'sigma_fx_ij', 'fx_m_ij', 'sigma_fx_m_ij', 'fx_ik', 'sigma_fx_ik', 'fx_m_ik', 'sigma_fx_m_ik', 'fy_ij', 'sigma_fy_ij', 'fy_m_ij', 'sigma_fy_m_ij', 'fy_ik', 'sigma_fy_ik', 'fy_m_ik', 'sigma_fy_m_ik', 'ax', 'sigma_ax', 'bx', 'sigma_bx', 'ay', 'sigma_ay', 'by', 'sigma_by'])
    ax: torch.Tensor
        alfa x
    sigma_ax: torch.Tensor
        sigma alfa x
    bx: torch.Tensor
        beta x
    sigma_bx: torch.Tensor
        sigma beta x
    ay: torch.Tensor
        alfa y
    sigma_ay: torch.Tensor
        sigma alfa y
    by: torch.Tensor
        beta y
    sigma_by: torch.Tensor
        sigma beta y

    Methods
    ----------
    __init__(self, model:'Model', table:'Table', limit:int=8, use_model:bool=False) -> None
        Twiss instance initialization.
    get_action(self, *, data_threshold:dict={'use': True, 'factor': 5.0}, data_dbscan:dict={'use': False, 'factor': 2.5}, data_local_outlier_factor:dict={'use': False, 'contamination': 0.01}, data_isolation_forest:dict={'use': False, 'contamination': 0.01}) -> None
        Estimate actions at each monitor location with optional data cleaning and estimate action center and spread.
    get_twiss_from_amplitude(self) -> None
        Estimate twiss from amplitude.
    phase_virtual(self, limit:int=None, exclude:list=None, **kwargs) -> None
        Estimate x & y phase for virtual locations.
    phase_correct(self, *, limit:int=None, **kwargs) -> None
        Correct x & y phase for monitor locations.
    phase_alfa(a_m:torch.Tensor, f_ij:torch.Tensor, f_m_ij:torch.Tensor, f_ik:torch.Tensor, f_m_ik:torch.Tensor, *, error:bool=True, model:bool=True, sigma_a_m:torch.Tensor=0.0, sigma_f_ij:torch.Tensor=0.0, sigma_f_m_ij:torch.Tensor=0.0, sigma_f_ik:torch.Tensor=0.0, sigma_f_m_ik:torch.Tensor=0.0) -> tuple
        Estimate twiss alfa at index (i) from given triplet (i, j, k) phase data.
    phase_beta(b_m:torch.Tensor, f_ij:torch.Tensor, f_m_ij:torch.Tensor, f_ik:torch.Tensor, f_m_ik:torch.Tensor, *, error:bool=True, model:bool=True, sigma_b_m:torch.Tensor=0.0, sigma_f_ij:torch.Tensor=0.0, sigma_f_m_ij:torch.Tensor=0.0, sigma_f_ik:torch.Tensor=0.0, sigma_f_m_ik:torch.Tensor=0.0) -> tuple
        Estimate twiss beta at index (i) from given triplet (i, j, k) phase data.
    get_twiss_from_phase(self, *, virtual:bool=True, error:bool=True, model:bool=False, use_correct:bool=False, use_correct_sigma:bool=False) -> None
        Estimate twiss from phase data.
    filter_twiss(self, plane:str = 'x', *, phase:dict={'use': True, 'threshold': 10.00}, model:dict={'use': True, 'threshold': 00.50}, value:dict={'use': True, 'threshold': 00.50}, sigma:dict={'use': True, 'threshold': 00.25}, limit:dict={'use': True, 'threshold': 05.00}) -> dict
        Filter twiss for given data plane and cleaning options.
    mask_range(self, limit:tuple) -> torch.Tensor
        Generate weight mask based on given range limit.
    mask_location(self, table:list) -> torch.Tensor
        Generate weight mask based on given range limit.
    mask_distance(self, function) -> torch.Tensor
        Generate weight mask based on given range limit.
    process_twiss(self, plane:str='x', *, weight:bool=True, mask:torch.Tensor=None) -> dict
        Process twiss data.
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
    def __init__(self, model:'Model', table:'Table', flag:torch.Tensor=None, limit:int=8, use_model:bool=False) -> None:
        """
        Twiss instance initialization.

        Parameters
        ----------
        model: 'Model'
            Model instance
        table: 'Table'
            Table instance
        flag: torch.Tensor
            external flags for each model location
        limit: int | tuple
            range limit to use, (min, max), 1 <= min <= max, mim is excluded, for full range min==max
        use_model: bool
            flag to use precomputed model data

        Returns
        -------
        None

        """
        self.model, self.table, self.limit, self.use_model = model, table, limit, use_model

        self.limit = self.limit if isinstance(self.limit, tuple) else (self.limit, self.limit)

        if self.use_model:
            if self.model.limit is None:
                raise Exception(f'TWISS: model limit is None')
            if self.model.limit < max(self.limit):
                raise Exception(f'TWISS: requested limit={self.limit} should be less than model limit={self.model.limit}')

        self.size, self.dtype, self.device = self.model.size, self.model.dtype, self.model.device

        if self.model.monitor_count != self.table.size:
            raise Exception(f'TWISS: expected {self.model.monitor_count} monitors in Model, got {self.table.size} in Table')

        if self.model.monitor_name != self.table.name:
            raise Exception(f'TWISS: expected monitor names to match')

        if flag is None:
            self.flag = [flag if kind == self.model._monitor else 0 for flag, kind in zip(self.model.flag, self.model.kind)]
            self.flag = torch.tensor(self.flag, dtype=torch.int64, device=self.device)
        else:
            if len(flag) != self.size:
                raise Exception(f'TWISS: external flag length {len(flag)}, expected length {self.size}')
            self.flag = flag.to(torch.int64).to(self.device)

        if self.use_model:
            self.count = self.model.count
            self.combo = self.model.combo
            self.index = self.model.index
        else:
            self.count = torch.tensor([limit*(2*limit - 1) for limit in range(1, max(self.limit) + 1)], dtype=torch.int64, device=self.device)
            self.combo = [generate_other(probe, max(self.limit), self.flag) for probe in range(self.size)]
            self.combo = torch.stack([generate_pairs(max(self.limit), 1 + 1, probe=probe, table=table, dtype=torch.int64, device=self.device) for probe, table in enumerate(self.combo)])
            self.index = mod(self.combo, self.size).to(torch.int64)

        self.shape = self.combo.shape

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
            raise Exception(f'TWISS: invalid limit={self.limit}')

        self.fx = torch.zeros_like(self.model.fx)
        self.fy = torch.zeros_like(self.model.fy)

        self.fx[self.model.monitor_index] = self.table.fx
        self.fy[self.model.monitor_index] = self.table.fy

        self.sigma_fx = torch.zeros_like(self.model.sigma_fx)
        self.sigma_fy = torch.zeros_like(self.model.sigma_fy)

        self.sigma_fx[self.model.monitor_index] = self.table.sigma_fx
        self.sigma_fy[self.model.monitor_index] = self.table.sigma_fy

        self.fx_correct, self.sigma_fx_correct = torch.clone(self.fx), torch.clone(self.sigma_fx)
        self.fy_correct, self.sigma_fy_correct = torch.clone(self.fy), torch.clone(self.sigma_fy)

        self.virtual_x, self.correct_x = {}, {}
        self.virtual_y, self.correct_y = {}, {}

        self.action, self.data_amplitude, self.data_phase = {}, {}, {}

        self.ax, self.sigma_ax = torch.zeros_like(self.model.ax), torch.zeros_like(self.model.sigma_ax)
        self.bx, self.sigma_bx = torch.zeros_like(self.model.bx), torch.zeros_like(self.model.sigma_bx)

        self.ay, self.sigma_ay = torch.zeros_like(self.model.ay), torch.zeros_like(self.model.sigma_ay)
        self.by, self.sigma_by = torch.zeros_like(self.model.by), torch.zeros_like(self.model.sigma_by)

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


    def get_action(self, *,
                   data_threshold:dict={'use': True, 'factor': 5.0},
                   data_dbscan:dict={'use': False, 'factor': 2.5},
                   data_local_outlier_factor:dict={'use': False, 'contamination': 0.01},
                   data_isolation_forest:dict={'use': False, 'contamination': 0.01}) -> None:
        """
        Estimate actions at each monitor location with optional data cleaning and estimate action center and spread.

        Parameters
        ----------
        data_threshold: dict
            parameters for threshold detector
        data_dbscan: dict
            parameters for dbscan detector
        data_local_outlier_factor: dict
            parameters for local outlier factor detector
        data_isolation_forest: dict
            parameters for isolation forest detector

        Returns
        -------
        None, update self.action dictionary

        """
        self.action = {}

        index = self.model.monitor_index

        jx = self.table.ax**2/(2.0*self.model.bx[index])
        jy = self.table.ay**2/(2.0*self.model.by[index])

        sigma_jx = self.table.ax/self.model.bx[index]*self.table.sigma_ax
        sigma_jy = self.table.ay/self.model.by[index]*self.table.sigma_ay

        mask = torch.clone(self.flag[index])
        mask = torch.stack([mask, mask]).to(torch.bool)

        data = standardize(torch.stack([jx, jy]), center_estimator=median, spread_estimator=biweight_midvariance)

        if data_threshold['use']:
            factor = data_threshold['factor']
            center = median(data)
            spread = biweight_midvariance(data).sqrt()
            min_value, max_value = center - factor*spread, center + factor*spread
            mask *= threshold(data, min_value, max_value)

        if data_dbscan['use']:
            factor = data_dbscan['factor']
            for case in range(1):
                mask[case] *= dbscan(data[case].reshape(-1, 1), epsilon=factor)

        if data_local_outlier_factor['use']:
            for case in range(1):
                mask[case] *= local_outlier_factor(data[case].reshape(-1, 1), contamination=data_local_outlier_factor['contamination'])

        if data_isolation_forest['use']:
            for case in range(1):
                mask[case] *= isolation_forest(data[case].reshape(-1, 1), contamination=data_isolation_forest['contamination'])

        mask_jx, mask_jy = mask
        mask_jx, mask_jy = mask_jx/sigma_jx**2, mask_jy/sigma_jy**2

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
        Estimate twiss from amplitude.

        Note, action dictionary should be precomputed

        Parameters
        ----------
        None

        Returns
        -------
        None, update self.twiss_from_amplitude dictionary

        """
        if self.action == {}:
            raise Exception('error: action dictionary is empty')

        self.data_amplitude = {}

        ax, sigma_ax = self.table.ax, self.table.sigma_ax
        ay, sigma_ay = self.table.ay, self.table.sigma_ay

        jx, sigma_jx = self.action['center_jx'], self.action['spread_jx']
        jy, sigma_jy = self.action['center_jy'], self.action['spread_jy']

        bx, by = ax**2/(2.0*jx), ay**2/(2.0*jy)

        sigma_bx = torch.sqrt(ax**2/jx**2*sigma_ax**2 + 0.25*ax**4/jx**4*sigma_jx**2)
        sigma_by = torch.sqrt(ay**2/jy**2*sigma_ay**2 + 0.25*ay**4/jy**4*sigma_jy**2)

        index = self.model.monitor_index
        bx_model, by_model = self.model.bx[index], self.model.by[index]

        self.data_amplitude['bx'], self.data_amplitude['sigma_bx'] = bx, sigma_bx
        self.data_amplitude['by'], self.data_amplitude['sigma_by'] = by, sigma_by


    def phase_virtual(self, limit:int=None, exclude:list=None, **kwargs) -> None:
        """
        Estimate x & y phase for virtual locations.

        Parameters
        ----------
        limit: int
            range limit to use
        exclude: list
            list of virtual location to exclude
        **kwargs:
            passed to Decomposition.phase_virtual

        Returns
        -------
        None, update self.virtual_x and self.virtual_y dictionaries

        """
        self.virtual_x, self.virtual_y = {}, {}

        limit = max(self.limit) if limit is None else limit
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
                                                    **kwargs)

        def auxiliary_y(probe):
            return Decomposition.phase_virtual(probe, limit, self.flag, nuy, NUY, fy, FY,
                                                    sigma_frequency=sigma_nuy, sigma_frequency_model=sigma_NUY,
                                                    sigma_phase=sigma_fy, sigma_phase_model=sigma_FY,
                                                    **kwargs)

        data_x = [auxiliary_x(probe) for probe in index]
        data_y = [auxiliary_y(probe) for probe in index]

        for count, probe in enumerate(index):
            self.virtual_x[probe], self.virtual_y[probe] = data_x[count], data_y[count]
            self.fx[probe], self.sigma_fx[probe] = self.virtual_x[probe].get('model')
            self.fy[probe], self.sigma_fy[probe] = self.virtual_y[probe].get('model')


    def phase_correct(self, *, limit:int=None, **kwargs) -> None:
        """
        Correct x & y phase for monitor locations.

        Note, this introduce strong bias towards model, do not use large range limit
        Note, phase at the location is not used

        Parameters
        ----------
        limit: int
            range limit
        **kwargs:
            passed to phase_virtual Decomposition method

        Returns
        -------
        None, update self.correct_x and self.correct_y dictionaries

        """
        self.correct_x, self.correct_y = {}, {}

        limit = max(self.limit) if limit is None else limit
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
            return Decomposition.phase_virtual(probe, limit, self.flag, nux, NUX, fx, FX,
                                                    sigma_frequency=sigma_nux, sigma_frequency_model=sigma_NUX,
                                                    sigma_phase=sigma_fx, sigma_phase_model=sigma_FX,
                                                    **kwargs)

        def auxiliary_y(probe):
            return Decomposition.phase_virtual(probe, limit, self.flag, nuy, NUY, fy, FY,
                                                    sigma_frequency=sigma_nuy, sigma_frequency_model=sigma_NUY,
                                                    sigma_phase=sigma_fy, sigma_phase_model=sigma_FY,
                                                    **kwargs)

        data_x = [auxiliary_x(probe) for probe in index]
        data_y = [auxiliary_y(probe) for probe in index]

        for count, probe in enumerate(index):
            self.correct_x[probe], self.correct_y[probe] = data_x[count], data_y[count]
            self.fx_correct[probe], self.sigma_fx_correct[probe] = self.correct_x[probe].get('model')
            self.fy_correct[probe], self.sigma_fy_correct[probe] = self.correct_y[probe].get('model')


    @staticmethod
    def phase_alfa(a_m:torch.Tensor,
                   f_ij:torch.Tensor, f_m_ij:torch.Tensor,
                   f_ik:torch.Tensor, f_m_ik:torch.Tensor,
                   *,
                   error:bool=True, model:bool=True,
                   sigma_a_m:torch.Tensor=0.0,
                   sigma_f_ij:torch.Tensor=0.0, sigma_f_m_ij:torch.Tensor=0.0,
                   sigma_f_ik:torch.Tensor=0.0, sigma_f_m_ik:torch.Tensor=0.0) -> tuple:
        """
        Estimate twiss alfa at index (i) from given triplet (i, j, k) phase data.

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
        (a, 0) or (a, sigma_a)

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
                   f_ij:torch.Tensor, f_m_ij:torch.Tensor,
                   f_ik:torch.Tensor, f_m_ik:torch.Tensor,
                   *,
                   error:bool=True, model:bool=True,
                   sigma_b_m:torch.Tensor=0.0,
                   sigma_f_ij:torch.Tensor=0.0, sigma_f_m_ij:torch.Tensor=0.0,
                   sigma_f_ik:torch.Tensor=0.0, sigma_f_m_ik:torch.Tensor=0.0) -> tuple:
        """
        Estimate twiss beta at index (i) from given triplet (i, j, k) phase data.

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
        (b, 0) or (b, sigma_b)

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


    def get_twiss_from_phase(self, *, virtual:bool=True, error:bool=True, model:bool=False,
                             use_correct:bool=False, use_correct_sigma:bool=False, use_model:bool=False) -> None:
        """
        Estimate twiss from phase data.

        Note, raw data is saved, no cleaning is performed
        Values (and errors) are computed for each triplet

        Parameters
        ----------
        error: bool
            flag to compute twiss errors
        model: bool
            flag to include model error
        use_correct: bool
            flag to use corrected phases
        use_correct_sigma: bool
            flag to use corrected phase errors
        use_model: bool
            flag to use precomputed model data

        Returns
        -------
        None, update self.twiss_from_phase dictionary

        """
        self.data_phase = {}

        fx = self.fx_correct if use_correct else self.fx
        fy = self.fy_correct if use_correct else self.fy

        sigma_fx = self.sigma_fx_correct if use_correct_sigma else self.sigma_fx
        sigma_fy = self.sigma_fy_correct if use_correct_sigma else self.sigma_fy

        ax_m, bx_m = self.model.ax, self.model.bx
        ay_m, by_m = self.model.ay, self.model.by

        index = self.combo.swapaxes(0, -1)

        value, sigma = Decomposition.phase_advance(*index, self.table.nux, fx, error=error, model=False, sigma_frequency=self.table.sigma_nux, sigma_phase=sigma_fx)
        fx_ij, fx_ik = value.swapaxes(0, 1)
        sx_ij, sx_ik = sigma.swapaxes(0, 1)

        value, sigma = Decomposition.phase_advance(*index, self.table.nuy, fy, error=error, model=False, sigma_frequency=self.table.sigma_nuy, sigma_phase=sigma_fy)
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

        self.data_phase['ax'], self.data_phase['sigma_ax'], self.data_phase['bx'], self.data_phase['sigma_bx'] = ax.T, sigma_ax.T, bx.T, sigma_bx.T
        self.data_phase['ay'], self.data_phase['sigma_ay'], self.data_phase['by'], self.data_phase['sigma_by'] = ay.T, sigma_ay.T, by.T, sigma_by.T


    def filter_twiss(self, plane:str = 'x', *,
                     phase:dict={'use': True, 'threshold': 10.00},
                     model:dict={'use': True, 'threshold': 00.50},
                     value:dict={'use': True, 'threshold': 00.50},
                     sigma:dict={'use': True, 'threshold': 00.25},
                     limit:dict={'use': True, 'threshold': 05.00}) -> dict:
        """
        Filter twiss for given data plane and cleaning options.

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
            clean outliers outside scaled interval
            used if 'use' is True

        Returns
        -------
        mask (torch.Tensor)

        """
        size, length, *_ = self.index.shape
        mask = torch.ones((size, length), device=self.device).to(torch.bool)

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

        return mask


    def mask_range(self, limit:tuple) -> torch.Tensor:
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


    def mask_location(self, table:list) -> torch.Tensor:
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


    def mask_distance(self, function) -> torch.Tensor:
        """
        Generate weight mask based on given range limit.

        Parameters
        ----------
        function: Callable
            function to apply to distance data

        Returns
        -------
        weight mask (torch.Tensor)

        """
        mask = torch.stack([function(distance) for distance in self.distance])
        mask = torch.stack([mask for _ in range(self.size)])
        return mask


    def process_twiss(self, plane:str='x', *,
                      weight:bool=True, mask:torch.Tensor=None) -> dict:
        """
        Process twiss data.

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

            return result

        weight = (mask.to(self.dtype)/sigma_a**2).nan_to_num()
        center = weighted_mean(a, weight=weight)
        spread = weighted_variance(a, weight=weight, center=center).sqrt()
        result['value_a'] = center
        result['sigma_a'] = spread
        result['error_a'] = (center - a_m)/a_m

        weight = (mask.to(self.dtype)/sigma_b**2).nan_to_num()
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


    def get_ax(self, index:int) -> torch.Tensor:
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


    def get_bx(self, index:int) -> torch.Tensor:
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


    def get_fx(self, index:int) -> torch.Tensor:
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


    def get_ay(self, index:int) -> torch.Tensor:
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


    def get_by(self, index:int) -> torch.Tensor:
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


    def get_fy(self, index:int) -> torch.Tensor:
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


    def get_twiss(self, index:int) -> dict:
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


    def __call__(self, limit:int=None) -> pandas.DataFrame:
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

        mask_x = self.filter_twiss(plane='x')
        mask_y = self.filter_twiss(plane='y')

        _ = self.process_twiss(plane='x', mask=mask_x, weight=True)
        _ = self.process_twiss(plane='y', mask=mask_y, weight=True)

        return self.get_table()


def main():
    pass

if __name__ == '__main__':
    main()