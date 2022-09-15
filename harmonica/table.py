"""
Table module.
Generate contaner with estimated parameters (frequencies, amplitudes and phases).

"""

import numpy
import torch

from .decomposition import Decomposition

class Table():
    """
    Returns
    ----------
    Table class instance.

    Parameters
    ----------
    name: list
        location names
    nux: torch.Tensor
        x frequency (fractional part)
    nuy: torch.Tensor
        y frequency (fractional part)
    ax: torch.Tensor
        x amplitude
    ay: torch.Tensor
        y amplitude
    fx: torch.Tensor
        x phase
    fy: torch.Tensor
        y phase
    sigma_nux: torch.Tensor:
        x frequency error
    sigma_nuy: torch.Tensor:
        y frequency error
    sigma_ax: torch.Tensor:
        x amplitude error
    sigma_ay: torch.Tensor:
        y amplitude error
    sigma_fx: torch.Tensor:
        x phase error
    sigma_fy: torch.Tensor:
        y phase error
    x & y: torch.Tensor
        x & y data
    dtype: torch.dtype
        data type
    device: torch.device
        data device

    Attributes
    ----------
    name: list
        location names
    size: int
        number of locations
    nux: torch.Tensor
        x frequency (fractional part)
    nuy: torch.Tensor
        y frequency (fractional part)
    ax: torch.Tensor
        x amplitude
    ay: torch.Tensor
        y amplitude
    fx: torch.Tensor
        x phase
    fy: torch.Tensor
        y phase
    sigma_nux: torch.Tensor:
        x frequency error
    sigma_nuy: torch.Tensor:
        y frequency error
    sigma_ax: torch.Tensor:
        x amplitude error
    sigma_ay: torch.Tensor:
        y amplitude error
    sigma_fx: torch.Tensor:
        x phase error
    sigma_fy: torch.Tensor:
        y phase error
    phase_x: torch.Tensor
        x phase advance from each location to the next one
    sigma_x: torch.Tensor
        x phase advance error from each location to the next one
    phase_y: torch.Tensor
        y phase advance from each location to the next one
    sigma_y: torch.Tensor
        y phase advance error from each location to the next one
    x & y: torch.Tensor
        x & y data
    accumulate: bool
        flag to accumulate phase
    dtype: torch.dtype
        data type
    device: torch.device
        data device

    Methods
    ----------
    __init__(self, name:list, nux:torch.Tensor, nuy:torch.Tensor, ax:torch.Tensor, ay:torch.Tensor, fx:torch.Tensor, fy:torch.Tensor, sigma_nux:torch.Tensor=None, sigma_nuy:torch.Tensor=None, sigma_ax:torch.Tensor=None, sigma_ay:torch.Tensor=None, sigma_fx:torch.Tensor=None, sigma_fy:torch.Tensor=None, *, x:torch.Tensor=None, y:torch.Tensor=None, dtype:torch.dtype=torch.float64, device:torch.device='cpu') -> None
        Table instance initialization.
    accumulate_phase(self) -> None
        Accumulate x & y phases (modify self.fx and self.fy).
    __repr__(self) -> str
        String representation.

    """
    def __init__(self,
                 name:list,
                 nux:torch.Tensor,
                 nuy:torch.Tensor,
                 ax:torch.Tensor,
                 ay:torch.Tensor,
                 fx:torch.Tensor,
                 fy:torch.Tensor,
                 sigma_nux:torch.Tensor=None,
                 sigma_nuy:torch.Tensor=None,
                 sigma_ax:torch.Tensor=None,
                 sigma_ay:torch.Tensor=None,
                 sigma_fx:torch.Tensor=None,
                 sigma_fy:torch.Tensor=None,
                 *,
                 x:torch.Tensor=None,
                 y:torch.Tensor=None,
                 accumulate:bool=False,
                 dtype:torch.dtype=torch.float64,
                 device:torch.device='cpu') -> None:
        """
        Table instance initialization.

        Parameters
        ----------
        name: list
            location names
        nux: torch.Tensor
            x frequency (fractional part)
        nuy: torch.Tensor
            y frequency (fractional part)
        ax: torch.Tensor
            x amplitude
        ay: torch.Tensor
            y amplitude
        fx: torch.Tensor
            x phase
        fy: torch.Tensor
            y phase
        sigma_nux: torch.Tensor:
            x frequency error
        sigma_nuy: torch.Tensor:
            y frequency error
        sigma_ax: torch.Tensor:
            x amplitude error
        sigma_ay: torch.Tensor:
            y amplitude error
        sigma_fx: torch.Tensor:
            x phase error
        sigma_fy: torch.Tensor:
            y phase error
        x & y: torch.Tensor
            x & y data
        accumulate: bool
            flag to accumulate phase
        dtype: torch.dtype
            data type
        device: torch.device
            data device

        Returns
        -------
        None

        """
        self.name = name
        self.size = len(name)

        self.dtype, self.device = dtype, device

        self.nux = nux.to(self.dtype).to(self.device)
        self.nuy = nuy.to(self.dtype).to(self.device)

        self.ax = ax.to(self.dtype).to(self.device)
        self.ay = ay.to(self.dtype).to(self.device)
        self.fx = fx.to(self.dtype).to(self.device)
        self.fy = fy.to(self.dtype).to(self.device)

        if accumulate:
            self.accumulate_phase()

        self.sigma_nux = torch.tensor(0.0, dtype=dtype, device=device) if sigma_nux is None else sigma_nux.to(self.dtype).to(self.device)
        self.sigma_nuy = torch.tensor(0.0, dtype=dtype, device=device) if sigma_nuy is None else sigma_nuy.to(self.dtype).to(self.device)

        zero = torch.zeros(self.size, dtype=self.dtype, device=self.device)

        self.sigma_ax = torch.zeros_like(zero) if sigma_ax is None else sigma_ax.to(self.dtype).to(self.device)
        self.sigma_ay = torch.zeros_like(zero) if sigma_ay is None else sigma_ay.to(self.dtype).to(self.device)
        self.sigma_fx = torch.zeros_like(zero) if sigma_fx is None else sigma_fx.to(self.dtype).to(self.device)
        self.sigma_fy = torch.zeros_like(zero) if sigma_fy is None else sigma_fy.to(self.dtype).to(self.device)

        probe = torch.tensor(range(self.size), dtype=torch.int64, device=self.device)
        other = probe + 1
        self.phase_x, self.sigma_x = Decomposition.phase_advance(probe, other, self.nux, self.fx, error=True, sigma_frequency=self.sigma_nux, sigma_phase=self.sigma_fx, model=False)
        self.phase_y, self.sigma_y = Decomposition.phase_advance(probe, other, self.nuy, self.fy, error=True, sigma_frequency=self.sigma_nuy, sigma_phase=self.sigma_fy, model=False)

        self.x, self.y = x, y


    def accumulate_phase(self) -> None:
        """
        Accumulate x & y phases (modify self.fx and self.fy).

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        for index in range(1, self.size):

            while self.fx[index] < self.fx[index - 1]:
                self.fx[index:] += 2.0*numpy.pi

            while self.fy[index] < self.fy[index - 1]:
                self.fy[index:] += 2.0*numpy.pi


    def __repr__(self) -> str:
        """
        String representation.

        """
        return f'Table({self.size})'


def main():
    pass

if __name__ == '__main__':
    main()