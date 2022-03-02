"""
Table module.

"""

import torch

class Table():

    def __init__(self, name:list,
                 nux:torch.Tensor, nuy:torch.Tensor,
                 ax:torch.Tensor, ay:torch.Tensor, fx:torch.Tensor, fy:torch.Tensor,
                 sigma_nux:torch.Tensor=None, sigma_nuy:torch.Tensor=None,
                 sigma_ax:torch.Tensor=None, sigma_ay:torch.Tensor=None,
                 sigma_fx:torch.Tensor=None, sigma_fy:torch.Tensor=None, *,
                 dtype:torch.dtype=torch.float64, device:torch.device='cpu') -> None:
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
        sigma_nuy: torch.Tensor:
            y phase error
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

        self.nux = nux
        self.nuy = nuy

        self.ax = torch.tensor(ax.cpu().numpy(), dtype=self.dtype, device=self.device)
        self.ay = torch.tensor(ay.cpu().numpy(), dtype=self.dtype, device=self.device)
        self.fx = torch.tensor(fx.cpu().numpy(), dtype=self.dtype, device=self.device)
        self.fy = torch.tensor(fy.cpu().numpy(), dtype=self.dtype, device=self.device)

        self.sigma_nux = None if sigma_nux is None else sigma_nux
        self.sigma_nuy = None if sigma_nuy is None else sigma_nuy

        self.sigma_ax = None if sigma_ax is None else torch.tensor(sigma_ax.cpu().numpy(), dtype=self.dtype, device=self.device)
        self.sigma_ay = None if sigma_ay is None else torch.tensor(sigma_ay.cpu().numpy(), dtype=self.dtype, device=self.device)
        self.sigma_fx = None if sigma_fx is None else torch.tensor(sigma_fx.cpu().numpy(), dtype=self.dtype, device=self.device)
        self.sigma_fy = None if sigma_fy is None else torch.tensor(sigma_fy.cpu().numpy(), dtype=self.dtype, device=self.device)


    def __repr__(self):
        """
        String representation.

        """
        return f'Table({self.size})'