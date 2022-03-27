"""
Window module.
Generate window data.
Initialize Window class instance.

"""

import torch
import numpy


class Window():
    """
    Returns
    ----------
    Window class instance.

    Window(length:int=1024, name:str=None, order:float=None, **kwargs)
    Window.from_cosine(length:int=1024, order:float=1.0, **kwargs)
    Window.from_kaiser(length:int=1024, order:float=1.0, **kwargs)

    Parameters
    ----------
    length: int
        window data length
    name: str
        window name ('cosine_window' or 'kaiser_window')
    order: float
        window order parameter, positive float
    **kwargs:
        passed to torch.ones (use dtype and device)

    Attributes
    ----------
    length: int
        window data length
    name: str
        window name ('cosine_window' or 'kaiser_window')
    order: float
        window order parameter, positive float
    dtype: torch.dtype
        window type
    device: torch.device
        window device
    window: torch.Tensor
        window container
    total: torch.Tensor
        window sum (property)

    Methods
    ----------
    __init__(self, length:int=1024, name:str=None, order:float=None, **kwargs) -> None
        Window instance initialization.
    cosine_window(length:int=1024, order:float=1.0, **kwargs) -> torch.Tensor
        Generate cosine window data (staticmethod).
    kaiser_window(length:int=1024, order:float=1.0, **kwargs) -> torch.Tensor
        Generate kaiser window data (staticmethod).
    set_data(self, *, data:torch.Tensor=None, name:str=None, order:float=None) -> None
        Set self.window container for given input data with matching length or given name and order.
    total(self) -> torch.Tensor
        Window sum (property).
    from_cosine(cls, length:int=1024, order:float, **kwargs) -> 'Window'
        Create Window instance using cosine window (classmethod).
    from_kaiser(cls, length:int=1024, order:float, **kwargs) -> 'Window'
        Create Window instance using kaiser window (classmethod).
    __repr__(self) -> str
        String representation.
    __len__(self) -> int
        Window length.
    __call__(self, *, data:torch.Tensor=None, name:str=None, order:float=None) -> None
        Invoke set_data() method.
        Set self.window container for given input data with matching length or given name and order.

    """
    def __init__(self, length:int=1024, name:str=None, order:float=None, **kwargs) -> None:
        """
        Window instance initialization.

        If name==None and order==None, self.window container is equal to ones.

        Parameters
        ----------
        length: int
            window length
        name: str
            window name ('cosine_window' or 'kaiser_window')
        order: float
            window order parameter, positive float
        **kwargs:
            passed to torch.ones (use dtype and device)

        Returns
        -------
        None

        """
        self.length = length
        self.window = torch.ones(self.length, **kwargs)
        self.dtype = self.window.dtype
        self.device = self.window.device
        self.name = name
        self.order = order
        if self.name != None and self.order != None:
            self.set_data(name=self.name, order=self.order)


    @staticmethod
    def cosine_window(length:int=1024, order:float=1.0, **kwargs) -> torch.Tensor:
        """
        Generate cosine window data (staticmethod).

        Parameters
        ----------
        length: int
            window length
        order: float
            window order parameter, positive float
        **kwargs:
            dtype, device

        Returns
        -------
        window data (torch.Tensor)

        """
        window = torch.linspace(0.0, (length - 1.0)/length, length, **kwargs)
        factor = 2.0**order*torch.exp(torch.lgamma(torch.tensor(1.0 + order, **kwargs)))**2
        factor = factor/torch.exp(torch.lgamma(torch.tensor(1.0 + 2.0*order, **kwargs)))
        torch.cos(2.0*numpy.pi*(window - 0.5), out=window)
        window.add_(1.0).pow_(order).mul_(factor)
        return window


    @staticmethod
    def kaiser_window(length:int=1024, order:float=1.0, **kwargs) -> torch.Tensor:
        """
        Generate kaiser window data (staticmethod).

        Parameters
        ----------
        length: int
            window length
        order: float
            window order parameter, positive float
        **kwargs:
            dtype, device

        Returns
        -------
        window data (torch.Tensor)

        """
        window = torch.linspace(0.0, (length - 1.0)/length, length, **kwargs)
        factor = 1.0/torch.i0(torch.tensor(numpy.pi*order, **kwargs))
        window.sub_(0.5).pow_(2.0).mul_(-4.0).add_(1.0)
        window.sqrt_().mul_(numpy.pi*order).i0_().mul_(factor)
        return window


    def set_data(self, *, data:torch.Tensor=None, name:str=None, order:float=None) -> None:
        """
        Set self.window container for given input data with matching length or given name and order.

        If data == None, generate window data using staticmethod for given name and order
        If data != None, copy given input data to self.window, other parameters are ignored

        Parameters
        ----------
        data: torch.Tensor
            window data with matching length
        name: str
            window name ('cosine_window' or 'kaiser_window')
        order: float
            window order parameter, positive float

        Returns
        -------
        None

        """
        if data == None and name != None and order != None:
            self.name = name
            self.order = order
            self.window = type.__getattribute__(self.__class__, self.name)(self.length, self.order, dtype=self.dtype, device=self.device)
            return

        if data != None:
            if data.shape != self.window.shape:
                raise Exception(f'WINDOW: expected input data length {self.length}, got {len(data)}')
            self.name = None
            self.order = None
            self.window.copy_(data)
            return

        raise Exception(f'WINDOW: wrong input arguments in set_data')


    @property
    def total(self) -> torch.Tensor:
        """
        Window sum (property).

        """
        return torch.sum(self.window)


    @classmethod
    def from_cosine(cls, length:int=1024, order:float=1.0, **kwargs) -> 'Window':
        """
        Create Window instance using cosine window (classmethod).

        Parameters
        ----------
        length: int
            window length
        order: float
            window order parameter, positive float
        **kwargs:
            dtype, device

        Returns
        -------
        Window instance (cosine window of given length and order)

        """
        return Window(length, 'cosine_window', order, **kwargs)


    @classmethod
    def from_kaiser(cls, length:int=1024, order:float=1.0, **kwargs) -> 'Window':
        """
        Create Window instance using kaiser window (classmethod).

        Parameters
        ----------
        length: int
            window length
        order: float
            window order parameter, positive float
        **kwargs:
            dtype, device

        Returns
        -------
        Window instance (kaiser window of given length and order)

        """
        return Window(length, 'kaiser_window', order, **kwargs)


    def __repr__(self) -> str:
        """
        String representation.

        """
        return f'{self.__class__.__name__}{self.length, self.name, self.order}'


    def __len__(self) -> int:
        """
        Window length.

        """
        return self.length


    def __call__(self, *, data:torch.Tensor=None, name:str=None, order:float=None) -> None:
        """
        Invoke set_data() method.
        Set self.window container for given input data with matching length or given name and order.

        If data == None, generate window data using staticmethod for given name and order
        If data != None, copy given input data to self.window, other parameters are ignored

        Parameters
        ----------
        data: torch.Tensor
            window data with matching length
        name: str
            window name ('cosine_window' or 'kaiser_window')
        order: float
            window order parameter, positive float

        Returns
        -------
        None

        """
        self.set_data(name=name, order=order, data=data)


def main():
    pass

if __name__ == '__main__':
    main()
