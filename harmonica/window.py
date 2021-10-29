"""
Window module.
Initialize Window class instance.

"""

import torch

class Window:
    """
    Returns
    ----------
    Window class instance.

    Window(length:int, name:str, order:float, **kwargs)
    Window.from_cosine(length:int, order:float, **kwargs)
    Window.from_kaiser(length:int, order:float, **kwargs)

    Parameters
    ----------
    length: int
        Window data length.
    name: str
        Window name (a static method of Window class).
        See cosine_window or kaiser_window implementations.
    order: float
        Window order pararameter.
    **kwargs:
        dtype, device

    Attributes
    ----------
    length: int
        Window data length.
    name: str
        Window name (a static method of Window class).
        See cosine_window or kaiser_window implementations.
    order: float
        Window order pararameter.
    dtype: torch.dtype
        Window data type.
    device: torch.device
        Window data device.
    data: torch.Tensor
        Window data.
    total: torch.Tensor
        Window sum (property).

    Methods
    ----------
    set_data(self, *, name:str=None, order:float=None, tensor:torch.Tensor=None) -> None
        Set (new) window data for given name and order or input tensor with matching length.
    cosine_window(length:int, order:float, data:torch.Tensor) -> None
        Cosine window generator (staticmethod).
    kaiser_window(length:int, order:float, data:torch.Tensor) -> None
        Cosine window generator (staticmethod).
    from_cosine(cls, length:int, order:float, **kwargs) -> 'Window'
        Create Window instance using cosine window (classmethod).
    from_kaiser(cls, length:int, order:float, **kwargs) -> 'Window'
        Create Window instance using kaiser window (classmethod).
    __repr__(self) -> str
        String representation.
    __len__(self) -> int
        Window length.
    __getitem__(self, idx:int) -> torch.Tensor:
        Return window data at given index.
    __call__(self, *, name:str=None, order:float=None, tensor:torch.Tensor=None) -> None
        Set (new) window data for given name and order or input tensor with matching length. Invoke set_data.

    """

    def __init__(self, length: int = 1024, name: str = None, order: float = None, **kwargs) -> None:
        self.length = length
        self.data = torch.ones(length, **kwargs)
        self.dtype = self.data.dtype
        self.device = self.data.device
        self.name = name
        self.order = order
        if name != None and order != None:
            self.set_data(name=name, order=order)


    def set_data(self, *, name:str=None, order:float=None, tensor:torch.Tensor=None) -> None:
        """
        Set (new) window data for given name and order or input tensor with matching length.

        Modify data container inplace.
        tensor != None, copy given tensor to window data.
        tensor == None, generate window data using a static method for given name and order.

        Parameters
        ----------
        name: str
            valid window name ('cosine_window' or 'kaiser_window')
        order: float
            window order
        tensor: torch.Tensor
            tensor with window data

        Returns
        -------
        None

        """
        if tensor == None and name != None and order != None:
            self.name = name
            self.order = order
            type.__getattribute__(self.__class__, self.name)(self.length, self.order, self.data)
            return

        if tensor != None and self.data.shape == tensor.shape:
            self.data.copy_(tensor)
            return

        raise Exception(f'WINDOW: wrong input arguments in set_data.')


    @property
    def total(self) -> torch.Tensor:
        """
        Window sum (property).

        """
        return torch.sum(self.data)


    @staticmethod
    @torch.jit.script
    def cosine_window(length:int, order:float, data:torch.Tensor) -> None:
        """
        Cosine window generator (staticmethod).

        Modify data container inplace.

        Parameters
        ----------
        length: int
            window data length
        order: float
            window order
        data: torch.Tensor
            window data

        Returns
        -------
        None

        """
        pi = 2.0*torch.acos(torch.zeros(1, dtype=data.dtype, device=data.device))
        num = 2.0**order*torch.exp(torch.lgamma(torch.tensor(1.0 + order, dtype=data.dtype, device=data.device)))**2
        den = torch.exp(torch.lgamma(torch.tensor(1.0 + 2.0*order, dtype=data.dtype, device=data.device)))
        factor = num/den
        torch.linspace(0.0, (length - 1.0)/length, length, out=data)
        torch.cos(2.0*pi*(data - 0.5), out=data)
        data.add_(1.0).pow_(order).mul_(factor)


    @staticmethod
    @torch.jit.script
    def kaiser_window(length:int, order:float, data:torch.Tensor) -> None:
        """
        Kaiser window generator (staticmethod).

        Modify data container inplace.

        Parameters
        ----------
        length: int
            window data length
        order: float
            window order
        data: torch.Tensor
            window data

        Returns
        -------
        None

        """
        pi = 2.0*torch.acos(torch.zeros(1, dtype=data.dtype, device=data.device))
        factor = 1.0/torch.i0(pi*order)
        torch.linspace(0.0, (length - 1.0) / length, length, out=data)
        data.sub_(0.5).pow_(2.0).mul_(-4.0).add_(1.0)
        torch.sqrt(data, out=data)
        data.mul_(pi*order)
        torch.i0(data, out=data)
        data.mul_(factor)


    @classmethod
    def from_cosine(cls, length:int, order:float, **kwargs) -> "Window":
        """
        Create Window instance using cosine window (classmethod).

        Parameters
        ----------
        length: int
            window length
        order: float
            window order
        **kwargs:
            dtype, device

        Returns
        -------
        Window instance (cosine window)

        """
        return Window(length, "cosine_window", order, **kwargs)


    @classmethod
    def from_kaiser(cls, length:int, order:float, **kwargs) -> "Window":
        """
        Create Window instance using kaise window (classmethod).

        Parameters
        ----------
        length: int
            window length
        order: float
            window order
        **kwargs:
            dtype, device

        Returns
        -------
        Window instance (kaise window)

        """
        return Window(length, "kaiser_window", order, **kwargs)


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


    def __getitem__(self, idx:int) -> torch.Tensor:
        """
        Return window data at given index.

        Parameters
        ----------
        idx: int
            index

        Returns
        -------
        torch.Tensor:
            value at idx

        """
        return self.data[idx]


    def __call__(self, *, name:str=None, order:float=None, tensor:torch.Tensor=None) -> None:
        """
        Set (new) window data for given name and order or input tensor with matching length.

        Invoke set_data.
        Modify data container inplace.
        tensor != None, copy given tensor to window data.
        tensor == None, generate window data using a static method for given name and order.

        Parameters
        ----------
        name: str
            valid window name ('cosine_window' or 'kaiser_window')
        order: float
            window order
        tensor: torch.Tensor
            tensor with window data

        Returns
        -------
        None

        """
        self.set_data(name=name, order=order, tensor=tensor)


def main():
    pass

if __name__ == '__main__':
    main()
