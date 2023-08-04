"""
Derivative module.
Compute mapping derivatives.

"""

import torch

from typing import Callable
from math import factorial

from .util import flatten


def derivative(degree:int,
               mapping:Callable[[torch.Tensor, ...], torch.Tensor],
               *args:tuple[torch.Tensor],
               jacobian:Callable[[Callable], Callable]=torch.func.jacfwd) -> tuple:
    """
    Compute derivatives for given mapping upto given total monomial degree.

    Note, derivatives are computed with respect to the first argument

    Parameters
    ----------
    degree: int
        maximum total monomial degree
    mapping: Callable[[torch.Tensor, ...], torch.Tensor]
        mapping
    *args: tuple[torch.Tensor]
        mapping arguments
    jacobian: Callable[[Callable], Callable]
        jacobian (jacrev or jacfwd)

    Returns
    -------
    tuple of derivative tensors (value, jacobian, hessian, ...) (tuple)

    """
    x, *xs = args
    def local(x, *xs):
        y = mapping(x, *xs)
        return y, y

    for _ in range(degree):
        def local(x, *xs, local=local):
            y, ys = jacobian(local, has_aux=True)(x, *xs)
            return y, (ys, y)

    _, y = local(x, *xs)

    return tuple(flatten(y))


def evaluate(state:torch.Tensor,
             table:tuple[torch.Tensor]) -> torch.Tensor:
    """
    Evaluate deviation state for a given tuple of derivatives (value, jacobian, hessian, ...).

    Parameters
    ----------
    state: torch.Tensor
        deviation state
    table: tuple[torch.Tensor]
        tuple of derivatives (value, jacobian, hessian, ...)

    Returns
    -------
    evaluation result (torch.Tensor)

    """
    result, *matrices = table
    result = torch.clone(result)
    for degree, matrix in enumerate(matrices):
        value = torch.clone(matrix)
        for _ in range(degree + 1):
            value @= state
        result += value/factorial(degree + 1)
    return result


def monomial_index(dimension:int,
                   degree:int,
                   *,
                   dtype:torch.dtype=torch.int64,
                   device:torch.device=torch.device('cpu')) -> torch.Tensor:
    """
    Generate monomial indeces for given dimension and total monomial degree.

    Note, table is generated only for a given degree

    Parameters
    ----------
    dimension: int
        dimension (number of input variables/parameters)
    degree: int
        maximum total monomial degree
    dtype: torch.dtype
        data type
    device: torch.device
        data device

    Returns
    -------
    indices table (torch.Tensor)

    """
    if degree == 1:
        return torch.eye(dimension, dtype=dtype, device=device)

    unit = monomial_index(dimension, 1, dtype=dtype, device=device)
    keys = monomial_index(dimension, degree - 1, dtype=dtype, device=device)

    return torch.cat([keys + row for row in unit])


def monomial_table(dimension:int,
                   degree:int,
                   table:tuple[torch.Tensor],
                   dtype:torch.dtype=torch.int64,
                   device:torch.device=torch.device('cpu')) -> dict:
    """
    Generate table representation of derivatives.

    Parameters
    ----------
    dimension: int
        dimension (number of input variables/parameters)
    degree: int
        maximum total monomial degree
    table: tuple[torch.Tensor]
        value, jacobian, hessian, ...
    dtype: torch.dtype
        data type
    device: torch.device
        data device

    Returns
    -------
    table representation of derivatives

    """
    result, *matrices = table

    mark = []
    grid = {}
    if degree != 0:
        for i in range(len(result)):
            grid[i] = {}

    for i in range(1, 1 + degree):
        keys = tuple(map(tuple, monomial_index(dimension, i, dtype=dtype, device=device).tolist()))
        mark.extend(dict.fromkeys(keys))
        factor = 1.0/factorial(i)
        for j in range(len(result)):
            values = factor*(table[i][j].flatten())
            for key, value in zip(keys, values):
                if key not in grid[j]:
                    grid[j][key]  = value
                else:
                    grid[j][key] += value

    if degree != 0:
        grid = dict(zip(mark, torch.stack([torch.stack([*grid[i].values()]) for i in range(len(result))]).T))

    zero = tuple(torch.zeros(dimension, dtype=dtype, device=device).tolist())

    return {**{zero: result}, **grid}


def evaluate_table(state:torch.Tensor,
                   table:dict[torch.Tensor]) -> torch.Tensor:
    """
    Evaluate deviation state for a given monomial table.

    Parameters
    ----------
    state: torch.Tensor
        deviation state
    table: dict[torch.Tensor]
        monomial table

    Returns
    -------
    evaluation result (torch.Tensor)

    """
    keys = torch.tensor([*table.keys()], dtype=torch.int64, device=state.device)
    values = torch.stack([*table.values()])
    return (values.T * (state**keys).prod(-1)).sum(-1)


def main():
    pass

if __name__ == '__main__':
    main()