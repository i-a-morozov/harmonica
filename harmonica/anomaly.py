"""
Anomaly detection module.
Basic threshold detector and sklean wrappers for dbscan, lof and isolation forest.

"""

import torch

from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def threshold(data:torch.Tensor,
              min_value:torch.Tensor,
              max_value:torch.Tensor) -> torch.Tensor:
    """
    Generate mask using min & max values.

    Note, data in (max_value, min_value) interval have True values.

    Parameters
    ----------
    data: torch.Tensor
        input 1D data
    min_value: float
        min value threshold
    max_value: float
        max value threshold

    Returns
    -------
    mask (torch.Tensor)

    """
    return (data > min_value.reshape(-1, 1))*(data < max_value.reshape(-1, 1))


def dbscan(data:torch.Tensor,
           epsilon:float,
           **kwargs) -> torch.Tensor:
    """
    Generate mask using DBSCAN.

    Note, data in the largest cluster have True values.

    Parameters
    ----------
    data: torch.Tensor
        input data with shape (n_samples, n_features)
    epsilon: float
        DBSCAN epsilon
    **kwargs:
        passed to DBSCAN()

    Returns
    -------
    mask (torch.Tensor)

    """
    group = DBSCAN(eps=epsilon, **kwargs).fit(data.cpu().numpy())
    label = Counter(group.labels_)
    label = max(label, key=label.get)
    return torch.tensor(group.labels_ == label).to(data.device)


def local_outlier_factor(data:torch.Tensor,
                         *,
                         contamination:float=0.01,
                         **kwargs) -> torch.Tensor:
    """
    Generate mask using LocalOutlierFactor.

    Note, data label 1 have True values.

    Parameters
    ----------
    data: torch.Tensor
        input data with shape (n_samples, n_features)
    contamination: float
        contamination fraction
    **kwargs:
        passed to LocalOutlierFactor()

    Returns
    -------
    mask (torch.Tensor)

    """
    group = LocalOutlierFactor(contamination=contamination, novelty=True, **kwargs).fit(data.cpu().numpy())
    label = group.predict(data.cpu().numpy())
    return torch.tensor(1 == label).to(data.device)


def isolation_forest(data:torch.Tensor,
                     *,
                     contamination:float=0.01,
                     **kwargs) -> torch.Tensor:
    """
    Generate mask using IsolationForest.

    Note, data label 1 have True values.

    Parameters
    ----------
    data: torch.Tensor
        input data with shape (n_samples, n_features)
    contamination: float
        contamination fraction
    **kwargs:
        passed to IsolationForest()

    Returns
    -------
    mask (torch.Tensor)

    """
    group = IsolationForest(contamination=contamination, **kwargs).fit(data.cpu().numpy())
    label = group.predict(data.cpu().numpy())
    return torch.tensor(1 == label).to(data.device)


def score(size:int,
          mask:torch.Tensor) -> torch.Tensor:
    """
    Count number of marked (with False) elements.

    Note, 1D mask is reshaped into (size, -1)

    Parameters
    ----------
    size: int
        reshape size
    mask: torch.Tensor
        mask

    Returns
    -------
    score (torch.Tensor)

    """
    return mask.logical_not().reshape(size, -1).sum(-1)


def main():
    pass

if __name__ == '__main__':
    main()