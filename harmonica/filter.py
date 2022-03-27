"""
Filter TbT data.
Noise estimation.

"""

import torch

from kneed import KneeLocator

from .data import Data


class Filter():
    """
    Returns
    ----------
    Filter class instance.

    Parameters
    ----------
    data: 'Data'
        Data instance
        Noise estimation is perfomed using work container
        Filtering is performed using work container (modify work container inplace)
    random_seed: int
        Random seed (randomized range estimation)

    Attributes
    ----------
    data: 'Data'
        Data instance
    random_seed: int
        Random seed (randomized range estimation)

    Methods
    ----------
    make_matrix(signal:torch.Tensor) -> torch.Tensor
        Compute Hankel matrix representation for a given batch of signals (staticmethod).
    make_signal(matrix:torch.Tensor) -> torch.Tensor
        Compute signal representation for a given batch of Hankel matrices (staticmethod).
    randomized_range(rank:int, count:int, matrix:torch.Tensor) -> torch.Tensor
        Randomized range estimation.
    randomized_range(rank:int, count:int, matrix:torch.Tensor, random_seed:int=0) -> torch.Tensor
        Randomized range estimation based on QR decomposition (staticmethod).
    svd_list(rank:int, matrix:torch.Tensor, cpu:bool=True)
        Compute list of singular values for a given batch of matrices (staticmethod).
    svd_list_randomized(cls, rank:int, matrix:torch.Tensor, buffer:int=8, count:int=8, random_seed:int=0, cpu:bool=True) -> torch.Tensor
        Compute list of singular values for a given batch of matrices (classmethod).
    locate(data:torch.Tensor, **kwargs) -> int
        Find the point of maximum curvature.
    svd_truncation(rank:int, matrix:torch.Tensor, cpu:bool=True) -> tuple
        Compute SVD truncation for given rank and a batch of matrices (staticmethod).
    svd_truncation_randomized(cls, rank:int, matrix:torch.Tensor, buffer:int=8, count:int=8, random_seed:int=0, cpu:bool=True) -> tuple
        Compute randomized SVD truncation for given rank and a batch of matrices (classmethod).
    svd_optimal(matrix:torch.Tensor, cpu:bool=True) -> tuple
        Estimate optimal truncation rank and noise value for a given batch of matrices.
    rpca_shrink(threshold:float, matrix:torch.Tensor) -> torch.Tensor
         Replace input matrix elements with zeros if the absolute value is less than a given threshold (staticmethod).
    rpca_threshold(threshold:float, matrix:torch.Tensor, cpu:bool=True) -> torch.Tensor
        SVD truncation based on singular values thresholding (staticmethod).
    rpca(cls, matrix:torch.Tensor, limit:int=128, factor:float=1.E-7, cpu:bool=True) -> tuple
        RPCA by principle component pursuit by alternating directions (classmethod).
    estimate_noise(self, *, limit:int=32, cpu:bool=True) -> tuple
        Estimate optimal truncation rank and noise value for each signal in TbT.
    filter_svd(self, *, rank:int=0, limit:int=32, random:bool=False, buffer:int=8, count:int=4, cpu:bool=True) -> torch.Tensor
        Perform TbT filtering based on (randomized) SVD truncation of full TbT matrix.
    filter_hankel(self, *, rank:int=0, size:int=32, random:bool=False, buffer:int=8, count:int=8, cpu:bool=True) -> torch.Tensor
        Perform TbT filtering based on (randomized) SVD truncation of individual TbT signals.
    filter_rpca(self, *, limit:int=512, factor:float=1.E-9, cpu:bool=True) -> tuple
        Perform TbT filtering based on RPCA of full TbT matrix.

    """
    def __init__(self, data:'Data', *, random_seed:int=0) -> None:
        self.data = data
        self.random_seed = random_seed


    @staticmethod
    def make_matrix(signal:torch.Tensor) -> torch.Tensor:
        """
        Compute Hankel matrix representation for a given batch of signals.

        If signal length is 2n, corresponding matrix shape is (n + 1, n).

        Parameters
        ----------
        signal: torch.Tensor
            batch of input signals

        Returns
        -------
        torch.Tensor
            batch of Hankel matrices

        """
        dtype = signal.dtype
        device = signal.device
        size, length = signal.shape
        length = length // 2
        matrix = torch.zeros((size, length + 1, length), dtype=dtype, device=device)
        for i in range(length + 1):
            matrix[:, i].copy_(signal[:, i:i + length])
        return matrix


    @staticmethod
    def make_signal(matrix:torch.Tensor) -> torch.Tensor:
        """
        Compute signal representation for a given batch of Hankel matrices.

        If matrix shape is (n + 1, n), corresponding output signal length is 2n.
        Each signal is computed by averaging skew diagonals of corresponding matrix.

        Parameters
        ----------
        matrix: torch.Tensor
            batch of input Hankel matrices

        Returns
        -------
        torch.Tensor
            batch of signals

        """
        dtype = matrix.dtype
        device = matrix.device
        matrix = torch.transpose(matrix, 1, 2).flip(1)
        size, length, _ = matrix.shape
        signal = torch.zeros((size, 2*length), dtype=dtype, device=device)
        for i, j in enumerate(range(-length + 1, length + 1)):
            signal[:, i] = torch.mean(torch.diagonal(matrix, dim1=1, dim2=2, offset=j), 1)
        return signal


    @staticmethod
    def randomized_range(rank:int, count:int, matrix:torch.Tensor, random_seed:int=0) -> torch.Tensor:
        """
        Randomized range estimation based on QR decomposition.

        Randomized SVD truncation auxiliary function.

        Parameters
        ----------
        rank: int
            range rank (number of columns)
        count: int
            number of iterations to use in randomized range
        matrix: torch.Tensor
            input batch of matrices
        random_seed: int
            random seed

        Returns
        -------
        torch.Tensor
            batch of estimated range matrices

        """
        dtype = matrix.dtype
        device = matrix.device
        generator = torch.Generator(device=device)
        generator.manual_seed(random_seed)
        size, m, n = matrix.shape
        transpose = torch.clone(torch.transpose(matrix, 1, 2))
        projection1 = torch.randn((size, n, rank), dtype=dtype, device=device, generator=generator)
        projection2 = torch.zeros((size, m, rank), dtype=dtype, device=device)
        for _ in range(count):
            projection2 = torch.linalg.qr(torch.matmul(matrix, projection1)).Q
            projection1 = torch.linalg.qr(torch.matmul(transpose, projection2)).Q
        return torch.linalg.qr(torch.matmul(matrix, projection1)).Q


    @staticmethod
    def svd_list(rank:int, matrix:torch.Tensor, cpu:bool=True) -> torch.Tensor:
        """
        Compute list of singular values for a given batch of matrices.

        Note, all singular values are computed, but only requested number is returned.

        Parameters
        ----------
        rank: int
            number of singular values to return
        matrix: torch.Tensor
            input batch of matrices
        cpu: bool
            flag to use CPU for SVD computation

        Returns
        -------
        torch.Tensor
            list of singular values

        """
        return torch.linalg.svdvals(matrix.cpu() if cpu else matrix)[:, :rank].to(matrix.device)


    @classmethod
    def svd_list_randomized(cls, rank:int, matrix:torch.Tensor,
                            buffer:int=8, count:int=8, random_seed:int=0, cpu:bool=True) -> torch.Tensor:
        """
        Compute list of singular values for a given batch of matrices.

        Parameters
        ----------
        rank: int
            number of singular values to return
        matrix: torch.Tensor
            input batch of matrices
        buffer: int
            number of extra dimensions (randomized range estimation)
        count: int
            number of iterations (randomized range estimation)
        random_seed: int
            random seed (randomized range estimation)
        cpu: bool
            flag to use CPU for SVD computation

        Returns
        -------
        torch.Tensor:
            list of singular values

        """
        projection = cls.randomized_range(rank + buffer, count, matrix, random_seed)
        matrix = torch.matmul(torch.transpose(projection, 1, 2), matrix)
        return torch.linalg.svdvals(matrix.cpu() if cpu else matrix)[:, :rank].to(matrix.device)


    @staticmethod
    def locate(data:torch.Tensor, **kwargs) -> int:
        """
        Find the point of maximum curvature.

        """
        return KneeLocator(range(len(data)), data.cpu().numpy(), **kwargs).knee


    @staticmethod
    def svd_truncation(rank:int, matrix:torch.Tensor, cpu:bool=True) -> tuple:
        """
        Compute SVD truncation for given rank and a batch of matrices.

        Note, all matrices are truncated using the same rank.

        Parameters
        ----------
        rank: int
            truncation rank (number of singular values to keep)
        matrix: torch.Tensor
            input batch of matrices
        cpu: bool
            flag to use CPU for SVD computation

        Returns
        -------
        tuple:
            SVD values and truncated batch of matrices

        """
        device = matrix.device
        u, s, v = torch.linalg.svd(matrix.cpu() if cpu else matrix, full_matrices=False)
        u = u[:, :, :rank].to(device)
        s = s[:, :rank].to(device)
        v = v[:, :rank, :].to(device)
        return s, torch.matmul(u, torch.matmul(torch.diag_embed(s), v))


    @classmethod
    def svd_truncation_randomized(cls, rank:int, matrix:torch.Tensor,
                                  buffer:int=8, count:int=8, random_seed:int=0, cpu:bool=True) -> tuple:
        """
        Compute randomized SVD truncation for given rank and a batch of matrices.

        Note, all matrices are truncated using the same rank.

        Parameters
        ----------
        rank: int
            truncation rank (number of singular values to keep)
        matrix: torch.Tensor
            input batch of matrices
        buffer: int
            number of extra dimensions (randomized range estimation)
        count: int
            number of iterations (randomized range estimation)
        random_seed: int
            random seed (randomized range estimation)
        cpu: bool
            flag to use CPU for SVD computation

        Returns
        -------
        tuple:
            SVD values and truncated batch of matrices

        """
        device = matrix.device
        projection = cls.randomized_range(rank + buffer, count, matrix, random_seed)
        matrix = torch.matmul(torch.transpose(projection, 1, 2), matrix)
        u, s, v = torch.linalg.svd(matrix.cpu() if cpu else matrix, full_matrices=False)
        u = u[:, :, :rank].to(device)
        s = s[:, :rank].to(device)
        v = v[:, :rank, :].to(device)
        u = torch.matmul(projection, u)
        return s, torch.matmul(u, torch.matmul(torch.diag_embed(s), v))


    @staticmethod
    def svd_optimal(matrix:torch.Tensor, cpu:bool=True) -> tuple:
        """
        Estimate optimal truncation rank and noise value for a given batch of matrices.

        Note, all singular values are computed.
        Approximate expression is used for estimation.

        Parameters
        ----------
        matrix: torch.Tensor
            input batch of matrices
        cpu: bool
            flag to use CPU for SVD computation

        Returns
        -------
        tuple:
            optimal rank and noise value for each matrix

        """
        dtype = matrix.dtype
        device = matrix.device
        size, m, n = matrix.shape
        s = torch.linalg.svdvals(matrix.cpu() if cpu else matrix).to(device)
        median = torch.median(s, dim=-1).values
        beta = torch.tensor(min(m, n)/max(m, n), dtype=dtype, device=device)
        omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
        psi = torch.sqrt(2.0*(beta + 1.0) + 8.0*beta/(beta + 1.0 + torch.sqrt(beta**2 + 14.0*beta + 1.0)))
        tau = omega*median
        lmbd = torch.sqrt(torch.tensor(max(m, n), dtype=dtype, device=device))
        return torch.sum((s - tau.reshape(-1, 1) > 0), 1), tau/psi/lmbd


    @staticmethod
    def rpca_shrink(threshold:float, matrix:torch.Tensor) -> torch.Tensor:
        """
        Replace input matrix elements with zeros if the absolute value is less than a given threshold.

        RPCA auxiliary function.

        Parameters
        ----------
        threshold: float
            threshold value
        matrix: torch.Tensor
            input matrix

        Returns
        -------
        torch.Tensor:
            thresholded matrix

        """
        dtype = matrix.dtype
        device = matrix.device
        return torch.sign(matrix)*torch.maximum(
            (torch.abs(matrix) - threshold), torch.zeros(matrix.shape, dtype=dtype, device=device))


    @staticmethod
    def rpca_threshold(threshold:float, matrix:torch.Tensor, cpu:bool=True) -> torch.Tensor:
        """
        SVD truncation based on singular values thresholding.

        RPCA auxiliary function.

        Parameters
        ----------
        threshold: float
            threshold value
        matrix: torch.Tensor
            input matrix
        cpu: bool
            flag to use CPU for SVD computation

        Returns
        -------
        torch.Tensor:
            thresholded matrix

        """
        dtype = matrix.dtype
        device = matrix.device
        u, s, v = torch.linalg.svd(matrix.cpu() if cpu else matrix, full_matrices=False)
        u = u.to(device)
        s = s.to(device)
        v = v.to(device)
        s = torch.diag(s)
        s = torch.sign(s)*torch.maximum((torch.abs(s) - threshold), torch.zeros(s.shape, dtype=dtype, device=device))
        return torch.matmul(torch.matmul(u, s), v)


    @classmethod
    def rpca(cls, matrix:torch.Tensor, limit:int=512, factor:float=1.E-9, cpu:bool=True) -> tuple:
        """
        RPCA by principle component pursuit by alternating directions.

        Note, acts on a single matrix.
        For normal noise estimation, factor value should be small.

        Parameters
        ----------
        matrix: torch.Tensor
            input matrix
        limit: int
            maximum number of iterations
        factor: float
            tolerance factor
        cpu: bool
            flag to use CPU for SVD computation

        Returns
        -------
        tuple:
            number of elapsed iterations, error at the last iteration, low rank matrix and sparse matrix

        """
        dtype = matrix.dtype
        device = matrix.device
        m, n = matrix.shape
        mu = 0.25/torch.linalg.norm(matrix, ord=1)*m*n
        mu_inv = 1/mu
        lmbd = 1/torch.sqrt(torch.tensor(max(m, n), dtype=dtype, device=device))
        tolerance = factor*torch.linalg.norm(matrix, ord='fro')
        sparse = torch.zeros_like(matrix)
        low = torch.zeros_like(matrix)
        work = torch.zeros_like(matrix)
        error = torch.zeros_like(matrix)
        count = 0
        while count < limit:
            error = mu_inv*work
            low = cls.rpca_threshold(mu_inv, matrix - sparse + error, cpu=cpu)
            sparse = cls.rpca_shrink(mu_inv*lmbd, matrix - low + error)
            error = matrix - low - sparse
            work += mu*error
            count += 1
            value = torch.linalg.norm(error, ord='fro')
            if value < tolerance:
                break
        return count, value, low, sparse


    def estimate_noise(self, *, limit:int=64, cpu:bool=True,
                       randomized:bool=False, buffer:int=8, count:int=8, random_seed:int=0) -> tuple:
        """
        Estimate optimal truncation rank and noise value for each signal in TbT.

        Note, data from work container is used for estimation.

        Parameters
        ----------
        limit: int
            number of columns to use for estimation
        cpu: bool
            flag to use CPU for SVD computation
        randomized: bool
            flag to use randomized SVD
        buffer: int
            number of extra dimensions (randomized range estimation)
        count: int
            number of iterations (randomized range estimation)
        random_seed: int
            random seed (randomized range estimation)

        Returns
        -------
        tuple:
            estimated optimal rank and noise value for each signal in TbT

        """
        if not randomized:
            return self.svd_optimal(self.make_matrix(self.data.work)[:, :, :limit], cpu=cpu)

        matrix = self.make_matrix(self.data.work)
        projection = self.randomized_range(limit + buffer, count, matrix, random_seed)
        matrix = torch.matmul(torch.transpose(projection, 1, 2), matrix)
        return self.svd_optimal(matrix, cpu=cpu)


    def filter_svd(self, *, rank:int=0, limit:int=32,
                   random:bool=False, buffer:int=8, count:int=8, cpu:bool=True) -> torch.Tensor:
        """
        Perform TbT filtering based on (randomized) SVD truncation of full TbT matrix.

        Input from work, result in work.
        If rank is zero, estimate rank by optimal SVD truncation.

        Parameters
        ----------
        rank: int
            truncation rank, if zero, rank is estimated with optimal SVD truncation
        limit: int
            number of columns to use for optimal SVD truncation, used if rank is zero
        random: bool
            flag to used randomized SVD for truncation
        buffer: int
            number of extra dimensions (randomized range estimation)
        count: int
            number of iterations (randomized range estimation)
        cpu: bool
            flag to use CPU for SVD computation

        Returns
        -------
        torch.Tensor:
            singular values

        """
        matrix = self.data.work.unsqueeze(0)

        if rank == 0:
            rank, _ = self.__class__.svd_optimal(matrix[:, :, :limit], cpu=cpu)
            rank = rank.item()

        if not random:
            value, matrix = self.__class__.svd_truncation(rank, matrix, cpu=cpu)
        else:
            value, matrix = self.__class__.svd_truncation_randomized(rank, matrix, buffer=buffer,
                                                                     count=count, random_seed=self.random_seed, cpu=cpu)

        self.data.work.copy_(matrix.squeeze())
        return value


    def filter_hankel(self, *, rank:int=0, limit:int=32,
                      random:bool=False, buffer:int=8, count:int=8, cpu:bool=True) -> torch.Tensor:
        """
        Perform TbT filtering based on (randomized) SVD truncation of individual TbT signals.

        Input from work, result in work.
        If rank is zero, estimate rank by optimal SVD truncation.
        Maximum rank is used for truncation.

        Parameters
        ----------
        rank: int
            truncation rank, if zero, rank is estimated with optimal SVD truncation
        limit: int
            number of columns to use for optimal SVD truncation, used if rank is zero
        random: bool
            flag to used randomized SVD for truncation
        buffer: int
            number of extra dimensions (randomized range estimation)
        count: int
            number of iterations (randomized range estimation)
        cpu: bool
            flag to use CPU for SVD computation

        Returns
        -------
        torch.Tensor:
            singular values for each signal

        """
        matrix = self.__class__.make_matrix(self.data.work)

        if rank == 0:
            rank, _ = self.__class__.svd_optimal(matrix[:, :, :limit], cpu=cpu)
            rank = rank.max().item()

        if not random:
            value, matrix = self.__class__.svd_truncation(rank, matrix, cpu=cpu)
        else:
            value, matrix = self.__class__.svd_truncation_randomized(rank, matrix, buffer=buffer,
                                                                     count=count, random_seed=self.random_seed, cpu=cpu)

        self.data.work.copy_(self.__class__.make_signal(matrix))
        return value


    def filter_rpca(self, *, limit:int=512, factor:float=1.E-9, cpu:bool=True) -> tuple:
        """
        Perform TbT filtering based on RPCA of full TbT matrix.

        Input from work, result in work.
        Note, if RPCA filtering is desired for an individual signal, create TbT data from it first.

        Parameters
        ----------
        limit: int
            maximum number of iterations
        factor: float
            tolerance factor
        cpu: bool
            flag to use CPU for SVD computation

        Returns
        -------
        tuple:
            number of elapsed iterations, last itration error and 'noise' matrix

        """
        count, error, matrix, noise = self.__class__.rpca(self.data.work, limit=limit, factor=factor, cpu=cpu)
        self.data.work.copy_(matrix)
        return count, error, noise


def main():
    pass

if __name__ == '__main__':
    main()