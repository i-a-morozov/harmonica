"""
Parameterization module.
Compute (coupled) twiss parameters.

"""

import numpy
import torch
import functorch

from .util import mod


def id_symplectic(d:int,
                  *,
                  dtype:torch.dtype=torch.float64,
                  device:torch.device=torch.device('cpu')) -> torch.Tensor:
    """
    Generate symplectic 'identity' matrix for a given input (configuration space) dimension.

    Note, symplectic block is [[0, 1], [-1, 0]]

    Parameters
    ----------
    d: int
        configuration space dimension
    dtype: torch.dtype
        data type
    device: torch.device
        data device

    Returns
    -------
    symplectic 'identity' matrix (torch.Tensor)

    """
    block = torch.tensor([[0, 1], [-1, 0]], dtype=dtype, device=device)
    return torch.block_diag(*[block for _ in range(d)])


def is_symplectic(matrix:torch.Tensor,
                  *,
                  epsilon:float=1.0E-12) -> bool:
    """
    Test symplectic condition for a given input matrix elementwise.

    Parameters
    ----------
    matrix: torch.Tensor
        input matrix
    epsilon: float
        tolerance epsilon

    Returns
    -------
    test result (bool)

    """
    d = len(matrix) // 2
    s = id_symplectic(d, dtype=matrix.dtype, device=matrix.device)
    return all(epsilon > (matrix.T @ s @ matrix - s).abs().flatten())


def to_symplectic(matrix:torch.Tensor) -> torch.Tensor:
    """
    Perform symplectification of a given input matrix.

    Parameters
    ----------
    matrix: torch.Tensor
        input matrix

    Returns
    -------
    symplectified matrix (torch.Tensor)

    """
    d = len(matrix) // 2
    i = torch.eye(2*d, dtype=matrix.dtype, device=matrix.device)
    s = id_symplectic(d, dtype=matrix.dtype, device=matrix.device)
    v = s @ (i - matrix) @ (i + matrix).inverse()
    w = 0.5*(v + v.T)
    return (s + w).inverse() @ (s - w)


def inverse_symplectic(matrix:torch.Tensor) -> torch.Tensor:
    """
    Compute inverse of a given input symplectic matrix.

    Parameters
    ----------
    matrix: torch.Tensor
        input matrix

    Returns
    -------
    inverse matrix (torch.Tensor)

    """
    d = len(matrix) // 2
    s = id_symplectic(d, dtype=matrix.dtype, device=matrix.device)
    return -s @ matrix.T @ s


def twiss_compute(matrix:torch.Tensor,
                  *,
                  epsilon:float=1.0E-12) -> tuple:
    """
    Compute fractional tunes, normalization matrix and Wolski twiss matrices for a given one-turn input matrix.

    Input matrix can have arbitrary even dimension
    In-plane 'beta' is used for ordering

    Symplectic block is [[0, 1], [-1, 0]]
    Complex block is 1/sqrt(2)*[[1, 1j], [1, -1j]]
    Rotation block is [[cos(alpha), sin(alpha)], [-sin(alpha), cos(alpha)]]

    Parameters
    ----------
    matrix: torch.Tensor
        input one-turn matrix
    epsilon: float
        tolerance epsilon

    Returns
    -------
    fractional tunes [T_1, ..., T_K], normalization matrix N and Wolski twiss matrices W = [W_1, ..., W_K] (tuple)
    M = N R N^-1 = ... + W_I S sin(2*pi*T_I) - (W_I S)**2 cos(2*pi*T_I) + ... for I = 1, ..., K

    """
    dtype = matrix.dtype
    device = matrix.device

    rdtype = torch.tensor(1, dtype=dtype).abs().dtype
    cdtype = (1j*torch.tensor(1, dtype=dtype)).dtype

    d = len(matrix) // 2

    b_p = torch.tensor([[1, 0], [0, 1]], dtype=rdtype, device=device)
    b_s = torch.tensor([[0, 1], [-1, 0]], dtype=rdtype, device=device)
    b_c = 0.5**0.5*torch.tensor([[1, +1j], [1, -1j]], dtype=cdtype, device=device)

    m_p = torch.stack([torch.block_diag(*[b_p*(i == j) for i in range(d)]) for j in range(d)])
    m_s = torch.block_diag(*[b_s for _ in range(d)])
    m_c = torch.block_diag(*[b_c for _ in range(d)])

    l, v = torch.linalg.eig(matrix)

    if (l.abs() - epsilon > 1).sum():
        return None, None, None

    l, v = l.reshape(d, -1), v.T.reshape(d, -1, 2*d)
    for i, (v1, v2) in enumerate(v):
        v[i] /= (-1j*(v1 @ m_s.to(cdtype) @ v2)).abs().sqrt()

    for i in range(d):
        o = torch.imag(l[i].log()).argsort()
        l[i], v[i] = l[i, o], v[i, o]

    t = 1.0 - l.log().abs().mean(-1)/(2.0*numpy.pi)

    n = torch.cat([*v]).H
    n = (n @ m_c).real
    w = torch.stack([n @ m_p[i] @ n.T for i in range(d)])

    o = torch.tensor([w[i].diag().argmax() for i in range(d)]).argsort()
    t, v = t[o], v[o]
    n = torch.cat([*v]).H
    n = (n @ m_c).real

    f = torch.stack(torch.hsplit(n.T @ m_s @ n - m_s, d)).abs().sum((1, -1)) > epsilon
    for i in range(d):
        if f[i]:
            t[i] = (1.0 - t[i]).abs()
            v[i] = v[i].conj()

    n = torch.cat([*v]).H
    n = (n @ m_c).real

    r = []
    for i in range(d):
        a = (n[2*i, 2*i + 1] + 1j*n[2*i, 2*i]).angle() - 0.5*numpy.pi
        b = torch.tensor([[a.cos(), a.sin()], [-a.sin(), a.cos()]])
        r.append(b)

    n = n @ torch.block_diag(*r)
    for i in range(d):
        w[i] = n @ m_p[i] @ n.T

    return t, n, w


def twiss_propagate(twiss:torch.Tensor,
                    matrix:torch.Tensor) -> torch.Tensor:
    """
    Propagate Wolski twiss matrices for a given batch of transport matrices.

    Parameters
    ----------
    twiss: torch.Tensor
        Wolski twiss matrices
    matrix: torch.Tensor
        batch of transport matrices

    Returns
    -------
    Wolski twiss matrices for each transport matrix (torch.Tensor)

    """
    return torch.stack([torch.matmul(matrix, torch.matmul(twiss[i], matrix.swapaxes(1, -1))) for i in range(len(twiss))]).swapaxes(0, 1)
    # return matrix.unsqueeze(1) @ twiss.unsqueeze(0) @ matrix.swapaxes(1, -1).unsqueeze(1)


def twiss_phase_advance(normal:torch.Tensor,
                        matrix:torch.Tensor) -> tuple:
    """
    Compute phase advances and final normalization matrices for a given normalization matrix and a given batch of transport matrices.

    Note, output phase advance is mod 2*pi

    Parameters
    ----------
    normal: torch.Tensor
        normalization matrix
    matrix: torch.Tensor
        batch of transport matrices

    Returns
    -------
    phase advances and final normalization matrices for each transport matrix (tuple)

    """
    d = len(normal) // 2

    index = torch.arange(d, dtype=torch.int64, device=normal.device)

    local = torch.matmul(matrix, normal)

    angle = mod(torch.arctan2(local[:, 2*index, 2*index + 1], local[:, 2*index, 2*index]), 2.0*numpy.pi).T
    angle_cos = angle.cos()
    angle_sin = angle.sin()

    rotation = torch.stack([+angle_cos, -angle_sin, +angle_sin, +angle_cos])
    rotation = torch.stack([torch.block_diag(*block.reshape(-1, 2, 2)) for block in rotation.swapaxes(0, -1)])

    return angle.T, local @ rotation


def normal_to_wolski(normal:torch.Tensor) -> torch.Tensor:
    """
    Compute Wolski twiss matrices for a given batch of normalization matrices.

    Parameters
    ----------
    normal: torch.Tensor
        batch of normalization matrices

    Returns
    -------
    Wolski twiss matrices for each normalization matrix (torch.Tensor)

    """
    dtype = normal.dtype
    device = normal.device

    *_, d = normal.shape
    d //= 2

    projection = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)
    projection = torch.stack([torch.block_diag(*[projection*(i == j) for i in range(d)]) for j in range(d)])

    return torch.stack([torch.matmul(normal, torch.matmul(projection[i], normal.swapaxes(1, -1))) for i in range(d)]).swapaxes(0, 1)


def wolski_to_normal(twiss:torch.Tensor) -> torch.Tensor:
    """
    Compute normalization matrices for a given batch of Wolski twiss matrices.

    Note, normalization matrix is computed using fake one-turn matrix with random tunes

    Parameters
    ----------
    twiss: torch.Tensor
        batch of Wolski twiss matrices

    Returns
    -------
    normalization matrices for each Wolski twiss (torch.Tensor)

    """
    dtype = twiss.dtype
    device = twiss.device

    s, d, *_ = twiss.shape

    projection = torch.tensor([[0, 1], [-1, 0]], dtype=dtype, device=device)
    projection = torch.block_diag(*[projection for _ in range(d)])

    tune = torch.rand(d, dtype=dtype, device=device)
    fake = torch.zeros((2*d, 2*d), dtype=dtype, device=device)

    result = []
    for i in range(s):
        for j in range(d):
            local = twiss[i, j] @ projection
            fake += local * tune[j].sin() - local @ local * tune[j].cos()
        _, normal, _ = twiss_compute(fake)
        fake.zero_()
        result.append(normal)

    return torch.stack(result)


def parametric_normal(n11:torch.Tensor,
                      n33:torch.Tensor,
                      n21:torch.Tensor,
                      n43:torch.Tensor,
                      n13:torch.Tensor,
                      n31:torch.Tensor,
                      n14:torch.Tensor,
                      n41:torch.Tensor,
                      *,
                      dtype:torch.dtype=torch.float64,
                      device:torch.device=torch.device('cpu')) -> torch.Tensor:

    """
    Generate 'parametric' 4x4 normalization matrix for given free elements.

    Note, elements denoted with X are computed for given free elements using symplectic condition constraints
    For n11 > 0 & n33 > 0, all matrix elements are not singular in uncoupled limit

        n11   0 n13 n14
        n21   X   X   X
        n31   X n33   0
        n41   X n43   X

    Parameters
    ----------
    n11, n33, n21, n43, n13, n31, n14, n41: torch.Tensor
        free matrix elements
    dtype: torch.dtype
        data type
    device: torch.device
        data device

    Returns
    -------
    normalization matrix (torch.Tensor)

    """
    normal = torch.zeros((4, 4), dtype=dtype, device=device)

    normal[0, 0] = n11
    normal[0, 1] = 0.0
    normal[0, 2] = n13
    normal[0, 3] = n14

    normal[1, 0] = n21
    normal[1, 1] = n33*(n11 + n14*(n33*n41 - n31*n43))/(n11*(n11*n33 - n13*n31))
    normal[1, 2] = (n13*n21 + n33*n41 - n31*n43)/n11
    normal[1, 3] = (n14*n21*n33 -n31 + n14*n31/n11*(n31*n43 - n13*n21 - n33*n41))/(n11*n33 - n13*n31)

    normal[2, 0] = n31
    normal[2, 1] = n14*n33/n11
    normal[2, 2] = n33
    normal[2, 3] = 0.0

    normal[3, 0] = n41
    normal[3, 1] = (n13*(-1.0 - (n14*n33*n41)/n11) + n14*n33*n43)/(n11*n33 - n13*n31)
    normal[3, 2] = n43
    normal[3, 3] = (n11 + n14*(n33*n41 - n31*n43))/(n11*n33 - n13*n31)

    return normal


def lb_normal(a1x:torch.Tensor,
              b1x:torch.Tensor,
              a2x:torch.Tensor,
              b2x:torch.Tensor,
              a1y:torch.Tensor,
              b1y:torch.Tensor,
              a2y:torch.Tensor,
              b2y:torch.Tensor,
              u:torch.Tensor,
              v1:torch.Tensor,
              v2:torch.Tensor,
              *,
              epsilon:float=1.0E-12,
              dtype:torch.dtype=torch.float64,
              device:torch.device=torch.device('cpu')) -> torch.Tensor:
    """
    Generate Lebedev-Bogacz normalization matrix.

    Note, a1x, b1x, a2y, b2y are 'in-plane' twiss parameters

    Parameters
    ----------
    a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2: torch.Tensor
        Lebedev-Bogacz twiss parameters
    epsilon: float
        tolerance epsilon
    dtype: torch.dtype
        data type
    device: torch.device
        data device

    Returns
    -------
    normalization matrix (torch.Tensor)
    M = N R N^-1

    """
    cv1, sv1 = v1.cos(), v1.sin()
    cv2, sv2 = v2.cos(), v2.sin()

    if b1x < epsilon: b1x *= 0.0
    if b2x < epsilon: b2x *= 0.0
    if b1y < epsilon: b1y *= 0.0
    if b2y < epsilon: b2y *= 0.0

    return torch.tensor(
        [
            [b1x.sqrt(), 0, b2x.sqrt()*cv2, -b2x.sqrt()*sv2],
            [-a1x/b1x.sqrt(), (1 - u)/b1x.sqrt(), (-a2x*cv2 + u*sv2)/b2x.sqrt(), (a2x*sv2 + u*cv2)/b2x.sqrt()],
            [b1y.sqrt()*cv1, -b1y.sqrt()*sv1, b2y.sqrt(), 0],
            [(-a1y*cv1 + u*sv1)/b1y.sqrt(), (a1y*sv1 + u*cv1)/b1y.sqrt(), -a2y/b2y.sqrt(), (1 - u)/b2y.sqrt()]
        ], dtype=dtype, device=device
    ).nan_to_num(posinf=0.0, neginf=0.0)


def cs_normal(ax:torch.Tensor,
              bx:torch.Tensor,
              ay:torch.Tensor,
              by:torch.Tensor,
              *,
              dtype:torch.dtype=torch.float64,
              device:torch.device=torch.device('cpu')) -> torch.Tensor:
    """
    Generate Courant-Snyder normalization matrix.

    Parameters
    ----------
    ax, bx, ay, by: torch.Tensor
        Courant-Snyder twiss parameters
    dtype: torch.dtype
        data type
    device: torch.device
        data device

    Returns
    -------
    normalization matrix (torch.Tensor)
    M = N R N^-1

    """
    return torch.tensor(
        [
            [bx.sqrt(), 0, 0, 0],
            [-ax/bx.sqrt(), 1/bx.sqrt(), 0, 0],
            [0, 0, by.sqrt(), 0],
            [0, 0, -ay/by.sqrt(), 1/by.sqrt()]
        ], dtype=dtype, device=device
    )


def wolski_to_lb(twiss:torch.Tensor) -> torch.Tensor:
    """
    Convert Wolski twiss matrices to Lebedev-Bogacz twiss parameters.

    [a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2]
    a1x, b1x, a2y, b2y are 'in-plane' twiss parameters

    Parameters
    ----------
    twiss: torch.Tensor
        Wolski twiss matrices

    Returns
    -------
    Lebedev-Bogacz twiss (torch.Tensor)

    """
    a1x = -twiss[0, 0, 1]
    b1x = +twiss[0, 0, 0]
    a2x = -twiss[1, 0, 1]
    b2x = +twiss[1, 0, 0]

    a1y = -twiss[0, 2, 3]
    b1y = +twiss[0, 2, 2]
    a2y = -twiss[1, 2, 3]
    b2y = +twiss[1, 2, 2]

    u = 1/2*(1 + a1x**2 - a1y**2 - b1x*twiss[0, 1, 1] + b1y*twiss[0, 3, 3])

    cv1 = (1/torch.sqrt(b1x*b1y)*twiss[0, 0, 2]).nan_to_num(nan=-1.0)
    sv1 = (1/u*(a1y*cv1 + 1/torch.sqrt(b1x)*(torch.sqrt(b1y)*twiss[0, 0, 3]))).nan_to_num(nan=0.0)

    cv2 = (1/torch.sqrt(b2x*b2y)*twiss[1, 0, 2]).nan_to_num(nan=+1.0)
    sv2 = (1/u*(a2x*cv2 + 1/torch.sqrt(b2y)*(torch.sqrt(b2x)*twiss[1, 1, 2]))).nan_to_num(nan=0.0)

    v1 = torch.arctan2(sv1, cv1)
    v2 = torch.arctan2(sv2, cv2)

    return torch.tensor([a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2], dtype=twiss.dtype, device=twiss.device)


def lb_to_wolski(a1x:torch.Tensor,
                 b1x:torch.Tensor,
                 a2x:torch.Tensor,
                 b2x:torch.Tensor,
                 a1y:torch.Tensor,
                 b1y:torch.Tensor,
                 a2y:torch.Tensor,
                 b2y:torch.Tensor,
                 u:torch.Tensor,
                 v1:torch.Tensor,
                 v2:torch.Tensor,
                 *,
                 epsilon:float=1.0E-12,
                 dtype:torch.dtype=torch.float64,
                 device:torch.device=torch.device('cpu')) -> torch.Tensor:
    """
    Convert Lebedev-Bogacz twiss parameters to Wolski twiss matrices.

    Parameters
    ----------
    a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2: torch.Tensor
        Lebedev-Bogacz twiss parameters
    epsilon: float
        tolerance epsilon
    dtype: torch.dtype
        data type
    device: torch.device
        data device

    Returns
    -------
    Wolski twiss matrices (torch.Tensor)

    """
    n = lb_normal(a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2, epsilon=epsilon, dtype=dtype, device=device)

    p1 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=n.dtype, device=n.device)
    p2 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=n.dtype, device=n.device)

    w1 = torch.matmul(n, torch.matmul(p1, n.T))
    w2 = torch.matmul(n, torch.matmul(p2, n.T))

    return torch.stack([w1, w2])


def wolski_to_cs(twiss:torch.Tensor) -> torch.Tensor:
    """
    Convert Wolski twiss matrices to Courant-Snyder twiss parameters.

    Parameters
    ----------
    twiss: torch.Tensor
        Wolski twiss matrices

    Returns
    -------
    Courant-Snyder twiss (torch.Tensor)

    """
    ax = -twiss[0, 0, 1]
    bx = +twiss[0, 0, 0]

    ay = -twiss[1, 2, 3]
    by = +twiss[1, 2, 2]

    return torch.tensor([ax, bx, ay, by], dtype=twiss.dtype, device=twiss.device)


def cs_to_wolski(ax:torch.Tensor,
                 bx:torch.Tensor,
                 ay:torch.Tensor,
                 by:torch.Tensor,
                 *,
                 dtype:torch.dtype=torch.float64,
                 device:torch.device=torch.device('cpu')) -> torch.Tensor:
    """
    Convert Courant-Snyder twiss parameters to Wolski twiss matrices.

    Parameters
    ----------
    ax, bx, ay, by: torch.Tensor
        Courant-Snyder twiss parameters
    dtype: torch.dtype
        data type
    device: torch.device
        data device

    Returns
    -------
    Wolski twiss matrices (torch.Tensor)

    """
    n = cs_normal(ax, bx, ay, by, dtype=dtype, device=device)

    p1 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=n.dtype, device=n.device)
    p2 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=n.dtype, device=n.device)

    w1 = torch.matmul(n, torch.matmul(p1, n.T))
    w2 = torch.matmul(n, torch.matmul(p2, n.T))

    return torch.stack([w1, w2])


@functorch.vmap
def matrix_uncoupled(ax1:torch.Tensor,
                     bx1:torch.Tensor,
                     ax2:torch.Tensor,
                     bx2:torch.Tensor,
                     fx12:torch.Tensor,
                     ay1:torch.Tensor,
                     by1:torch.Tensor,
                     ay2:torch.Tensor,
                     by2:torch.Tensor,
                     fy12:torch.Tensor) -> torch.Tensor:
    """
    Generate uncoupled transport matrices using CS twiss data between given locations.

    Input twiss parameters should be 1D tensors with matching length

    Parameters
    ----------
    ax1, bx1, ay1, by1: torch.Tensor
        twiss parameters at the 1st location(s)
    ax2, bx2, ay2, by2: torch.Tensor
        twiss parameters at the 2nd location(s)
    fx12, fy12: torch.Tensor
        twiss phase advance between locations

    Returns
    -------
    uncoupled transport matrices (torch.Tensor)

    """
    cx = torch.cos(fx12)
    sx = torch.sin(fx12)

    mx11 = torch.sqrt(bx2/bx1)*(cx + ax1*sx)
    mx12 = torch.sqrt(bx1*bx2)*sx
    mx21 = -(1 + ax1*ax2)/torch.sqrt(bx1*bx2)*sx + (ax1 - ax2)/torch.sqrt(bx1*bx2)*cx
    mx22 = torch.sqrt(bx1/bx2)*(cx - ax2*sx)

    rx1 = torch.stack([mx11, mx12])
    rx2 = torch.stack([mx21, mx22])

    mx = torch.stack([rx1, rx2])

    cy = torch.cos(fy12)
    sy = torch.sin(fy12)

    my11 = torch.sqrt(by2/by1)*(cy + ay1*sy)
    my12 = torch.sqrt(by1*by2)*sy
    my21 = -(1 + ay1*ay2)/torch.sqrt(by1*by2)*sy + (ay1 - ay2)/torch.sqrt(by1*by2)*cy
    my22 = torch.sqrt(by1/by2)*(cy - ay2*sy)

    ry1 = torch.stack([my11, my12])
    ry2 = torch.stack([my21, my22])

    my = torch.stack([ry1, ry2])

    return torch.block_diag(mx, my)


@functorch.vmap
def matrix_coupled(normal1:torch.Tensor,
                   normal2:torch.Tensor,
                   advance12:torch.Tensor) -> torch.Tensor:
    """
    Generate coupled transport matrices using normalization matrices and phase advances between given locations.

    Parameters
    ----------
    normal1: torch.Tensor
        normalization matrices at the 1st location(s)
    normal2: torch.Tensor
        normalization matrices at the 2nd location(s)
    advance12: torch.Tensor
        phase advance between locations

    Returns
    -------
    coupled transport matrices (torch.Tensor)

    """
    cos_advance12 = advance12.cos()
    sin_advance12 = advance12.sin()

    r1 = torch.stack([+cos_advance12, -sin_advance12])
    r2 = torch.stack([+sin_advance12, +cos_advance12])

    rotation = torch.block_diag(*torch.stack([r1, r2]).swapaxes(0, -1))

    return normal2 @ rotation @ normal1.inverse()


@functorch.vmap
def matrix_rotation(advance12:torch.Tensor) -> torch.Tensor:
    """
    Generate rotation matrices using phase advances between given locations.

    Parameters
    ----------
    advance12: torch.Tensor
        phase advance between locations

    Returns
    -------
    rotation matrices (torch.Tensor)

    """
    cos_advance12 = advance12.cos()
    sin_advance12 = advance12.sin()

    r1 = torch.stack([+cos_advance12, -sin_advance12])
    r2 = torch.stack([+sin_advance12, +cos_advance12])

    return torch.block_diag(*torch.stack([r1, r2]).swapaxes(0, -1))


def invariant_uncoupled(normal:torch.Tensor,
                        trajectory:torch.Tensor) -> torch.Tensor:
    """
    Compute uncoupled invariants for given normalization matrix and trajectory.

    Note, trajectory is assumed to have the form [..., [qx_i, px_i, qy_i, py_i], ...]

    Parameters
    ----------
    normal: torch.Tensor
        normalization matrix
    trajectory: torch.Tensor
        trajectory

    Returns
    -------
    [jx, jy] for each turn (torch.Tensor)

    """
    qx, px, qy, py = trajectory.T

    n11, n12, n13, n14 = normal[0]
    n21, n22, n23, n24 = normal[1]
    n31, n32, n33, n34 = normal[2]
    n41, n42, n43, n44 = normal[3]

    jx = 0.5*n11**2*px**2 - n11*n21*px*qx + 0.5*(1.0/n11**2 + n21**2)*qx**2
    jy = 0.5*n33**2*py**2 - n33*n43*py*qy + 0.5*(1.0/n33**2 + n43**2)*qy**2

    return torch.stack([jx, jy])


def invariant_coupled(normal:torch.Tensor,
                      trajectory:torch.Tensor) -> torch.Tensor:
    """
    Compute coupled invariants for given normalization matrix and trajectory.

    Note, trajectory is assumed to have the form [..., [qx_i, px_i, qy_i, py_i], ...]

    Parameters
    ----------
    normal: torch.Tensor
        normalization matrix
    trajectory: torch.Tensor
        trajectory

    Returns
    -------
    [jx, jy] for each turn (torch.Tensor)

    """
    qx, px, qy, py = trajectory.T

    n11, n12, n13, n14 = normal[0]
    n21, n22, n23, n24 = normal[1]
    n31, n32, n33, n34 = normal[2]
    n41, n42, n43, n44 = normal[3]

    jx  = (n11**2*px**2)/2
    jx += -n11*n21*px*qx
    jx += n11*n31*px*py
    jx += -n11*n41*px*qy
    jx += ((-2*n11**3*n13*n21**2*n31*n33+n11**4*n21**2*n33**2+n11**2*(n13**2*n21**2*n31**2+n33**2)+2*n11*n14*n33**2*(n33*n41-n31*n43)+n14**2*n33**2*(n33*n41-n31*n43)**2)*qx**2)/(2*n11**2*(n13*n31-n11*n33)**2)
    jx += -(((-n11**2*n13*n21*n31**2+n11**3*n21*n31*n33+n11*n14*n33**2+n14**2*n33**2*(n33*n41-n31*n43))*py*qx)/(n11**2*(-n13*n31+n11*n33)))
    jx += ((-2*n11**3*n13*n21*n31*n33*n41+n11**4*n21*n33**2*n41+n13*n14**2*n33**2*n41*(-n33*n41+n31*n43)+n11**2*(-n13*n33+n13**2*n21*n31**2*n41+n14*n33**2*n43)+n11*n14*n33*(n14*n33*n43*(n33*n41-n31*n43)+n13*(-2*n33*n41+n31*n43)))*qx*qy)/(n11**2*(n13*n31-n11*n33)**2)
    jx += 1/2*(n31**2+(n14**2*n33**2)/n11**2)*py**2
    jx += -(((-n11**2*n13*n31**2*n41+n11**3*n31*n33*n41-n13*n14**2*n33**2*n41+n11*n14*n33*(-n13+n14*n33*n43))*py*qy)/(n11**2*(-n13*n31+n11*n33)))
    jx += ((-2*n11**3*n13*n31*n33*n41**2+n11**4*n33**2*n41**2+n13**2*n14**2*n33**2*n41**2+2*n11*n13*n14*n33*n41*(n13-n14*n33*n43)+n11**2*(n13**2*(1+n31**2*n41**2)-2*n13*n14*n33*n43+n14**2*n33**2*n43**2))*qy**2)/(2*n11**2*(n13*n31-n11*n33)**2)

    jy  = 1/2*(n13**2+n14**2)*px**2
    jy += (((n13**2+n14**2)*n31*(n13*n21+n33*n41-n31*n43)-n11*(-n14*n31+n14**2*n21*n33+n13*n33*(n13*n21+n33*n41-n31*n43)))*px*qx)/(n11*(-n13*n31+n11*n33))
    jy += n13*n33*px*py
    jy += ((-n13**2*n31*n43+n14**2*(n33*n41-n31*n43)+n11*(n14+n13*n33*n43))*px*qy)/(n13*n31-n11*n33)
    jy += (((n13**2+n14**2)*n31**2*(n13*n21+n33*n41-n31*n43)**2-2*n11*n31*(n13*n21+n33*n41-n31*n43)*(-n14*n31+n14**2*n21*n33+n13*n33*(n13*n21+n33*n41-n31*n43))+n11**2*(n33**2*(n13**2*n21**2+n14**2*n21**2+2*n13*n21*n33*n41+n33**2*n41**2)-2*n31*n33*(n14*n21+n33*(n13*n21+n33*n41)*n43)+n31**2*(1+n33**2*n43**2)))*qx**2)/(2*n11**2*(n13*n31-n11*n33)**2)
    jy += -((n33*(n13*n21+n33*n41-n31*n43)*py*qx)/n11)
    jy += -(((-n31*(n13*n21+n33*n41-n31*n43)*(n13**2*n31*n43+n14**2*(-n33*n41+n31*n43))+n11**2*(-n33*(n14*n21+n33*(n13*n21+n33*n41)*n43)+n31*(1+n33**2*n43**2))+n11*(2*n13**2*n21*n31*n33*n43+n14*(-2*n31+n14*n21*n33)*(-n33*n41+n31*n43)+n13*n31*(n14*n21+2*n33*n43*(n33*n41-n31*n43))))*qx*qy)/(n11*(n13*n31-n11*n33)**2))
    jy += (n33**2*py**2)/2
    jy += -n33*n43*py*qy
    jy += ((n13**2*n31**2*n43**2+n14**2*(n33*n41-n31*n43)**2+n11**2*(1+n33**2*n43**2)-2*n11*(n13*n31*n33*n43**2+n14*(-n33*n41+n31*n43)))*qy**2)/(2*(n13*n31-n11*n33)**2)

    return torch.stack([jx, jy])


def invariant(normal:torch.Tensor,
              trajectory:torch.Tensor) -> torch.Tensor:
    """
    Compute invariants for given normalization matrix and trajectory.

    Note, trajectory is assumed to have the form [..., [qx_i, px_i, qy_i, py_i], ...]

    Parameters
    ----------
    normal: torch.Tensor
        normalization matrix
    trajectory: torch.Tensor
        trajectory

    Returns
    -------
    [jx, jy] for each turn (torch.Tensor)

    """
    qx, px, qy, py = torch.inner(inverse_symplectic(normal), trajectory)

    jx = 1/2*(qx**2 + px**2)
    jy = 1/2*(qy**2 + py**2)

    return torch.stack([jx, jy])


def momenta(matrix:torch.Tensor,
            qx1:torch.Tensor,
            qx2:torch.Tensor,
            qy1:torch.Tensor,
            qy2:torch.Tensor) -> torch.Tensor:
    """
    Compute momenta at position 1 for given transport matrix and coordinates at 1 & 2.

    Parameters
    ----------
    matrix: torch.Tensor
        transport matrix between
    qx1, qx2, qy1, qy2: torch.Tensor
        x & y coordinates at 1 & 2

    Returns
    -------
    px and py at 1 (torch.Tensor)

    """
    m11, m12, m13, m14 = matrix[0]
    m21, m22, m23, m24 = matrix[1]
    m31, m32, m33, m34 = matrix[2]
    m41, m42, m43, m44 = matrix[3]

    px1  = qx1*(m11*m34 - m14*m31)/(m14*m32 - m12*m34)
    px1 += qx2*m34/(m12*m34 - m14*m32)
    px1 += qy1*(m13*m34 - m14*m33)/(m14*m32 - m12*m34)
    px1 += qy2*m14/(m14*m32 - m12*m34)

    py1  = qx1*(m11*m32 - m12*m31)/(m12*m34 - m14*m32)
    py1 += qx2*m32/(m14*m32 - m12*m34)
    py1 += qy1*(m12*m33 - m13*m32)/(m14*m32 - m12*m34)
    py1 += qy2*m12/(m12*m34 - m14*m32)

    return torch.stack([px1, py1])


def main():
    pass

if __name__ == '__main__':
    main()