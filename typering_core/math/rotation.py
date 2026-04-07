import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_to_matrix(r:np.ndarray) -> np.ndarray:
    ''' Convert quaternions to rotation matrices.
    Args:
        r: np.ndarray, [*, 4], the quaternions, in (x, y, z, w) scalar-last format.
    Returns:
        np.ndarray, [*, 3, 3], the rotation matrices.
    '''
    rotations = R.from_quat(r.reshape(-1, 4))
    return rotations.as_matrix().reshape(*r.shape[:-1], 3, 3).astype(r.dtype)


def matrix_to_quat(r:np.ndarray) -> np.ndarray:
    ''' Reverse of quat_to_matrix.
    '''
    rotations = R.from_matrix(r.reshape(-1, 3, 3))
    return rotations.as_quat().reshape(*r.shape[:-2], 4).astype(r.dtype)


def quat_to_rotvec(r:np.ndarray) -> np.ndarray:
    ''' Convert quaternions to rotation vectors.
    Args:
        r: np.ndarray, [*, 4], the quaternions.
    Returns:
        np.ndarray, [*, 3], the rotation vectors.
    '''
    rotations = R.from_quat(r.reshape(-1, 4))
    return rotations.as_rotvec().reshape(*r.shape[:-1], 3).astype(r.dtype)


def rotvec_to_quat(r:np.ndarray) -> np.ndarray:
    ''' Reverse of quat_to_rotvec.
    '''
    rotations = R.from_rotvec(r.reshape(-1, 3))
    return rotations.as_quat().reshape(*r.shape[:-1], 4).astype(r.dtype)


def matrix_to_rotvec(r:np.ndarray) -> np.ndarray:
    ''' Convert rotation matrices to rotation vectors.
    Args:
        r: np.ndarray, [*, 3, 3], the rotation matrices.
    Returns:
        np.ndarray, [*, 3], the rotation vectors.
    '''
    rotations = R.from_matrix(r.reshape(-1, 3, 3))
    return rotations.as_rotvec().reshape(*r.shape[:-2], 3).astype(r.dtype)


def rotvec_to_matrix(r:np.ndarray) -> np.ndarray:
    ''' Reverse of matrix_to_rotvec.
    '''
    rotations = R.from_rotvec(r.reshape(-1, 3))
    return rotations.as_matrix().reshape(*r.shape[:-1], 3, 3).astype(r.dtype)


def partial_rotvec_to_matrix(r:np.ndarray) -> np.ndarray:
    ''' Calculate the partial derivative of rotation matrices over rotation vectors.
    Math:
        Suppose R = rotvec2matrix(r), then out = \cfrac{\partial R}{\partial r}.
    Args:
        r: np.ndarray, [*, 3], the rotation vectors.
    Returns:
        np.ndarray, [*, 9, 3], the partial derivative, 9 is the row expansion of the rotation matrices.
    '''
    r = r.reshape(-1, 3)
    theta = np.linalg.norm(r, axis=1)
    n = r / np.clip(theta[:,None], 1e-8, None)
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    s, c = np.sin(theta), np.cos(theta)
    zeros = np.zeros_like(theta)
    res = np.array([
        [2*nx*(1-c),    zeros,      zeros   ],
        [ny*(1-c),      nx*(1-c),   -s      ],
        [nz*(1-c),      s,          nx*(1-c)],
        [ny*(1-c),      nx*(1-c),   s       ],
        [zeros,         2*ny*(1-c), zeros   ],
        [-s,            nz*(1-c),   ny*(1-c)],
        [nz*(1-c),      -s,         nx*(1-c)],
        [s,             nz*(1-c),   ny*(1-c)],
        [zeros,         zeros,      2*nz*(1-c)]])
    res = (1/np.clip(theta, 1e-8, None))[:,None,None] * res.transpose(2, 0, 1)
    return res.reshape(*r.shape[:-1], 9, 3)
    

def matrix_to_r6d(r:np.ndarray) -> np.ndarray:
    ''' Convert rotation matrices to r6d format.
    Args:
        r: np.ndarray, [*, 3, 3], the rotation matrices.
    Returns:
        np.ndarray, [*, 6], the r6d format.
    '''
    shape = r.shape
    r = r.reshape(-1, 3, 3)[:, :, :2].transpose(0, 2, 1)
    r = r.reshape(*shape[:-2], 6)
    return r


def r6d_to_matrix(r:np.ndarray) -> np.ndarray:
    ''' Reverse of matrix_to_r6d.
    '''
    shape = r.shape
    r = r.reshape(-1, 2, 3).transpose(0, 2, 1)
    z = np.cross(r[:, :, 0], r[:, :, 1])[:,:,None]
    r = np.concatenate([r, z], axis=2)
    r = r.reshape(*shape[:-1], 3, 3)
    return r


def normalize_vector(v:np.ndarray) -> np.ndarray:
    ''' Normalize 3d vectors.
    Args:
        v: np.ndarray, [*, 3], the 3d vectors.
    Returns:
        np.ndarray, [*, 3], the normalized vectors.
    '''
    shape = v.shape
    v = v.reshape(-1, 3)
    v /= np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-8, None)
    v = v.reshape(shape)
    return v


def normalize_r6d(r:np.ndarray) -> np.ndarray:
    ''' Normalize r6d rotation, such that the two vectors are unit and orthogonal.
    Args:
        r: np.ndarray, [*, 6], rotations in r6d format.
    Returns:
        np.ndarray, [*, 6], normalized rotations.
    '''
    shape = r.shape
    r = r.reshape(-1, 2, 3)
    x, y = normalize_vector(r[:, 0, :]), normalize_vector(r[:, 1, :])
    sum, diff = normalize_vector(x + y), normalize_vector(x - y)
    x, y = (sum + diff) / 2**0.5, (sum - diff) / 2**0.5
    r = np.concatenate([x, y], axis=1)
    return r.reshape(*shape[:-1], 6)


def quat_mul(q1:np.ndarray, q2:np.ndarray) -> np.ndarray:
    ''' Multiply two quaternions.
    Args:
        q1: np.ndarray, [*, 4], in (x, y, z, w) scalar-last format.
        q2: np.ndarray, [*, 4], in (x, y, z, w) scalar-last format.
    Returns:
        np.ndarray, [*, 4], the product of the two quaternions.
    '''
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    v1, v2 = q1[:, :3], q2[:, :3]
    s1, s2 = q1[:, 3:], q2[:, 3:]
    v = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    s = s1 * s2 - np.sum(v1 * v2, axis=1, keepdims=True)
    q = np.concatenate([v, s], axis=1).reshape(q1.shape)
    return q


def quat_inv(q:np.ndarray) -> np.ndarray:
    ''' Inverse of quaternions.
    Args:
        q: np.ndarray, [*, 4], in (x, y, z, w) scalar-last format.
    Returns:
        np.ndarray, [*, 4], the inverse of the quaternions.
    '''
    q = q.reshape(-1, 4)
    q = np.concatenate([-q[:, :3], q[:, 3:]], axis=1)
    return q.reshape(q.shape)


def quat_power(q:np.ndarray, p:float) -> np.ndarray:
    ''' Power of quaternions.
    Args:
        q: np.ndarray, [*, 4], in (x, y, z, w) scalar-last format.
        p: float, the power.
    Returns:
        np.ndarray, [*, 4], the power of the quaternions.
    '''
    q = q.reshape(-1, 4)
    half_theta = np.arccos(q[:, 3:])
    n = normalize_vector(q[:, :3])
    v = np.sin(half_theta * p) * n
    s = np.cos(half_theta * p)
    q = np.concatenate([v, s], axis=1).reshape(q.shape)
    return q


def quat_slerp(q1:np.ndarray, q2:np.ndarray, t:float) -> np.ndarray:
    ''' Spherical linear interpolation between two sequences of quaternions.
        Result q = slerp(q1, q2, t) = q1 * (1-t) + q2 * t (under the semantics of slerp).
    Args:
        q1: np.ndarray, [*, 4], in (x, y, z, w) scalar-last format.
        q2: np.ndarray, [*, 4], in (x, y, z, w) scalar-last format.
        t: float, the interpolation factor.
    Returns:
        np.ndarray, [*, 4], the interpolated quaternions.
    '''
    return quat_mul(quat_power(quat_mul(q2, quat_inv(q1)), t), q1)


def matrix_slerp(r1:np.ndarray, r2:np.ndarray, t:float) -> np.ndarray:
    ''' Spherical linear interpolation between two sequences of rotation matrices.
        Result r = slerp(r1, r2, t) = r1 * (1-t) + r2 * t (under the semantics of slerp).
    Args:
        r1: np.ndarray, [*, 3, 3], the first rotation matrices.
        r2: np.ndarray, [*, 3, 3], the second rotation matrices.
        t: float, the interpolation factor.
    Returns:
        np.ndarray, [*, 3, 3], the interpolated rotation matrices.
    '''
    q1, q2 = matrix_to_quat(r1), matrix_to_quat(r2)
    q = quat_slerp(q1, q2, t)
    return quat_to_matrix(q)


def rotvec_slerp(r1:np.ndarray, r2:np.ndarray, t:float) -> np.ndarray:
    ''' Spherical linear interpolation between two sequences of rotation vectors.
        Result r = slerp(r1, r2, t) = r1 * (1-t) + r2 * t (under the semantics of slerp).
    Args:
        r1: np.ndarray, [*, 3], the first rotation vectors.
        r2: np.ndarray, [*, 3], the second rotation vectors.
        t: float, the interpolation factor.
    Returns:
        np.ndarray, [*, 3], the interpolated rotation vectors.
    '''
    q1, q2 = rotvec_to_quat(r1), rotvec_to_quat(r2)
    q = quat_slerp(q1, q2, t)
    return quat_to_rotvec(q)


if __name__ == '__main__':
    pass
