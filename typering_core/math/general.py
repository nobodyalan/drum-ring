import torch
from torch import Tensor
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate as interp
from typing import Tuple, List, Any, Dict


def fit_least_square(ys:np.ndarray):
    xs = np.arange(ys.shape[0])
    ex = np.mean(xs)
    ey = np.mean(ys)
    ex2 = np.mean(xs**2)
    exy = np.mean(xs*ys)
    k = (exy - ex*ey)/(ex2 - ex*ex)
    b = ey - k*ex
    return k, b
    

def smooth_imu_timestamps(ts:np.ndarray, n_window:int=1000):
    length = ts.shape[0]
    assert length > 0
    n_segment = (length - 1) // n_window + 1
    ks, bs = [], []
    for i in range(n_segment):
        start = i * n_window
        end = min((i+1) * n_window, length)
        k, b = fit_least_square(ts[start:end])
        ks.append(k)
        bs.append(b)
    if n_segment == 1:  # no need to adjust k and b
        return ks[0] * np.arange(length) + bs[0]
    smoothed = []
    # i == 0
    k = 0.5 * ks[0] + (0.5/n_window) * (bs[1]-bs[0])
    b = bs[0]
    smoothed.append(k * np.arange(n_window) + b)
    # i = 1, ..., n_segment-2
    for i in range(1, n_segment-1):
        k = 0.5 * (ks[i]-ks[i-1]) + (0.5/n_window) * (bs[i+1]-bs[i-1])
        b = 0.5 * (ks[i-1]*n_window + bs[i-1] + bs[i])
        smoothed.append(k * np.arange(n_window) + b)
    # i == n_segment-1
    l = length % n_window
    k = ks[-1] - 0.5*ks[-2]*n_window/l + (0.5/l) * (bs[-1]-bs[-2])
    b = 0.5 * (ks[-2]*n_window + bs[-2] + bs[-1])
    smoothed.append(k * np.arange(l) + b)
    smoothed = np.concatenate(smoothed)
    return smoothed


def smooth_imu_timestamps2(ts:np.ndarray, fps:float=200.0):
    N = ts.shape[0]
    res = []
    s = 0
    for i in range(N-1):
        if ts[i+1] - ts[i] < 0.01: continue
        res.append(np.linspace(ts[i]-(i-s)/fps, ts[i], num=i-s+1, endpoint=True))
        s = i + 1
    if s < N:
        res.append(np.linspace(ts[-1]-(N-1-s)/fps, ts[-1], num=N-s, endpoint=True))
    res = np.concatenate(res)
    # remove jittering
    delta = np.linspace(0, (N-1)/fps, num=N, endpoint=True)
    res = res - delta
    while True:
        modified = False
        l = 1 # l, r in range [1, N-2]
        while l < N-1:
            r = l + 1
            while r < N-1 and abs(res[r]-res[l]) < 1e-3:
                r += 1
            diff = min(res[l]-res[l-1], res[r-1]-res[r])
            if diff > 0:
                modified = True
                res[l:r] = res[l:r] - diff
            else: l = r
        if not modified: break
    res = res + delta
    return res


def fix_ring_calib_timestamps(server_start_ts:np.ndarray,
        server_end_ts:np.ndarray, client_ts:np.ndarray):
    ''' Used when len(server_end_ts) == len(client_ts) == len(server_start_ts) - 1.
    '''
    # find missing idx
    n = len(server_start_ts)
    l, r = 0, n - 1
    while l != r:
        idx = (l + r) // 2
        ts = np.concatenate([server_start_ts[:idx], server_start_ts[idx+1:]])
        min_diff = np.min(server_end_ts - ts)
        if min_diff <= 0: l = idx + 1
        else: r = idx
    
    # add missing data
    missing_server_end_t = server_start_ts[l] + 0.050   # 50 ms   
    if l == 0: missing_client_t = client_ts[0] - 0.005
    elif l == n - 1: missing_client_t = client_ts[n-2] + 0.005
    else: missing_client_t = (client_ts[l-1] + client_ts[l]) / 2
    client_ts = np.concatenate([client_ts[:l],
        np.array([missing_client_t]), client_ts[l:]])
    server_end_ts = np.concatenate([server_end_ts[:l],
        np.array([missing_server_end_t]), server_end_ts[l:]])
    
    return server_start_ts, server_end_ts, client_ts


def ring_calib_time_params(data:dict, n_valid:int=50):
    ''' Input ring calibration time data.
        Fit b and k using linear regression.
    '''
    client_ts = data['client_timestamps']
    server_start_ts = data['server_start_timestamps']
    server_end_ts = data['server_end_timestamps']
    server_ts = (server_start_ts + server_end_ts) / 2
    server_dt = server_end_ts - server_start_ts

    length = len(client_ts)
    ts1 = list(zip(server_ts[:length//2], client_ts[:length//2], server_dt[:length//2]))
    ts1 = sorted(ts1, key=lambda x:x[2])[:n_valid]
    ts2 = list(zip(server_ts[length//2:], client_ts[length//2:], server_dt[length//2:]))
    ts2 = sorted(ts2, key=lambda x:x[2])[:n_valid]
    ts = ts1 + ts2
    _server_ts, _client_ts, _ = zip(*ts)
    _server_ts = np.array(_server_ts)
    _client_ts = np.array(_client_ts)
    X_b = np.c_[np.ones((len(_client_ts), 1)), _client_ts]
    b, k = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(_server_ts)
    return b, k


def resample(data:np.ndarray, axis:int, ratio:float, kind='quadratic') -> np.ndarray:
    ''' Resample data by spline interpolation.
    args:
        data: np.ndarray, with any shape and dtype.
        axis: int, the samping axis.
        ratio: float, the number of resampled data points over
            the number of original data points.
        kind: str, the interpolation time.
    returns:
        np.ndarray, the resampled data, dtype = the original dtype.
    '''
    N = data.shape[axis]
    M = int(np.round(N * ratio))
    t = np.arange(N, dtype=np.float64)
    x = np.linspace(0, N-1, num=M, endpoint=True)
    interp_func = interp.interp1d(t, data, kind=kind, axis=axis)
    return interp_func(x)


def rcs(data:np.ndarray, d:float=2.0) -> np.ndarray:
    ''' Rate-of-change score, from paper TapType.
    args:
        data: np.ndarray, with shape (N, D).
    returns:
        np.ndarray, with shape (N-1,), the rate-of-change score.
    '''
    norm = np.linalg.norm(data, axis=1)
    q = (1/d) ** np.arange(data.shape[0]-1)[::-1]
    res = np.abs(np.diff(norm)) * q
    res = np.cumsum(res) / q
    return res


def imu_feature(data:Tensor, d:float=2.0):
    ''' Extract IMU features.
        Extract acc/gyro RCS score.
    Args:
        data: Tensor, [B, T, 6], the 6-axis IMU data.
    Returns: 
        Tensor, [B, T, C], the extracted features.
    '''
    norm = torch.concat([
        torch.norm(data[:, :, :3], dim=2, keepdim=True),
        torch.norm(data[:, :, 3:], dim=2, keepdim=True)], dim=2)
    q = (1/d) ** torch.arange(data.shape[1]-2, -1, -1, device=data.device)[None,:,None]
    res = torch.abs(torch.diff(norm, dim=1)) * q
    res = torch.cumsum(res, dim=1) / q
    res = torch.cat([res, torch.zeros_like(res[:, -1:]),], dim=1)
    return res


def board_coordinates(p0, p1, p2, p3, p4):
    ''' Give the four corners of the board and an interioior point,
        compute the coordinates within the board.
        Restrict u, v in [-0.5, 1.5], otherwise return None.
        e.g.:
            p1 ---------- p2
            |             |
            |        p0   |
            |             |
            p4 ---------- p3
    '''
    y34, y21 = p3[1] - p4[1], p2[1] - p1[1]
    y40, y10 = p4[1] - p0[1], p1[1] - p0[1]
    x34, x21 = p3[0] - p4[0], p2[0] - p1[0]
    x40, x10 = p4[0] - p0[0], p1[0] - p0[0]
    
    a = y34 * x21 - y21 * x34
    b = y34 * x10 + y40 * x21 - y21 * x40 - y10 * x34
    c = y40 * x10 - y10 * x40
    u1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    u2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    u = np.nan
    if -0.5 <= u1 and u1 < 1.5: u = u1
    elif -0.5 <= u2 and u2 < 1.5: u = u2
    y32, y41 = p3[1] - p2[1], p4[1] - p1[1]
    y20, y10 = p2[1] - p0[1], p1[1] - p0[1]
    x32, x41 = p3[0] - p2[0], p4[0] - p1[0]
    x20, x10 = p2[0] - p0[0], p1[0] - p0[0]
    a = y32 * x41 - y41 * x32
    b = y32 * x10 + y20 * x41 - y41 * x20 - y10 * x32
    c = y20 * x10 - y10 * x20
    v1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    v2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    v = np.nan
    if -0.5 <= v1 and v1 < 1.5: v = v1
    elif -0.5 <= v2 and v2 < 1.5: v = v2
    return u, v


def board_coordinates2(p0, p1, p2, p3, p4):
    ''' Give the four corners of the board and an interioior point,
        compute the coordinates within the board.
        Do not restrict u, v, return u from u1, u2 that are closer to 0.5.
        e.g.:
            p1 ---------- p2
            |             |
            |        p0   |
            |             |
            p4 ---------- p3
    '''
    y34, y21 = p3[1] - p4[1], p2[1] - p1[1]
    y40, y10 = p4[1] - p0[1], p1[1] - p0[1]
    x34, x21 = p3[0] - p4[0], p2[0] - p1[0]
    x40, x10 = p4[0] - p0[0], p1[0] - p0[0]    
    a = y34 * x21 - y21 * x34
    b = y34 * x10 + y40 * x21 - y21 * x40 - y10 * x34
    c = y40 * x10 - y10 * x40
    if a != 0.0:
        u1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        u2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
        u = u1 if abs(u1-0.5) < abs(u2-0.5) else u2
    else: u = -c / b
    
    y32, y41 = p3[1] - p2[1], p4[1] - p1[1]
    y20, y10 = p2[1] - p0[1], p1[1] - p0[1]
    x32, x41 = p3[0] - p2[0], p4[0] - p1[0]
    x20, x10 = p2[0] - p0[0], p1[0] - p0[0]
    a = y32 * x41 - y41 * x32
    b = y32 * x10 + y20 * x41 - y41 * x20 - y10 * x32
    c = y20 * x10 - y10 * x20
    if a != 0.0:
        v1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        v2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
        v = v1 if abs(v1-0.5) < abs(v2-0.5) else v2
    else: v = -c / b
    return u, v
    
    
def gaussion_distribution(x:np.ndarray, y:np.ndarray):
    ''' Given two 1D arrays, compute the 2D Gaussian distribution.
        Return mu, cov, and the prob distribution function PDF.
    '''
    mu = np.array([np.mean(x), np.mean(y)])
    cov = np.cov(x, y)
    
    def pdf(_x, _y):
        ''' Return probability and mdis.
            Note: mdis is the Mahalanobis distance,
                eq: mdis = sqrt((x-mu)^T @ inv(cov) @ (x-mu))
            e.g. points within n sigma range <=> mdis <= n.
        '''
        xy = np.vstack([_x, _y])
        dxy = xy - mu[:, None]
        # TODO: may encounter singular matrix error
        mdis = np.sqrt(np.einsum('bi,ib->b', dxy.T, np.linalg.inv(cov) @ dxy))
        p = 1 / (2*np.pi*np.sqrt(np.linalg.det(cov))) * np.exp(-0.5*mdis*mdis)
        return p, mdis
        
    return mu, cov, pdf


def calc_uer_from_phrases(phrases1:List[str], phrases2:List[str]) -> float:
    assert len(phrases1) == len(phrases2)
    total_cnt, err_cnt = 0, 0
    for phrase1, phrase2 in zip(phrases1, phrases2):
        phrase1 = phrase1.strip().split()
        phrase2 = phrase2.strip().split()
        total_cnt += len(phrase1)
        min_length = min(len(phrase1), len(phrase2))
        for w1, w2 in zip(phrase1[:min_length], phrase2[:min_length]):
            if w1 != w2: err_cnt += 1
        err_cnt += abs(len(phrase1) - len(phrase2))
    assert total_cnt > 0
    return err_cnt / total_cnt


def char2finger_bhatt_coeff(p1:dict, p2:dict) -> float:
    res = []
    alphabet = 'qwertyuiopasdfghjklzxcvbnm'
    fingers = ['left index', 'left middle', 'left ring/pinky',
        'right index', 'right middle', 'right ring/pinky']
    for c in alphabet:
        coeff = 0.0
        for finger in fingers:
            a, b = p1[c][finger], p2[c][finger]
            coeff += np.sqrt(a * b)
        res.append(coeff)
    return np.mean(res)


def sample_finger_identities(text:str, char2finger:Dict, labels:List[Any]) -> List[Any]:
    '''Sample finger identities for each character in text using roulette wheel selection.
    Args:
        text: Input string containing only 26 English letters and spaces
        char2finger: Dictionary mapping each letter to probability array of 6 finger identities
        labels: List of 6 labels corresponding to the 6 finger identities
    Returns:
        List of sampled labels with same length as text, spaces remain as spaces
    '''
    res = [np.random.choice(labels, p=char2finger[c]) if c != ' ' else ' ' for c in text]
    return ''.join(res)


if __name__ == '__main__':
    data = np.array([0.1, 0.4, 0.2, 0.5, 2.0, 0.6, 0.3, 0.5, 0.2])
    rcs1 = rcs(data[:, None])
    rcs2 = rcs(data[1:, None])
    print(rcs1-rcs1[1])
    print(rcs2-rcs2[0])