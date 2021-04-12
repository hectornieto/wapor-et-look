# -*- coding: utf-8 -*-
"""
Savitzky-Gloay NDVI filtering
author: Laust Færch @ DHI GRAS
Created on 2020/10/27

Based on the article:
Chen, J., Jönsson, P., Tamura, M., Gu, Z., Matsushita, B., & Eklundh, L. 2004. A simple method for reconstructing a
high-quality NDVI time-series data set based on the Savitzky-Golay filter.

"""


import numpy as np
import pandas as pd

from numba import jit
from scipy import interpolate
from scipy.signal import savgol_filter
from tqdm import tqdm

def _interpolate_1d(data):
    # we need at least 2 non-nan elements
    if np.sum(~np.isnan(data) * 1) < 2:
        return data

    good = ~np.isnan(data)

    # scipy interpolation
    finterp = interpolate.interp1d(np.flatnonzero(good),
                                   data[good], kind='linear',
                                   fill_value=np.nan,
                                   bounds_error=False)

    data = finterp(np.arange(data.shape[0]))

    return data


# @jit
def _prepare_smoothing(N0):
    # long term change trend fitting:
    m = list(np.array([4, 5, 6, 7]) * 2 + 1)
    d = [2, 3, 4]

    parameters = np.array(np.meshgrid(m, d)).T.reshape(-1, 2)  # not Numba supported

    long_term_trend = np.zeros((N0.shape[0], parameters.shape[0]))

    for i, param in enumerate(parameters):
        long_term_trend[:, i] = savgol_filter(N0, param[0], param[1])

    # find minimum square error
    square_error = (long_term_trend - N0[:, None]) ** 2
    min_square_error_arg = np.argmin(np.sum(square_error, axis=0))

    Ntr = long_term_trend[:, min_square_error_arg]

    # Determination of NDVI weights for each point (based on difference from original data)
    W = np.ones_like(N0)
    di = np.abs(N0 - Ntr)
    W_full = 1 - di / np.max(di)
    W[N0 < Ntr] = W_full[N0 < Ntr]

    return Ntr, W


# @jit
def _recursive_smoothing(N0, Ntr, W, niter=0):
    max_iter = 5
    ndvi_stopping_difference = 0.05

    # generate new NDVI timeseries
    N1 = N0.copy()
    N1[N0 < Ntr] = Ntr[N0 < Ntr]

    # Fitting new trend using savgol (m=4, d=6)
    Nk = savgol_filter(N1, 9, 6)

    # calculate fitting-effect index (changed to max ndvi diff)
    max_diff = np.max(np.abs(Nk - N0))

    if (max_diff >= ndvi_stopping_difference) & (niter <= max_iter):
        N1 = _recursive_smoothing(Nk, Ntr, W, niter + 1)

    return N1


# @jit
def _savgol_reconstruct_1d(N0):

    # dont smooth if all nan
    if np.all(np.isnan(N0)):
        return N0

    Ntr, W = _prepare_smoothing(N0)
    Nk = _recursive_smoothing(N0, Ntr, W)
    return Nk


def savgol_reconstruct(data, axis=0, invert=False):
    """
    :param data: np.array((y,x,t))
    NDVI/Albedo raster timeseries. Each image in timeseries is shape y,x.
    :param axis: int
    Axis for the time-dimension in the array. Filtering/interpolation will be apply along this axis
    :param invert: bool
    Inversion of the data and output. This is useful for albedo where bad cloud masking will force the values up,
    instead of down for NDVI.

    :return: y_smoothed: np.array(y,x,t)
    """
    if invert:
        data = data * -1

    print('interpolating...')
    data_interp = np.apply_along_axis(_interpolate_1d, axis, data)

    print('reconstructing...')
    data_recons = np.apply_along_axis(_savgol_reconstruct_1d, axis, data_interp)

    if invert:
        data_recons = data_recons * -1
        data_interp = data_interp * -1

    return data_recons, data_interp


def interpolate_and_reconstruct(timeseries, dates):
    timeseries = temporal_interpolate(timeseries, dates)
    timeseries = _savgol_reconstruct_1d(timeseries.reshape(-1))
    return timeseries


def temporal_interpolate(timeseries, dates):

    timeseries = pd.DataFrame(timeseries, index=pd.to_datetime(dates))
    timeseries = timeseries.interpolate(method="time",
                                        axis=0,
                                        limit_direction="both").to_numpy()

    return timeseries


def savgol_reconstruct_optimized(data, dates):
    """
    :param data: np.array((t,y,x))
    raster timeseries. Each image in timeseries is shape y,x.
    :param dates: array_like(t)
    List of Datetime objects

    :return: y_smoothed: np.array(t,y,x)
    """
    print('interpolating and reconstructing...')
    iijj = [(i, j) for i in range(data.shape[1]) for j in range(data.shape[2])]
    for i, j in tqdm(iijj):
        # we need at least 2 non-nan elements
        if np.sum(~np.isnan(data[:, i, j])) < 2:
            continue
        data[:, i, j] = interpolate_and_reconstruct(data[:, i, j], dates)

    return data


def iterpolate_reconstruct(data, dates, slice_size=1e6):

    dims = data.shape
    data = data.reshape(data.shape[0], -1)
    n_elements = data.shape[1]
    indices = int(np.ceil(n_elements / slice_size))

    for i in tqdm(range(indices)):
        index = int(i * slice_size)
        index_end = int(np.minimum(index + slice_size, data.shape[1]))

        data[:, index: index_end] = temporal_interpolate(data[:, index: index_end],
                                                         dates)

    data = data.reshape(dims)
    return data


def savgol_reconstruct_vectorized(data, dates, slice_size=1e5):

    dims = data.shape
    data = data.reshape(data.shape[0], -1)
    data_sg = data.copy()
    n_elements = data.shape[1]
    indices = int(np.ceil(n_elements / slice_size))

    for i in tqdm(range(indices)):
        index = int(i * slice_size)
        index_end = int(np.minimum(index + slice_size, data.shape[1]))

        data[:, index: index_end] = temporal_interpolate(data[:, index: index_end], dates)
        data[:, index: index_end][np.isnan(data[:, index: index_end])] = 0
        data_sg[:, index: index_end] = savgol_filter(data[:, index: index_end],
                                                     axis=0, polyorder=3, window_length=5)

    data = data.reshape(dims)
    data_sg = data_sg.reshape(dims)
    return data, data_sg

