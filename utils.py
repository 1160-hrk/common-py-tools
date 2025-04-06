# !/usr/bin/env python3.
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:02:50 2023.

@author: hirokitsusaka
"""


import csv
import math as mt
import pickle

import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.signal import find_peaks




def get_ind(array:np.ndarray, values:np.ndarray)->np.ndarray:
    """Find the indices of the nearest values in the array for each value in values.

    Parameters
    ----------
        array (numpy.ndarray): The input array to search in.
        values (list or numpy.ndarray): The values to find the nearest indices for.

    Returns
    -------
        numpy.ndarray: An array of indices corresponding to the nearest values in the input array.
    """
    # Ensure input is a numpy array
    array = np.asarray(array)
    values = np.asarray(values)
    
    # Get the indices that would sort the array
    sorted_indices = np.argsort(array)
    
    # Find the insertion points for values in the sorted array
    insertion_points = np.searchsorted(array[sorted_indices], values)
    
    # Clip the insertion points to ensure they are within the array bounds
    insertion_points = np.clip(insertion_points, 1, len(array) - 1)
    
    # Find the indices of the nearest values
    left = sorted_indices[insertion_points - 1]
    right = sorted_indices[insertion_points]
    
    # Choose the index with the nearest value
    indices = np.where(np.abs(values - array[left]) <= np.abs(values - array[right]), left, right)
    
    return indices


def get_subarray(x, xi=None, xf=None, return_ind=False):
    """Get subarray of the list-type input.

    Parameters
    ----------
    x : list
        input.
    i : int
        initial value.
    f : int
        final value.

    Returns
    -------
    x_out : list
        subarray of x.
    [ii, ff] : list
            index of initial and final value of x.

    """
    if xi is None:
        i = 0
    else:
        i = get_ind(i, x)
    if xf is None:
        xx = x[i:]
    else:
        f = get_ind(xf, x) + 1
        xx = x[i:f]
    if return_ind:
        return xx, [i, f]
    else:
        return xx


def get_subarray_2D(x, y, xi=None, xf=None, return_ind=False):
    """Get subarray of the 2D list-type input.

    Parameters
    ----------
    x : list
        input, horizontal axis.
    x : list
        input, vertical axis.
    i : int
        initial value of x.
    f : int
        final value of x.
    return_ind : bool, default value is False
        wheser or not to return the index.

    Returns
    -------
    x_out : list
        subarray of x.
    y_out : list
        subarray of y.

    """
    if xi is None:
        i = 0
    else:
        i = get_ind(x, xi)
    if xf is None:
        xx = x[i:]
        yy = y[i:]
    else:
        f = get_ind(x, xf) + 1
        xx = x[i:f]
        yy = y[i:f]
    if return_ind:
        return xx, yy, [i, f]
    else:
        return xx, yy


def get_ind_max_xi_xf(x, y, xi=None, xf=None):
    xx, yy, [i, f] = get_subarray_2D(x, y, xi=xi, xf=xf)
    ii = np.argmax(yy)
    return ii + i


def get_ind_xi_xf(v, x, y, xi=None, xf=None):
    xx, yy, [i, f] = get_subarray_2D(x, y, xi=xi, xf=xf)
    ii = get_ind(v, yy)
    return ii + i


def get_inds_peak_xi_xf(v, x, y, xi=None, xf=None, **kwargs):
    xx, yy, [i, f] = get_subarray_2D(x, y, xi=xi, xf=xf)
    ii = np.array(find_peaks(v, **kwargs)[0])
    return ii + i



def fitting_w_range(fit, x, y, xi=None, xf=None,
                    p0=None,
                    bounds=None,
                    return_x=False, return_pcov=False):
    """Execute the fitting for given function and horizontal range.

    Parameters
    ----------
    fit : function
        fitting function.
    x : numpy.array
        horizontal data.
    y : numpy.array
        vertical data.
    i : float
        initial value of horizontal data.
    f : float
        final value of horizontal data.
    p_ini : list
        initial fitting parameter.

    Returns
    -------
    popt : list
        fitting parameter.
    pcov : list
        covariance.
    xx : numpy.array
        part of horizontal data used for fitting.

    """
    xx, yy = get_subarray_2D(x, y, xi=xi, xf=xf)
    if bounds == None:
        popt, pcov = curve_fit(fit, xx, yy, p0=p0)
    else:
        popt, pcov = curve_fit(fit, xx, yy, p0=p0, bounds=bounds)
    if return_pcov and return_x:
        return popt, pcov, xx
    elif return_pcov:
        return popt, pcov
    elif return_x:
        return popt, xx
    else:
        return popt


def ndarray_from_txtfile(fullname, manner):
    fid = open(fullname, 'r')
    a = np.array([])
    b = np.array([])
    for line in fid:
        currentrow = line[:-1].split(manner)
        if currentrow[0] != '':
            cr = np.array(list(map(float,currentrow)))
            a = np.append(a,cr[0])
            b = np.append(b,cr[1])
    fid.close()
    return a,b


def ndarray_from_csvfile(path):
    """Get the data from .csv file.

    Parameters
    ----------
    path : str
        full path you would like to open.

    Returns
    -------
    a : numpy.ndarray
        data you get.
    b : numpy.ndarray
        data you get.

    """
    with open(path, newline='', encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile, delimiter=',', quotechar='|')
        a = np.array([])
        b = np.array([])
        for row in rows:
            if row[0] != '':
                cr = np.array(list(map(float, row)))
                a = np.append(a, cr[0])
                b = np.append(b, cr[1])
    return a, b


def pickle_dump(obj, path):
    """Save the data as binary file.

    Parameters
    ----------
    obj : object
        object you would like to save.
    path : str
        full path where you would like to save the data.

    Returns
    -------
    None.

    """
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    """Load the data from binary file.

    Parameters
    ----------
    path : str
        full path you would like to open.

    Returns
    -------
    data : object
        object you would like to open.

    """
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data
###############################################################################
#                           Other Functions(end)                              #
###############################################################################
###############################################################################
#                         Function Definition(end)                            #
###############################################################################


# filepath = '/Users/hirokitsusaka/Research@the_Umiversity_of_TOKYO/\
# experiments/FTIR/20220704/sample.csv'
# wn_FTIR, transmittance_FTIR = ndarray_from_txtfile(filepath, ',')
# wn_FTIR_base, transmittance_FTIR_base = get_subarray_2D(
#     wn_FTIR, transmittance_FTIR, 2400, 2600
#     )
# transmittance_FTIR_base = np.average(transmittance_FTIR_base)
# wn_FTIR, transmittance_FTIR = get_subarray_2D(
#     wn_FTIR, transmittance_FTIR, 2200, 2400
#     )
# absorbance_FTIR = -np.log10(transmittance_FTIR/transmittance_FTIR_base)*1000
# wl_FTIR = 10**7/wn_FTIR
if __name__ == "__main__":
    prop = [
        {
            'name': 'wavelength',
            'dim': 'length',
            'unit': 'nm',
            'unit_factor': 1e-9
            },
        {
            'name': 'absorbance_change',
            'dim': 'none',
            'unit': 'mOD',
            'unit_factor': 1e-3
            }
        ]
