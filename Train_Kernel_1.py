# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:08:31 2019

@author: mata3103
"""

from scipy.io import wavfile
import numpy as np

PATH = 'C:\Matlab_Bazi\Gausssian Process\McGill M1'

fs, data = wavfile.read(PATH+'\MA_S1.wav')

# https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.ipynb

from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)

framelength = 128
Z = rolling(data,framelength)

#keep every S
S = 32
ZZ = Z[::S,:]

 

    
    