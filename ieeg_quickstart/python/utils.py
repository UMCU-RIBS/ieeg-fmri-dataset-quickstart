import re
from fractions import Fraction
from scipy.signal import resample_poly
import numpy as np

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

def resample(x, sr1, sr2, axis=0):
    '''sr1: target, sr2: source'''
    a, b = Fraction(sr1, sr2)._numerator, Fraction(sr1, sr2)._denominator
    return resample_poly(x, a, b, axis).astype(np.float32)

def smooth_signal(y, n):
    box = np.ones(n)/n
    ys = np.convolve(y, box, mode='same')
    return ys

def zscore(x):
    return (x - np.mean(x, 0, keepdims=True)) / np.std(x, 0, keepdims=True)