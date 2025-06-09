"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the
software.

Patients' data is held in an inflammation table (2D array) where each
row contains inflammation data for a single patient taken over a number
of days and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):  
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=",")

def patient_normalise(data):
    """Normalising patient data."""
    is_zero = data<0
    if is_zero.any():
        raise ValueError(f"{np.where(is_zero)}")
    if not isinstance(data, np.ndarray):
        raise TypeError("This data is not a nd array.")
    if len(data.shape) != 2:
        raise ValueError('inflammation array should be 2-dimensional')
    patient_max_inflammation = np.max(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / patient_max_inflammation[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    return normalised


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array."""
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array."""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array."""
    return np.min(data, axis=0)

class Patient:
    """A patient class."""
    def __init__(self, name):
        self.name = name
    def get_name(self):
        """Return patient's name."""
        return self.name
