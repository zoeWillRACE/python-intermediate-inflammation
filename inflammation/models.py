"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the
software.

Patients' data is held in an inflammation table (2D array) where each
row contains inflammation data for a single patient taken over a number
of days and each column represents a single day across all patients.
"""

import numpy as np
import json


def load_csv(filename):  
    """Load a Numpy array from a CSV
    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=",")

def patient_normalise(data):
    """Normalising patient data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("This data is not a nd array.")
    is_zero = data<0
    if is_zero.any():
        raise ValueError(f"{np.where(is_zero)}")
    if len(data.shape) != 2:
        raise ValueError('inflammation array should be 2-dimensional')
    patient_max_inflammation = np.max(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / patient_max_inflammation[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    return normalised

def load_json(filename):
    """Load a numpy array from a JSON document.
    
    Expected format:
    [
      {
        "observations": [0, 1]
      },
      {
        "observations": [0, 2]
      }    
    ]
    :param filename: Filename of CSV to load
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data_as_json = json.load(file)
        return [np.array(entry['observations']) for entry in data_as_json]

def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array."""
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array."""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array."""
    return np.min(data, axis=0)

def compute_standard_deviation_by_day(data):
    """Calculates the standard deviation by day between datasets.
    Gets all the inflammation data from CSV files within a directory, works out the mean
    inflammation value for each day across all datasets, then visualises the
    standard deviation of these means on a graph."""
    means_by_day = map(daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)
    return daily_standard_deviation

def analyse_data(data_source):
    """Calculate the standard deviation by day between datasets
    Gets all the inflammation csvs within a directory, works out the mean
    inflammation value for each day across all datasets, then graphs the
    standard deviation of these means."""
    data = data_source.load_inflammation_data()
    daily_standard_deviation = compute_standard_deviation_by_day(data)

    return daily_standard_deviation
