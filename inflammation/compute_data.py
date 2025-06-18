"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import numpy as np

from inflammation.models import load_csv, load_json, daily_mean, compute_standard_deviation_by_day


class CSVDataSource:
    """
    Loads all the inflammation CSV files within a specified directory.
    """
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def load_inflammation_data(self):
        """
        Loads all the inflammation CSV files within a specified directory.
        """
        data_file_paths = glob.glob(os.path.join(self.dir_path, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation CSV files found in path {self.dir_path}")
        data = map(load_csv, data_file_paths)
        return list(data)

class JSONDataSource:
    """
    Loads patient data with inflammation values from JSON files within a specified folder.
    """
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def load_inflammation_data(self):
        """
        Loads patient data with inflammation values from JSON files within a specified folder.
        """
        data_file_paths = glob.glob(os.path.join(self.dir_path, 'inflammation*.json'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation JSON files found in path {self.dir_path}")
        data = map(load_json, data_file_paths)
        return list(data)