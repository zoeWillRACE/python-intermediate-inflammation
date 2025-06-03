"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_max, daily_min

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize('test_input, expected_test_result', [
    ([[1, 6], [2, 5], [3, 4]], [2, 5]),
    ([[-2, -7], [-3, -6], [-4, -5]], [-3, -6])], 
     ids=["positive ints", "negative ints"])
def test_daily_mean_integers(test_input, expected_test_result):
    """Test that mean function works for an array of positive integers."""
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), expected_test_result)

@pytest.mark.parametrize('test_input, expected_test_result', [
    ([[1, 6], [2, 5], [3, 4]], [3, 6]),
    ([[-2, -7], [-3, -6], [-4, -5]], [-2, -5])], 
     ids=["positive ints", "negative ints"])
def test_daily_max(test_input, expected_test_result):
    npt.assert_array_equal(daily_max(test_input), expected_test_result)


@pytest.mark.parametrize('test_input, expected_test_result', [
    ([[1, 6], [2, 5], [3, 4]], [1, 4]),
    ([[-2, -7], [-3, -6], [-4, -5]], [-4, -7])
], ids=["positive ints", "negative ints"]) #gives the tests cases a name
def test_daily_min(test_input, expected_test_result):
    npt.assert_array_equal(daily_min(test_input), expected_test_result)