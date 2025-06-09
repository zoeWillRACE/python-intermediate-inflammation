"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import patient_normalise
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


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]])
    ])
def test_patient_normalise(test, expected):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""
    result = patient_normalise(np.array(test))
    npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)

def test_patient_normalise_fail():
    argument = "This data is not a nd array."
    with pytest.raises(TypeError, match=argument):
        result = patient_normalise([1, 2, 3, 4, 5, 6, 7, 8, 9])

@pytest.mark.parametrize('test, expected, expect_raises',
    [
        # previous test cases here, with None for expect_raises, except for the next one - add ValueError
        # as an expected exception (since it has a negative input value)
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            None
        ),
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            None
        ),
        (
            [[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [1, 1, 1], [1, 1, 1]],
            None
        ),
        (
            [[1, 2, 3], [4, 5, float('nan')], [7, 8, 9]],
            [[0.33, 0.67, 1], [0, 0, 0], [0.78, 0.89, 1]],
            None
        ),
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None
        ),
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        ),
    ])
def test_patient_normalise_all(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers."""
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            patient_normalise(np.array(test))
    else:
        result = patient_normalise(np.array(test))
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)