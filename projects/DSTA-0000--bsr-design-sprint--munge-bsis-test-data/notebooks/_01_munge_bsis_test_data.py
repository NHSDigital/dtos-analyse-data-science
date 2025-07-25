"""
Helper functions for the notebook of the same name
"""

import logging
import random

import numpy as np
import pandas as pd
from IPython.display import display

def set_up_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s - %(name)s",
        datefmt="%H:%M:%S",
    )


def _perturb_value(value, apply_random_number_below_threshold=10, above_threshold_smear_factor=0.5):
    if value > apply_random_number_below_threshold:
        deviation = value * above_threshold_smear_factor
        lower_bound = max(0, int(value - deviation))
        upper_bound = int(value + deviation)
    else:
        lower_bound, upper_bound = 0, apply_random_number_below_threshold
    return random.randint(lower_bound, upper_bound)


def perturb_numeric_values_where_possible(value, apply_random_number_below_threshold=10, above_threshold_smear_factor=0.5):
    """
    Perturb numeric values, where possible.

    If 'value' can't be converted to numeric, it is returned unchanged.
    If 'value' is a float then a random number between 0 and 100 is returned.
    If 'value' is an int above 'apply_random_number_below_threshold' then
        it is perturbed by a deviation governed by 'above_threshold_smear_factor'.
    If 'value' is an int below 'apply_random_number_below_threshold' then
        a random number between 0 and 'apply_random_number_below_threshold' is returned.
    """

    try:
        value = pd.to_numeric(value)
        if isinstance(value, np.float64):
            value = round(random.uniform(0, 100), 2)
        elif isinstance(value, np.int64):
            value = _perturb_value(value, apply_random_number_below_threshold, above_threshold_smear_factor)
    except:
        pass
    return value


def xcheck__perturbation_worked_as_expected(df, df_other):
    """
    Compare dataframes before and after perturbation.

    Only numeric values are checked.
    """

    mask__numeric = df.map(lambda x: isinstance(x, (int, float)) and not isinstance(x, bool))
    assert (
        mask__numeric.sum().sum() == df.map(type).stack().value_counts().drop(str, axis=0).sum()
    ), "Expect mask and cross check of numeric value counts to match"

    diff = df[mask__numeric].astype(float) - df_other[mask__numeric].astype(float)
    # display(diff.describe().round(0))
    display(diff.abs().describe().round(0))
