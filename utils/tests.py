import os

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.numeric import NumericDtype

from utils import (
    add_day_of_month,
    add_day_of_week,
    add_day_of_year,
    convert_dummies,
    load_data_frame,
    polar_to_rectangular,
)


@pytest.fixture
def joined_df():
    return load_data_frame('var')


@pytest.fixture
def config():
    with open(os.path.join('utils', 'config.yml')) as f:
        config = yaml.safe_load(f)

    return config


@pytest.fixture
def preprocessed_df(joined_df, config):
    joined_df = add_day_of_week(joined_df)
    joined_df = add_day_of_month(joined_df)
    joined_df = add_day_of_year(joined_df)

    joined_df = convert_dummies(joined_df, config['dummy_cols'])

    return joined_df


def test_load_data_frame(joined_df):
    assert isinstance(
        joined_df, pd.DataFrame
    ), 'Loaded data is not in Pandas DataFrame format'
    assert joined_df.index.is_unique, 'Index is not unique'
    assert joined_df.shape[1] == 19, 'DataFrame does not have 19 columns'
    assert joined_df.notna().all().all(), 'DataFrame has null entries'


def test_pre_processing(preprocessed_df, config):
    assert isinstance(
        preprocessed_df, pd.DataFrame
    ), 'Preprocessed data is not in Pandas DataFrame format'
    assert preprocessed_df.index.is_unique, 'Index is not unique'
    assert preprocessed_df.shape[1] == 78, 'DataFrame does not have 78 columns'
    assert preprocessed_df.notna().all().all(), 'DataFrame has null entries'
    for col, dtype in preprocessed_df[config['x_cols']].dtypes.items():
        assert isinstance(
            dtype, (BooleanDtype, NumericDtype)
        ), f'{col} is not a numeric type'


def test_polar_to_rectangular():
    input_radius = 1.5
    input_angle = np.pi / 4
    x, y = polar_to_rectangular(input_radius, input_angle)
    calc_radius = np.sqrt(x**2 + y**2)
    calc_angle = np.arctan(y / x)

    assert np.isclose(
        calc_radius, input_radius
    ), f'The radius returned by `polar_to_rectangular`({calc_radius}) fails to match input radius({input_radius})'
    assert np.isclose(
        calc_angle, input_angle
    ), f'The angle returnrd by `polar_to_rectangular({calc_angle / np.pi} * pi) fails to match input angle({input_angle / np.pi} * pi)'
