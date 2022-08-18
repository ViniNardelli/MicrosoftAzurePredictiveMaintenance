import pandas as pd
import numpy as np
from typing import Tuple
from os.path import join, exists
from os import remove
from random import seed, choice


def split_train_test_csv(source_df: pd.DataFrame,
                         train_size: float = 0.8,
                         random_seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to separate the data into two different random ones
    :param source_df: data
    :param train_size: fraction of the source to be used in the train split
    :param random_seed: to set random state
    :return: one dataframes containing train_size * source_df rows and another with the rest rows
    """
    seed(random_seed)

    machines_relative_frequencies = (source_df.groupby(['machineID']).size() / len(source_df)).to_dict()
    train_machines = []
    sum_relative_frequencies = 0
    while sum_relative_frequencies < train_size:
        machine_id = choice(list(machines_relative_frequencies.keys()))
        train_machines.append(machine_id)
        sum_relative_frequencies += machines_relative_frequencies[machine_id]
        del machines_relative_frequencies[machine_id]

    return source_df[source_df['machineID'].isin(train_machines)], \
           source_df[~source_df['machineID'].isin(train_machines)]


def create_cycle_datetime(source_df: pd.DataFrame, datetime_column: str = 'datetime') -> pd.DataFrame:
    """
    Encoding the timestamp data cyclically.
    :param source_df: dataframe with the data
    :param datetime_column: column name containing datetime information
    :return: dataframe with cycle hour, day and month columns
    """
    hour = source_df[datetime_column].dt.hour
    day = source_df[datetime_column].dt.day
    month = source_df[datetime_column].dt.month

    hours_in_day = 24
    days_in_month = 30
    month_in_year = 12

    source_df['sin_hour'] = np.sin(2 * np.pi * hour / hours_in_day)
    source_df['cos_hour'] = np.cos(2 * np.pi * hour / hours_in_day)
    source_df['sin_day'] = np.sin(2 * np.pi * day / days_in_month)
    source_df['cos_day'] = np.cos(2 * np.pi * day / days_in_month)
    source_df['sin_month'] = np.sin(2 * np.pi * month / month_in_year)
    source_df['cos_month'] = np.cos(2 * np.pi * month / month_in_year)

    return source_df


def merge_failures_telemetry_error_datasets(failures: pd.DataFrame, telemetry: pd.DataFrame,
                                            errors: pd.DataFrame) -> pd.DataFrame:
    """
    Merge failures and telemetry datasets, set the correct data type and apply hot one encoding to failures
    :param errors: errors data
    :param failures: failures data
    :param telemetry: telemetry data
    :return: a dataframe containing both information
    """
    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])

    errors['datetime'] = pd.to_datetime(errors['datetime'])
    errors['errorID'] = errors['errorID'].astype('category')
    errors = pd.get_dummies(errors, columns=['errorID'], prefix='', prefix_sep='')

    failures['datetime'] = pd.to_datetime(failures['datetime'])
    failures['failure'] = failures['failure'].astype('category')
    failures['day'] = failures['datetime'].dt.date
    failures = pd.get_dummies(failures, columns=['failure'], prefix='', prefix_sep='')

    df = pd.merge(telemetry, errors, on=['machineID', 'datetime'], how='left', suffixes=('', '_errors'))
    df['day'] = df['datetime'].dt.date
    df = pd.merge(df, failures, on=['machineID', 'day'], how='left', suffixes=('', '_failures'))
    df.drop(columns=['day', 'datetime_failures'], inplace=True)
    df.fillna(0, inplace=True)

    df = df.groupby(['machineID', 'datetime']).agg({'volt': 'first',
                                                    'rotate': 'first',
                                                    'pressure': 'first',
                                                    'vibration': 'first',
                                                    'error1': 'sum',
                                                    'error2': 'sum',
                                                    'error3': 'sum',
                                                    'error4': 'sum',
                                                    'error5': 'sum',
                                                    'comp1': 'sum',
                                                    'comp2': 'sum',
                                                    'comp4': 'sum'})
    return df.reset_index()


def pre_process(failures_csv: str, telemetry_csv: str, errors_csv: str, output_path: str = '') -> Tuple[str, str]:
    """
    Merge the telemetry and failures data into one dataset, create the cycle time columns and split into train and test
    :param errors_csv: path to the errors csv
    :param failures_csv: path to the failure csv
    :param telemetry_csv: path to the telemetry csv
    :param output_path: where the output data will be saved
    """
    failures_df = pd.read_csv(failures_csv, header=0, index_col=0)
    telemetry_df = pd.read_csv(telemetry_csv, header=0, index_col=0)
    errors_df = pd.read_csv(errors_csv, header=0, index_col=0)

    df = merge_failures_telemetry_error_datasets(failures_df, telemetry_df, errors_df)
    df = create_cycle_datetime(df)
    df_train, df_test = split_train_test_csv(df)

    train_dataset_path = join(output_path, 'train_dataset.csv')
    test_dataset_path = join(output_path, 'test_dataset.csv')

    for path, dataframe in zip([train_dataset_path, test_dataset_path], [df_train, df_test]):
        if exists(path):
            remove(path)
        dataframe.to_csv(path, index=False)

    return train_dataset_path, test_dataset_path
