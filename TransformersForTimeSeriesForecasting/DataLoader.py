from torch.utils.data import Dataset
from torch import tensor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from joblib import dump


class MachineDataset(Dataset):

    def __init__(self, csv_path: str, training_window: int, forecast_window: int, comp_failure: str) -> None:
        self.df = pd.read_csv(csv_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.machine_indices_id = {i: machine_id for i, machine_id in enumerate(self.df['machineID'].unique())}
        self.transform = MinMaxScaler()
        self.training_window = training_window
        self.forecast_window = forecast_window
        self.comp_failure = comp_failure

    def __len__(self):
        return len(self.machine_indices_id)

    def __getitem__(self, item):
        machine_id = self.machine_indices_id[item]
        start = np.random.randint(0, len(self.df[self.df['machineID'] == machine_id]) - self.training_window - self.forecast_window)

        index_input = tensor([*range(start, start + self.training_window)])
        index_target = tensor([*range(start + self.training_window, start + self.training_window + self.forecast_window)])
        _input = tensor(self.df[self.df['machineID'] == machine_id][
                                  [self.comp_failure, 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']][
                              start: start + self.training_window].values)
        target = tensor(self.df[self.df['machineID'] == machine_id][
                                  [self.comp_failure, 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']][
                              start + self.training_window: start + self.training_window + self.forecast_window].values)

        # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
        # scalar is fit only for humidity, as the timestamps are already scaled
        # scalar input/output of shape: [n_samples, n_features].
        scaler = self.transform

        scaler.fit(_input[:, 0].unsqueeze(-1))
        _input[:, 0] = tensor(scaler.transform(_input[:, 0].unsqueeze(-1)).squeeze(-1))
        target[:, 0] = tensor(scaler.transform(target[:, 0].unsqueeze(-1)).squeeze(-1))

        # save the scalar to be used later when inverse translating the data for plotting.
        dump(scaler, 'scalar_item.joblib')

        return index_input, index_target, _input, target, str(machine_id)

    def get_datetime_labels(self, index_input, index_target):
        return self.df.iloc[index_input]['datetime'].values, self.df.iloc[index_target]['datetime'].values
