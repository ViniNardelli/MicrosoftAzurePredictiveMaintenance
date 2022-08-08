from argparse import ArgumentParser
from Preprocessing.utils import clean_directories
from Preprocessing.preprocessing_data import pre_process
from os.path import join
from DataLoader import MachineDataset
from torch.utils.data import DataLoader
from inference import inference
from train_with_sampling import transformer


def main(path_data: str,
         epoch: int = 1000,
         k: int = 60,
         batch_size: int = 1,
         frequency: int = 100,
         training_length: int = 48,
         forecast_window: int = 24,
         failures_csv: str = 'PdM_failures_model3.csv',
         telemetry_csv: str = 'PdM_telemetry_model3.csv',
         path_to_save_model: str = 'save_model/',
         path_to_save_loss: str = 'save_loss/',
         path_to_save_predictions: str = 'save_predictions/',
         device: str = 'cpu') -> None:
    """

    :param path_data:
    :param epoch:
    :param k:
    :param batch_size:
    :param frequency:
    :param training_length:
    :param forecast_window:
    :param failures_csv:
    :param telemetry_csv:
    :param path_to_save_model:
    :param path_to_save_loss:
    :param path_to_save_predictions:
    :param device:
    """
    clean_directories(path_to_save_model, path_to_save_loss, path_to_save_predictions)
    train_csv_path, test_csv_path = pre_process(join(path_data, failures_csv), join(path_data, telemetry_csv), path_data)

    train_dataset = MachineDataset(csv_path=train_csv_path, training_window=training_length, forecast_window=forecast_window)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = MachineDataset(csv_path=test_csv_path, training_window=training_length, forecast_window=forecast_window)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    best_model = transformer(train_dataloader, train_dataset, epoch, k,
                             path_to_save_model, path_to_save_loss, path_to_save_predictions, device)
    inference(path_to_save_predictions,test_dataset, forecast_window, test_dataloader, device, path_to_save_model, best_model)


if __name__ == '__main__':
    parser = ArgumentParser(prog='Transformers for Time Series Forecasting',
                            usage="""
                            Descrever como data deve ser""",
                            description="""
                            testando
                            hahaha""")
    parser.add_argument('path_data', type=str)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--k', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--frequency', type=int, default=100)
    parser.add_argument('--path_to_save_model', type=str, default='save_model/')
    parser.add_argument('--path_to_save_loss', type=str, default='save_loss/')
    parser.add_argument('--path_to_save_predictions', type=str, default='save_predictions/')
    parser.add_argument('--failures_csv', type=str, default='PdM_failures_model3.csv')
    parser.add_argument('--telemetry_csv', type=str, default='PdM_telemetry_model3.csv')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    main(**vars(args))
