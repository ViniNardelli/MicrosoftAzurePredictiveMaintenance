from model import Transformer
from DataLoader import MachineDataset
import logging
from joblib import load
import math, random
import torch
from helpers import log_loss
from plot import plot_loss, plot_training_3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
                    datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)


def flip_from_probability(p):
    return random.random() < p


def transformer(dataloader,
                dataset: MachineDataset,
                EPOCH,
                k,
                path_to_save_model,
                path_to_save_loss,
                path_to_save_predictions,
                device):
    device = torch.device(device)

    model = Transformer().double().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss = 0

        ## TRAIN -- TEACHER FORCING
        model.train()
        for index_in, index_tar, _input, target, machine_number in dataloader:

            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]

            optimizer.zero_grad()
            src = _input.permute(1, 0, 2).double().to(device)[:-1, :, :]  # torch.Size([24, 1, 7])
            target = _input.permute(1, 0, 2).double().to(device)[1:, :, :]  # src shifted by 1.
            sampled_src = src[:1, :, :]  # t0 torch.Size([1, 1, 7])

            for i in range(len(target) - 1):

                prediction = model(sampled_src, device)  # torch.Size([1xw, 1, 1])
                # for p1, p2 in zip(params, model.parameters()):
                #     if p1.data.ne(p2.data).sum() > 0:
                #         ic(False)
                # ic(True)
                # ic(i, sampled_src[:,:,0], prediction)
                # time.sleep(1)
                """
                # to update model at every step
                # loss = criterion(prediction, target[:i+1,:,:1])
                # loss.backward()
                # optimizer.step()
                """

                if i < 24:  # One day, enough data to make inferences about cycles
                    prob_true_val = True
                else:
                    ## coin flip
                    v = k / (k + math.exp(epoch / k))  # probability of heads/tails depends on the epoch, evolves with time.
                    prob_true_val = flip_from_probability(v)  # starts with over 95 % probability of true val for each flip in epoch 0.
                    ## if using true value as new value

                if prob_true_val:  # Using true value as next value
                    sampled_src = torch.cat((sampled_src.detach(), src[i + 1, :, :].unsqueeze(0).detach()))
                else:  # using prediction as new value
                    positional_encodings_new_val = src[i + 1, :, :-3].unsqueeze(0)
                    predicted_fail = torch.cat((positional_encodings_new_val, prediction[-1, :, :].unsqueeze(0)),
                                               dim=2)
                    sampled_src = torch.cat((sampled_src.detach(), predicted_fail.detach()))

            """To update model after each sequence"""
            loss = criterion(target[:-1, :, -3:], prediction)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}.pth"

        if epoch % 10 == 0:  # Plot 1-Step Predictions

            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            scaler_list = load('scalar_item.joblib')
            # for i, scaler in enumerate(scaler_list):
            #     scaler.fit(_input[:, i].unsqueeze(-1))
            #     _input[:, i] = tensor(scaler.transform(_input[:, i].unsqueeze(-1)).squeeze(-1))
            #     target[:, i] = tensor(scaler.transform(target[:, i].unsqueeze(-1)).squeeze(-1))
            # sampled_src_humidity = scaler_list.inverse_transform(sampled_src[:, :, :4].cpu())  # torch.Size([35, 1, 7])
            # src_humidity = scaler_list.inverse_transform(src[:, :, :4].cpu())  # torch.Size([35, 1, 7])
            # target_humidity = scaler_list.inverse_transform(target[:, :, :4].cpu())  # torch.Size([35, 1, 7])
            # prediction_humidity = scaler_list.inverse_transform(prediction[:, :,
            #                                                0].detach().cpu().numpy())  # torch.Size([35, 1, 7])
            # TODO: plot
            plot_training_3(epoch,
                            path_to_save_predictions,
                            src[:, :, -3:],
                            sampled_src[:, :, -3:],
                            prediction,
                            machine_number,
                            *dataset.get_datetime_labels(index_in.tolist()[0], index_tar.tolist()[0]))

        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)

    # TODO: plot
    plot_loss(path_to_save_loss, train=True)
    return best_model
