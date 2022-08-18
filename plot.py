import matplotlib.pyplot as plt
from helpers import EMA
import matplotlib.dates as mdates
from numpy import concatenate


def plot_loss(path_to_save, train=True):
    plt.rcParams.update({'font.size': 10})
    with open(path_to_save + "/train_loss.txt", 'r') as f:
        loss_list = [float(line) for line in f.readlines()]
    if train:
        title = "Train"
    else:
        title = "Validation"
    EMA_loss = EMA(loss_list)
    plt.plot(loss_list, label="loss")
    plt.plot(EMA_loss, label="EMA loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title+"_loss")
    plt.savefig(path_to_save+f"/{title}.png")
    plt.close()

def plot_prediction(title, path_to_save, src, tgt, prediction, machine_number, index_in, index_tar):

    # idx_scr = index_in[0, 1:].tolist()
    # idx_tgt = index_tar[0].tolist()
    # idx_pred = [i for i in range(idx_scr[0] +1, idx_tgt[-1])] #t2 - t61

    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8))
    fig.suptitle("Forecast from Machine ID " + str(machine_number[0]), fontsize=24)

    # connect with last elemenet in src
    # tgt = np.append(src[-1], tgt.flatten())
    # prediction = np.append(src[-1], prediction.flatten())

    for i, (ax, ax_title) in enumerate(zip(axes.flatten(), ('Comp1 Failure', 'Comp2 Failure', 'Comp4 Failure'))):
        ax.plot(index_in[1:], src[:, 0, i], '-', color='blue', label='Input', linewidth=2)
        ax.plot(index_tar, tgt[:, 0, i], '-', color='indigo', label='Target', linewidth=2)
        ax.plot(concatenate((index_in[2:], index_tar[:-1])), prediction[:,  0, i],'--', color='limegreen', label='Forecast', linewidth=2)
        ax.grid(b=True, which='major', linestyle='-')
        ax.grid(b=True, which='minor', linestyle='--', alpha=0.5)
        ax.minorticks_on()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y\n%Hh%M'))
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right', fontsize=8)
        ax.set_ylim((-0.1, 1.1))
        ax.set_title(ax_title)
        ax.legend()

    # save
    plt.savefig(path_to_save+f"Prediction_{title}.png")
    plt.close()


def plot_training(epoch, path_to_save, src, prediction, sensor_number, index_in, index_tar):

    # idx_scr = index_in.tolist()[0]
    # idx_tar = index_tar.tolist()[0]
    # idx_pred = idx_scr.append(idx_tar.append([idx_tar[-1] + 1]))

    idx_scr = [i for i in range(len(src))]
    idx_pred = [i for i in range(1, len(prediction)+1)]

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 18})
    plt.grid(b=True, which='major', linestyle = '-')
    plt.grid(b=True, which='minor', linestyle = '--', alpha=0.5)
    plt.minorticks_on()

    plt.plot(idx_scr, src, 'o-.', color = 'blue', label = 'input sequence', linewidth=1)
    plt.plot(idx_pred, prediction, 'o-.', color = 'limegreen', label = 'prediction sequence', linewidth=1)

    plt.title("Teaching Forcing from Sensor " + str(sensor_number[0]) + ", Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.savefig(path_to_save+f"/Epoch_{str(epoch)}.png")
    plt.close()


def plot_training_3(epoch, path_to_save, src, sampled_src, prediction, machine_number, index_in, index_tar):

    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8))
    fig.suptitle("Teaching Forcing from Machine " + str(machine_number[0]) + ", Epoch " + str(epoch), fontsize=24)
    # plt.rcParams.update({"font.size": 18})

    ## REMOVE DROPOUT FOR THIS PLOT TO APPEAR AS EXPECTED !! DROPOUT INTERFERES WITH HOW THE SAMPLED SOURCES ARE PLOTTED

    for i, (ax, title) in enumerate(zip(axes.flatten(), ('Comp1 Failure', 'Comp2 Failure', 'Comp4 Failure'))):
        ax.plot(index_in[1:], sampled_src[:, 0, i].numpy(), 'o-.', color='red', label='sampled source', linewidth=1, markersize=10)
        ax.plot(index_in[:-1], src[:, 0, i].numpy(), 'o-.', color='blue', label='input sequence', linewidth=1)
        ax.plot(index_in[2:], prediction[:, 0, i].detach().numpy(), 'o-.', color='limegreen', label='prediction sequence', linewidth=1)
        ax.grid(b=True, which='major', linestyle='-')
        ax.grid(b=True, which='minor', linestyle='--', alpha=0.5)
        ax.minorticks_on()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y\n%Hh%M'))
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right', fontsize=8)
        ax.set_ylim((-0.1, 1.1))
        ax.set_title(title)
        ax.legend()

    plt.savefig(path_to_save+f"/Epoch_{str(epoch)}.png")
    plt.close()
