import os
import csv
import click
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


@click.group()
def cli():
    pass


def moving_average(x, N):
    """calculate moving average to smooth the data
    
    Arguments:
        x {array of numbers} -- contains data points to be passed through moving average
        N {integer} -- the number of points taking into account
    
    Returns:
        array of numbers -- the smoothed data array
    """
    return np.convolve(x, np.ones((N,)) / N)[(N - 1) :]


def draw_graph(epoch_list, x, y, x_label, y_label, title, legend, is_moving_average):
    fig_acc = plt.figure()
    if is_moving_average:
        x = moving_average(x, 100)
        y = moving_average(y, 100)
    plt.plot(epoch_list, x, "b")
    plt.plot(epoch_list, y, "g")
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.grid(color="k", linestyle="-", linewidth=1)
    #  plt.ylim(0, 1.0)
    plt.title(title)
    plt.legend(legend, loc="lower right")
    return fig_acc


@cli.command()
@click.option("--moving_average", "is_moving_average", is_flag=True)
@click.option("--show", "is_show_plots", is_flag=True)
def plot(is_moving_average, is_show_plots):
    LOO = 16
    experiment_name_template = (
        "eye_v3-s3-model_v21_multiclass-softmax-cce-lw_1_0.01-hsv-loo_{0}-lr_1e_2-bn"
    )

    graphs_dir = os.path.join("data", "comparison")
    csv_dir = os.path.join(graphs_dir, "csv")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    history_list = []
    average_last_2000_output1_val_acc_list = []
    sd_last_2000_output1_val_acc_list = []
    max_train_list = []
    max_val_list = []

    best_of_all = {"accuracy": 0.0, "epoch": 0, "fold": 0}

    for l in range(LOO):
        # define variables and constants
        experiment_name = experiment_name_template.format(l + 1)
        moving_average_string = "-moving_average" if is_moving_average else ""
        output1_acc_filename = (
            f"{experiment_name}{moving_average_string}-output1_acc.png"
        )
        # define paths
        output1_acc_file = os.path.join(graphs_dir, output1_acc_filename)
        dataset_path = os.path.join("data", experiment_name)
        training_log_file = os.path.join(dataset_path, "training.csv")
        # read data from csv file
        history_data = pd.read_csv(training_log_file)
        # plot a graph of each fold
        graph_title = f"{experiment_name}\nOutput 1 Model Accuracy"
        graph_title += " (moving average)" if is_moving_average else ""
        fig_loo = draw_graph(
            history_data["epoch"],
            history_data["output1_acc"],
            history_data["val_output1_acc"],
            "Accuracy",
            "Epoch",
            graph_title,
            ["Train Accuracy", "Validation Accuracy"],
            is_moving_average,
        )

        """
        calculate statistics figures from all points of data
        """

        # find max accuracy of this epoch
        max_train = calculate_max(history_data["output1_acc"])
        max_val = calculate_max(history_data["val_output1_acc"])
        max_train_list.append(max_train)
        max_val_list.append(max_val)

        # find average of last 2000 epochs
        average_last_2000_output1_val_acc_train = calculate_average_last_n(
            history_data["val_output1_acc"], 2000
        )
        average_last_2000_output1_val_acc_list.append(
            average_last_2000_output1_val_acc_train
        )
        # find standard deviation of last 2000 epochs
        sd_last_2000_output1_val_acc_train = calculate_sd_last_n(
            history_data["val_output1_acc"], 2000
        )
        sd_last_2000_output1_val_acc_list.append(sd_last_2000_output1_val_acc_train)

        """
        find the max accuracy of this fold to be used in the other models
        by using only every 100 epoch data
        """

        # replace the others apart from every 100 epochs with 0
        # as we don't have their weights yet
        l_history_data = copy.deepcopy(history_data)
        l_history_data["val_output1_acc"] = [
            acc if i % 100 == 0 else 0.0
            for i, acc in enumerate(l_history_data["val_output1_acc"])
        ]

        # find the max accuracy in this fold with argmax
        arg_max_val_acc = np.argmax(np.array(l_history_data["val_output1_acc"]))
        max_val_acc = l_history_data["val_output1_acc"][arg_max_val_acc]
        max_val_epoch = l_history_data["epoch"][arg_max_val_acc]
        if max_val_acc > best_of_all["accuracy"]:
            best_of_all["accuracy"] = max_val_acc
            best_of_all["epoch"] = max_val_epoch
            best_of_all["fold"] = l + 1  # plus 1 as it starts from 0

        """
        draw and save figures of this fold
        """

        stats_text = f"MAX train = {max_train}\nMAX validation = {max_val}\nAVG 2000 val = {average_last_2000_output1_val_acc_train} ± {sd_last_2000_output1_val_acc_train}"
        # overlay statistics values on the graph figure
        anchored_text = AnchoredText(stats_text, loc=3)  # 3=lower left
        ax = fig_loo.gca()
        ax.add_artist(anchored_text)
        # store the figure on disk
        fig_loo.savefig(output1_acc_file, bbox_inches="tight")
        # store history data to be used to calculate the average values
        history_list.append(history_data)
    epoch_list = [a["epoch"] for a in history_list]
    min_epochs = min(map(len, epoch_list))
    epoch_list = range(min_epochs)

    # calculate the average of accuracies of output1 layer on training set
    output1_acc_list = [a["output1_acc"] for a in history_list]
    output1_acc_array = np.transpose(np.array(output1_acc_list))
    avg_output1_acc_list = [np.mean(a) for a in output1_acc_array]

    # calculate the average of accuracies of output1 layer on validation set
    output1_val_acc_list = [a["val_output1_acc"] for a in history_list]
    output1_val_acc_array = np.transpose(np.array(output1_val_acc_list))
    avg_output1_val_acc_list = [np.mean(a) for a in output1_val_acc_array]

    # plot graph of the average of all folds
    average_name = f"avg_1_{LOO}"
    experiment_name = experiment_name_template.format(average_name)
    output1_avg_acc_filename = f"{experiment_name}{moving_average_string}.png"
    output1_avg_acc_file = os.path.join(graphs_dir, output1_avg_acc_filename)
    graph_title = f"{experiment_name}\nOutput 1 Model Average Accuracy"
    graph_title += " (moving average)" if is_moving_average else ""
    fig_avg = draw_graph(
        epoch_list,
        avg_output1_acc_list,
        avg_output1_val_acc_list,
        "Accuracy",
        "Epoch",
        graph_title,
        ["Train Accuracy", "Validation Accuracy"],
        is_moving_average,
    )
    #  fig_avg.text(3, 8, 'boxed italics text in data coords', style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    max_train = calculate_max(avg_output1_acc_list)
    max_val = calculate_max(avg_output1_val_acc_list)
    stats_text = f"MAX train = {max_train}\nMAX validation = {max_val}"
    anchored_text = AnchoredText(stats_text, loc=3)  # 3=lower left
    ax = fig_avg.gca()
    ax.add_artist(anchored_text)
    # store figure on disk
    fig_avg.savefig(output1_avg_acc_file, bbox_inches="tight")

    """
    Write statistics data to a file
    """

    # save train/val accuracy lists to csv file
    acc_csv_filename = f"{experiment_name}{moving_average_string}.csv"
    acc_csv_file = os.path.join(csv_dir, acc_csv_filename)
    with open(acc_csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        # prepare data
        header_list = list(range(1, LOO + 1))
        header_list.insert(0, "type")
        header_list.append("average")
        # average of all last 2000 epochs average data
        average_last_2000_output1_val_acc_list.append(
            average_from_string_list(average_last_2000_output1_val_acc_list)
        )
        # average of all last 2000 epochs standard deviation data
        sd_last_2000_output1_val_acc_list.append(
            average_from_string_list(sd_last_2000_output1_val_acc_list)
        )
        average_last_2000_output1_val_acc_list.insert(0, "average 2000 val")
        sd_last_2000_output1_val_acc_list.insert(0, "SD 2000 val")
        max_train_list.append(average_from_string_list(max_train_list))
        max_train_list.insert(0, "max train")
        max_val_list.append(average_from_string_list(max_val_list))
        max_val_list.insert(0, "max val")
        # write data to file
        csv_writer.writerow(header_list)
        csv_writer.writerow(max_train_list)
        csv_writer.writerow(max_val_list)
        csv_writer.writerow(average_last_2000_output1_val_acc_list)
        csv_writer.writerow(sd_last_2000_output1_val_acc_list)

    print("===== EVERY 100 EPOCH STATS =====")
    print(
        f"Best of all: accuracy={best_of_all['accuracy']} in fold={best_of_all['fold']} at epoch={best_of_all['epoch']}"
    )

    # immediately show plotted graphs
    if is_show_plots:
        plt.show()


def format_percent(number):
    return format(number, "3.2f")


def calculate_max(data):
    max_percent = np.max(data) * 100
    return format_percent(max_percent)


def calculate_average_last_n(data, n):
    average = np.mean(data[-n:]) * 100
    return format_percent(average)


def calculate_sd_last_n(data, n):
    sd = np.std(data[-n:])
    return format_percent(sd)


def average_from_string_list(data):
    return format_percent(np.mean(list(map(float, data))))


def sd_from_string_list(data):
    return format_percent(np.std(list(map(float, data))))


if __name__ == "__main__":
    cli()
