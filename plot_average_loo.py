import os
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


@click.group()
def cli():
    pass


def moving_average(x, N):
    return np.convolve(x, np.ones((N, )) / N)[(N - 1):]


def draw_graph(epoch_list, x, y, x_label, y_label, title, legend, is_moving_average):
    fig_acc = plt.figure()
    if is_moving_average:
        x = moving_average(x, 100)
        y = moving_average(y, 100)
    plt.plot(epoch_list, x, 'b')
    plt.plot(epoch_list, y, 'g')
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.grid(color='k', linestyle='-', linewidth=1)
    #  plt.ylim(0, 1.0)
    plt.title(title)
    plt.legend(legend, loc='lower right')
    return fig_acc


@cli.command()
@click.option('--moving_average', 'is_moving_average', is_flag=True)
@click.option('--show', 'is_show_plots', is_flag=True)
def plot(is_moving_average, is_show_plots):
    LOO = 16
    #  experiment_name_template = "eye_v3-baseline_v12_multiclass-softmax-cce-lw_1_0.01-loo_{0}-lr_1e_2-bn"
    experiment_name_template = "eye_v3-baseline_v12_multiclass-softmax-cce-lw_1_0.01-ycbcr-loo_{0}-lr_1e_2-bn"

    graphs_dir = os.path.join('data', 'comparison')

    history_list = []
    for l in range(LOO):
        experiment_name = experiment_name_template.format(l+1)
        moving_average_string = "-moving_average" if is_moving_average else ""
        output1_acc_filename = f"{experiment_name}{moving_average_string}-output1_acc.png"
        output1_acc_file = os.path.join(graphs_dir, output1_acc_filename)
        dataset_path = os.path.join('data', experiment_name)
        training_log_file = os.path.join(dataset_path, 'training.csv')
        history_data = pd.read_csv(training_log_file)
        graph_title = f"{experiment_name}\nOutput 1 Model Accuracy"
        graph_title += " (moving average)" if is_moving_average else ""
        fig_loo = draw_graph(history_data['epoch'], history_data['output1_acc'],
                   history_data['val_output1_acc'], 'Accuracy', 'Epoch',
                   graph_title,
                   ['Train Accuracy', 'Validation Accuracy'], is_moving_average)
        fig_loo.savefig(output1_acc_file, bbox_inches='tight')
        history_list.append(history_data)
    epoch_list = [a['epoch'] for a in history_list]
    min_epochs = min(map(len, epoch_list))
    epoch_list = range(min_epochs)

    output1_acc_list = [a['output1_acc'] for a in history_list]
    output1_acc_array = np.transpose(np.array(output1_acc_list))
    avg_output1_acc_list = [np.mean(a) for a in output1_acc_array]

    output1_val_acc_list = [a['val_output1_acc'] for a in history_list]
    output1_val_acc_array = np.transpose(np.array(output1_val_acc_list))
    avg_output1_val_acc_list = [np.mean(a) for a in output1_val_acc_array]

    average_name = f"avg_1_{LOO}"
    experiment_name = experiment_name_template.format(average_name)
    output1_avg_acc_file = os.path.join(
        graphs_dir,
        experiment_name + '.png'
    )
    graph_title = f"{experiment_name}\nOutput 1 Model Average Accuracy"
    graph_title += " (moving average)" if is_moving_average else ""
    fig_avg = draw_graph(epoch_list, avg_output1_acc_list, avg_output1_val_acc_list,
               'Accuracy', 'Epoch',
               graph_title,
               ['Train Accuracy', 'Validation Accuracy'], is_moving_average)
    #  fig_avg.text(3, 8, 'boxed italics text in data coords', style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    max_train = format(np.max(avg_output1_acc_list) * 100, '3.2f')
    max_val = format(np.max(avg_output1_val_acc_list) * 100, '3.2f')
    stats_text = f"MAX train = {max_train}\nMAX validation = {max_val}"
    anchored_text = AnchoredText(stats_text, loc=3) # 3=lower left
    ax = fig_avg.gca()
    ax.add_artist(anchored_text)
    fig_avg.savefig(output1_avg_acc_file, bbox_inches='tight')

    print(f"MAX train = {max_train}")
    print(f"MAX validation = {max_val}")

    # immediately show plotted graphs
    if is_show_plots:
        plt.show()


if __name__ == "__main__":
    cli()
