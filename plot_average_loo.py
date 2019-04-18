import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_graph(epoch_list, x, y, x_label, y_label, title, legend, save_name):
    fig_acc = plt.figure()
    plt.plot(epoch_list, x, 'b')
    plt.plot(epoch_list, y, 'g')
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.grid(color='k', linestyle='-', linewidth=1)
    #  plt.ylim(0, 1.0)
    plt.title(title)
    plt.legend(legend, loc='lower right')
    fig_acc.savefig(save_name, bbox_inches='tight')


def plot(experiment_name):
    # define paths
    dataset_path = os.path.join('data', experiment_name)
    training_log_file = os.path.join(dataset_path, 'training.csv')

    graphs_dir = os.path.join(dataset_path, 'graphs')
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    output1_acc_file = os.path.join(graphs_dir, "output1_acc.png")
    output_iris_acc_file = os.path.join(graphs_dir, "output_iris_acc.png")
    output1_loss_file = os.path.join(graphs_dir, "output1_loss.png")
    output_iris_loss_file = os.path.join(graphs_dir, "output_iris_loss.png")

    history_data = pd.read_csv(training_log_file)
    print(history_data.columns)

    # plot graphs
    plot_graph(history_data['epoch'], history_data['output1_acc'],
               history_data['val_output1_acc'], 'Accuracy', 'Epoch',
               f"{experiment_name} - Output 1 Model Accuracy",
               ['Train Accuracy', 'Validation Accuracy'], output1_acc_file)
    plot_graph(history_data['epoch'], history_data['output_iris_acc'],
               history_data['val_output_iris_acc'], 'Accuracy', 'Epoch',
               f"{experiment_name} - Output Iris Model Accuracy",
               ['Train Accuracy', 'Validation Accuracy'], output_iris_acc_file)
    plot_graph(history_data['epoch'], history_data['output1_loss'],
               history_data['val_output1_loss'], 'Loss', 'Epoch',
               f"{experiment_name} - Output 1 Model Loss (cce)",
               ['Train Loss', 'Validation Loss'], output1_loss_file)
    plot_graph(history_data['epoch'], history_data['output_iris_loss'],
               history_data['val_output_iris_loss'], 'Loss', 'Epoch',
               f"{experiment_name} - Output Iris Model Loss (diff_iris_area)",
               ['Train Loss', 'Validation Loss'], output_iris_loss_file)

    # immediately show plotted graphs
    plt.show()


def plot_average():
    LOO = 6

    graphs_dir = os.path.join('data', 'comparison')
    output1_avg_acc_file = os.path.join(
        graphs_dir,
        f"eye_v3-baseline_v12_multiclass-softmax-cce-lw_1_0.01-loo_1_{LOO}-lr_1e_2-bn.png"
    )

    history_list = []
    for l in range(5):
        experiment_name = f"eye_v3-baseline_v12_multiclass-softmax-cce-lw_1_0.01-loo_{l+1}-lr_1e_2-bn"
        output1_acc_file = os.path.join(graphs_dir,
                                        f"{experiment_name}-output1_acc.png")
        dataset_path = os.path.join('data', experiment_name)
        training_log_file = os.path.join(dataset_path, 'training.csv')
        history_data = pd.read_csv(training_log_file)
        plot_graph(history_data['epoch'], history_data['output1_acc'],
                   history_data['val_output1_acc'], 'Accuracy', 'Epoch',
                   f"{experiment_name} - Output 1 Model Accuracy",
                   ['Train Accuracy', 'Validation Accuracy'], output1_acc_file)
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

    plot_graph(epoch_list, avg_output1_acc_list, avg_output1_val_acc_list,
               'Accuracy', 'Epoch',
               f"{experiment_name} - Output 1 Model Average Accuracy",
               ['Train Accuracy', 'Validation Accuracy'], output1_avg_acc_file)

    # immediately show plotted graphs
    plt.show()


if __name__ == "__main__":
    plot_average()
