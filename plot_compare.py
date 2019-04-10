import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_graph(figure_num, x_label, y_label, title, save_name, data_list):
    fig_acc = plt.figure(figure_num)
    legend_list = []
    for experiment_name, epoch, y in data_list:
        p = plt.plot(epoch, y)
        legend_list.append(experiment_name)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.grid(color='k', linestyle='-', linewidth=1)
    #  plt.ylim(0, 1.0)
    plt.title(title)
    plt.legend(legend_list, loc='lower right')
    fig_acc.savefig(save_name, bbox_inches="tight")


def plot(experiment_list):
    graphs_dir = os.path.join('data', 'comparison')
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    output1_acc_file = os.path.join(graphs_dir, "output1_acc.png")

    #  output_iris_acc_file = os.path.join(graphs_dir, "output_iris_acc.png")
    #  output1_loss_file = os.path.join(graphs_dir, "output1_loss.png")
    #  output_iris_loss_file = os.path.join(graphs_dir, "output_iris_loss.png")
    data_list = []
    for experiment_name in experiment_list:
        print(experiment_name)
        # define paths
        dataset_path = os.path.join('data', experiment_name)
        training_log_file = os.path.join(dataset_path, 'training.csv')

        history_data = pd.read_csv(training_log_file)
        print(history_data.columns)
        data_list.append((experiment_name, history_data['epoch'],
                          history_data['output1_acc']))

    # plot graphs
    plot_graph(1, 'Accuracy', 'Epoch',
               f"Comparison of Output 1 Model Accuracy", output1_acc_file,
               data_list)
    #  plot_graph(2, history_data['epoch'], history_data['output_iris_acc'],
    #  history_data['val_output_iris_acc'], 'Accuracy', 'Epoch',
    #  f"{experiment_name} - Output Iris Model Accuracy",
    #  ['Train Accuracy', 'Validation Accuracy'],
    #  output_iris_acc_file)
    #  plot_graph(3, history_data['epoch'], history_data['output1_loss'],
    #  history_data['val_output1_loss'], 'Loss', 'Epoch',
    #  f"{experiment_name} - Output 1 Model Loss (cce)",
    #  ['Train Loss', 'Validation Loss'], output1_loss_file)
    #  plot_graph(
    #  4, history_data['epoch'], history_data['output_iris_loss'],
    #  history_data['val_output_iris_loss'], 'Loss', 'Epoch',
    #  f"{experiment_name} - Output Iris Model Loss (diff_iris_area)",
    #  ['Train Loss', 'Validation Loss'], output_iris_loss_file)

    # immediately show plotted graphs
    plt.show()


def main():
    experiment_list = [
        'eye_v2-baseline_v9_multiclass-softmax-cce-lw_1_0.1-lr_1e_2-bn',
        'eye_v2-baseline_v10_multiclass-softmax-cce-lw_1_0.01-lr_1e_2-bn',
        'eye_v2-baseline_v11_multiclass-softmax-cce-lw_1_0.01-lr_1e_2-bn'
    ]
    print("hi")
    plot(experiment_list)


if __name__ == "__main__":
    main()
