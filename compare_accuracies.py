import os
import csv

import numpy as np
import pandas as pd


def write_compared_accuracies():
    # define paths
    experiment_names = [
        "eye_v3-model_v12_multiclass-softmax-cce-lw_1_0.01-hsv-loo_{0}-lr_1e_2-bn",
        "eye_v3-model_v15_multiclass-softmax-cce-lw_1_0.01-hsv-loo_{0}-lr_1e_2-bn",
        "eye_v3-s4-model_v26_multiclass-softmax-cce-lw_1_0.01-hsv-loo_{0}-lr_1e_2-bn",
    ]
    output_rows = []
    for fold in range(1, 16 + 1):
        for epoch in range(499, 5000, 500):
            output_row = []
            output_row.append(fold)
            output_row.append(epoch + 1)
            for experiment_name in experiment_names:
                # read data from csv file
                dataset_path = os.path.join("data", experiment_name.format(fold))
                training_log_file = os.path.join(dataset_path, "training.csv")
                history_data = pd.read_csv(training_log_file)
                # get accuracy of validation set at specific epoch
                val_output1_acc_list = history_data["val_output1_acc"]
                fold_epoch_accuracy = val_output1_acc_list[epoch]
                output_row.append(convert_to_percentage(fold_epoch_accuracy))
            output_rows.append(output_row)

    # # write outputs to file
    csv_dir = os.path.join("data", "comparison", "csv")
    acc_csv_filename = f"model_v12_v15_v26.csv"
    acc_csv_file = os.path.join(csv_dir, acc_csv_filename)
    with open(acc_csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        # prepare data
        header_list = ["fold", "epoch", "model_v12", "model_v15", "model_v26"]
        csv_writer.writerow(header_list)

        for output_row in output_rows:
            csv_writer.writerow(output_row)


def format_percent(number):
    return format(number, "3.2f")


def convert_to_percentage(number):
    return format_percent(number * 100)


if __name__ == "__main__":
    write_compared_accuracies()
