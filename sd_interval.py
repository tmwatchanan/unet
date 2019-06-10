import os

import numpy as np
import pandas as pd


def print_out_sd():
    # define paths
    experiment_name = (
        "eye_v3-s3-model_v21_multiclass-softmax-cce-lw_1_0.01-hsv-loo_14-lr_1e_2-bn"
    )
    dataset_path = os.path.join("data", experiment_name)
    training_log_file = os.path.join(dataset_path, "training.csv")

    # read data from csv file
    history_data = pd.read_csv(training_log_file)

    val_output1_acc_list = history_data["val_output1_acc"]

    interval_1 = val_output1_acc_list[0:5000].values
    interval_2 = val_output1_acc_list[4999:10000].values
    interval_3 = val_output1_acc_list[9999:15000].values
    interval_4 = val_output1_acc_list[14999:20000].values

    sd_interval_1 = np.std(interval_1)
    sd_interval_2 = np.std(interval_2)
    sd_interval_3 = np.std(interval_3)
    sd_interval_4 = np.std(interval_4)

    print(sd_interval_1)
    print(sd_interval_2)
    print(sd_interval_3)
    print(sd_interval_4)


if __name__ == "__main__":
    print_out_sd()
