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


@cli.command()
@click.pass_context
def training(ctx):
    experiment_name_template = (
        "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_{0}-lr_1e_2-bn"
    )
    data_dir = "data"

    fold = 1
    best_val_output1_acc_folds = []
    for fold in range(1, 4 + 1):
        experiment_name = experiment_name_template.format(fold)
        experiment_name_dir = os.path.join(data_dir, experiment_name)
        training_log_file = os.path.join(experiment_name_dir, "training.csv")

        training_log = pd.read_csv(training_log_file)
        output1_acc = training_log["output1_acc"]
        val_output1_acc = training_log["val_output1_acc"]

        max_val_output1_acc, arg_max_val_output1_acc = find_best_accuracy(
            val_output1_acc
        )
        max_val_output1_acc_epoch = arg_max_val_output1_acc + 1
        max_output1_acc = output1_acc[arg_max_val_output1_acc]
        best_val_output1_acc_folds.append(
            (fold, max_val_output1_acc_epoch, max_output1_acc, max_val_output1_acc)
        )

        print(f"fold {fold}, max accuracy @ epoch # {max_val_output1_acc_epoch}")
        print(f"max training accuracy = {convert_to_percentage(max_output1_acc)}")
        print(f"max validation accuracy = {convert_to_percentage(max_val_output1_acc)}")
    print("-----------------------------------")
    val_output1_acc_of_all_folds = list(map(lambda x: x[3], best_val_output1_acc_folds))
    best_fold_index = np.argmax(np.array(val_output1_acc_of_all_folds))
    best_fold = best_val_output1_acc_folds[best_fold_index]
    print(f"all fold {best_fold[0]} @ epoch = {best_fold[1]}")
    print(f"best training accuracy = {convert_to_percentage(best_fold[2])}")
    print(f"best validation accuracy = {convert_to_percentage(best_fold[3])}")


def find_best_accuracy(accuracy_values):
    arg_max_output1_acc = np.argmax(np.array(accuracy_values))
    max_output1_acc = accuracy_values[arg_max_output1_acc]
    return max_output1_acc, arg_max_output1_acc


def format_percent(number):
    return format(number, "3.2f")


def convert_to_percentage(number):
    return format_percent(number * 100)


def calculate_max(data):
    max_percent = np.max(data) * 100
    return format_percent(max_percent)


if __name__ == "__main__":
    cli()
