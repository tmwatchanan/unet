import os
import csv
import matplotlib.pyplot as plt

DATASET_NAME = "eye-multiclass-baseline_v2-softmax-cce-lr1e_2"
DATASET_DIR_NAME = DATASET_NAME
DATASET_PATH = os.path.join('data', DATASET_NAME)
LOSS_ACC_FILE = os.path.join(DATASET_PATH, f"{DATASET_DIR_NAME}-loss-acc.csv")
fig_acc_file = os.path.join(DATASET_PATH, f"{DATASET_DIR_NAME}-acc.png")
fig_cce_file = os.path.join(DATASET_PATH, f"{DATASET_DIR_NAME}-cce.png")
fig_mse_file = os.path.join(DATASET_PATH, f"{DATASET_DIR_NAME}-mse.png")
fig_mae_file = os.path.join(DATASET_PATH, f"{DATASET_DIR_NAME}-mae.png")
fig_mape_file = os.path.join(DATASET_PATH, f"{DATASET_DIR_NAME}-mape.png")

epoch_list = []
acc_list = []
val_acc_list = []
loss_list = []
val_loss_list = []
mse_list = []
val_mse_list = []
mae_list = []
val_mae_list = []
mape_list = []
val_mape_list = []

first_line = False
with open(LOSS_ACC_FILE) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if not first_line:
            print(f"Column names are {', '.join(row)}")
            first_line = True
        else:
            print(f"\tepoch {row[0]} | {row[1]}, {row[2]}, {row[3]}, {row[4]}")
            epoch_list.append(float(row[0]))
            acc_list.append(float(row[1]))
            val_acc_list.append(float(row[2]))
            loss_list.append(float(row[3]))
            val_loss_list.append(float(row[4]))
            mse_list.append(float(row[5]))
            val_mse_list.append(float(row[6]))
            mae_list.append(float(row[7]))
            val_mae_list.append(float(row[8]))
            mape_list.append(float(row[9]))
            val_mape_list.append(float(row[10]))


def plot_graph(figure_num, epoch_list, x, y, x_label, y_label, title, legend,
               save_name):
    fig_acc = plt.figure(figure_num)
    plt.plot(epoch_list, x, 'b')
    plt.plot(epoch_list, y, 'g')
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.grid(color='k', linestyle='-', linewidth=1)
    #  plt.ylim(0, 1.0)
    plt.title(title)
    plt.legend(legend, loc='upper left')
    fig_acc.savefig(save_name)


plot_graph(1, epoch_list, acc_list, val_acc_list, 'Accuracy', 'Epoch',
           f"{DATASET_DIR_NAME} - Model Accuracy",
           ['Train Accuracy', 'Validation Accuracy'], fig_acc_file)
plot_graph(2, epoch_list, loss_list, val_loss_list, 'Loss', 'Epoch',
           f"{DATASET_DIR_NAME} - Model Loss\n(categorical crossentropy)",
           ['Train Loss', 'Validation Loss'], fig_cce_file)
plot_graph(3, epoch_list, mse_list, val_mse_list, 'Loss', 'Epoch',
           f"{DATASET_DIR_NAME} - Model Loss\n(mean squared error)",
           ['Train Loss', 'Validation Loss'], fig_mse_file)
plot_graph(4, epoch_list, mae_list, val_mae_list, 'Loss', 'Epoch',
           f"{DATASET_DIR_NAME} - Model Loss\n(mean absolute error)",
           ['Train Loss', 'Validation Loss'], fig_mae_file)
plot_graph(
    5, epoch_list, mape_list, val_mape_list, 'Loss', 'Epoch',
    f"{DATASET_DIR_NAME} - Model Loss\n(mean absolute percentage error)",
    ['Train Loss', 'Validation Loss'], fig_mape_file)
plt.show()

#  # plot accuracy
#  fig_acc=plt.figure(1)
#  plt.plot(epoch_list, acc_list, 'b')
#  plt.plot(epoch_list, val_acc_list, 'g')
#  plt.grid(color = 'k', linestyle = '-', linewidth = 1)
#  plt.ylim(0, 1.0)
#  plt.title(f"{DATASET_DIR_NAME} - Model Accuracy")
#  plt.ylabel('Accuracy')
#  plt.xlabel('Epoch')
#  plt.legend(['Train Accuracy', 'Validation Accuracy'], loc = 'upper left')
#  fig_acc.savefig(fig_acc_file)

#  # plot loss
#  fig_loss=plt.figure(2)
#  plt.plot(epoch_list, loss_list, 'y')
#  plt.plot(epoch_list, val_loss_list, 'r')
#  plt.grid(color = 'k', linestyle = '-', linewidth = 1)
#  plt.title(f"{DATASET_DIR_NAME} - Model Loss")
#  plt.ylabel('Loss')
#  plt.xlabel('Epoch')
#  plt.legend(['Train Loss', 'Validation Loss'], loc = 'upper left')
#  fig_loss.savefig(fig_loss_file)
