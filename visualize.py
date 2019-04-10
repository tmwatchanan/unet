import os
import csv
import matplotlib.pyplot as plt

#  DATASET_NAME = "eye_v2-baseline_v7_multiclass-softmax-cce-lw_421-lr_1e_3"
DATASET_NAME = "eye_v2-baseline_v8_multiclass-softmax-cce-lw_0.1_4_2_1-lr_1e_3"
DATASET_DIR_NAME = DATASET_NAME
DATASET_PATH = os.path.join('data', DATASET_NAME)
LOSS_ACC_FILE = os.path.join(DATASET_PATH, f"loss-acc.csv")
fig_acc1_file = os.path.join(DATASET_PATH, f"acc1.png")
fig_acc2_file = os.path.join(DATASET_PATH, f"acc2.png")
fig_acc3_file = os.path.join(DATASET_PATH, f"acc3.png")

epoch_list = []
acc1_list = []
val_acc1_list = []
acc2_list = []
val_acc2_list = []
acc3_list = []
val_acc3_list = []

first_line = False
with open(LOSS_ACC_FILE) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if not first_line:
            print(f"Column names are {', '.join(row)}")
            first_line = True
        else:
            #  print(
            #  f"\tepoch {row[0]} | {row[1]}, {row[2]}, {row[3]}, {row[4]}, {row[5]}, {row[6]}"
            #  )
            epoch_list.append(float(row[0]))
            acc1_list.append(float(row[1]))
            val_acc1_list.append(float(row[2]))
            acc2_list.append(float(row[3]))
            val_acc2_list.append(float(row[4]))
            acc3_list.append(float(row[5]))
            val_acc3_list.append(float(row[6]))


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


plot_graph(1, epoch_list, acc1_list, val_acc1_list, 'Loss', 'Epoch',
           f"{DATASET_DIR_NAME} - Output 1 Model Accuracy",
           ['Train Accuracy', 'Validation Accuracy'], fig_acc1_file)
plot_graph(2, epoch_list, acc2_list, val_acc2_list, 'Loss', 'Epoch',
           f"{DATASET_DIR_NAME} - Output 2 Model Accuracy",
           ['Train Accuracy', 'Validation Accuracy'], fig_acc2_file)
plot_graph(3, epoch_list, acc3_list, val_acc3_list, 'Loss', 'Epoch',
           f"{DATASET_DIR_NAME} - Output 3 Model Accuracy",
           ['Train Accuracy', 'Validation Accuracy'], fig_acc3_file)
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
