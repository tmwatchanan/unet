import os
import csv
import matplotlib.pyplot as plt

DATASET_NAME = "eye"
DATASET_PATH = os.path.join('data', DATASET_NAME)
LOSS_ACC_FILE = os.path.join(DATASET_PATH, f"unet-{DATASET_NAME}-loss-acc.csv")
fig_acc_file = os.path.join(DATASET_PATH, f"unet-{DATASET_NAME}-acc.png")
fig_loss_file = os.path.join(DATASET_PATH, f"unet-{DATASET_NAME}-loss.png")

epoch_list = []
acc_list = []
val_acc_list = []
loss_list = []
val_loss_list = []

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

# plot accuracy
fig_acc = plt.figure(1)
plt.plot(epoch_list, acc_list, 'b')
plt.plot(epoch_list, val_acc_list, 'g')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')
fig_acc.savefig(fig_acc_file)

# plot loss
fig_loss = plt.figure(2)
plt.plot(epoch_list, loss_list, 'y')
plt.plot(epoch_list, val_loss_list, 'r')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
fig_loss.savefig(fig_loss_file)
plt.show()
