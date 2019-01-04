import csv
import matplotlib.pyplot as plt

LOSS_ACC_FILE = 'unet-eye-loss-acc.csv'

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
plt.figure(1)
plt.plot(epoch_list, acc_list, 'b')
plt.plot(epoch_list, val_acc_list, 'g')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')

# plot loss
plt.figure(2)
plt.plot(epoch_list, loss_list, 'y')
plt.plot(epoch_list, val_loss_list, 'r')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
plt.show()
