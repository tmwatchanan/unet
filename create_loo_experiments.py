import os
import natsort
import shutil
import copy

EXPERIMENT_NAME = (
    "eye_v3-baseline_v11_multiclass-softmax-cce-lw_1_0.01-hsv-loo_{}-lr_1e_2-bn"
)

eye_v3_dir = os.path.join("datasets", "eye_v3")
eye_v3_images_dir = os.path.join(eye_v3_dir, "images")
eye_v3_labels_dir = os.path.join(eye_v3_dir, "labels")


def main():
    #  filenames = [
    #  os.path.join(eye_v3_images_dir, fn)
    #  for fn in next(os.walk(eye_v3_images_dir))[2]
    #  ]
    filenames = next(os.walk(eye_v3_images_dir))[2]
    filenames = natsort.natsorted(filenames, reverse=False)
    print(filenames)

    for (idx, filename) in enumerate(filenames, 0):
        sub_exp_path = os.path.join("data", EXPERIMENT_NAME.format(idx + 1))
        print(idx, filename, sub_exp_path)

        train_path = os.path.join(sub_exp_path, "train")
        validation_path = os.path.join(sub_exp_path, "validation")

        test_path = os.path.join(sub_exp_path, "test", "images")
        shutil.copytree(eye_v3_images_dir, test_path)

        train_images_path = os.path.join(train_path, "images")
        train_labels_path = os.path.join(train_path, "labels")

        validation_images_path = os.path.join(validation_path, "images")
        validation_labels_path = os.path.join(validation_path, "labels")

        try:
            os.makedirs(train_images_path)
        except FileExistsError:  # directory already exists
            print(f"{train_images_path} is already existed.")

        try:
            os.makedirs(train_labels_path)
        except FileExistsError:  # directory already exists
            print(f"{train_labels_path} is already existed.")

        try:
            os.makedirs(validation_images_path)
        except FileExistsError:  # directory already exists
            print(f"{validation_images_path} is already existed.")

        try:
            os.makedirs(validation_labels_path)
        except FileExistsError:  # directory already exists
            print(f"{validation_labels_path} is already existed.")

        train_images_file_list = copy.deepcopy(filenames)
        del train_images_file_list[idx]
        validation_images_file_list = filenames[idx]

        for img in train_images_file_list:
            src = os.path.join(eye_v3_images_dir, img)
            dest = os.path.join(train_images_path, img)
            shutil.copy(src, dest)

            src = os.path.join(eye_v3_labels_dir, img)
            dest = os.path.join(train_labels_path, img)
            shutil.copy(src, dest)

        src = os.path.join(eye_v3_images_dir, validation_images_file_list)
        dest = os.path.join(validation_images_path, validation_images_file_list)
        shutil.copy(src, dest)

        src = os.path.join(eye_v3_labels_dir, validation_images_file_list)
        dest = os.path.join(validation_labels_path, validation_images_file_list)
        shutil.copy(src, dest)


if __name__ == "__main__":
    main()
