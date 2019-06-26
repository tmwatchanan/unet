import click
import os
from termcolor import colored, cprint

from model_v12_cv import create_model as v12_create_model
from model_v12_cv import get_test_data as v12_get_test_data
from model_v12_cv import test_generator as v12_test_generator
from model_v12_cv import save_result as v12_save_result
from model_v15_cv import create_model as v15_create_model
from model_v15_cv import get_test_data as v15_get_test_data
from model_v15_cv import test_generator as v15_test_generator
from model_v15_cv import save_result as v15_save_result


def create():
    TARGET_SIZE = (256, 256)
    NUM_CLASSES = 3
    SAVE_EACH_LAYER = False
    PREDICT_VERBOSE = 1  # 0 = silent, 1
    BATCH_NORMALIZATION = True

    dataset_dir = os.path.join("datasets", "eye_v5-s1")

    segment_inputs = [
        {  # s1
            "name": "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_{0}-lr_1e_2-bn",
            "color_model": "rgb",
            "weights": [4138, 4468, 3396, 3445],
            "input_size": (256, 256, 3),
            "prefix": "v12",
            "segment_id": "s1"
        },
        {  # s3
            "name": "eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_{0}-lr_1e_2-bn",
            "color_model": "hsv",
            "weights": [3874, 3437, 4470, 4996],
            "input_size": (256, 256, 2),
            "prefix": "v15",
            "segment_id": "s3"
        },
    ]
    for s, segment_input in enumerate(segment_inputs):
        model_v = segment_input["prefix"]
        create_model_func = f"{model_v}_create_model"
        get_test_data_func = f"{model_v}_get_test_data"
        test_generator_func = f"{model_v}_test_generator"
        save_result_func = f"{model_v}_save_result"
        for i, weight in enumerate(segment_input["weights"]):
            fold = i + 1
            for set_name in ["test", "train", "validation"]:
                set_dir = os.path.join(dataset_dir, f"fold_{fold}", set_name)
                segment_id = segment_input["segment_id"]
                predicted_result_dir = os.path.join(set_dir, "segments", segment_id)
                if not os.path.exists(predicted_result_dir):
                    os.makedirs(predicted_result_dir)

                experiment_name = segment_input["name"].format(fold)
                experiment_dir = os.path.join("data", experiment_name)
                weights_dir = os.path.join(experiment_dir, "weights")

                trained_weights_filename = f"{weight:08d}.hdf5"
                trained_weights_file = os.path.join(
                    weights_dir, trained_weights_filename
                )

                input_size = segment_input["input_size"]

                # load pretrained model
                model = globals()[create_model_func](
                    pretrained_weights=trained_weights_file,
                    input_size=input_size,
                    num_classes=NUM_CLASSES,
                    batch_normalization=BATCH_NORMALIZATION,
                    is_summary=False,
                )

                color_model = segment_input["color_model"]
                # test the model
                test_data_dict = dict(
                    test_path=set_dir, target_size=TARGET_SIZE, image_color=color_model
                )
                test_flow, test_files = globals()[get_test_data_func](**test_data_dict)
                test_gen = globals()[test_generator_func](test_flow, color_model)

                predict_steps = len(test_files)
                results = model.predict_generator(
                    test_gen, steps=predict_steps, verbose=PREDICT_VERBOSE
                )
                globals()[save_result_func](
                    predicted_result_dir,
                    results,
                    file_names=test_files,
                    weights_name=weight,
                    target_size=TARGET_SIZE,
                    num_class=NUM_CLASSES,
                    save_each_layer=SAVE_EACH_LAYER,
                )
    cprint(
        f"> `create_segment_inputs` command was successfully run, the predicted result will be in ",
        color="green",
        end="",
    )
    cprint(f"{predicted_result_dir}", color="green", attrs=["bold"])


if __name__ == "__main__":
    create()
