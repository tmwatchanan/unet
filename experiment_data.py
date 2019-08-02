def get_experiment_pool():
    from ensemble_aggregation_predict import EnsembleMode

    experiment_pool = {
        # "eye_v4-model_v12-rgb": {
        #     "input_size": (256, 256, 3),
        #     "color_model": "rgb",
        #     "file": "model_v12_cv",
        #     "dataset": "eye_v4",
        #     "experiments": [
        #         (
        #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
        #             4991,
        #         ),
        #         (
        #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
        #             3974,
        #         ),
        #         (
        #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
        #             4853,
        #         ),
        #         (
        #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
        #             4345,
        #         ),
        #     ],
        # },
        # "eye_v4-model_v12-hsv": {
        #     "input_size": (256, 256, 3),
        #     "color_model": "hsv",
        #     "file": "model_v12_cv",
        #     "dataset": "eye_v4",
        #     "experiments": [
        #         (
        #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn",
        #             2702,
        #         ),
        #         (
        #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn",
        #             4385,
        #         ),
        #         (
        #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn",
        #             3730,
        #         ),
        #         (
        #             "eye_v4-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn",
        #             4865,
        #         ),
        #     ],
        # },
        # "eye_v4-model_v15-hsv": {
        #     "input_size": (256, 256, 2),
        #     "color_model": "hsv",
        #     "file": "model_v15_cv",
        #     "dataset": "eye_v4",
        #     "experiments": [
        #         (
        #             "eye_v4-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn",
        #             3526,
        #         ),
        #         (
        #             "eye_v4-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn",
        #             4984,
        #         ),
        #         (
        #             "eye_v4-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn",
        #             4287,
        #         ),
        #         (
        #             "eye_v4-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn",
        #             4873,
        #         ),
        #     ],
        # },
        # "eye_v4-model_v13-rgb": {
        #     "input_size": (256, 256, 5),
        #     "color_model": "rgb",
        #     "file": "model_v13_cv",
        #     "dataset": "eye_v4",
        #     "experiments": [
        #         (
        #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
        #             4925,
        #         ),
        #         (
        #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
        #             4829,
        #         ),
        #         (
        #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
        #             3061,
        #         ),
        #         (
        #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
        #             4808,
        #         ),
        #     ],
        # },
        # "eye_v4-model_v13-hsv": {
        #     "input_size": (256, 256, 5),
        #     "color_model": "hsv",
        #     "file": "model_v13_cv",
        #     "dataset": "eye_v4",
        #     "experiments": [
        #         (
        #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn",
        #             2321,
        #         ),
        #         (
        #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn",
        #             2076,
        #         ),
        #         (
        #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn",
        #             4462,
        #         ),
        #         (
        #             "eye_v4-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn",
        #             2940,
        #         ),
        #     ],
        # },
        "eye_v5-model_v12-rgb": {
            "abbreviation": "s1",
            "input_size": (256, 256, 3),
            "color_model": "rgb",
            "file": "model_v12_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
                    4138,
                ),
                (
                    "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
                    4468,
                ),
                (
                    "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
                    3396,
                ),
                (
                    "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
                    3445,
                ),
            ],
        },
        "eye_v5-model_v12-hsv": {
            "abbreviation": "s2",
            "input_size": (256, 256, 3),
            "color_model": "hsv",
            "file": "model_v12_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn",
                    3722,
                ),
                (
                    "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn",
                    4884,
                ),
                (
                    "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn",
                    4971,
                ),
                (
                    "eye_v5-model_v12_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn",
                    4283,
                ),
            ],
        },
        "eye_v5-model_v15-hsv": {
            "abbreviation": "s3",
            "input_size": (256, 256, 2),
            "color_model": "hsv",
            "file": "model_v15_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn",
                    2637,
                ),
                (
                    "eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn",
                    4811,
                ),
                (
                    "eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn",
                    4122,
                ),
                (
                    "eye_v5-model_v15_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn",
                    4388,
                ),
            ],
        },
        "eye_v5-model_v13-rgb": {
            "abbreviation": "s4",
            "input_size": (256, 256, 5),
            "color_model": "rgb",
            "file": "model_v13_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
                    4909,
                ),
                (
                    "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
                    4619,
                ),
                (
                    "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
                    4692,
                ),
                (
                    "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
                    4399,
                ),
            ],
        },
        "eye_v5-model_v13-hsv": {
            "abbreviation": "s5",
            "input_size": (256, 256, 5),
            "color_model": "hsv",
            "file": "model_v13_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_1-lr_1e_2-bn",
                    4684,
                ),
                (
                    "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_2-lr_1e_2-bn",
                    4513,
                ),
                (
                    "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_3-lr_1e_2-bn",
                    4111,
                ),
                (
                    "eye_v5-model_v13_multiclass-softmax-cce-lw_1_0-hsv-fold_4-lr_1e_2-bn",
                    4886,
                ),
            ],
        },
        # "eye_v5-model_v23-rgb": {
        #     "input_size": (256, 256, 9),
        #     "color_model": "rgb",
        #     "file": "model_v23_cv",
        #     "dataset": "eye_v5-pass_1",
        #     "experiments": [
        #         (
        #             "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
        #             2456,
        #         ),
        #         (
        #             "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
        #             4480,
        #         ),
        #         (
        #             "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
        #             4012,
        #         ),
        #         (
        #             "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
        #             3308,
        #         ),
        #     ],
        # },
        "eye_v5-model_v23-rgb": {
            "input_size": (256, 256, 9),
            "color_model": "rgb",
            "file": "model_v23_cv",
            "dataset": "eye_v5-pass_1",
            "experiments": [
                (
                    "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
                    3742,
                ),
                (
                    "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
                    4903,
                ),
                (
                    "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
                    3637,
                ),
                (
                    "eye_v5-model_v23_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
                    4617,
                ),
            ],
        },
        "eye_v5-model_v28-hsv": {
            "abbreviation": "s6",
            "input_size": (256, 256, 8),
            "color_model": None,
            "file": "model_v28_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v28_multiclass-softmax-cce-lw_1_0-fold_1-lr_1e_2-bn",
                    4186,
                ),
                (
                    "eye_v5-model_v28_multiclass-softmax-cce-lw_1_0-fold_2-lr_1e_2-bn",
                    4717,
                ),
                (
                    "eye_v5-model_v28_multiclass-softmax-cce-lw_1_0-fold_3-lr_1e_2-bn",
                    4737,
                ),
                (
                    "eye_v5-model_v28_multiclass-softmax-cce-lw_1_0-fold_4-lr_1e_2-bn",
                    4185,
                ),
            ],
        },
        "eye_v5-model_v35-rgb": {
            "abbreviation": "s31",
            "input_size": (256, 256, 3),
            "color_model": "rgb",
            "file": "model_v35_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v35_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
                    3709,
                ),
                (
                    "eye_v5-model_v35_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
                    3988,
                ),
                (
                    "eye_v5-model_v35_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
                    3731,
                ),
                (
                    "eye_v5-model_v35_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
                    4898,
                ),
            ],
        },
        "eye_v5-unet_v2-rgb": {
            "abbreviation": "u2",
            "input_size": (256, 256, 3),
            "color_model": "rgb",
            "file": "unet_v2_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-unet_v2_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_4-bn",
                    4500,
                ),
                (
                    "eye_v5-unet_v2_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_4-bn",
                    3900,
                ),
                (
                    "eye_v5-unet_v2_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_4-bn",
                    4500,
                ),
                (
                    "eye_v5-unet_v2_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_4-bn",
                    4500,
                ),
            ],
            "weight_name": "unet_v2_best",
        },
        "eye_v5-unet_v3-rgb": {
            "abbreviation": "u3",
            "input_size": (256, 256, 3),
            "color_model": "rgb",
            "file": "unet_v3_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-unet_v3_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_4-bn",
                    3800,
                ),
                (
                    "eye_v5-unet_v3_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_4-bn",
                    3400,
                ),
                (
                    "eye_v5-unet_v3_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_4-bn",
                    2700,
                ),
                (
                    "eye_v5-unet_v3_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_4-bn",
                    4900,
                ),
            ],
            "weight_name": "unet_v3_best",
        },
        "eye_v5-unet_v4-rgb": {
            "abbreviation": "u4",
            "input_size": (256, 256, 3),
            "color_model": "rgb",
            "file": "unet_v4_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-unet_v4_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_4-bn",
                    3200,
                ),
                (
                    "eye_v5-unet_v4_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_4-bn",
                    4500,
                ),
                (
                    "eye_v5-unet_v4_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_4-bn",
                    4800,
                ),
                (
                    "eye_v5-unet_v4_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_4-bn",
                    2100,
                ),
            ],
            "weight_name": "unet_v4_best",
        },
        "eye_v5-unet-rgb": {
            "abbreviation": "u1",
            "input_size": (256, 256, 3),
            "color_model": "rgb",
            "file": "unet_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-unet_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_4-bn",
                    3669,
                ),
                (
                    "eye_v5-unet_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_4-bn",
                    4205,
                ),
                (
                    "eye_v5-unet_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_4-bn",
                    1961,
                ),
                (
                    "eye_v5-unet_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_4-bn",
                    2693,
                ),
            ],
            "weight_name": "unet_best",
        },
        "eye_v5-segnet-rgb": {
            "abbreviation": "sg1",
            "input_size": (256, 256, 3),
            "color_model": "rgb",
            "file": "segnet_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-segnet_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_4-bn",
                    4438,
                ),
                (
                    "eye_v5-segnet_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_4-bn",
                    2949,
                ),
                (
                    "eye_v5-segnet_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_4-bn",
                    748,
                ),
                (
                    "eye_v5-segnet_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_4-bn",
                    2808,
                ),
            ],
            "weight_name": "segnet_best",
        },
        "sum_confidences_s1s2s3s4s5": {
            "dataset": "eye_v5",
            "ensemble_mode": EnsembleMode.summation,
            "models": [
                "eye_v5-model_v12-rgb",
                "eye_v5-model_v12-hsv",
                "eye_v5-model_v15-hsv",
                "eye_v5-model_v13-rgb",
                "eye_v5-model_v13-hsv",
            ],
        },
        "eye_v5-model_v35-rgb": {
            "abbreviation": "s31",
            "input_size": (256, 256, 3),
            "color_model": "rgb",
            "file": "model_v35_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v35_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
                    3709,
                ),
                (
                    "eye_v5-model_v35_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
                    3988,
                ),
                (
                    "eye_v5-model_v35_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
                    3731,
                ),
                (
                    "eye_v5-model_v35_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
                    4898,
                ),
            ],
        },
        "eye_v5-model_v36-rgb": {
            "abbreviation": "s34",
            "input_size": (256, 256, 5),
            "color_model": "rgb",
            "file": "model_v36_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v36_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
                    4379,
                ),
                (
                    "eye_v5-model_v36_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
                    4151,
                ),
                (
                    "eye_v5-model_v36_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
                    4833,
                ),
                (
                    "eye_v5-model_v36_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
                    4814,
                ),
            ],
        },
        "eye_v5-model_v37": {
            "abbreviation": "s36",
            "input_size": (256, 256, 8),
            "color_model": None,
            "file": "model_v37_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v37_multiclass-softmax-cce-lw_1_0-fold_1-lr_1e_2-bn",
                    4826,
                ),
                (
                    "eye_v5-model_v37_multiclass-softmax-cce-lw_1_0-fold_2-lr_1e_2-bn",
                    4699,
                ),
                (
                    "eye_v5-model_v37_multiclass-softmax-cce-lw_1_0-fold_3-lr_1e_2-bn",
                    4974,
                ),
                (
                    "eye_v5-model_v37_multiclass-softmax-cce-lw_1_0-fold_4-lr_1e_2-bn",
                    4897,
                ),
            ],
        },
        "eye_v5-model_v38-rgb": {
            "input_size": (256, 256, 15),
            "color_model": None,
            "file": "model_v38_cv",
            "dataset": "eye_v5-pass_1",
            "experiments": [
                (
                    "eye_v5-model_v38_multiclass-softmax-cce-lw_1_0-fold_1-lr_1e_4-bn",
                    147,
                ),
                (
                    "eye_v5-model_v38_multiclass-softmax-cce-lw_1_0-fold_2-lr_1e_4-bn",
                    483,
                ),
                (
                    "eye_v5-model_v38_multiclass-softmax-cce-lw_1_0-fold_3-lr_1e_4-bn",
                    447,
                ),
                (
                    "eye_v5-model_v38_multiclass-softmax-cce-lw_1_0-fold_4-lr_1e_4-bn",
                    495,
                ),
            ],
        },
        "eye_v5-model_v39-rgb": {
            "abbreviation": "s41",
            "input_size": (256, 256, 3),
            "color_model": "rgb",
            "file": "model_v39_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v39_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
                    4977,
                ),
                (
                    "eye_v5-model_v39_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
                    4556,
                ),
                (
                    "eye_v5-model_v39_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
                    4786,
                ),
                (
                    "eye_v5-model_v39_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
                    4492,
                ),
            ],
            "weight_name": "model_v39_best",
        },
        "eye_v5-model_v40-rgb": {
            "abbreviation": "s44",
            "input_size": (256, 256, 5),
            "color_model": "rgb",
            "file": "model_v40_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v40_multiclass-softmax-cce-lw_1_0-rgb-fold_1-lr_1e_2-bn",
                    3799,
                ),
                (
                    "eye_v5-model_v40_multiclass-softmax-cce-lw_1_0-rgb-fold_2-lr_1e_2-bn",
                    4704,
                ),
                (
                    "eye_v5-model_v40_multiclass-softmax-cce-lw_1_0-rgb-fold_3-lr_1e_2-bn",
                    4976,
                ),
                (
                    "eye_v5-model_v40_multiclass-softmax-cce-lw_1_0-rgb-fold_4-lr_1e_2-bn",
                    4553,
                ),
            ],
            "weight_name": "model_v40_best",
        },
        "eye_v5-model_v41": {
            "abbreviation": "s44",
            "input_size": (256, 256, 8),
            "color_model": None,
            "file": "model_v41_cv",
            "dataset": "eye_v5",
            "experiments": [
                (
                    "eye_v5-model_v41_multiclass-softmax-cce-lw_1_0-fold_1-lr_1e_2-bn",
                    4940,
                ),
                (
                    "eye_v5-model_v41_multiclass-softmax-cce-lw_1_0-fold_2-lr_1e_2-bn",
                    3756,
                ),
                (
                    "eye_v5-model_v41_multiclass-softmax-cce-lw_1_0-fold_3-lr_1e_2-bn",
                    4822,
                ),
                (
                    "eye_v5-model_v41_multiclass-softmax-cce-lw_1_0-fold_4-lr_1e_2-bn",
                    3541,
                ),
            ],
            "weight_name": "model_v41_best",
        },
        "sum_confidences_s31u4sg1": {
            "dataset": "eye_v5",
            "ensemble_mode": EnsembleMode.summation,
            "models": [
                "eye_v5-model_v35-rgb",
                "eye_v5-unet_v4-rgb",
                "eye_v5-segnet-rgb",
            ],
        },
    }
    return experiment_pool
