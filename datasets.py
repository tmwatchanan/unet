class Dataset:
    def __init__(self, name):
        if name == "eye_v4":
            self.set_eye_v4()
        elif name == "eye_v5":
            self.set_eye_v5()
        else:
            raise ValueError(f"{name} is invalid dataset name")

    def set_eye_v4(self):
        self.name = "eye_v4"
        self.train_batch_size = 4
        self.validation_batch_size = 4
        self.train_steps_per_epoch = 4
        self.validation_steps_per_epoch = 2

    def set_eye_v5(self):
        self.name = "eye_v5"
        self.train_batch_size = 4
        self.validation_batch_size = 4
        self.train_steps_per_epoch = 8
        self.validation_steps_per_epoch = 4
