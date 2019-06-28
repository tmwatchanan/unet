import pytest
from datasets import Dataset


class TestDataset:
    def test_invalid_name(self):
        with pytest.raises(ValueError, match=r".*is invalid dataset name.*"):
            _ = Dataset("eye")


class TestEyeV4:
    def test_name(self):
        dataset = Dataset("eye_v4")
        assert dataset.name == "eye_v4"

    def test_train_batch_size(self):
        dataset = Dataset("eye_v4")
        assert dataset.train_batch_size == 4

    def test_validation_batch_size(self):
        dataset = Dataset("eye_v4")
        assert dataset.validation_batch_size == 4

    def test_train_steps_per_epoch(self):
        dataset = Dataset("eye_v4")
        assert dataset.train_steps_per_epoch == 4

    def test_validation_steps_per_epoch(self):
        dataset = Dataset("eye_v4")
        assert dataset.validation_steps_per_epoch == 2


class TestEyeV5:
    def test_name(self):
        dataset = Dataset("eye_v5")
        assert dataset.name == "eye_v5"

    def test_train_batch_size(self):
        dataset = Dataset("eye_v5")
        assert dataset.train_batch_size == 4

    def test_validation_batch_size(self):
        dataset = Dataset("eye_v5")
        assert dataset.validation_batch_size == 4

    def test_train_steps_per_epoch(self):
        dataset = Dataset("eye_v5")
        assert dataset.train_steps_per_epoch == 8

    def test_validation_steps_per_epoch(self):
        dataset = Dataset("eye_v5")
        assert dataset.validation_steps_per_epoch == 4
