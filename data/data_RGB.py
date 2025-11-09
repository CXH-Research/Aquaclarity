import os
from .dataset_RGB import DataLoaderTrain, DataLoaderTest


def get_training_data(rgb_dir, target_class, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, target_class, img_options)


def get_test_data(rgb_dir, target_class, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, target_class, img_options)