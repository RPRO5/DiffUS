import argparse

from datasets.base import Dataset
from datasets.generation import get_dataset


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def dataset(folder,
            image_size,
            exts=['jpg', 'jpeg', 'png', 'tiff'],
            augment_flip=False,
            convert_image_to=None,
            condition=0,
            equalizeHist=False,
            crop_patch=True,
            sample=False, 
            generation=False):
    if generation:
          dataset_import = "base"
    else:
        dataset_import = "base"
        
        args = dict2namespace(args)
        config = dict2namespace(config)
        return get_dataset(args, config)[0]
