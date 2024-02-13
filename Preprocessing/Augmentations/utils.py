'''
Module that implements auxiliary augmentation utils.

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''
from pathlib import Path
from xml.etree import ElementTree as et
import logging
import shutil


def merge_augmented_instances(images_dir: Path, labels_dir: Path) -> None:
    '''
    Merges all augmented images into main images and labels directories.

    :param images_dir: Path to images directory.
    :param labels_dir: Path to labels directory.
    '''

    try:
        assert images_dir.exists()
    except AssertionError as err:
        logging.error(f'{images_dir} not found')
        raise err
    try:
        assert labels_dir.exists()
    except AssertionError as err:
        logging.error(f'{labels_dir} not found')
        raise err

    augmented_images_directories = [subdir for subdir in images_dir.iterdir() if subdir.is_dir()]
    augmented_labels_directories = [subdir for subdir in labels_dir.iterdir() if subdir.is_dir()]


    for i in range(len(augmented_images_directories)):
        augmented_images = list(augmented_images_directories[i].glob('*'))
        augmented_labels = list(augmented_labels_directories[i].glob('*'))

        for j in range(len(augmented_images)):
            shutil.move(augmented_images[j], images_dir.joinpath(augmented_images[j].name))
            shutil.move(augmented_labels[j], labels_dir.joinpath(augmented_labels[j].name))

        augmented_images_directories[i].rmdir()
        augmented_labels_directories[i].rmdir()

