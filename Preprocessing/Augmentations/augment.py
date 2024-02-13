'''
Module that implements augmentations utilites.

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''


import math
import albumentations as A
from PIL import Image
from pathlib import Path
from multiprocessing import Pool
import multiprocessing
import numpy as np
import logging


def perform_augmentations(
        data_dir: Path,
        augmentations_file: Path,
        class_list: list) -> None:
    '''
    Function to do image augmentations using multiprocessing.

    :param data_dir: Directory to fetch images.
    :param augmentations_file: YAML file with albumentations style.
    :param class_list: list of classes
    '''
    try:
        assert data_dir.exists()
    except AssertionError as err:
        logging.error(f"{data_dir} not found.")
        raise err

    try:
        assert augmentations_file.exists()
    except AssertionError as err:
        logging.error(f"{augmentations_file} not found.")
        raise err

    Augmented_Image_Dir = data_dir.joinpath(augmentations_file.stem)
    if not Path.exists(Augmented_Image_Dir):
        Augmented_Image_Dir.mkdir(exist_ok=True)
    transforms = A.load(str(augmentations_file), data_format='yaml')


    list_dir = list(data_dir.rglob('*'))


    num_cpus = multiprocessing.cpu_count()
    pool = Pool(int(num_cpus / 2))
    lim_inf = 0
    lim_sup = math.floor(len(list_dir) / num_cpus)
    batch = lim_sup

    for i in range(num_cpus):

        pool.apply_async(compute_kernel,
                         args=(augmentations_file.stem,
                               list_dir[lim_inf:lim_sup],
                               transforms,
                               Augmented_Image_Dir))
        lim_inf = lim_sup
        if lim_sup > len(list_dir):
            lim_sup = len(list_dir) - 1
        else:
            lim_sup += batch

    pool.close()
    pool.join()


def compute_kernel(
        transform_name: str,
        list_dir: list,
        transforms: object,
        Augmented_Image_Dir: Path) -> None:
    '''
    Core function that performs image augmentations.

    :param transform_name: Name of the transformation applied.
    :param list_dir: list of images to augment
    :param transforms: Albumentations object that performs augmentation
    :param Augmented_Image_Dir: Directory to store augmented images
    '''
    list_dir2 = [x.stem for x in list_dir if x.is_file()]

    i = 0
    for image_file in list_dir:

        if image_file.is_file():

            image = np.array(Image.open(str(image_file)))

            if image is not None:

                try:
                    transformed = transforms(image=image)

                    transformed_image = Image.fromarray(transformed['image'])

                    transformed_image.save(
                        str(Augmented_Image_Dir.joinpath(f"{transform_name}_{image_file.name}")))

                    print("Image" +
                          str(list_dir2[i]) +
                          ".png processed and saved")

                except Exception as e:
                    print(e)
                    raise
                i += 1


if __name__ == '__main__':
    perform_augmentations(
        Path('../../Data/FootballerDetection/raw_data/images'),
        Path('../../Data/FootballerDetection/raw_data/labels'),
        Path('./transformations/RandomRain.yml'),
        [
            'referee',
            'player',
            'ball',
            'goalkeeper'])
