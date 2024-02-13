import wandb
import argparse
import logging
from Augmentations.augment import perform_augmentations
from Augmentations.utils import merge_augmented_instances
from pathlib import Path
import sys


def run(args):
    log_filename = 'Augmentations.log'

    # setting logging
    logging.basicConfig(
        filename=log_filename,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w')
    logging.root.setLevel(logging.INFO)


    run = wandb.init(job_type="Preprocessing_Augmentations")
    run.config.update(args)

    logging.info(
        f"Performing {Path(args.augmentation_file).stem} augmentations...")
    perform_augmentations(
        Path(
            args.data_dir), Path(
            args.augmentation_file), args.class_list.split(','))
    logging.info(f"Augmentations successfully finished.")

    if args.merge_augmented_images.upper() == "TRUE":
        logging.info(f"Merging all augmentations applied.")
        merge_augmented_instances(
            Path(
            args.image_directory), Path(
            args.labels_directory))
        logging.info(f"Merging successfully done.")

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )
    artifact.add_file(str(args.augmentation_file))
    artifact.add_file(log_filename)
    run.log_artifact(artifact)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Module to perform Preprocessing related to Object Detection Datasets')

    parser.add_argument(
        '--class_list',
        type=str,
        help='List of classes to detect. E.g: --class_list Class1,Class2,...,ClassN',
        required=True)


    parser.add_argument('--data_dir', type=str,
                        help='Directory to fetch images.', required=True
                        )


    parser.add_argument(
        '--augmentation_file',
        type=str,
        help='YAML file with albumentations style',
        required=True)

    parser.add_argument(
        '--merge_augmented_images',
        type=str,
        help='True if want to merge automatically augmented instances with images and labels dir',
        required=True)

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Name of the artifact",
        required=True)

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type of the artifact",
        required=True)

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="A brief description of this artifact",
        required=True)

    args = parser.parse_args()
    run(args)
