import argparse
import wandb
import logging
from .Train.Train_Tensorflow.train import train
from pathlib import Path

def run(args):
    log_filename = 'Training.log'

    # setting logging
    logging.basicConfig(
        filename=log_filename,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w')
    logging.root.setLevel(logging.INFO)

    run = wandb.init(job_type="Training")
    run.config.update(args)

    logging.info("Creating training and validation datasets...")

    try:
        assert Path(args.input_dir).exists()
    except AssertionError as err:
        logging.error(f"{Path(args.input_dir)} not found.")
        raise err



    if args.framework == 'tensorflow':
        model =
        train()





    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )

    artifact.add_file(log_filename)
    artifact.add_dir(str(Path(args.output_dir)))
    run.log_artifact(artifact)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Module to perform model training')

    parser.add_argument(
        '--input_dir',
        type=str,
        help='Folder where images are stored',
        required=True)

    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for results',
        required=True)

    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model. Must be the name of the file under model_repo (without file extension)",
        required=True)

    parser.add_argument(
        "--train_config",
        type=str,
        help="Training configuration dictionary. Must be given a string formatted with json.",
        required=True)

    parser.add_argument('--framework', type=str, help='Run training through either pytorch or tensorflow',
                        choices=['pytorch', 'tensorflow'])

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