import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
import mlflow

_steps = [
    "DATAGATHER",
    "EDA",
    "EDA_VISUALIZER",
    "PREPROCESSING_AUGMENTATIONS",
    "MODELING_MAIN",
    "MODELING_MODELSELECTION",
    "MODELING_INFERENCE",
    "DEPLOY"
]

@hydra.main(config_name='config')
def run(config: DictConfig):
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    working_directory = Path.cwd().parents[2]

    if "PREPROCESSING_AUGMENTATIONS" in active_steps:
        # Download file and load in W&B
        _ = mlflow.run(
            str(Path(hydra.utils.get_original_cwd()).joinpath("Preprocessing")),
            "main_augmentations",
            parameters={
                'class_list': config["PREPROCESSING"]["class_list"],
                'data_dir': str(working_directory.joinpath(Path(config["PREPROCESSING"]["AUGMENTATIONS"]["data_dir"]))),
                'augmentation_file': str(working_directory.joinpath('Preprocessing','Augmentations','transformations',config["PREPROCESSING"]["AUGMENTATIONS"]["augmentations_file"])),
                'merge_augmented_images': config["PREPROCESSING"]["AUGMENTATIONS"]["merge_augmented_images"],
                "artifact_name": config["PREPROCESSING"]["AUGMENTATIONS"]["artifact_name"],
                "artifact_type": config["PREPROCESSING"]["AUGMENTATIONS"]["artifact_type"],
                "artifact_description": config["PREPROCESSING"]["AUGMENTATIONS"]["artifact_description"]
            },
            env_manager='local'
        )

if __name__ == '__main__':
    run()