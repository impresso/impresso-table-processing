import os
import sys
import json
import glob
import click
import torch

from pathlib import Path

sys.path.append("/home/amvernet/dhSegment-torch/")
from dh_segment_torch.training import Trainer
from dh_segment_torch.data.annotation import ColorLabels
from dh_segment_torch.models import Model


@click.command()
@click.option('--run-path', '-r',
              help="""Path to a folder with a config file to run a dhSegment
              experiment. It must be under
              <experiment_name>/dhSegment_models/<run_name> and the necessary
              files to run an experiment should be in the <experiment_name>
              folder.""")
@click.option('--model-path', '-m', default=None,
            help="""Path to a model from which to resume training.
            Default: None""")
@click.option('--device', '-d', default='cuda:0',
              help='GPU on which to run the training. Default: cuda:0')
def train_dhSegment(run_path, model_path, device):

    run_path = os.path.abspath(run_path)
    run_name = os.path.basename(run_path)
    experiment_path = os.path.abspath(os.path.join(run_path, "../.."))
    experiment_name = os.path.basename(experiment_path)

    with open(os.path.join(run_path, "config.json"), "r") as f:
        config = json.load(f)

    print(f"The experiment {experiment_name} will be saved under {os.path.abspath(run_path)}. The following configuration is used: {config}")
    dataset_name = config['dataset_name']
    params = {
            "color_labels": {"label_json_file": os.path.join(experiment_path, "color_labels.json")},
            "train_dataset": {
                "type": "image_csv", # Image csv dataset
                "csv_filename": os.path.join(experiment_path, f"{dataset_name}_train.csv"),
                "repeat_dataset": config["train_dataset"]["repeat_dataset"],
                "compose": {"transforms": [{"type": "rotate", "limit": 1},
                                           {"type": "random_scale", "scale_limit": (0.9, 1.1), "p": 1.0},
                                           config['fixed_size_resize']
                                           ]}
            },
            "val_dataset": {
                "type": "image_csv", # Validation dataset
                "csv_filename": os.path.join(experiment_path, f"{dataset_name}_val.csv"),
                "repeat_dataset": config["val_dataset"]["repeat_dataset"],
                "compose": {"transforms": [config['fixed_size_resize']]}
            },
            "model": { # Model definition, original dhSegment
                "encoder": {"type": config["encoder"]["type"],
                            "pretrained": config['pretrained'],
                            "normalization": config['encoder']['normalization']
                           },
                "decoder": {
                    "decoder_channels": [512, 256, 128, 64, 32],
                    "max_channels": 512
                }
            },
            "regularizer": config["regularizer"],
            "metrics": [['miou', 'iou'], ['iou', {"type": 'iou', "average": None}]], # Metrics to compute
            "optimizer": config['optimizer'], # Learning rate # by default Adam optimizer # weight decay acts as L2 regularizer
            "lr_scheduler": config['lr_scheduler'], # Exponential decreasing learning rate
            "val_metric": config['val_metric'], # Metric to observe to consider a model better than another, the + indicates that we want to maximize
            "early_stopping": config['early_stopping'], # Number of validation steps without increase to tolerate, stops if reached
            "model_out_dir": run_path, # Path to model output
            "num_epochs": config['num_epochs'], # Number of epochs for training
            "evaluate_every_epoch": config['evaluate_every_epoch'], # Number of epochs between each validation of the model
            "batch_size": config['batch_size'], # Batch size (to be changed if the allocated GPU has little memory)
            "num_data_workers": 4,
            "num_accumulation_steps": 1,
            "track_train_metrics": False,
            "loggers": [
                {"type": 'wandb',
                 "dir": run_path,
                 "log_every": 4,
                 "log_images_every": 60,
                 "exp_name": experiment_name + "_dhSegment",
                 "name": run_name,
                 "config": config
                 }
               ],
            "device": device,
            "train_checkpoint": {
                "type": "iteration",
                "every_n_iterations": 1000, # Make a checkpoint every 100 iterations
                "permanent_every_n_iterations": 2000, # Make a permanent checkpoint every 2000 iterations
                "checkpoints_to_keep": 2, # Number of checkpoints to keep, in this case only the last 2 checkpoints are kept
                # checkpoint_dir and prefix are set by the trainer, the first one is where to save the checkpoint, the second is a string to prepend to the checkpoint name
            },
          "val_checkpoint": {
              "checkpoints_to_keep": 4
          }
    }

    trainer = Trainer.from_params(params)

    if model_path is not None:
        model_state_dict = torch.load(model_path)
        trainer.model.load_state_dict(model_state_dict, strict=False)

    trainer.train()

    print("Training over.")


if __name__ == '__main__':
    train_dhSegment()
