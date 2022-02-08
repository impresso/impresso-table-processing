import os
import sys
import cv2
import glob
import click
import numpy as np

from tqdm import tqdm
from PIL import Image

sys.path.append("/home/amvernet/dhSegment-torch/")
from dh_segment_torch.config import Params
from dh_segment_torch.inference import PredictProcess
from dh_segment_torch.data.annotation import ColorLabels


@click.command()
@click.option('--experiment-path', '-e',
              help="""Path where an experiment for dhSegment has been set up.""")
@click.option('--model-path', '-m',
              help="""Path to the model to be used for predictions.""")
@click.option('--dataset-name', '-d',
              help="""The name of the dataset to use.
              Default: Test set of the dataset associated with the experiment name.""")
@click.option('--device', default='cuda:0',
              help='GPU on which to run the inference. Default: cuda:0.')
def generate_predictions(experiment_path, model_path, dataset_name, device):
    
    color_labels = ColorLabels.from_labels_json_file(os.path.join(experiment_path, "color_labels.json"))
    tagset = color_labels.labels

    dataset_params = {
        "type": "csv",
        "csv_path": os.path.join(experiment_path, f"{dataset_name}.csv"),
        "pre_processing": {"transforms": [{"type": "fixed_resize", "height": 1333, "width": 800}]}
    }

    model_params = {
        "model": {
                "encoder": "resnet50",
                "decoder": {"decoder_channels": [512, 256, 128, 64, 32], "max_channels": 512}
            },
        "num_classes": len(color_labels.labels),
        "model_state_dict": model_path,
        "device": device
    }

    batch_size = 2
    process_params = Params({
        'data': dataset_params,
        'model': model_params,
        'batch_size': batch_size,
        'num_workers': 32,
        'add_path': True
    })

    # generate predictions
    run_name = os.path.basename(os.path.dirname(model_path))
    predictions_path = os.path.join(experiment_path, "dhSegment_predictions", run_name, dataset_name, "npy")
    os.makedirs(predictions_path, exist_ok=True)

    predict_annots = PredictProcess.from_params(process_params)
    print(f"Generating the predictions {batch_size} by {batch_size} under {predictions_path} ...")
    predict_annots.process_to_probas_files(predictions_path)


if __name__ == '__main__':
    generate_predictions()
