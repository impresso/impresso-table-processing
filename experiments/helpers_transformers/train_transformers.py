import os
import json
import torch
import click
import wandb
import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

from PIL import Image
from datasets_transformers import *
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LayoutXLMTokenizer, LayoutLMv2ForSequenceClassification
from transformers import TrainingArguments, Trainer, LayoutLMv2FeatureExtractor, LayoutLMv2Processor, LayoutLMv2Model, LayoutXLMProcessor
from sklearn.preprocessing import LabelEncoder


@click.command()
@click.option('--run-path', '-r',
              help="""Path to a folder with a config file to run a dhSegment
              experiment. It must be under
              <experiment_name>/transformer_models/<run_name> and the necessary
              files to run an experiment should be in the <experiment_name>
              folder.""")
@click.option('--metadata-path', '-m',
              default="/scratch/students/amvernet/datasets/metadata/",
              help="""Path to folder with JSON files containing the original and resized
              height and width of the pictures to be used.\n
              Default: /scratch/students/amvernet/datasets/metadata""")
@click.option('--images-path', '-i',
              default="/scratch/students/amvernet/datasets/images/",
              help="""Path to the folder containing the images to be used.
              Default: /scratch/students/amvernet/datasets/images/""")
@click.option('--device', '-d', default='cuda:0',
              help="""GPU on which to run the training.
              Default: cuda:0.""")
@click.option('--ocr', '-o', is_flag=True, default=False,
              help="""Whether or not to apply OCR when using LayoutXLM.
              Default: False.""")
def train_transformers(run_path, metadata_path, images_path, device, ocr):

    run_path = os.path.abspath(run_path)
    run_name = os.path.basename(run_path)
    experiment_path = os.path.abspath(os.path.join(run_path, "../.."))
    experiment_name = os.path.basename(experiment_path)

    with open(os.path.join(run_path, "config.json"), "r") as f:
        config = json.load(f)

    metadata = {}
    onlyfiles = [f for f in listdir(metadata_path) if isfile(join(metadata_path, f))]
    for file in onlyfiles:
        if file.endswith(".json"):
            with (open(join(metadata_path, file), "r")) as f:
                metadata.update(json.loads(f.read()))

    model_name = config['model_name']

    valid_model_names = {"microsoft/layoutlm-base-uncased", "microsoft/layoutxlm-base", "xlm-roberta-base"}
    if model_name not in valid_model_names:
        print(f"Only {valid_model_names} are valid models.")
        return

    model_name_to_display = model_name
    if '/' in model_name:
        model_name_to_display = model_name.split('/')[1]

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["WANDB_PROJECT"] = experiment_name + f"_{model_name_to_display}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[-1])

    df_train = pd.read_parquet(config['train_set_path'])
    df_val = pd.read_parquet(config['val_set_path'])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=len(df_train['tag'].unique()))

    LE = LabelEncoder()
    LE.fit(df_train['tag'])

    print("The datasets are being prepared...")
    if model_name == "microsoft/layoutlm-base-uncased":
        train_dataset = LayoutLMDataset(df_train, metadata, tokenizer, LE)
        val_dataset = LayoutLMDataset(df_val, metadata, tokenizer, LE)
    elif model_name == "microsoft/layoutxlm-base":
        feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=ocr)
        tokenizer = LayoutXLMTokenizer.from_pretrained('microsoft/layoutxlm-base')
        processor = LayoutXLMProcessor(feature_extractor, tokenizer)
        train_dataset = LayoutXLMDataset(df_train, metadata, processor, LE, images_path, ocr)
        val_dataset = LayoutXLMDataset(df_val, metadata, processor, LE, images_path, ocr)
    elif model_name == "xlm-roberta-base":
        train_dataset = BERTDataset(df_train, tokenizer, LE)
        val_dataset = BERTDataset(df_val, tokenizer, LE)

    training_args = TrainingArguments(output_dir=run_path,
                                      evaluation_strategy=config["evaluation_strategy"],
                                      save_strategy=config["evaluation_strategy"],
                                      num_train_epochs=config['num_train_epochs'],
                                      run_name=run_name,
                                      per_device_train_batch_size=config["batch_size"],
                                      per_device_eval_batch_size=config["batch_size"],
                                      load_best_model_at_end=True,
                                      metric_for_best_model=config["metric_for_best_model"],
                                      report_to="wandb",
                                      save_total_limit=5)

    def compute_metrics(eval_pred):
        metric = load_metric(config["metric"])
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    wandb.init(dir=run_path, name=run_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("Training starts...")
    trainer.train()
    print("Training over.")
    model.save_pretrained(os.path.join(run_path, f"best_model_{model_name_to_display}"))


if __name__ == '__main__':
    train_transformers()
