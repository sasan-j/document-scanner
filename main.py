import logging
import pathlib
import shutil
from subprocess import call
from typing import List
from pprint import pprint


import cv2
import pandas as pd
import typer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import wandb

import dataprocessor
from dataprocessor import DatasetType
from experiment import Experiment
import utils
from model import ScannerModel

app = typer.Typer()
logger = logging.getLogger("zebel-scanner")


@app.command()
def process_smartdoc(
    src: pathlib.Path = typer.Argument(..., help="The path to smartdoc dataset"),
    dst: pathlib.Path = typer.Argument(..., help="The path to the processed dataset"),
):
    """This command converts smartdoc videos into images and labels."""
    if dst.exists() and dst.is_file():
        typer.echo("The destination path is a file.")
        typer.Exit(code=1)
    if not dst.exists():
        dst.mkdir(parents=True)
    for dir in tqdm(src.iterdir(), desc="Processing backgrounds"):
        if dir.is_dir() and dir.name.startswith("background"):
            for video in tqdm(dir.iterdir(), desc="Processing videos"):
                bg_dst_dir = dst / dir.name
                bg_dst_dir.mkdir(parents=True, exist_ok=True)
                if video.is_file() and video.suffix == ".avi":
                    tqdm.write(f"Processing {video}")
                    vid_dst_dir = bg_dst_dir / video.stem
                    vid_dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(
                        video.parent / f"{video.stem}.gt.xml",
                        vid_dst_dir / f"{video.stem}.gt",
                    )
                    call(
                        [
                            "ffmpeg",
                            "-i",
                            video,
                            vid_dst_dir / "%03d.jpg",
                            "-loglevel",
                            "panic",
                        ]
                    )


@app.command()
def document_data_generator(
    src: pathlib.Path = typer.Argument(..., help="The path to the processed dataset"),
    dst: pathlib.Path = typer.Argument(
        ..., help="The path to store the generated document data"
    ),
    dataset: DatasetType = typer.Option(
        DatasetType.SMARTDOC, help="The dataset to use"
    ),
):
    """This command generates the data to be used for training the document model."""
    if dst.exists() and dst.is_file():
        typer.echo("The destination path is a file.")
        typer.Exit(code=1)
    if not dst.exists():
        dst.mkdir(parents=True)
        (dst / "images").mkdir(parents=True, exist_ok=True)

    if dataset == DatasetType.SMARTDOC:
        dataset_loader = dataprocessor.SmartDocDirectories(str(src))
    # elif dataset == Dataset.SELFCOLLECTED:
    #     dataset_test = dataprocessor.dataset.SelfCollectedDataset(src)
    else:
        typer.echo("Unknown dataset.")
        typer.Exit(code=1)
        return

    # with open(dst / "gt.csv", "a") as csvfile:
    #     writer = csv.writer(
    #         csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    #     )
    data = []
    counter = 0
    for data_elem in tqdm(dataset_loader.myData):
        img_path = data_elem[0]
        # target = data_elem[1].reshape((4, 2))
        target = data_elem[1]
        img = cv2.imread(img_path)
        counter += 1
        f_name = f"{str(counter).zfill(8)}.jpg"
        cv2.imwrite(str(dst / "images" / f_name), img)
        data.append([f_name, target])
        # writer.writerow((f_name, list(target)))
    df = pd.DataFrame(data, columns=["image", "annotation"])
    df.to_feather(str(dst / "annotation.feather"))


@app.command()
def train(
    batch_size: int = typer.Option(default=10, help="The batch size"),
    lr: float = typer.Option(default=0.0001, help="The learning rate"),
    epochs: int = typer.Option(default=3, help="The number of epochs"),
    schedule: List[int] = typer.Option(
        default=[10, 20, 30], help="The schedule for decreasing the learning rate"
    ),
    gammas: List[int] = typer.Option(
        default=[0.1, 0.1, 0.1],
        help="LR is multiplied by gamma[k] on schedule[k], number of gammas should be equal to schedule",
    ),
    arch: str = typer.Option(
        default="unet",
        help="The architecture to use, one of Unet,UnetPlusPlus,MAnet,Linknet,FPN,PSPNet,DeepLabV3,DeepLabV3Plus,PAN",
    ),
    encoder: str = typer.Option(
        default="mobilenet_v2",
        help="The endcoder to use, one of mobilenet_v2,resnet18,resnet34,resnet50,resnet101,resnet152",
    ),
    encoder_weights: str = typer.Option(
        default="imagenet", help="The weights to use for the encoder"
    ),
    width: int = typer.Option(default=224, help="The width of the input image"),
    height: int = typer.Option(default=224, help="The height of the input image"),
    momentum: float = typer.Option(default=0.9, help="The momentum for SGD"),
    cuda: bool = typer.Option(default=True, help="Use CUDA"),
    debug: bool = typer.Option(default=False, help="Use debug mode"),
    seed: int = typer.Option(default=42, help="The seed for the random generator"),
    log_interval: int = typer.Option(
        default=10, help="The interval for logging - batches"
    ),
    name: str = typer.Option(default="untitled", help="Name of the experiment"),
    out_dir: pathlib.Path = typer.Option(
        default="../experiments/",
        help='Directory to store the results; a new folder "DDMMYYYY" will be created '
        "in the specified directory to save the results.",
    ),
    decay: float = typer.Option(default=0.00001, help="The decay for the optimizer"),
    train_dir: pathlib.Path = typer.Option(
        default="../train/", help="The directory to store the training data"
    ),
    valid_dir: pathlib.Path = typer.Option(
        default="../valid/", help="The directory to store the validation data"
    ),
):

    config = {
        "seed": seed,
        "batch_size": batch_size,
        "architecture": arch,
        "encoder": encoder,
        "encoder_weights": encoder_weights,
        "epochs": epochs,
        "width": width,
        "height": height,
        "auto_lr_find": True,
    }

    wandb.init(project="zebel-scanner", config=config)

    # Define an experiment.
    my_experiment = Experiment(name, locals(), out_dir)
    # tb_writer = SummaryWriter(log_dir=str(my_experiment.path / "tensorboard"))

    # Add logging support
    logger = utils.setup_logger(my_experiment.path, level="info")

    cuda = cuda and torch.cuda.is_available()

    # Get the right dataset based on model_type
    dataset_train = dataprocessor.SmartDocDataset(
        directory=train_dir, height=height, width=width
    )
    dataset_val = dataprocessor.SmartDocDataset(
        directory=valid_dir, height=height, width=width
    )

    # Fix the seed.
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    kwargs = {"num_workers": 4, "pin_memory": True} if cuda else {}

    train_dataloader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, **kwargs
    )
    valid_dataloader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, **kwargs
    )

    model = ScannerModel(arch, encoder, encoder_weights, 3, 1)
    tb_logger = TensorBoardLogger(my_experiment.path, name=name, sub_dir="tb")
    wandb_logger = WandbLogger(name=name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=my_experiment.path / "checkpoints_top",
        save_top_k=2,
        monitor="valid_dataset_iou",
    )
    # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        default_root_dir=str(my_experiment.path),
        gpus=1,
        # strategy="ddp",
        strategy=DDPPlugin(find_unused_parameters=False),
        precision=16,
        max_epochs=epochs,
        # max_time="00:12:00:00",
        logger=[tb_logger, wandb_logger],
        callbacks=[checkpoint_callback, lr_monitor],
        flush_logs_every_n_steps=100,
        auto_lr_find=True,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

    torch.save(
        model.state_dict(),
        my_experiment.path / f"{arch}-{encoder}-{encoder_weights}.pb",
    )


if __name__ == "__main__":
    app()
