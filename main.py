import logging
import pathlib
import shutil
from subprocess import call
from typing import List


import cv2
import pandas as pd
import typer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


import dataprocessor
from dataprocessor import DatasetType
from experiment import Experiment
from models import Model
import trainer
import utils
from unet import UNet

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
    lr: float = typer.Option(default=0.001, help="The learning rate"),
    epochs: int = typer.Option(default=10, help="The number of epochs"),
    schedule: List[int] = typer.Option(
        default=[10, 20, 30], help="The schedule for decreasing the learning rate"
    ),
    gammas: List[int] = typer.Option(
        default=[0.1, 0.1, 0.1],
        help="LR is multiplied by gamma[k] on schedule[k], number of gammas should be equal to schedule",
    ),
    momentum: float = typer.Option(default=0.9, help="The momentum for SGD"),
    cuda: bool = typer.Option(default=True, help="Use CUDA"),
    debug: bool = typer.Option(default=False, help="Use debug mode"),
    seed: int = typer.Option(default=42, help="The seed for the random generator"),
    log_interval: int = typer.Option(
        default=10, help="The interval for logging - batches"
    ),
    model: Model = typer.Option(default=Model.RESNET20, help="The model type"),
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

    # Define an experiment.
    my_experiment = Experiment(name, locals(), out_dir)
    tb_writer = SummaryWriter(log_dir=str(my_experiment.path / "tensorboard"))

    # Add logging support
    logger = utils.setup_logger(my_experiment.path, level="info")

    cuda = cuda and torch.cuda.is_available()

    # Get the right dataset based on model_type
    dataset_train = dataprocessor.SmartDocDataset(directory=train_dir)
    dataset_val = dataprocessor.SmartDocDataset(directory=valid_dir)

    # Fix the seed.
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    kwargs = {"num_workers": 0, "pin_memory": True} if cuda else {}

    train_dataloader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, **kwargs
    )
    valid_dataloader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=True, **kwargs
    )

    # # Get the required model
    myModel = UNet(retain_dim=True)
    myModel = myModel.type(torch.float16)

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)

    # Applying it to our net
    myModel.apply(init_weights)

    if cuda:
        myModel.cuda()

    # Define the optimizer used in the experiment
    optimizer = torch.optim.SGD(
        # filter(lambda p: p.requires_grad, myModel.parameters()),
        myModel.parameters(),
        lr,
        momentum=momentum,
        weight_decay=decay,
        nesterov=True,
    )

    # Trainer object used for training
    my_trainer = trainer.Trainer(
        train_dataloader, valid_dataloader, myModel, cuda, optimizer, tb_writer
    )

    # Running epochs_class epochs
    for epoch in range(0, epochs):
        logger.info("Epoch : %d", epoch)
        my_trainer.update_lr(epoch, schedule, gammas)
        my_trainer.train(epoch)
        tb_writer.flush()

    torch.save(myModel.state_dict(), my_experiment.path / f"{model}.pb")
    my_experiment.store_json()


if __name__ == "__main__":
    app()
