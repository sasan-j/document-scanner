from enum import Enum
import pathlib
import shutil
from subprocess import call
from typing import List

import cv2
import numpy as np
import typer
from tqdm import tqdm
import torch

import dataprocessor
from dataprocessor import DatasetType
import utils
from experiment import Experiment

app = typer.Typer()


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


class Model(str, Enum):
    RESNET32 = "resnet32"
    RESNET20 = "resnet20"
    DENSENET = "densenet"


class Loader(str, Enum):
    RAM = "ram"
    DISK = "disk"


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

    if dataset == DatasetType.SMARTDOC:
        dataset_loader = dataprocessor.SmartDocDirectories(str(src))
    # elif dataset == Dataset.SELFCOLLECTED:
    #     dataset_test = dataprocessor.dataset.SelfCollectedDataset(src)
    else:
        typer.echo("Unknown dataset.")
        typer.Exit(code=1)
        return

    import csv

    with open(dst / "gt.csv", "a") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        counter = 0
        for data_elem in tqdm(dataset_loader.myData):
            img_path = data_elem[0]
            target = data_elem[1].reshape((4, 2))
            img = cv2.imread(img_path)

            if dataset == DatasetType.SELFCOLLECTED:
                target = target / (img.shape[1], img.shape[0])
                target = target * (1920, 1920)
                img = cv2.resize(img, (1920, 1920))

            corner_cords = target

            for angle in range(0, 271, 90):
                img_rotate, gt_rotate = utils.rotate(img, corner_cords, angle)
                for random_crop in range(0, 16):
                    counter += 1
                    f_name = f"{str(counter).zfill(8)}.jpg"

                    img_crop, gt_crop = utils.random_crop(img_rotate, gt_rotate)
                    img_crop = cv2.resize(img_crop, (64, 64))
                    gt_crop = np.array(gt_crop)

                    cv2.imwrite(str(dst / f_name), img_crop)
                    writer.writerow((f_name, tuple(list(gt_crop))))


@app.command()
def corner_data_generator(
    src: pathlib.Path = typer.Argument(..., help="The path to the processed dataset"),
    dst: pathlib.Path = typer.Argument(
        ..., help="The path to store the generated corner data"
    ),
    dataset: DatasetType = typer.Option(
        DatasetType.SMARTDOC, help="The dataset to use"
    ),
):
    """This command generates the data to be used for training the corner model."""
    if dst.exists() and dst.is_file():
        typer.echo("The destination path is a file.")
        typer.Exit(code=1)
    if not dst.exists():
        dst.mkdir(parents=True)

    if dataset == DatasetType.SMARTDOC:
        dataset_loader = dataprocessor.SmartDocDirectories(str(src))
    # elif dataset == Dataset.SELFCOLLECTED:
    #     dataset_test = dataprocessor.dataset.SelfCollectedDataset(src)
    else:
        typer.echo("Unknown dataset.")
        typer.Exit(code=1)
        return

    import csv

    with open(dst / "gt.csv", "a") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        counter = 0
        for data_elem in tqdm(dataset_loader.myData):
            img_path = data_elem[0]
            target = data_elem[1].reshape((4, 2))
            img = cv2.imread(img_path)

            if dataset == DatasetType.SELFCOLLECTED:
                target = target / (img.shape[1], img.shape[0])
                target = target * (1920, 1920)
                img = cv2.resize(img, (1920, 1920))

            corner_cords = target

            for angle in range(0, 1, 90):
                img_rotate, gt_rotate = utils.rotate(img, corner_cords, angle)
                for random_crop in range(0, 1):
                    img_list, gt_list = utils.get_corners(img_rotate, gt_rotate)
                    for a in range(0, 4):
                        counter += 1
                        f_name = f"{str(counter).zfill(8)}.jpg"
                        gt_store = list(np.array(gt_list[a]) / (300, 300))
                        img_store = cv2.resize(img_list[a], (64, 64))
                        cv2.imwrite(
                            str(dst / f_name),
                            img_store,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                        )
                        writer.writerow((f_name, tuple(gt_store)))


@app.command()
def train(
    batch_size: int = typer.Option(default=32, help="The batch size"),
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
    pretrain: bool = typer.Option(default=False, help="Use pretrained weights"),
    debug: bool = typer.Option(default=False, help="Use debug mode"),
    seed: int = typer.Option(default=42, help="The seed for the random generator"),
    log_interval: int = typer.Option(
        default=10, help="The interval for logging - batches"
    ),
    model: Model = typer.Option(default=Model.RESNET32, help="The model type"),
    name: str = typer.Option(default="", help="Name of the experiment"),
    out_dir: pathlib.Path = typer.Option(
        default="../",
        help='Directory to store the results; a new folder "DDMMYYYY" will be created '
        "in the specified directory to save the results.",
    ),
    decay: float = typer.Option(default=0.00001, help="The decay for the optimizer"),
    model_type: str = typer.Option(
        default="document", help="Which model to train, document or corner"
    ),
    loader: Loader = typer.Option(
        default=Loader.DISK,
        help="Loader to load data; hdd for reading from the hdd and ram for loading all data in the memory",
    ),
    train_dir: pathlib.Path = typer.Option(
        default="../train/", help="The directory to store the training data"
    ),
    valid_dir: pathlib.Path = typer.Option(
        default="../valid/", help="The directory to store the validation data"
    ),
):

    # Define an experiment.
    my_experiment = Experiment(name, locals(), out_dir)

    # Add logging support
    # logger = utils.setup_logger(my_experiment.path)

    cuda = cuda and torch.cuda.is_available()

    # Get the right dataset based on model_type
    dataset = dataprocessor.DatasetFactory.get_dataset(train_dir, model_type)
    dataset_val = dataprocessor.DatasetFactory.get_dataset(valid_dir, model_type)

    # Fix the seed.
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    train_dataset_loader = dataprocessor.LoaderFactory.get_loader(
        loader, dataset.myData, transform=dataset.train_transform, cuda=cuda
    )
    # Loader used for training data
    val_dataset_loader = dataprocessor.LoaderFactory.get_loader(
        loader, dataset_val.myData, transform=dataset.test_transform, cuda=cuda
    )
    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}

    # # Iterator to iterate over training data.
    # train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
    #                                             batch_size=args.batch_size, shuffle=True, **kwargs)
    # # Iterator to iterate over training data.
    # val_iterator = torch.utils.data.DataLoader(val_dataset_loader,
    #                                         batch_size=args.batch_size, shuffle=True, **kwargs)

    # # Get the required model
    # myModel = model.ModelFactory.get_model(args.model_type, args.dataset)
    # if cuda:
    #     myModel.cuda()

    # # Should I pretrain the model on CIFAR?
    # if args.pretrain:
    #     trainset = dataprocessor.DatasetFactory.get_dataset(None, "CIFAR")
    #     train_iterator_cifar = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    #     # Define the optimizer used in the experiment
    #     cifar_optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
    #                                     weight_decay=args.decay, nesterov=True)

    #     # Trainer object used for training
    #     cifar_trainer = trainer.CIFARTrainer(train_iterator_cifar, myModel, args.cuda, cifar_optimizer)

    #     for epoch in range(0, 70):
    #         logger.info("Epoch : %d", epoch)
    #         cifar_trainer.update_lr(epoch, [30, 45, 60], args.gammas)
    #         cifar_trainer.train(epoch)

    #     # Freeze the model
    #     counter = 0
    #     for name, param in myModel.named_parameters():
    #         # Getting the length of total layers so I can freeze x% of layers
    #         gen_len = sum(1 for _ in myModel.parameters())
    #         if counter < int(gen_len * 0.5):
    #             param.requires_grad = False
    #             logger.warning(name)
    #         else:
    #             logger.info(name)
    #         counter += 1

    # # Define the optimizer used in the experiment
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, myModel.parameters()), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.decay, nesterov=True)

    # # Trainer object used for training
    # my_trainer = trainer.Trainer(train_iterator, myModel, args.cuda, optimizer)

    # # Evaluator
    # my_eval = trainer.EvaluatorFactory.get_evaluator("rmse", args.cuda)
    # # Running epochs_class epochs
    # for epoch in range(0, args.epochs):
    #     logger.info("Epoch : %d", epoch)
    #     my_trainer.update_lr(epoch, args.schedule, args.gammas)
    #     my_trainer.train(epoch)
    #     my_eval.evaluate(my_trainer.model, val_iterator)

    # torch.save(myModel.state_dict(), my_experiment.path + args.dataset + "_" + args.model_type+ ".pb")
    # my_experiment.store_json()


if __name__ == "__main__":
    app()
