from enum import Enum
import pathlib
import shutil
from subprocess import call

import cv2
import numpy as np
import typer
from tqdm import tqdm

import dataprocessor
import utils

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


class Dataset(str, Enum):
    SMARTDOC = "smartdoc"
    SELFCOLLECTED = "selfcollected"


@app.command()
def document_data_generator(
    src: pathlib.Path = typer.Argument(..., help="The path to the processed dataset"),
    dst: pathlib.Path = typer.Argument(
        ..., help="The path to store the generated document data"
    ),
    dataset: Dataset = typer.Option(Dataset.SMARTDOC, help="The dataset to use"),
):
    """This command generates the data to be used for training the document model."""
    if dst.exists() and dst.is_file():
        typer.echo("The destination path is a file.")
        typer.Exit(code=1)
    if not dst.exists():
        dst.mkdir(parents=True)

    if dataset == Dataset.SMARTDOC:
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

            if dataset == Dataset.SELFCOLLECTED:
                target = target / (img.shape[1], img.shape[0])
                target = target * (1920, 1920)
                img = cv2.resize(img, (1920, 1920))

            corner_cords = target

            for angle in range(0, 270, 90):
                img_rotate, gt_rotate = utils.rotate(img, corner_cords, angle)
                for random_crop in range(0, 16):
                    counter += 1
                    f_name = f"{str(counter).zfill(8)}.jpg"

                    img_crop, gt_crop = utils.random_crop(img_rotate, gt_rotate)
                    img_crop = cv2.resize(img_crop, (64, 64))
                    gt_crop = np.array(gt_crop)

                    cv2.imwrite(str(dst / f_name), img_crop)
                    writer.writerow((f_name, tuple(list(gt_crop))))


if __name__ == "__main__":
    app()
