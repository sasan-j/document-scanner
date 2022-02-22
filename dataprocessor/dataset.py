import csv
from enum import Enum
import logging
import os
import pathlib
from typing import List, Union

import cv2
import numpy as np
import pandas as pd
import transforms as T
import xml.etree.ElementTree as ET

import utils


logger = logging.getLogger("zebel-scanner")


class DatasetType(str, Enum):
    SMARTDOC = "smartdoc"
    SELFCOLLECTED = "selfcollected"


class Dataset:
    """
    Base class to represent a Dataset
    """

    def __init__(self, name):
        self.name = name
        self.data = []
        self.labels = []


class SmartDocDirectories(Dataset):
    """
    Class to parse out the data from the SmartDoc dataset
    """

    def get_ground_truth(self, image: pathlib.Path, list_gt) -> np.ndarray:
        list_of_points = {}

        for point in list_gt[int(image.stem) - 1].iter("point"):
            myDict = point.attrib
            # TODO: should we throw away the decimal part?
            list_of_points[myDict["name"]] = (
                int(float(myDict["x"])),
                int(float(myDict["y"])),
            )

        ground_truth = np.asarray(
            (
                list_of_points["tl"],
                list_of_points["tr"],
                list_of_points["br"],
                list_of_points["bl"],
            )
        )
        return utils.sort_gt(ground_truth)

    def get_ground_truth_from_xml(self, gt_path: pathlib.Path) -> List:
        tree = ET.parse(gt_path)
        root = tree.getroot()
        return [x for x in root.iter("frame")]

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []

        if not isinstance(directory, pathlib.Path):
            directory = pathlib.Path(directory)

        for bg_dir in directory.iterdir():
            if bg_dir.is_dir():
                for images_dir in bg_dir.iterdir():
                    if images_dir.is_dir():
                        list_gt = self.get_ground_truth_from_xml(
                            images_dir / f"{images_dir.name}.gt"
                        )
                        for image in images_dir.iterdir():
                            if image.is_file() and image.name.endswith(".jpg"):
                                self.data.append(str(image))
                                self.labels.append(
                                    self.get_ground_truth(image, list_gt)
                                )

        self.labels = np.array(self.labels)
        self.labels = np.reshape(self.labels, (-1, 8))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = []
        for a in range(len(self.data)):
            self.myData.append([self.data[a], self.labels[a]])


class SmartDoc(Dataset):
    """
    Class to load the data from the SmartDoc dataset
    """

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []
        if type(directory) is not list:
            self.dirs = [directory]
        for d in self.dirs:
            logger.info("Pass train/test data paths here")
            self.classes_list = {}

            file_names = []
            print(d, "gt.csv")
            with open(os.path.join(d, "gt.csv"), "r") as csvfile:
                spamreader = csv.reader(
                    csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
                )
                import ast

                for row in spamreader:
                    file_names.append(row[0])
                    self.data.append(os.path.join(d, row[0]))
                    test = row[1].replace("array", "")
                    self.labels.append((ast.literal_eval(test)))
        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 8))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = [self.data, self.labels]


class SmartDocCorner(Dataset):
    """
    Class to load the data from the SmartDoc corner dataset
    """

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []
        if type(directory) is not list:
            self.dirs = [directory]

        for d in self.dirs:
            logger.info("Pass train/test data paths here")
            self.classes_list = {}
            file_names = []
            with open(d / "gt.csv", "r") as csvfile:
                spamreader = csv.reader(
                    csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
                )
                import ast

                for row in spamreader:
                    file_names.append(row[0])
                    self.data.append(d / row[0])
                    test = row[1].replace("array", "")
                    self.labels.append((ast.literal_eval(test)))
        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 2))
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

        self.myData = [self.data, self.labels]


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(directory, type="document") -> Union[SmartDoc, SmartDocCorner]:
        if type == "document":
            return SmartDoc(directory)
            # then it must be corner
        else:
            return SmartDocCorner(directory)


class SmartDocDataset(Dataset):
    """The new dataset class used for training"""

    def __init__(
        self,
        directory="data",
        annotations_file="annotation.feather",
        images_dir="images",
    ):
        if isinstance(directory, str):
            directory = pathlib.Path(directory)
        self.directory = directory
        self.annotations_file = annotations_file
        self.images_dir = images_dir
        self.data = pd.read_feather(directory / annotations_file)
        self.transforms = T.Compose(
            [
                T.ToPILImage(),
                T.Resize([512, 512]),
                # transforms.ColorJitter(1.5, 1.5, 0.9, 0.5),
                T.PILToTensor(),
            ]
        )

    def __getitem__(self, index):
        img_path = self.directory / self.images_dir / self.data.image[index]
        target = self.data.annotation[index].reshape((4, 2))
        # img = read_image(str(img_path))
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        return self.transforms(img, SmartDocDataset.generate_mask(img, target))

    def __len__(self):
        # return min(len(self.data), 5000)
        return len(self.data)

    @staticmethod
    def generate_mask(image, annotation):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, [annotation], 255)
        return mask
