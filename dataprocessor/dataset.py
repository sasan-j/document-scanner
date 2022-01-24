import logging
import pathlib
from typing import List

import numpy as np
import xml.etree.ElementTree as ET

import utils


logger = logging.getLogger(__name__)


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
