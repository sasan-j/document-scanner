import logging

import torch.utils.data as td
import tqdm
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class HddLoader(td.Dataset):
    def __init__(self, data, transform=None, cuda=False):
        self.data = data

        self.transform = transform
        self.cuda = cuda
        self.len = len(data[0])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        Replacing this with a more efficient implemnetation selection; removing c
        :param index:
        :return:
        """
        assert index < len(self.data[0])
        assert index < self.len
        img = Image.open(self.data[0][index])
        target = self.data[1][index]
        if self.transform is not None:
            img = self.transform(img)

        return img, target


class RamLoader(td.Dataset):
    def __init__(self, data, input_size, transform=None, cuda=False):
        self.data = data

        self.transform = transform
        self.cuda = cuda
        self.len = len(data[0])
        self.input_size = input_size
        self.loadInRam()

        data_transforms = {  # noqa
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }

    def loadInRam(self):
        self.loaded_data = []
        logger.info("Loading data in RAM")
        for i in tqdm.tqdm(self.data[0]):
            img = Image.open(i)
            if self.transform is not None:
                img = self.transform(img)
            self.loaded_data.append(img)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        Replacing this with a more efficient implemnetation selection; removing c
        :param index:
        :return:
        """
        assert index < len(self.data[0])
        assert index < self.len
        target = self.data[1][index]
        img = self.loaded_data[index]
        return img, target


class LoaderFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_loader(type, data, transform=None, cuda=False) -> td.Dataset:
        if type == "hdd":
            return HddLoader(data, transform=transform, cuda=cuda)
        elif type == "ram":
            return RamLoader(data, transform=transform, cuda=cuda)
