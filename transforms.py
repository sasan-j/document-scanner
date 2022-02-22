import numpy as np
import torch
from torchvision.transforms import functional as TVF
from torchvision import transforms


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = TVF.resize(image, self.size)
        target = TVF.resize(
            target, self.size, interpolation=transforms.InterpolationMode.NEAREST
        )
        return image, target


class ToPILImage:
    def __call__(self, image, target):
        image = TVF.to_pil_image(image)
        target = TVF.to_pil_image(target)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = TVF.pil_to_tensor(image).type(torch.float16) / 255
        target = torch.as_tensor(np.array(target), dtype=torch.float16) / 255
        return image, target
