import os
import torch
from PIL import Image
from os import listdir


class CocoDataloader(object):

    def __init__(self, data_dir, transform=None):
        """
        The constructor to initialized paths to coco images
        :param data_dir: directory to coco images
        :param transform: image transformations
        """
        self.transform = transform
        self.image_names = [os.path.join(data_dir, img) for img in listdir(data_dir) if os.path.join(data_dir, img)]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image = Image.open(self.image_names[idx])

        if self.transform:
            image = self.transform(image)

        return image


class GreyToColor(object):
    """
    Converts grey tensor images to tensor with 3 channels
    """

    def __init__(self, size):
        """
        @param size: image size
        """
        self.image = torch.zeros([3, size, size])

    def __call__(self, image):
        """
        @param image: image as a torch.tensor
        @return: transformed image if it is grey scale, otherwise original image
        """

        out_image = self.image

        if image.shape[0] == 3:
            out_image = image
        else:
            out_image[0, :, :] = torch.unsqueeze(image, 0)
            out_image[1, :, :] = torch.unsqueeze(image, 0)
            out_image[2, :, :] = torch.unsqueeze(image, 0)

        return out_image


class Food101Dataloader(object):

    def __init__(self, data_dir, img_names, transform=None):
        """
        The constructor to initialized paths to Food-101 images
        :param data_dir: directory to images
        :params: img_names from meta data
        :param transform: image transformations
        """
        self.transform = transform
        self.image_names = [os.path.join(data_dir, img) for img in img_names]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image = Image.open(self.image_names[idx].split('\n')[0] + '.jpg')

        if self.transform:
            image = self.transform(image)

        return image