import numpy as np
from augmentation.StatefulTransform import StatefulTransform
from augmentation.Augmentations import MSLANetAugmentation
from dataloaders.DataLoader import DataLoader
from typing import Optional, Tuple
import torch

from typing import Optional

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import pandas as pd
from config import BATCH_SIZE, IMAGE_SIZE, NORMALIZE, RANDOM_SEED
import random

random.seed(RANDOM_SEED)


class ImagesAndSegmentationDataLoader(DataLoader):
    """
    This class is used to load the images and create the dataloaders.
    The dataloder will output a tuple of (images, labels, segmentations), if segmentations are available (for training and validation, not for testing).
    The images are not segmented, and they are resized only if the resize_dim parameter is set.
    """

    def __init__(self,
                 limit: Optional[int] = None,
                 transform: Optional[transforms.Compose] = None,
                 dynamic_load: bool = False,
                 # If None, no resize is performed
                 resize_dim: Optional[Tuple[int, int]] = IMAGE_SIZE,
                 upscale_train: bool = True,
                 normalize: bool = NORMALIZE,
                 normalization_statistics: tuple = None,
                 batch_size: int = BATCH_SIZE,
                 load_segmentations: bool = True,
                 load_synthetic: bool = True,
                 return_image_name: bool = False):
        super().__init__(limit=limit,
                         transform=transform,
                         dynamic_load=dynamic_load,
                         upscale_train=upscale_train,
                         normalize=normalize,
                         normalization_statistics=normalization_statistics,
                         batch_size=batch_size,
                         always_rotate=False,
                         load_synthetic=load_synthetic)
        self.resize_dim = resize_dim
        self.load_segmentations = load_segmentations
        self.return_image_name = return_image_name

        assert return_image_name or load_segmentations, "Returning both image name and segmentation is still not supported"
        if self.resize_dim is not None:
            self.stateful_transform = StatefulTransform(
                height=resize_dim[0],
                width=resize_dim[1],
                always_rotate=self.always_rotate)
            self.transform = transforms.Compose([
                transforms.Resize(resize_dim,
                                  interpolation=Image.BILINEAR),
                transforms.ToTensor()
            ])
        else:
            self.stateful_transform = StatefulTransform(
                always_rotate=self.always_rotate)

        self.mslanet_transform = MSLANetAugmentation(
            resize_dim=self.resize_dim).transform

    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int):
        img = metadata.iloc[idx]
        load_segmentations = "train" in img and self.load_segmentations
        label = img['label']
        image = Image.open(img['image_path'])
        if load_segmentations:
            segmentation = Image.open(img['segmentation_path']).convert('1')
            if img["augmented"]:
                image, segmentation = self.stateful_transform(
                    image, segmentation)
            else:
                image = self.transform(image)
                segmentation = self.transform(segmentation)
        # Only load images
        else:
            if img["augmented"]:
                image = (np.array(image)).astype(np.uint8)
                image = self.mslanet_transform(image=image)["image"] / 255
            else:
                image = self.transform(image)
        if load_segmentations:
            return image, label, segmentation

        if self.return_image_name:
            return image, label, img["image_id"]
        return image, label

    def load_images_and_labels(self, metadata: pd.DataFrame):
        images = []
        segmentations = []
        labels = []

        for index, (row_index, img) in tqdm(enumerate(metadata.iterrows()), desc=f'Loading images'):
            load_segmentations = "train" in img and self.load_segmentations
            if load_segmentations:
                image, label, segmentation = self.load_images_and_labels_at_idx(
                    idx=index, metadata=metadata)
                segmentations.append(segmentation)
            else:
                image, label = self.load_images_and_labels_at_idx(
                    idx=index, metadata=metadata)
            images.append(image)
            labels.append(label)
        images = torch.stack(images)
        if load_segmentations:
            segmentations = torch.stack(segmentations)
        labels = torch.tensor(labels, dtype=torch.long)

        print(f"---Data Loader--- Images uploaded: " + str(len(images)))

        if load_segmentations:
            return images, labels, segmentations
        return images, labels
