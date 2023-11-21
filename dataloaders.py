from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
import os
from PIL import Image
from tqdm import tqdm
import random
import math

from config import DATASET_TEST_DIR, DATASET_TRAIN_DIR, METADATA_TEST_DIR, METADATA_TRAIN_DIR, SEGMENTATION_DIR, BATCH_SIZE


class ImageDataset(Dataset):
    def __init__(self,
                 metadata: pd.DataFrame,
                 train: bool = True,
                 # Control the data augmentation process aim to solve class imbalance
                 balance_data: bool = True,
                 # If balace_data, controls how many images to remove from the majority class in percentage. 1 if you don't want to apply undersampling.
                 undersampling_majority_class_weight: float = 0.5,
                 normalize: bool = False,  # Control the application of z-score normalization
                 mean: float = 0,
                 std: float = 0,
                 # Adjustment value to avoid division per zero during normalization
                 std_epsilon: float = 0.01,
                 transform: Optional[transforms.Compose] = None,
                 balance_transform: Optional[transforms.Compose] = None):
        self.metadata = metadata
        self.transform = transform

        unique_labels = self.metadata['dx'].unique()
        label_dict = {label: idx for idx, label in enumerate(unique_labels)}
        assert len(
            label_dict) == 7, "There should be 7 unique labels, increase the limit"
        labels_encoded = self.metadata['dx'].map(label_dict)
        self.metadata['label'] = labels_encoded
        self.metadata['augmented'] = False
        self.metadata = self.metadata
        self.train = train
        self.balance_data = balance_data
        self.balance_transform = balance_transform
        self.normalize = normalize
        self.mean = mean
        self.std = std
        if std_epsilon <= 0:
            raise ValueError("std_epsilon must be a positive number")
        else:
            self.std_epsilon = std_epsilon
        if undersampling_majority_class_weight <= 0 and undersampling_majority_class_weight > 1:
            raise ValueError(
                "undersampling_majority_class_weight must be a value in the range (0, 1]")
        else:
            self.undersampling_majority_class_weight = undersampling_majority_class_weight

        scale_factor = 0.1
        ORIGINAL_HEIGHT, ORIGINAL_WIDTH = 450, 600
        height, width = int(
            ORIGINAL_HEIGHT * scale_factor), int(ORIGINAL_WIDTH * scale_factor)
        if self.transform is None:
            self.transform = transforms.Compose([
                # transforms.Resize((height, width)),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        '''
        if self.balance_transform is None:
            self.balance_transform = transforms.Compose([
                transforms.Resize((height, width)), #TO DO: TRY TO PUT TOGETHER SELF.TRASFORM AND BALANCE TRANSFORM
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                #transforms.RandomResizedCrop(size=(self.height, self.width), scale=(0.9, 1.1)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor()
        ])
        '''

        if self.balance_data:
            self.balance_dataset()

        if self.train:
            self.images, self.labels, self.segmentations = self.load_images_and_labels()
        else:
            self.images, self.labels = self.load_images_and_labels()

        # TODO: maybe load other information,
        # encode it in one-hot vectors and concatenate them to the images in order to feed it to the NN

    def balance_dataset(self):
        # Count images associated to each label
        labels_counts = Counter(self.metadata['label'])
        max_label, max_count = max(
            labels_counts.items(), key=lambda x: x[1])  # Majority class
        _, second_max_count = labels_counts.most_common(
            2)[-1]  # Second majority class

        print(
            f"Max count is {max_count}, while second max count is {second_max_count}")

        # Undersampling most common class
        max_label_images_to_remove = max_count - max(
            math.floor(max_count*self.undersampling_majority_class_weight), second_max_count)

        print(f"Max labels to remove is {max_label_images_to_remove}")
        label_indices = self.metadata[self.metadata['label']
                                      == max_label].index
        removal_indices = random.sample(
            label_indices.tolist(), k=max_label_images_to_remove)
        self.metadata = self.metadata.drop(index=removal_indices)
        self.metadata.reset_index(drop=True, inplace=True)
        labels_counts = Counter(self.metadata['label'])
        max_label, max_count = max(labels_counts.items(), key=lambda x: x[1])

        # Oversampling of the other classes
        for label in self.metadata['label'].unique():
            label_indices = self.metadata[self.metadata['label']
                                          == label].index
            current_images = len(label_indices)

            if current_images < max_count:
                num_images_to_add = max_count - current_images
                aug_indices = random.choices(
                    label_indices.tolist(), k=num_images_to_add)
                self.metadata = pd.concat(
                    [self.metadata, self.metadata.loc[aug_indices]])
                # Apply data augmentation only to the augmented subset
                self.metadata.loc[aug_indices, 'augmented'] = True
                label_indices = self.metadata[self.metadata['label']
                                              == label].index

    def load_images_and_labels(self):
        not_found_files = []
        images = []
        segmentations = []
        labels = []
        for _, img in tqdm(self.metadata.iterrows(), desc=f'Loading {"train" if self.train else "test"} images'):
            if not os.path.exists(img['image_path']):
                not_found_files.append(img['image_path'])
                continue
            labels.append(img['label'])
            if self.balance_data:
                # CHANGE WITH NEW SIZES DINAMICALLY!
                stateful_transform = StatefulTransform(45, 60)
                if not os.path.exists(img['segmentation_path']):
                    not_found_files.append(img['segmentation_path'])
                    continue
                if img['augmented']:
                    ti, ts = stateful_transform(Image.open(
                        img['image_path']), Image.open(img['segmentation_path']))
                    images.append(ti)
                    segmentations.append(ts)
                else:
                    images.append(self.transform(
                        Image.open(img['image_path'])))
                    segmentations.append(self.transform(
                        Image.open(img['segmentation_path'])))

            elif self.train:
                if not os.path.exists(img['segmentation_path']):
                    not_found_files.append(img['segmentation_path'])
                    continue
                images.append(self.transform(Image.open(img['image_path'])))
                segmentations.append(self.transform(
                    Image.open(img['segmentation_path'])))
            else:
                images.append(self.transform(Image.open(img['image_path'])))
        if self.train:
            segmentations = torch.stack(segmentations)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        print(f"---Data Loader--- Images uploaded: " + str(len(images)))

        print(
            f"Loading complete, some files ({len(not_found_files)}) were not found: {not_found_files}")
        if self.train:
            return images, labels, segmentations
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.normalize:
            image = (image - self.mean) / (self.std + self.std_epsilon)
        if self.train:
            segmentation = self.segmentations[idx]
            return image, label, segmentation
        return image, label


class StatefulTransform:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, seg):
        img = transforms.Resize((224, 224))(img)
        seg = transforms.Resize((224, 224))(seg)

        if random.random() > 0.5:
            img = TF.hflip(img)
            seg = TF.hflip(seg)

        if random.random() > 0.5:
            img = TF.vflip(img)
            seg = TF.vflip(seg)

        if random.random() > 0.5:
            angle = random.randint(1, 360)
            img = TF.rotate(img, angle)
            seg = TF.rotate(seg, angle)

        img = transforms.ToTensor()(img)
        seg = transforms.ToTensor()(seg)

        return img, seg


def calculate_normalization_statistics(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    images_for_normalization = []

    for _, img in tqdm(df[:100].iterrows(), desc=f'Calculating normalization statistics'):
        if not os.path.exists(img['image_path']):
            continue
        image = transforms.ToTensor()(Image.open(img['image_path']))
        images_for_normalization.append(image)

    images_for_normalization = torch.stack(images_for_normalization)
    mean = torch.tensor([torch.mean(images_for_normalization[:, :, :, channel])
                        for channel in range(3)]).reshape(3, 1, 1)
    std = torch.tensor([torch.std(images_for_normalization[:, :, :, channel])
                       for channel in range(3)]).reshape(3, 1, 1)

    print("---Normalization--- Normalization flag set to True: Images will be normalized with z-score normalization")
    print(
        f"---Normalization--- Statistics for normalization (per channel) -> Mean: {mean.view(-1)}, Variance: {std.view(-1)}, Epsilon (adjustment value): 0.01")

    return mean, std


def load_metadata(train: bool = True,
                  limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame] or pd.DataFrame:
    metadata = pd.read_csv(METADATA_TRAIN_DIR if train else METADATA_TEST_DIR)
    if limit is not None and limit > len(metadata):
        print(
            f"Ignoring limit for {METADATA_TRAIN_DIR if train else METADATA_TEST_DIR} because it is bigger than the dataset size")
        limit = None
    if limit is not None:
        metadata = metadata.sample(n=limit, random_state=42)
    metadata['image_path'] = metadata['image_id'].apply(
        lambda x: os.path.join(DATASET_TRAIN_DIR if train else DATASET_TEST_DIR, x + '.jpg'))

    if train:
        metadata['segmentation_path'] = metadata['image_id'].apply(
            lambda x: os.path.join(SEGMENTATION_DIR, x + '_segmentation.png'))

        # Assuming `df` is your DataFrame
        df_train, df_val = train_test_split(
            metadata, test_size=0.2, random_state=42, stratify=metadata['dx'])

        return df_train, df_val

    return metadata


def check_distribution(df: pd.DataFrame, name: Optional[str] = None):
    labels_counts = Counter(df['dx'])
    label_percentage = {label: count/len(df) for label,
                        count in labels_counts.items()}
    percentages = [(x[0], round(x[1], 2)) for x in sorted(
        label_percentage.items(), key=lambda x: x[0])]
    if name is not None:
        print(f"Labels percentages for {name} are {percentages}")
    return label_percentage


def compute_weights(df: pd.DataFrame, labels_encoding: Dict[str, int]):
    inverted_encoding = {v: k for k, v in labels_encoding.items()}
    label_percentage = check_distribution(df)
    weights = [label_percentage[inverted_encoding[i]] for i in range(7)]
    return weights


def create_dataloaders(normalize: bool = True, limit: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    df_train, df_val = load_metadata(limit=limit)
    df_test = load_metadata(train=False, limit=limit)

    print("------Check distribution-----")
    check_distribution(df_train, "train")
    check_distribution(df_val, "val")
    check_distribution(df_test, "test")
    print("-----------------------------")

    # Calculate and store normalization statistics for the training dataset
    train_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    train_std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    # if normalize:
    #    train_mean, train_std = calculate_normalization_statistics(df_train)
    # print(train_mean, train_std)

    train_dataset = ImageDataset(
        df_train, normalize=normalize, mean=train_mean, std=train_std, balance_data=True)
    val_dataset = ImageDataset(
        df_val, train=True, normalize=normalize, mean=train_mean, std=train_std, balance_data=False)
    test_dataset = ImageDataset(
        df_test, train=False, normalize=normalize, mean=train_mean, std=train_std, balance_data=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = create_dataloaders(normalize=True)

    batch: torch.Tensor
    labels: torch.Tensor
    segmentations: torch.Tensor
    for (batch, labels, segmentations) in train_loader:
        print(f"Batch shape is {batch.shape}")
        print(f"Labels shape is {labels.shape}")
        print(f"Segmentation shape is {segmentations.shape}")
        break
