import os
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
import torch
import numpy as np

from typing import Optional

from PIL import Image, ImageDraw
from tqdm import tqdm
from torchvision import transforms
import pandas as pd
import torchvision.transforms.functional as TF
from augmentation.Augmentations import MSLANetAugmentation
from config import BATCH_SIZE, DATA_DIR, DATASET_TRAIN_DIR, IMAGE_SIZE, METADATA_TRAIN_DIR, NORMALIZE, RANDOM_SEED, SYNTHETIC_METADATA_TRAIN_DIR
import random

from dataloaders.DataLoader import DataLoader
from datasets.MSLANetDataset import MSLANetDataset
from models.GradCAM import GradCAM
from shared.constants import IMAGENET_STATISTICS

random.seed(RANDOM_SEED)


class MSLANetDataLoader(DataLoader):
    def __init__(self,
                 limit: Optional[int] = None,
                 transform: Optional[transforms.Compose] = None,
                 dynamic_load: bool = False,
                 resize_dim: Optional[Tuple[int, int]] = IMAGE_SIZE,
                 upscale_train: bool = False,
                 normalize: bool = NORMALIZE,
                 normalization_statistics: tuple = None,
                 batch_size: int = BATCH_SIZE,
                 load_synthetic: bool = False):
        super().__init__(limit=limit,
                         transform=transform,
                         dynamic_load=dynamic_load,
                         upscale_train=upscale_train,
                         normalize=normalize,
                         normalization_statistics=normalization_statistics,
                         batch_size=batch_size,
                         always_rotate=False,
                         data_dir=os.path.join(DATA_DIR, "gradcam_output"),
                         load_synthetic=load_synthetic)
        self.resize_dim = resize_dim
        self.load_synthetic = load_synthetic
        self.mslanet_transform = MSLANetAugmentation(
            resize_dim=self.resize_dim).transform
        self.transform = transforms.Compose([
            transforms.Resize(resize_dim,
                              interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ])
        self.gradcam = GradCAM()

    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int):
        LOW_THRESHOLD = 70
        HIGH_THRESHOLD = 110
        img = metadata.iloc[idx]
        label = img['label']
        image_ori = Image.open(img['image_path'])
        # image_low = Image.open(img['image_path_low'])
        # image_high = Image.open(img['image_path_high'])
        if img["augmented"]:
            image_ori = (np.array(image_ori)).astype(np.uint8)
            image_ori = self.mslanet_transform(image=image_ori)["image"] / 255
        else:
            image_ori = self.transform(image_ori)

        _, image_low, _ = self.gradcam.generate_cam(
            image=image_ori, threshold=LOW_THRESHOLD)
        _, image_high, _ = self.gradcam.generate_cam(
            image=image_ori, threshold=HIGH_THRESHOLD)
        # image_ori = TF.resize(image_ori, size=self.resize_dim,
        #                       interpolation=Image.BILINEAR)
        # image_ori = TF.to_tensor(image_ori)
        # image_low = TF.to_tensor(image_low)
        # image_high = TF.to_tensor(image_high)
        return (image_ori, image_low, image_high), label

    def load_images_and_labels(self, metadata: pd.DataFrame):
        images = []
        images_low = []
        images_high = []
        labels = []

        for index, (row_index, img) in tqdm(enumerate(metadata.iterrows()), desc=f'Loading images'):
            (image_ori, image_low, image_high), label = self.load_images_and_labels_at_idx(
                idx=index, metadata=metadata)
            images.append(image_ori)
            images_low.append(image_low)
            images_high.append(image_high)
            labels.append(label)
        images = torch.stack(images)
        images_low = torch.stack(images_low)
        images_high = torch.stack(images_high)
        labels = torch.tensor(labels, dtype=torch.long)

        print(f"---Data Loader--- Images uploaded: " + str(len(images)))

        return (images, images_low, images_high), labels

    def _init_metadata(self,
                       limit: Optional[int] = None):
        metadata = pd.read_csv(METADATA_TRAIN_DIR)
        synthetic_metadata = pd.read_csv(SYNTHETIC_METADATA_TRAIN_DIR)
        label_dict = {'nv': 0, 'bkl': 1, 'mel': 2,
                      'akiec': 3, 'bcc': 4, 'df': 5, 'vasc': 6}  # 2, 3, 4 malignant, otherwise begign
        labels_encoded = metadata['dx'].map(label_dict)
        metadata['label'] = labels_encoded

        print(f"LOADED METADATA HAS LENGTH {len(metadata)}")
        # if self.load_synthetic and limit is not None:
        #     limit = limit // 2

        if limit is not None and limit > len(metadata):
            print(
                f"Ignoring limit for because it is bigger than the dataset size")
            limit = None
        if limit is not None:
            print(f"---LIMITING REAL DATASET TO {limit} ENTRIES---")
            metadata = metadata.sample(n=limit, random_state=42)
        ori_data_dir = DATASET_TRAIN_DIR
        low_data_dir = os.path.join(DATA_DIR, "gradcam_output_70")
        high_data_dir = os.path.join(DATA_DIR, "gradcam_output_110")
        metadata['image_path'] = metadata['image_id'].apply(
            lambda x: os.path.join(ori_data_dir, x + '.jpg'))
        metadata['image_path_low'] = metadata['image_id'].apply(
            lambda x: os.path.join(low_data_dir, x + '.jpg'))
        metadata['image_path_high'] = metadata['image_id'].apply(
            lambda x: os.path.join(high_data_dir, x + '.jpg'))

        df_train, df_test = train_test_split(
            metadata,
            test_size=0.1,  # 15% test, 85% train
            random_state=RANDOM_SEED,
            stratify=metadata['dx'])

        df_train, df_val = train_test_split(
            df_train,
            test_size=0.2,  # Of the 85% train, 10% val, 90% train
            random_state=RANDOM_SEED,
            stratify=df_train['dx'])

        if self.load_synthetic:
            # Merge train dataset with synthetic dataset (I want to use the synthetic dataset only for training)
            print(f"---LOADING SYNTHETIC DATA IN THE TRAINING SET---")
            df_train = df_train[["image_id", "dx", "label", "image_path"]]
            df_train["synthetic"] = False
            labels_encoded = synthetic_metadata['dx'].map(label_dict)
            synthetic_metadata['label'] = labels_encoded
            synthetic_metadata['image_path'] = synthetic_metadata['image_id'].apply(
                lambda x: os.path.join(self.synthetic_data_dir, x + '.png'))

            augmented_low_data_dir = os.path.join(
                DATA_DIR, "augmented_gradcam_output_70")
            augmented_high_data_dir = os.path.join(
                DATA_DIR, "augmented_gradcam_output_110")

            synthetic_metadata['image_path'] = synthetic_metadata['image_id'].apply(
                lambda x: os.path.join(self.synthetic_data_dir, x + '.png'))
            synthetic_metadata['image_path_low'] = synthetic_metadata['image_id'].apply(
                lambda x: os.path.join(augmented_low_data_dir, x + '.png'))
            synthetic_metadata['image_path_high'] = synthetic_metadata['image_id'].apply(
                lambda x: os.path.join(augmented_high_data_dir, x + '.png'))

            if limit is not None and limit > len(synthetic_metadata):
                print(
                    f"Ignoring limit for because it is bigger than the dataset size")
                limit = None
            if limit is not None:
                print(f"---LIMITING SYNTHETIC DATASET TO {limit} ENTRIES---")
                synthetic_metadata = synthetic_metadata.sample(
                    n=limit, random_state=42)
            df_train = pd.concat(
                [df_train, synthetic_metadata], ignore_index=True)

        assert len(df_train['label'].unique(
        )) == 7, f"Number of unique labels in metadata is not 7, it's {len(df_train['label'].unique())}, increase the limit"
        assert len(df_val['label'].unique(
        )) == 7, f"Number of unique labels in metadata is not 7, it's {len(df_val['label'].unique())}, increase the limit"
        # TODO: Uncomment
        assert len(df_test['label'].unique(
        )) == 7, f"Number of unique labels in metadata is not 7, it's {len(df_test['label'].unique())}, increase the limit"

        df_train["train"] = True
        # df_val["train"] = False
        # df_test["train"] = False

        print(f"---TRAIN---: {len(df_train)} entries")
        print(f"---VAL---: {len(df_val)} entries")
        print(f"---TEST---: {len(df_test)} entries")
        return df_train, df_val, df_test

    def get_train_dataloder(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        if self.normalize:
            self.normalization_statistics = IMAGENET_STATISTICS
        train_dataset = MSLANetDataset(
            self.train_df,
            load_data_fn=self.load_data,
            normalize=self.normalize,
            mean=self.normalization_statistics[0] if self.normalize else None,
            std=self.normalization_statistics[1] if self.normalize else None,
            balance_data=self.upscale_train,
            resize_dims=IMAGE_SIZE,
            dynamic_load=self.dynamic_load)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
        )
        return train_dataloader

    def get_val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.normalize:
            self.normalization_statistics = IMAGENET_STATISTICS

        val_dataset = MSLANetDataset(
            self.val_df,
            load_data_fn=self.load_data,
            normalize=self.normalize,
            mean=self.normalization_statistics[0] if self.normalize else None,
            std=self.normalization_statistics[1] if self.normalize else None,
            balance_data=False,
            resize_dims=IMAGE_SIZE,
            dynamic_load=self.dynamic_load)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
        )
        return val_dataloader

    def get_test_dataloader(self):
        if self.normalize:
            self.normalization_statistics = IMAGENET_STATISTICS

        test_dataset = MSLANetDataset(
            self.test_df,
            load_data_fn=self.load_data,
            normalize=self.normalize,
            mean=self.normalization_statistics[0] if self.normalize else None,
            std=self.normalization_statistics[1] if self.normalize else None,
            balance_data=False,
            resize_dims=IMAGE_SIZE,
            dynamic_load=self.dynamic_load)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
        )
        return test_dataloader
