import os
import random
from typing import Optional, Tuple, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

TEST_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class UnpairedDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        transforms: Optional[transforms.Compose] = None,
        test_size: float = 0.2,
        use_big_dataset: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        fraction_dataset: float = 1.0,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        A_dir = os.path.join(data_dir, "monet_jpg")
        B_dir = os.path.join(data_dir, "photo_jpg")

        dataset_size_A = len(os.listdir(A_dir))
        dataset_size_B = len(os.listdir(B_dir))

        dataset_size_A = int(dataset_size_A * fraction_dataset)
        dataset_size_B = int(dataset_size_B * fraction_dataset)

        length_dataset = (
            max(dataset_size_A, dataset_size_B)
            if use_big_dataset
            else min(dataset_size_A, dataset_size_B)
        )
        files_A = [
            os.path.join(A_dir, name)
            for name in sorted(os.listdir(A_dir))[:length_dataset]
        ]
        files_B = [
            os.path.join(B_dir, name)
            for name in sorted(os.listdir(B_dir))[:length_dataset]
        ]

        test_size_A = int(len(files_A) * test_size)
        test_size_B = int(len(files_B) * test_size)
        train_size_A = len(files_A) - test_size_A
        train_size_B = len(files_B) - test_size_B

        if mode == "train":
            self.files_A = files_A[:train_size_A]
            self.files_B = files_B[:train_size_B]
        elif mode == "test":
            self.files_A = files_A[test_size_A:]
            self.files_B = files_B[test_size_B:]

        self.transforms = transforms
        self.use_big_dataset = use_big_dataset
        self.shuffle = shuffle

    def __len__(self) -> int:
        if self.use_big_dataset:
            return max(len(self.files_A), len(self.files_B))
        else:
            return min(len(self.files_A), len(self.files_B))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index_A = index % len(self.files_A)
        if self.shuffle:
            index_B = np.random.randint(0, len(self.files_B))
        else:
            index_B = index % len(self.files_B)

        file_A = self.files_A[index_A]
        file_B = self.files_B[index_B]

        img_A = Image.open(file_A)
        img_B = Image.open(file_B)

        if self.transforms is not None:
            img_A = self.transforms(img_A)
            img_B = self.transforms(img_B)
        else:
            img_A = transforms.ToTensor()(img_A)
            img_B = transforms.ToTensor()(img_B)

        img_A = cast(torch.Tensor, img_A)
        img_B = cast(torch.Tensor, img_B)

        return img_A, img_B


def create_dataloaders(
    data_dir: str,
    batch_size: int = 5,
    num_workers: int = 4,
    pin_memory: bool = False,
    test_size: float = 0.2,
    seed: int = 42,
    use_big_dataset: bool = False,
    shuffle: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders
    """
    train_dataset = UnpairedDataset(
        data_dir,
        mode="train",
        transforms=TRAIN_TRANSFORMS,
        test_size=test_size,
        seed=seed,
        use_big_dataset=use_big_dataset,
        shuffle=shuffle,
    )
    test_dataset = UnpairedDataset(
        data_dir,
        mode="test",
        transforms=TEST_TRANSFORMS,
        test_size=test_size,
        seed=seed,
        use_big_dataset=use_big_dataset,
        shuffle=shuffle,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
