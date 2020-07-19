from os.path import isdir, join
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset.utils import encode_img, decode_img, array_yxc2cyx, array_cyx2yxc


DEFAULT_DSET_PATH = "/home/igor/datasets/faces6k/aligned"
DEFAULT_TARGET_SIZE = (64, 64)
DEFAULT_RGB_MEAN = (0.5566640322357532, 0.44162517425904774, 0.3831630928626849)
DEFAULT_RGB_STD = (0.2790375660010676, 0.2526975687092722, 0.24888596084747636)


class FacesDataset:

    def make_tensor(self, path_to_image):
        image = cv2.imread(path_to_image)
        resize_image = (image.shape[1], image.shape[0]) != self.output_size
        if resize_image:
            image = cv2.resize(image, self.output_size)
        return encode_img(image, mean=self.mean, std=self.std)

    def __init__(self, root=DEFAULT_DSET_PATH, target_size=DEFAULT_TARGET_SIZE,
                 mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)):
        assert isdir(root)

        self.root = root
        self.output_size = target_size
        self.sample_ptr = 0
        self.mean = mean
        self.std = std

        self.img_paths = glob(join(root, "*.jpg"))

        self.imgs = list(map(self.make_tensor, self.img_paths))
        return

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self[i] for i in range(*item.indices(len(self)))]
        elif isinstance(item, int):
            return self.imgs[item]
        else:
            raise ValueError("Invalid index(-ices) to __get_item__ method")

    def __iter__(self):
        return self

    def __next__(self):
        if self.sample_ptr < len(self.imgs):
            out = self.sample_ptr
            self.sample_ptr += 1
            return self.__getitem__(out)
        else:
            self.sample_ptr = 0
            raise StopIteration

    def __str__(self):
        return "AlignedFaces6k"


def make_dataloader(dataset, batch_size=1, shuffle_dataset=True):
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


if __name__ == "__main__":
    dataset = FacesDataset("/home/igor/datasets/faces6k/small", target_size=(64, 64),
                           mean=DEFAULT_RGB_MEAN, std=DEFAULT_RGB_STD)
    train_dloader = make_dataloader(dataset, 10, True)
    print("Dataset size", len(dataset))
    print("Train", len(train_dloader))

    iterator = iter(train_dloader)
    batch = next(iterator)
    print("Sample shape: ", batch.shape)
    print("Sample dtype: ", type(batch.dtype))

    for idx, img in enumerate(batch):
        denorm = decode_img(img.numpy().copy(), mean=DEFAULT_RGB_MEAN, std=DEFAULT_RGB_STD)
        cv2.imwrite("/home/igor/temp/temp"+str(idx)+".jpg", denorm)
    print("Saved")

