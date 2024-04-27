from os.path import join, isfile, splitext, basename

from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class NPYDatasets(Dataset):
    def __init__(self, config, transformer, train=True):
        super(NPYDatasets, self)
        """Initializes image paths and preprocessing module."""
        base_path = os.path.join(config["dataset_path"], "train" if train else "val")
        support_types = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 'TIF', 'TIFF']
        support_types = set(support_types)
        self.origin_image_path = join(base_path, "images")
        self.ground_truth_path = join(base_path, "masks")
        self.gt_format = config["gt_format"]
        self.image_paths = [join(self.origin_image_path, f) for f in os.listdir(self.origin_image_path)
                            if isfile(join(self.origin_image_path, f)) and splitext(f)[1][1:] in support_types]
        self.transformer = transformer

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        origin_image_name, extension = splitext(basename(image_path))
        filename = self.get_gt_file_name(origin_image_name, extension)
        msk_path = join(self.ground_truth_path, filename)
        img = np.array(Image.open(image_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2)
        msk[msk < 128] = 0
        msk[msk >= 128] = 1
        img, msk = self.transformer((img, msk))
        return img, msk, origin_image_name

    def get_gt_file_name(self, origin_image_name: str, extension: str) -> str:
        try:
            return self.gt_format.format(origin_image_name)
        except:
            return origin_image_name + "_segmentation.png"

    def __len__(self):
        return len(self.image_paths)
