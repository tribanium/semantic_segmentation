from torch.utils.data import Dataset
import os
from PIL import Image


class TGSDataset(Dataset):
    """
    Loads the dataset contained in images/ and masks/ folders
    Resizes images and masks from 101x101 to 128x128 so that the size matches the network's.
    Apply torch transforms passed as inputs

    __getitem__ returns a tuple containing the input image and its mask.
    """

    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.all_images = os.listdir(img_path)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        file_name = self.all_images[index]
        input_img = (
            Image.open(os.path.join(self.img_path, file_name))
            .convert("L")
            .resize((128, 128))
        )
        mask_img = (
            Image.open(os.path.join(self.mask_path, file_name))
            .convert("L")
            .resize((128, 128))
        )
        if self.transform is not None:
            input_img = self.transform(input_img)
            mask_img = self.transform(mask_img)
        return input_img, mask_img
