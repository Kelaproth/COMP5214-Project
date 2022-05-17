from torch.utils.data import Dataset
import json
import os
from PIL import Image

from utils import ImageSampler, image_converter

class BaseDataset(Dataset):

    def __init__(self, dataset_name: str, type: str) -> None:
        self.dataset_name = dataset_name
        self.type = type.lower()

        assert self.type in {'train', 'test'}, \
            "Unknwon dataset type"

        if self.type == 'train':
            with open(os.path.join(os.getcwd(), 'data', 'train', 
            (dataset_name + '_image_list.json')), 'r') as f:
                self.image_list = json.load(f)
        else:
            with open(os.path.join(os.getcwd(), 'data', 'test', 
            (dataset_name + '_image_list.json')), 'r') as f:
                self.image_list = json.load(f)
        
        assert self.image_list is not None, \
            "Load image list error"
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = Image.open(self.image_list[index], mode='r')
        img = img.convert('RGB')
        return img

class SRSamplingDataset(BaseDataset):
    '''
    This dataset allow to use the common CV dataset (COCO) to 
    generate SR dataset, instead of using the dedicaded dataset
    (like DIV2K). Note: The list of the dataset should only contain
    the image path.
    '''

    def __init__(self, dataset_name: str, type: str, 
        crop_size, scaling_factor, lr_img_type, hr_img_type) -> None:
        super().__init__(dataset_name, type)

        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        # If this is a training dataset, then crop dimensions must be perfectly divisible by the scaling factor
        # (If this is a test dataset, images are not cropped to a fixed size, so this variable isn't used)
        if self.type == 'train':
            assert self.crop_size % self.scaling_factor == 0, \
            "Crop dimensions are not perfectly divisible by scaling factor! "
            "This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"

        self.transform = ImageSampler(split=self.type,
                                        crop_size=self.crop_size,
                                        scaling_factor=self.scaling_factor,
                                        lr_img_type=self.lr_img_type,
                                        hr_img_type=self.hr_img_type)

    def __getitem__(self, index):
        img = super().__getitem__(index)
        lr_img, hr_img = self.transform(img)
        return lr_img, hr_img

class SROriginDataset(BaseDataset):
    '''
    This dataset use the dedicaded dataset (like DIV2K). 
    Note: The list of the dataset should only contain the image path. 
    You need 2 separate call to load hr and lr images separately.
    '''
    def __init__(self, dataset_name: str, type: str, 
        img_type) -> None:
        super().__init__(dataset_name, type)

        self.img_type = img_type

        assert img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        self.convertor = image_converter

    def __getitem__(self, index):
        img = super().__getitem__(index)
        img = self.convertor(img, source='pil', target=self.img_type)
        return img