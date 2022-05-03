import json
import os
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import functional


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def generate_image_lists(data_dir: str = './data', min_size: int = 100):
    """This function will generate image list based on data_folder,
    which is expect to have two folders, train and test folder

    Args:
        data_dir (str): dataset
        min_size (int): the minimum size of the image to be included
    """
    print("[INFO] Start data conversion.")
    original_path = os.path.abspath(data_dir)
    train_data_path = os.path.join(original_path, 'train')
    train_dirs = next(os.walk(train_data_path))[1]
    for d in train_dirs:
        if os.path.exists(os.path.join(train_data_path, (d + '_image_list.json'))):
            print('List JSON exist and pass it')
            continue
        train_images = list()
        dataset_path = os.path.join(train_data_path, d)
        for img in os.listdir(dataset_path):
            img_path = os.path.join(dataset_path, img)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
        print(f"[INFO] Dataset {d} for training, has {len(train_images)} images.")
        with open(os.path.join(train_data_path, (d + '_image_list.json')), 'w') as f:
            json.dump(train_images, f)
    
    test_data_path = os.path.join(original_path, 'test')
    test_dirs = next(os.walk(test_data_path))[1]
    for d in test_dirs:
        if os.path.exists(os.path.join(test_data_path, (d + '_image_list.json'))):
            print('List JSON exist and pass it')
            continue
        test_images = list()
        dataset_path = os.path.join(test_data_path, d)
        for img in os.listdir(dataset_path):
            img_path = os.path.join(dataset_path, img)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
        print(f"[INFO] Dataset {d} for testing, has {len(test_images)} images.")
        with open(os.path.join(test_data_path, (d + '_image_list.json')), 'w') as f:
            json.dump(test_images, f)
    print('[INFO] Data conversion is done.')

def image_converter(img, source, target):
    """A image converter. Source from 
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution#tutorial-in-progress

    Args:
        ing (_type_): _description_
        source (_type_): _description_
        target (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target
    
        # Convert from source to [0, 1]
    if source == 'pil':
        img = functional.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = functional.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img


class ImageSampler:

    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
        """
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: LR images will be downsampled from the HR images by this factor
        :param lr_img_type: the target format for the LR image; see convert_image() above for available formats
        :param hr_img_type: the target format for the HR image; see convert_image() above for available formats
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        """

        # Crop
        if self.split == 'train':
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            left = np.random.randint(1, img.width - self.crop_size)
            top = np.random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)

        # Sanity check
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # Convert the LR and HR image to the required type
        lr_img = image_converter(lr_img, source='pil', target=self.lr_img_type)
        hr_img = image_converter(hr_img, source='pil', target=self.hr_img_type)

        return lr_img, hr_img



if __name__ == "__main__":
    generate_image_lists()