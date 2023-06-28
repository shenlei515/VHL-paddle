import logging

import numpy as np
import paddle
import paddle.io as data
from PIL import Image
from paddle.vision.datasets.cifar import Cifar10
import paddle.vision.transforms as transforms

from data_preprocessing.utils.utils import Cutout

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')



def data_transforms_cifar10(resize=32, augmentation="default", dataset_type="full_dataset",
                            image_resolution=32):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([])
    test_transform = transforms.Compose([])

    image_size = 32

    if dataset_type == "full_dataset":
        pass
    elif dataset_type == "sub_dataset":
        # train_transform.transforms.append(transforms.ToPILImage())
        pass
    else:
        raise NotImplementedError

    if resize is 32:
        pass
    else:
        image_size = resize
        train_transform.transforms.append(transforms.Resize(resize))
        test_transform.transforms.append(transforms.Resize(resize))

    if augmentation == "default":
        train_transform.transforms.append(transforms.RandomCrop(image_size, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    elif augmentation == "no":
        pass
    else:
        raise NotImplementedError

    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    if augmentation == "default":
        pass
        # train_transform.transforms.append(Cutout(16))
    elif augmentation == "no":
        pass
    else:
        raise NotImplementedError

    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    return CIFAR_MEAN, CIFAR_STD, train_transform, test_transform


    # if augmentation == "default":
    #     if resize is 32:
    #         train_transform = transforms.Compose([
    #             transforms.ToPILImage(),
    #             transforms.RandomCrop(32, padding=4),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    #         ])
    #         test_transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    #         ])
    #         logging.info(f"Cifar10 dataset augmentation: default, resize: {resize}")
    #     else:
    #         train_transform = transforms.Compose([
    #             transforms.ToPILImage(),
    #             transforms.Resize(resize),
    #             transforms.RandomCrop(resize, padding=4),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    #         ])
    #         test_transform = transforms.Compose([
    #             transforms.ToPILImage(),
    #             transforms.Resize(resize),
    #             transforms.ToTensor(),
    #             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    #         ])
    #     train_transform.transforms.append(Cutout(16))
    # else:
    #     if resize is 32:
    #         train_transform = transforms.Compose([
    #             transforms.ToPILImage(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    #         ])
    #         test_transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    #         ])
    #     else:
    #         train_transform = transforms.Compose([
    #             transforms.ToPILImage(),
    #             transforms.Resize(resize),
    #             transforms.ToTensor(),
    #             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    #         ])
    #         test_transform = transforms.Compose([
    #             transforms.ToPILImage(),
    #             transforms.Resize(resize),
    #             transforms.ToTensor(),
    #             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    #         ])





class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        cifar_dataobj = Cifar10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            targets = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            targets = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)




class CIFAR10_truncated_WO_reload(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                full_dataset=None):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.full_dataset = full_dataset

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # print("download = " + str(self.download))
        # cifar_dataobj = Cifar10(self.root, self.train, self.transform, self.target_transform, self.download)
        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = self.full_dataset.data
            targets = np.array(self.full_dataset.targets)
        else:
            data, targets = list(zip(*self.full_dataset.data))
            data = paddle.to_tensor(data, dtype=paddle.float32)
            targets = paddle.to_tensor(targets, dtype=paddle.float32)

        if self.dataidxs is not None:
            # print("data", data)
            # print("targets", targets)
            # print("self.dataidx", self.dataidxs)
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        # print("device in datasets", paddle.device.get_device())
        # index = paddle.to_tensor(index, dtype= paddle.int64)
        # print("index", index)
        # print("index.dtype", type(index))
        # index = paddle.to_tensor(index, place=paddle.CPUPlace())
        img, targets = self.data[index], self.targets[index]
        img = np.reshape(img, [3, 32, 32])
        img = img.transpose([1, 2, 0])
        
        if self.transform is not None:
            Image.fromarray(np.uint8(img)).convert('RGB')
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)





