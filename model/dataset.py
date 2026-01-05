import torch
import torch.nn as nn

from PIL import Image
import glob
import numpy as np

from util.util import transform


class dataset(torch.utils.data.Dataset):
    """class to create a dataset

    Args:
        torch (_type_): _description_
    """
    def __init__(self, img_dir, label_dir, transform = transform, C = 11, S = 7):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.C = C
        self.S = S
        # path to list of individual label paths
        self.label_path_list = sorted(label_dir.glob("*.txt"))
        # path to list of individual img paths
        self.img_path_list = sorted(img_dir.glob("*.jpg"))

    # This fn returns the total number of data in dataset
    def __len__(self):
        return int(len(self.img_path_list)/2.0)
    
    def __getitem__(self, idx):
        return [self.img_parser(idx, self.transform), self.label_parser(idx)]

    def img_parser(self, idx, transform):
        """This function returns the required index img in Tensor format from the dataset

        Args:
            idx (int): idx to get the required data from dataset
            img_dir (pathlib.Path): path to the img dataset
            transform (torchvision.transforms): transform needed to convert a img into Tensor with resizing and maybe augmentation
        """
        # we using 2*idx + 1 rather than idx because we want to avoid dublicate data
        img_path = self.img_path_list[2*idx + 1]

        # open the image
        img = Image.open(img_path)
        img_tensor = transform(img)
        return img_tensor
    
    def label_parser(self, idx):
        """This function returns the label format accepted in model

        Args:
            idx (int): index to get the required label data from dataset
            label_dir (pathlib.Path): path to the label dataset
        """
        label_path = self.label_path_list[2*idx + 1]
        step_size = 1.0/self.S

        # converting the given label format into the label format needed in the model
        zero_label = torch.zeros(size = (self.S, self.S, self.C + 5))
        with open(label_path, "r") as f:
            for line in f:
                lbl = [float(x) for x in line.split()]
                k, l = 0, 0
                stop = False
                for i in np.arange(0, 1, step_size):
                    if stop == True:
                        break
                    elif i <= lbl[1] < i + step_size:                    
                        for j in np.arange(0, 1, step_size):
                            if j <= lbl[2] < j + step_size:
                                zero_label[k, l, int(lbl[0])] = 1.0
                                zero_label[k, l, self.C:(self.C + 5)] = torch.Tensor([1, lbl[1], lbl[2], lbl[3], lbl[4]])
                                stop = True
                                break
                            else:
                                l += 1   
                    else:
                        k += 1
        transformed_label = zero_label
        return transformed_label
        




