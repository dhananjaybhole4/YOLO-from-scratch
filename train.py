from model.yolo import YOLOv1
from model.loss import YoloLoss
from model.training_loop import train
from model.dataset import dataset
from util.util import *

from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader

from pathlib import Path

def main():

    # paths
    cwd = Path.cwd()
    train_img_dir = cwd / "dataset/image_train"
    test_img_dir = cwd / "dataset/image_test"
    train_label_dir = cwd / "dataset/label_train"
    test_label_dir = cwd / "dataset/label_test"
    print(train_img_dir)
    
    # Dataset
    train_dataset = dataset(train_img_dir,
                            train_label_dir)
    test_dataset = dataset(test_img_dir,
                           test_label_dir)
    print(train_dataset.__len__())
    # DataLoader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size = 1,
                                  shuffle = True,
                                  )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size = 1,
                                 shuffle = True,
                                 )
    
    # model
    model = YOLOv1()

    # loss
    loss = YoloLoss()

    # optimizer
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)

    # device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    count = train(epochs = 1,
                  model = model,
                  train_dataloader = train_dataloader,
                  test_dataloader = test_dataloader,
                  loss_fn = loss,
                  optimizer = optimizer,
                  device = device)
    
    plot_loss_graph(count)

if __name__ == "__main__":
    main()