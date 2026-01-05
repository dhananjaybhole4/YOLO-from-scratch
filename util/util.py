import torch
import torch.nn as nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

def intersection_over_union(predictions, target):
    """calculate iou between predictions and targets

    Args:
        predictions (_type_): prediction done by model in shape [Batch Size, S, S, 4]
        target (_type_): target label available in shape [Batch Size, S, S, 4]
    """

    # parameters for predicted bbox
    prediction_x1 = predictions[..., 0:1] - predictions[..., 2:3]/2.0
    prediction_y1 = predictions[..., 1:2] - predictions[..., 3:4]/2.0
    prediction_x2 = predictions[..., 0:1] + predictions[..., 2:3]/2.0
    prediction_y2 = predictions[..., 1:2] + predictions[..., 3:4]/2.0

    # parameters for target bbox
    target_x1 = target[..., 0:1] - target[..., 2:3]/2.0
    target_y1 = target[..., 1:2] - target[..., 3:4]/2.0
    target_x2 = target[..., 0:1] + target[..., 2:3]/2.0
    target_y2 = target[..., 1:2] + target[..., 3:4]/2.0

    # intersected rectangle points
    x1 = torch.max(target_x1, prediction_x1)
    y1 = torch.max(target_y1, prediction_y1)
    x2 = torch.min(target_x2, prediction_x2)
    y2 = torch.min(target_y2, prediction_y2)


    intersection_area = (x2 - x1).clamp(0)*(y2 - y1).clamp(0)

    prediction_area = torch.abs(prediction_x2 - prediction_x1)*torch.abs(prediction_y2 - prediction_y1)
    target_area = torch.abs(target_x2 - target_x1)*torch.abs(target_y2 - target_y1)

    intersection_over_union = intersection_area/(prediction_area + target_area - intersection_area)

    return intersection_over_union


transform = transforms.Compose([
    transforms.Resize(size = (448,448)),
    transforms.ToTensor()
])

def plot_loss_graph(count):
    epoch_count = count[0]
    train_loss_count = count[1]
    test_loss_count = count[2]

    plt.figure(figsize = (12, 9))
    plt.plot(epoch_count, train_loss_count, label = "train_loss_curve")
    plt.plot(epoch_count, test_loss_count, label = "test_loss_curve")
    plt.title("loss_curve")
