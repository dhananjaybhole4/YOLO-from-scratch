import torch
import torch.nn as nn

from tqdm.auto import tqdm

def training_step(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim,
                  device: str):
    train_loss = 0
    for index, value in enumerate(dataloader):
        X , Y = value
        # getting the data on given device
        X, Y = X.to(device), Y.to(device)
        # getting model on given device
        model.to(device)
        # train mode
        model.train()
        # prediction
        Y_pred = model(X)
        # loss
        loss = loss_fn(Y_pred, Y)
        # zero grad
        optimizer.zero_grad()
        # back propagation
        loss.backward()
        # step
        optimizer.step()

        # loss accumulation
        train_loss += loss
    
    # total train loss per epoch
    train_loss /= len(dataloader)
    return train_loss

def testing_step(model: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module,
                 device: str):
    
    test_loss = 0
    for index, value in enumerate(dataloader):
        X, Y = value
        # getting the data on given device
        X, Y = X.to(device), Y.to(device)
        # getting model on given device
        model.to(device)
        # model in eval mode
        model.eval()
        with torch.inference_mode():
            Y_pred = model(X)
        # loss
        loss = loss_fn(Y_pred, Y)
        
        # loss accumulation
        test_loss += loss

    # total test loss per epoch
    test_loss /= len(dataloader)
    return test_loss    

def train(epochs: int,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim,
          device: str):
    epochs_count = []
    train_loss_count = []
    test_loss_count = []
    for epoch in tqdm(range(epochs)):
        train_loss = training_step(model,
                                   train_dataloader,
                                   loss_fn,
                                   optimizer,
                                   device)
        test_loss = testing_step(model,
                                    test_dataloader,
                                    loss_fn,
                                    device)
        epochs_count.append(epoch + 1)
        train_loss_count.append(train_loss)
        test_loss_count.append(test_loss)
    
    return [epochs_count, train_loss_count, test_loss_count]
