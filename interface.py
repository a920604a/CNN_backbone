from CNN import AlexNet,VGG
import os 
import torch
import torch.nn as nn

from utils import get_training_dataloader,get_valid_dataloader,get_test_dataloader

import functools
import time

def time_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func_return_val = func(*args, **kwargs)
        end= time.perf_counter()
        print('{0:<1}.{1:<8} : {2:<8}sec'.format(func.__name__,'function',end-start))
        return func_return_val
    return wrapper

@time_wrapper
def train(train_loader, model, criterion, optimizer, epoch):
    CUDA = torch.cuda.is_available()
    total_train = 0
    correct_train = 0
    train_loss = 0
    for batch_index, (images, labels)  in enumerate(train_loader):
        if CUDA:
            images, labels = images.cuda(), labels.cuda()
        # clear gradient
        optimizer.zero_grad()

        # Forward propagation
        output = model(images) 
        loss = criterion(output, labels) 

        # Calculate gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index *images.size()[0] + len(images),
            total_samples=len(train_loader.dataset)
        ))

def train_loop(train_loader,valid_loader, model, criterion, optimizer, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        
        evaluate(valid_loader, model, criterion, epoch)
    
def evaluate(valid_loader, model, criterion, num_epochs): 
    CUDA = torch.cuda.is_available()
    model.eval()
    total_valid = 0
    correct_1 = 0
    valid_loss = 0
    for batch_idx,  (images, labels) in enumerate(valid_loader):        
        if CUDA:
            images, labels = images.cuda(), labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels) 
        
        
        valid_loss += loss.item()
        
        _, pred = outputs.topk(k=1, dim=1, largest=True, sorted=True)
        labels = labels.view(labels.size(0), -1).expand_as(pred)
        correct = pred.eq(labels).float()
        correct_1 += correct[:, :1].sum()
            
    print('evaluate loss ',valid_loss / len(valid_loader.dataset))
    print("Top 1 accuracy: ", 100*correct_1 / len(valid_loader.dataset))              



def main():
    num_epochs = 5
    lr = 0.01
    num_classes=2
    batch_size = 16
    
    train_loader = get_training_dataloader(root_path= './datasets/DogCat/training_set',\
                                       batch_size=batch_size, num_workers=2, shuffle=True, size=(128, 128))
    valid_loader = get_valid_dataloader(root_path= './datasets/DogCat/test_set',\
                                           batch_size=batch_size, num_workers=2, shuffle=False, size=(128, 128))


    CUDA = torch.cuda.is_available()
    
    device = torch.device("cuda" if CUDA else "cpu")
    
    

    model = VGG.Vgg16( num_classes=num_classes)
#     model = AlexNet.AlexNet( num_classes=num_classes)
    print(model)
    if CUDA:
        model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_loop(train_loader,valid_loader, model, criterion, optimizer, num_epochs)


if __name__=='__main__':
    main()