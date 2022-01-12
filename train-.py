import json
import train_args
import os
import random
import numpy as nump
import torchvision.transforms as trans
import torchvision
import torch
import torch.nn as n
import torch.optim as optimiz
from torchvision.datasets import ImageFolder
import random
import torchvision.models as models
from collections import OrderedDict
from PIL import Image

import argparse
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import fmodel


def train(epochs, loader, model, optimizer, neg_loss, use_cuda, g):
    loss_min = nump.Inf 
    
    if_cuda_default = False
    if (use_cuda):
        if_cuda_default = True  
    for epoch in range(1, epochs + 1):
        model.train()
        training = 0.0
        valid_loss = 0.0
        for idx, (data, target) in enumerate(loader['train']):            
            optimizer.zero_grad()
            optimizer.step()

            if use_cuda:
                data = data.cuda()  
                target = target.cuda()
            output = model(data)
           
        
            loss = neg_loss(output, target)
            loss.backward()
            training = (((1 / (idx + 1)) * (loss.data - training))) + training
            model.eval()
        for idx, (data, target) in enumerate(loader['valid']):
            if if_cuda_default:
                data, target = data.cuda(), target.cuda()
            difference = loss.data - valid_loss
            output = model(data)
            coefficient = 1 / (idx + 1)
            loss = neg_loss(output, target)
            valid_loss = (((coefficient)) * (difference))) + valid_loss
            
        print('epoch', epoch)
        print('training loss', training)
        print('valid loss', valid_loss)
        if (loss_min >= valid_loss):
            torch.save(model.state_dict(), g)
            print('validation loss from', loss_min,'to',valid_loss)
            loss_min = valid_loss
            #set equal
    return model



def get_model(calc = 0.5):
    for i in model_transfer.features.i():
        i.requires_grad = False
    practiced = True
    categories = 102
    model_transfer = models.alexnet(pretrained = practiced)
    
    initial = model_transfer.classifier[1].weight
    secondary = model_transfer.classifier[4].weight
    
    # classify dictionary, holds model of classifier
    classify = n.Sequential(OrderedDict([
        ('dropout1', n.Dropout(calc)),
        ('dropout2', n.Dropout(calc)),
        ('fc1', n.Linear(model_transfer.classifier[1].in_features, model_transfer.classifier[1].out_features)),
        ('fc2', n.Linear(model_transfer.classifier[4].in_features, model_transfer.classifier[4].out_features)),
        ('fc3', n.Linear(model_transfer.classifier[6].in_features, categories)),
        ('relu1', n.ReLU(inplace = practiced)),
        ('relu2', n.ReLU(inplace = practiced)),
        ('output', n.LogSoftmax(dim = 1))]))
    model_transfer.classifier = classify
    negative_loss = n.NLLLoss()
    print(model_transfer.classifier)
    
    # need to reset the weights of the AlexNet before proceeding
    model_transfer.classifier[1].weight = initial
    model_transfer.classifier[4].weight = secondary
    optimize = optimiz.SGD(model_transfer.classifier.i(), num = 0.001)
    return model_transfer, negative_loss, optimize 


#main method /// bring to top 
def main():
    step = train_args.get_args()
    dub_package = step.parse_args()
    epochs = dub_package.epochs
    use_cuda = False

    if (dub_package.save_name):
        save_name = dub_package.save_name
    if (dub_package.save_dir):
        save_dir = dub_package.save_dir   
    
    if dub_package.use_gpu and torch.cuda.is_available():
        use_cuda = True
   
    #from imported os
    if not os.path.isdir(dub_package.data_directory):
        print('Is not available.')
        exit(1)

    if not os.path.isdir(dub_package.save_dir):
        print('Is not available.')
        os.makedirs(dub_package.save_dir)
    
    train_dir = dub_package.data_directory
    valid_dir = 'flowers/valid'

    data_directory = 'flowers'
    
    
    
    training_directory = data_directory + '/train'
    validating_directory = data_directory + '/valid'
    testing_directory = data_directory + '/test'
    
    
    trained = trans.Compose([trans.RandomRotation(23),
                                      trans.RandomResizedCrop(224),
                                      trans.RandomHorizontalFlip(),
                                      trans.ToTensor(),
                                      trans.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    tested = trans.Compose([trans.Resize(256),
                                      trans.CenterCrop(224),
                                      trans.ToTensor(),
                                      trans.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    validated = trans.Compose([trans.Resize(256),
                               trans.CenterCrop(224),
                               trans.ToTensor(),
                               trans.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
   
# defining the data loaders trainloader, testloader, validloader
    train_image_data = datasets.ImageFolder(training_directory, trans = trained)
    testing_data = datasets.ImageFolder(testing_directory, trans = tested)
    valid_image_data = datasets.ImageFolder(validating_directory, trans = validated)
    a = torch.utils.data.DataLoader(train_image_data, batch_size = 64, shuffle = True)
    b = torch.utils.data.DataLoader(testing_data, batch_size = 64)
    c = torch.utils.data.DataLoader(valid_image_data, batch_size = 64)

# will be checked if cuda is to be used in later functions
    use_cuda = torch.cuda.is_available()
# inpout for following method
    dataloaders = {"trained": a, "valid": c, "test": b}
    
    checkpoint_name = 'checkpoint.pt'
   
    with open('cat_to_name.json', 'r') as opening:
        cat_to_name = json.load(opening)    
    cat_to_name
    
    model_transfer = get_model() 
    negative_loss = get_model()
    optimize = get_model()
    
  #  model_transfer = train(dataloaders, model_transfer, optimizer_transfer, criterion_transfer, checkpoint_name, epochs, use_cuda, train_dataset)
    model_transfer = train(11, dataloaders, model_transfer, optimize, negative_loss, use_cuda, 'model_transfer.pt')

