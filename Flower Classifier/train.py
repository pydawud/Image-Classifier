#imports
from torch import nn, optim
from torchvision import datasets, transforms, models

import classifier
import process_dataset
import processes_image
import process_json

import argparse

parser = argparse.ArgumentParser(description='Train Image Classifier')

# Command line arguments
parser.add_argument('--arch', type = str, default = 'vgg', help = 'Architecture')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')
parser.add_argument('--hidden_units', type = int, default = 5000, help = 'Neurons in the Hidden Layer')
parser.add_argument('--epochs', type = int, default = 15, help = 'Epochs')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'Device [GPU , CPU]')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'checkpoint path')

arguments = parser.parse_args()

# Image data directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#load the data loaders
train_dataloader, valid_dataloader, test_dataloader = process_dataset.load_dataloaders(train_dir, valid_dir, test_dir)

#define the model
model = classifier.create_classifier(arguments.hidden_units, arguments.arch)    

#Criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)

#train the classifier    
classifier.train_classifier(model, optimizer, criterion, arguments.epochs, train_dataloader, valid_dataloader, arguments.gpu)
    
#Test Accuracy of the classifier    
classifier.accuracy(model, test_dataloader, criterion, arguments.gpu)

#to be use in saving class_to_idx
train_image_data = process_dataset.get_train_dataset(train_dir)

#save the classifier for future  prediction
classifier.save_checkpoint(model, train_image_data, arguments.arch, arguments.epochs, arguments.learning_rate, arguments.hidden_units)  

    