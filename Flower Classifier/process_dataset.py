#imports
import torch
from torchvision import transforms, datasets


def transform_images():
    '''
    return transform of train, valid and test datasets
    '''
    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]),
    
        'valid': transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])]),
    
        'test': transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
        }
    return data_transforms

#iterateble dataloaders
def load_dataloaders(train_dir, valid_dir, test_dir):
    '''
    This function will process the data and return a data loader for train, test and validation set
    '''  
    data_transforms = transform_images()

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])

        }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
    }
    
    return dataloaders['train'], dataloaders['valid'], dataloaders['test']

# get image train datasets
def get_train_dataset(train_dir):
    
    data_transforms = transform_images()
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    
    return train_dataset