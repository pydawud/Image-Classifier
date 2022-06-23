#imports
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

import processes_image
from workspace_utils import active_session



#create the model
def create_classifier(hidden_units, arch):
    '''
    create the model base on the architecture i.e either vgg or alexnet and the number of hidden units
    '''
    if arch == 'vgg':
        input_units = 25088
        model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        input_units == 9216
        model = models.alexnet(pretrained=True)
    # Freeze pretrained model parameters to avoid backpropogating through them
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    # Build custom classifier
    classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1)
                  )
    model.classifier = classifier   
    
    return model    

#Training the model        
def train_classifier(model, optimizer, criterion, epochs, train_dataloader, valid_dataloader, device):
    '''
    To Train the model
    '''
    model.to(device)
    #Training the Network as learned in the last lesson on Transfer Learning
    with active_session():
               
        for epoch in range(epochs):
            running_loss = 0
            for images, labels in train_dataloader:
                # Move input and label tensors to the default device
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()

                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
        
            else:
                               
                valid_loss = 0
                accuracy = 0
                model.eval()
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for images, labels in valid_dataloader:
                        # Move input and label tensors to the default device
                        images, labels = images.to(device), labels.to(device)

                        log_ps = model(images)
                        valid_loss += criterion(log_ps, labels)

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(train_dataloader)),
                  "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_dataloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(valid_dataloader)))
            
            running_loss = 0
            model.train() 

#accuracy test function
def accuracy(model, test_dataloader, criterion, device):
    '''
    To test the accuracy of the model in %
    '''
    
    model.eval()
    model.to(device)
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
    
        accuracy = 0
    
        for images, labels in test_dataloader:
    
            images, labels = images.to(device), labels.to(device)
    
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        print("Accuracy: {:.2f}%".format(100*accuracy/len(test_dataloader)))
 
#predict function           
def predict(image_path, model, topk=5, device='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file (got help from stackoverflow)
    # switch to cuda and turn to evaluation mode
    model.to(device)
    model.eval()
    
    #get the image and convert it to tensor
    image = processes_image.process_image(image_path)
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    image = image.unsqueeze(0)
    image.to(device)
    
    output = model(image)
    
    ps = torch.exp(output)
    top_probs, top_indices = ps.topk(topk)
    
    # Convert to lists
    top_probs = top_probs.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    # Convert topk_indices to the actual class labels using class_to_idx
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    #print(idx_to_class)
    
    top_classes = [idx_to_class[index] for index in top_indices] 
        
    return top_probs, top_classes

#save checkpoint function
def save_checkpoint(model, train_dataset, arch, epochs, lr, hidden_units):
    '''
    A funnction to save the checkpoint of the model
    '''

    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {'hidden_layer_units': hidden_units,
                  'learning_rate': lr,
                  'model_arch': arch,
                  'model_state_dict': model.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx
                  }

    torch.save(checkpoint, 'checkpoint.pth')
    print('checkpoint saved Successfully')

#laod checkpoint function    
def load_checkpoint(filepath):
    '''
    To load the checkpoint state to the model
    '''
    checkpoint = torch.load(filepath)
    
    if checkpoint['model_arch'] == 'vgg':
        input_units = 25088
        model = models.vgg16(pretrained=True)
        
    elif checkpoint['model_arch'] == 'alexnet':
        input_units = 9216
        model = models.alexnet(pretrained=True)
       
           
    for param in model.parameters():
            param.requires_grad = False    
    
    model.class_to_idx = checkpoint['class_to_idx']

    classifier = nn.Sequential(nn.Linear(input_units, checkpoint['hidden_layer_units']),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(checkpoint['hidden_layer_units'], 102),
                                        nn.LogSoftmax(dim=1)
                              )

    model.classifier = classifier
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model            