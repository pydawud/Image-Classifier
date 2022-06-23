#imports
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns
#import os
#os.environ['QT_QPA_PLATFORM']='offscreen'

import matplotlib.pyplot as plt

from PIL import Image



#process Image function
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        
    args:
        image ---- an image to be processed
    return:
        a normalize image in numpy
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    #Resize
    pil_image.thumbnail((256,256), Image.ANTIALIAS)
    
    #Crop: get help from stackoverflow
    width, height = (pil_image.width,pil_image.height)   # Get dimensions
    new_width, new_height = (224,224)
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
   
    pil_image = pil_image.crop((left, top, right, bottom))
    
    np_image = np.array(pil_image)/255
    
    #Normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

#displaying image function
def imshow(image, ax=None, title=None):
    """Imshow for Tensor.
    args:
        image --- an image
        ax ---- axis to display the image with Default None
        title ---- Title of the image label with Default None
   return:
        ax ---- an axis
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#display predicted image
def display_image(image_dir, cat_to_name, probs, classes):
    '''
    Display function to display the predicted image and plot the top probability labels
    args:
        image_dir ---- image path
        cat_to_name ---- category of the flower
        classes --- label classes
    return:
        None
    
    '''
    # Plot flower input image
    plt.figure(figsize = (6,8))
    ax_1 = plt.subplot(2,1,1)

    image = process_image(image_dir)
    
    key = image_dir.split('/')[-2]

    flower_title = cat_to_name[key]

    imshow(image, ax_1, title=flower_title);

    # Convert from the class integer encoding to actual flower names
    flower_names = [cat_to_name[i] for i in classes]

    # Plot the probabilities for the top 5 classes as a bar graph
    plt.subplot(2,1,2)

    sns.barplot(x=probs, y=flower_names, color=sns.color_palette()[0]);

    plt.show()
    print('Done')
 