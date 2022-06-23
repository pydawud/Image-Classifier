#imports
import torch
import classifier
import process_dataset
import processes_image
import process_json

import argparse

parser = argparse.ArgumentParser(description='Prediction')

# Command line arguments
parser.add_argument('--image_dir', type = str, default = 'flowers/test/1/image_06754.jpg', help = 'Image path')
parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')
parser.add_argument('--topk', type = int, default = 5, help = 'Top k classes and probabilities')
parser.add_argument('--json', type = str, default = 'cat_to_name.json', help = 'class_to_name json file')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'device = [GPU , CPU]')

arguments = parser.parse_args()

# Load in a mapping from category label to category name
class_to_name_dict = process_json.load_json(arguments.json)

# Load pretrained network
model = classifier.load_checkpoint(arguments.checkpoint)
#print(model)  

checkpoint = torch.load(arguments.checkpoint)

# preprocess image
image = processes_image.process_image(arguments.image_dir)

# Display image
processes_image.imshow(image)

# Highest k probabilities and the indices of those probabilities corresponding to the classes (converted to the actual class labels)
top_probs, top_classes = classifier.predict(arguments.image_dir, model, arguments.topk, arguments.gpu)  

print(top_probs)
print(top_classes)

# Display the image along with the top 5 classes
processes_image.display_image(arguments.image_dir, class_to_name_dict, top_probs, top_classes)
