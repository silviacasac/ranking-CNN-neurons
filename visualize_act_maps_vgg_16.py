""" Visualizing the top PCACE channels activation maps with VGG-16 """
# Uses PyTorch
# Caps indicate variables changable by user
# Example with VGG-16

import io
import requests
import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from ace import ace
from ace import model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Can apply to different architectures
model_id = 4
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # This is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
elif model_id==4:
    net = models.vgg16(pretrained=True)
    finalconv_name = 'features'
elif model_id == 5:
    net = models.densenet121(pretrained=True)
    finalconv_name = 'features'

# All available architectures are at: https://pytorch.org/docs/stable/torchvision/models.html 

net.eval()

# Parameters to specify
# LAYER_NAME: name of the convolutional layer to which we apply PCACE. Follows the structure 'features[x]'.
# ORDER is an array containing the PCACE channels sorted from lowest to highest value

IMG_URL = good_urls[img_num] # img_num: which image one wishes to visualize
print('image no:', img_num)

response = requests.get(IMG_URL)
img_pil = Image.open(io.BytesIO(response.content))
img_pil.save('test.jpg')

img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))

h1 = net.LAYER_NAME.register_forward_hook(getActivation('matrix_name')) 

out = net(img_variable)

for i in ORDER:
  plt.imshow(activation['matrix_name'][0, i, :, :], cmap='viridis')
  plt.title('Channel %i' %i) 
  plt.colorbar()
  plt.show()
