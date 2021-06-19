""" Code for the PCACE algorithm in VGG-16 """
# Uses PyTorch
# Caps indicate variables changable by user
# Example with VGG-16
# Need to import the ACE package

import io
import requests
import numpy as np
import cv2
import pdb
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
    finalconv_name = 'features' # this is the last conv layer of the network
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

activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

# Obtain input images for PCACE from the ImageNet URLs
# Txt file should include the URLs to the input images
NUM_IMAGES = 300 
CLASS = Egyptian_cat
with open ("CLASS.txt", "r") as myfile:
    data2 = myfile.readlines() 

good_urls = []
for i in range(len(data2)):
    good_urls.append(data2[i].strip('\n'))

results = []

# PCACE parameters to specify:
# LAYER_NAME: name of the convolutional layer to which we apply PCACE. Follows the structure 'features[x]'.
# NUM_CHANNELS: number of channels in LAYER_NAME.
# SIZE: number of pixels in the activation maps of LAYER_NAME.
# SIZE_X, SIZE_Y: height and width of the activation maps. Must have SIZE = SIZE_X*SIZE_Y.
# CLASS_IDX: before the softmax, which index corresponds to the class score (class of the set of input images).
# PCA_COMP: number of components to which PCA wishes to be reduced to.

for z in range(NUM_CHANNELS):  # There are 64 channels in the first conv1 layer
  iterations = NUM_IMAGES  # Number of input images passing in
  w, h = iterations, SIZE 
  channelact = [[0 for x in range(w)] for y in range(h)]  # Each row is a predictor variable of length SIZE
  classscore = []  # This will store the class number before the softmax

  h1 = net.LAYER_NAME.register_forward_hook(getActivation('matrix_name'))
  h2 = net.classifier[6].register_forward_hook(getActivation('classifier[6]'))

  # loop
  for k in range(iterations):
    IMG_URL = good_urls[k] 

    normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      normalize
    ])

    response = requests.get(IMG_URL)
    img_pil = Image.open(io.BytesIO(response.content))

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    out = net(img_variable)

    classscore.append(activation['classifier[6]'][0][CLASS_IDX].item())
    # In the case of regression tasks, such as our air pollution use case, we would use
    # classscore.append(activation['classifier[6]'][0].item()), as there is no CLASS_IDX

    s = 0
    for i in range(SIZE_X):  
      for j in range(SIZE_Y):
        channelact[s][k] = (activation['matrix_name'][0, z, i, j]).item()  # z is the channel
        s += 1

  # Detach the hooks
  h1.remove()
  h2.remove() 

  x = channelact
  y = classscore
  s = np.transpose(x)
  c = s.tolist()
  x_std = StandardScaler().fit_transform(c)
  pca = PCA(n_components=PCA_COMP)
  nou_x = pca.fit_transform(x_std)

  s = np.transpose(nou_x)
  c = s.tolist()
  x = c
  y = classscore
  myace = ace.ACESolver()
  myace.specify_data_set(x, y)
  myace.solve()
  val = np.corrcoef(myace.x_transforms[0], myace.y_transform)[0,1]

  results.append(abs(val))

# At the end, "results" contains all the PCACE values of all the channels
