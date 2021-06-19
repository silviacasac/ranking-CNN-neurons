""" Visualizing the filter activation maximization images with VGG-16 """
# Uses Keras
# Caps indicate variables changable by user
# Example with VGG-16
# Needs tf-keras-vis and --upgrade tf-keras-vis tensorflow matplotllib
# Following the code from https://github.com/keisen/tf-keras-vis

%reload_ext autoreload
%autoreload 2

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16 as Model
from matplotlib import pyplot as plt
%matplotlib inline

import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.callbacks import Print

_, gpus = num_of_gpus()
print('{} GPUs'.format(gpus))

# Load model
model = Model(weights='imagenet', include_top=True)
model.summary()

# Parameters to specify:
# LAYER_NAME: where is the channel whose feature visualization we are trying to see.
# FILTER_NUMBER: which channel within that layer.

def model_modifier(current_model):
    target_layer = current_model.get_layer(name=LAYER_NAME)
    new_model = tf.keras.Model(inputs=current_model.inputs,
                               outputs=target_layer.output)
    new_model.layers[-1].activation = tf.keras.activations.linear
    return new_model

activation_maximization = ActivationMaximization(model,
                                                 model_modifier,
                                                 clone=False)

def loss(output):
    return output[..., FILTER_NUMBER]

%%time
# Generate max activation
activation = activation_maximization(loss,
                                     callbacks=[Print(interval=50)])
image = activation[0].astype(np.uint8)

# Render
subplot_args = { 'nrows': 1, 'ncols': 1, 'figsize': (3, 3),
                 'subplot_kw': {'xticks': [], 'yticks': []} }
f, ax = plt.subplots(**subplot_args)
ax.imshow(image)
ax.set_title('filter[{:03d}]'.format(FILTER_NUMBER), fontsize=14)
plt.tight_layout()
plt.show()
