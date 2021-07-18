# PCACE: A Statistical Approach to Ranking Neurons for CNN Interpretability

PCACE is a new algorithm for ranking neurons in a CNN architecture in order of importance towards the final classification. PCACE is a statistical method combining Alternating Condition Expectation with Principal Component Analysis to find the maximal correlation coefficient between a hidden neuron and the final class score. This yields a rigorous and standardized method for quantifying the relevance of each neuron towards the final model classification.

## Summary of Usage
1) `pcace_resnet_18.py`: code for the PCACE algorithm in the ResNet-18 architecture. Uses PyTorch to load the model  and requires the ACE package. Caps indicate variables changeable by the user: 
`NUM_IMAGES`: the number of input images for PCACE.
`CLASS`: the class to which the input images belong to. 
`LAYER_NAME`: name of the convolutional layer to which we apply PCACE. Follows the structure `layerx[y].convz`. `NUM_CHANNELS`: number of channels in `LAYER_NAME`. 
`SIZE`: number of pixels in the activation maps of `LAYER_NAME`. 
`SIZE_X`, `SIZE_Y`: height and width of the activation maps. Must have `SIZE` = `SIZE_X`*`SIZE_Y`.
`CLASS_IDX`: before the softmax, which index corresponds to the class score (class of the set of input images).
`PCA_COMP`: number of components to which PCA wishes to be reduced to.
After the algorithm runs, it provides an array `results` with the PCACE values of all channels, which can then be sorted.

2) `pcace_vgg_16.py`: same code an functionality as `pcace_resnet_18.py` but in the VGG-16 architecture instead of ResNet-18. Computes the PCACE values for any layer in the VGG-16 architecture.

3) `activation_maximization.py`: code to visualize the filter activation maximization images with VGG-16 following the code from https://github.com/keisen/tf-keras-vis. Uses Keras to load the model and requires teh tf-keras-vis package. Caps indicate variables changeable by the user:
`LAYER_NAME`: where is the channel whose feature visualization we are trying to see.
`FILTER_NUMBER`: which channel within that layer.

4) `visualize_act_maps_resnet_18.py`: code to visualize the activation maps of the top PCACE channels with ResNet-18. As in `pcace_resnet_18.py`, it uses PyTorch to load the model. Caps indicate variables changeable by the user:
`LAYER_NAME`: name of the convolutional layer to which we apply PCACE. Follows the structure `layerx[y].convz`.
`ORDER`: an array containing the PCACE channels sorted from lowest to highest value.
The `good_urls` refer to a list containing the URLs of the images that one wishes to visualize.

5) `visualize_act_maps_vgg_16.py`: same functionality as in the `visualize_act_maps_resnet_18.py` code (i.e., visualize the activation maps of the top PCACE channels), but in the VGG-16 architecture instead of ResNet-18.

6) `visualizing_cam.py`: producing CAM visualizations with ResNet-18 following the code from https://github.com/zhoubolei/CAM. Uses PyTorch to load the model. Returns the CAM visualization of the input image (in this case, given with a URL).

7) `london_kdd_examples_slevel.csv`: The .csv file contains metadata for the 300 street level images we used in our experiments. In our experiments we used images from Google Street View. More information on these images and how to use them are available from here: https://developers.google.com/maps/documentation/streetview/overview. `gsv_panoid`: correspods to the 'pano' parameter, which is a specific panorama ID for the image. `gsv_lat, gsv_lng`: corresponds the the location coordinates for the image. Both `gsv_panoid` and `gsv_lat`, `gsv_lng` parameters can be used to access the images used in our experiments. 
