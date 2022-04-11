from tensorflow.keras.models i
port Model
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
#import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import SimpleITK as sitk
from scipy.ndimage import zoom
import argparse


def get_data(proj_dir, filename1, filename2):
    
    """
    data preprocessing for gradCAM
    data shape: (32, 118, 118, 1)
    small data shape: (32, 32, 32, 1)
    """

    ## create numpy array 
    nrrd1 = sitk.ReadImage(os.path.join(proj_dir, filename1), sitk.sitkFloat32)
    img1 = sitk.GetArrayFromImage(nrrd1)
    #s = img1.shape
    #img1 = img1.shape((1, s[0], s[1], s[2], 1))
    img1 = img1.reshape((1, 32, 118, 118, 1))

    nrrd2 = sitk.ReadImage(os.path.join(proj_dir, filename2), sitk.sitkFloat32)
    img2 = sitk.GetArrayFromImage(nrrd2)
    #s = img2.shape
    #img2 = img2.shape((1, s[0], s[1], s[2], 1))
    img2 = img2.reshape((1, 32, 32, 32, 1))

    return img1, img2


def find_target_layer(model, saved_model):

    """
    find the final conv layer by looping layers in reverse order
    """
    for layer in reversed(model.layers):
        # check to see if the layer has a 4D output
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


def compute_heatmap(proj_dir, saved_model, img1, img2, conv, 
                    activation='sigmoid', pred_index=1):

    """
    construct our gradient model by supplying (1) the inputs
    to our pre-trained model, (2) the output of the (presumably)
    final conv layer in the network, and (3) the output of the
    sigmoid activations from the model
    """

    model = load_model(os.path.join(proj_dir, saved_model))
    #model.summary() 
    
    gradModel = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(conv).output, model.output]
        )
    # record operations for automatic differentiation
    with tf.GradientTape() as tape:
        """
        cast the image tensor to a float-32 data type, pass the
        image through the gradient model, and grab the loss
        associated with the specific class index
        """
        input1 = tf.cast(img1, tf.float32)
        input2 = tf.cast(img2, tf.float32)
        inputs = (input1, input2)

        conv_output, preds = gradModel(inputs)
        #print(preds)
        #print(preds.shape)
        if activation == 'sotfmax':
            class_channel = preds[:, pred_index]
        elif activation == 'sigmoid':
            class_channel = preds
    # use automatic differentiation to compute the gradients
    grads = tape.gradient(class_channel, conv_output)
    """
    This is a vector where each entry is the mean intensity of the gradient
    over a specific feature map channel
    """
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    """
    We multiply each channel in the feature map array
    by "how important this channel is" with regard to the top predicted class
    then sum all the channels to obtain the heatmap class activation
    """
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap


def save(heatmap, out_dir, img1, img2, alpha, inputs, slice_n, conv, ID, zoomfactor):
    
    # create foler to save files based on ID and case type
    folder = str(ID) + '_' + str(inputs)
    save_dir = os.path.join(out_dir, folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Rescale heatmap to a range 0-255
    #print(heatmap.shape)
    heatmap = heatmap[slice_n, :, :]
    heatmap = np.uint8(255 * heatmap)
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    #print(jet_heatmap.shape)
    # resize heatmap
    heatmap_img0 = keras.preprocessing.image.array_to_img(jet_heatmap)
    heatmap_img1 = heatmap_img0.resize((32, 32))
    if inputs == 'bbox':
        heatmap_img1 = heatmap_img0.resize((118, 118))
        img = img1
        img = img.reshape((32, 118, 118))
    elif inputs == 'small':
        heatmap_img1 = heatmap_img0.resize((118,118))
        img = img2
        #print(img.shape)
        img = img.reshape((32, 32, 32))
    heatmap_arr = keras.preprocessing.image.img_to_array(heatmap_img1)
    # resize arr to match heatmap
    img = zoom(img, (zoomfactor, 1, 1))
    img = img[slice_n, :, :]
    img = np.repeat(img[..., np.newaxis], 3, axis=-1)
    #print('img shape:', img.shape)
    CT = keras.preprocessing.image.array_to_img(img)
    
    # Superimpose the heatmap on original image
    gradcam = heatmap_arr * alpha + img
    gradcam = keras.preprocessing.image.array_to_img(gradcam)

    # Save the superimposed image
    fn_gradcam = 'GradCAM_' + str(ID) + '_' + str(inputs) + '_' + \
                 str(conv) + '_' + str(slice_n) + '.png'
    fn_CT = 'CT_' + str(ID) + '_' + str(inputs) + '_' + \
            str(conv) + '_' + str(slice_n) + '.png'
    fn2 = 'heatmap.png'
    fn3 = 'heatmap_raw.png'
    gradcam.save(os.path.join(save_dir, fn_gradcam))
    #heatmap_img1.save(os.path.join(out_dir, 'heatmap.png'))
    #heatmap_img0.save(os.path.join(out_dir, 'heatmap_raw.png'))
    CT.save(os.path.join(save_dir, fn_CT))


if __name__ == '__main__':
    
    proj_dir = '/mnt/aertslab/USERS/Ben/E3311_gradCAM'
    out_dir = '/mnt/aertslab/USERS/Ben/E3311_gradCAM/output'
    fn1 = 'E3311_E3311-33015_1_testcase.nrrd'
    fn2 = 'E3311_E3311-33015_1_small_testcase.nrrd'
    saved_model = '1.h5'
    alpha = 0.005
    inputs = 'bbox'

    if inputs == 'bbox':
        ID = fn1.split('-')[1].split('_t')[0].strip()
    elif inputs == 'small':
        ID = fn2.split('-')[1].split('_s')[0].strip()
    
    img1, img2 = get_data(
        proj_dir=proj_dir,
        filename1=fn1,
        filename2=fn2)

    for conv in ['conv2', 'conv3a', 'conv3b', 'conv4a', 'conv4b']:
        if conv == 'conv2s':
            zoomfactor = 0.5
            slice_range = 15
        elif conv == 'conv3as':
            zoomfactor = 0.25
            slice_range = 8
        elif conv == 'conv3bs':
            zoomfactor = 0.25
            slice_range = 8
        elif conv == 'conv4as':
            zoomfactor = 0.125
            slice_range = 4
        elif conv == 'conv4bs':
            zoomfactor = 0.125
            slice_range = 4
        if conv == 'conv2':
            zoomfactor = 0.5
            slice_range = 15
        elif conv == 'conv3a':
            zoomfactor = 0.25
            slice_range = 8
        elif conv == 'conv3b':
            zoomfactor = 1/8
            slice_range = 4
        elif conv == 'conv4a':
            zoomfactor = 1/16
            slice_range = 2
        elif conv == 'conv4b':
            zoomfactor = 1/16
            slice_range = 2

        print(conv)
        heatmap = compute_heatmap(
            proj_dir=proj_dir,
            saved_model=saved_model,
            img1=img1,
            img2=img2,
            conv=conv)
        
        for slice_n in range(slice_range):
            print(slice_n)
            save(
                heatmap, 
                out_dir=out_dir, 
                img1=img1, 
                img2=img2, 
                alpha=alpha, 
                inputs=inputs, 
                slice_n=slice_n,
                conv=conv,
                ID=ID,
                zoomfactor=zoomfactor)





#    parser = argparse.ArgumentParser()
#    parser.add_argument('--proj_dir', default='/mnt/aertslab/USERS/Ben/E3311_gradCAM',
#                        type = str, help = 'proj path')
#    parser.add_argument('--out_dir', default='/mnt/aertslab/USERS/Ben/E3311_gradCAM/output',
#                        type = str, help = 'output path')
#    parser.add_argument('--fn1', default='E3311_E3311-33011_1_testcase.nrrd',
#                        type = str, help = 'filename of CT scan')
#    parser.add_argument('--fn2', default='E3311_E3311-33011_1_small_testcase.nrrd',
#                        type = str, help = 'file name')
#    parser.add_argument('--saved_model', default='1.h5', type = str, help = 'filename of CT scan')
#    parser.add_argument('--alpha', default=0.005, type = str, help = 'alpha')
#    parser.add_argument('--inputs', default='small', type = str, help = 'inputs (small | bbox)')
#    parser.add_argument('--conv', default='conv3as', type = str, help = 'conv layer')
#    parser.add_argument('--ID', default='33011_1', type = str, help = 'case ID')
#
#    args = parser.parse_args()
