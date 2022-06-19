import skimage
import numpy as np
import imagej

ij = imagej.init(mode='interactive')
img = skimage.data.astronaut()
img = np.mean(img[10:190,140:310], axis=2)
java_img = ij.py.to_java(img)

dataset_2d = ij.io().open('test_image.tif')

ij.ui().show(dataset_2d)