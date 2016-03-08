# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from scipy.misc import imread
import operator
from numpy.random import normal

#TODO Switch from k-means clustering to http://keras.io/

rgb = {} #rgb[[0,0,0] = 3
n_colors = 30
for row in np.array(imread("marcus.jpg")):
    for color in row:
        key = ','.join([str(c) for c in color.tolist()])
        rgb[key] = rgb[key]+1 if key in rgb else 0

sorted_rgb = sorted(rgb.items(), key=operator.itemgetter(1))
for k,v in sorted_rgb:
    if v > 0:
        print k,"=",v

pic = np.array(imread("marcus.jpg"), dtype=np.float64) / 255
w, h, d = original_shape = tuple(pic.shape)
image_array = np.reshape(pic, (w * h, d))
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
labels = kmeans.predict(image_array)

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
plt.show()
