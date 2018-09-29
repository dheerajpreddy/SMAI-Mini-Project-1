
# coding: utf-8

# In[1]:


import sys, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def read_image(im) :

    # Opening image as greyscale
    img = Image.open(im).convert('L')
    img.load()

    # Converting image into a numpy array
    data = np.asarray(img, dtype="int32")

    return data


# In[3]:


def PCA_decompose(img, num_comp):

    # Find mean and center the image
    centered = (img - np.mean(img.T, axis=1)).T

    # Find covariance matrix
    cov = np.cov(centered)

    # Eigen decomposition, get eigenvalues and eigenvectors
    [eig_val, eig_vec] = np.linalg.eig(cov)

    # Getting top num_comp eigen vectors
    if num_comp>=0 and num_comp<=256:
        eig_vec = eig_vec[:,range(num_comp)]

    # Finally, finding projection of the image
    projection = np.dot(eig_vec.T, centered)

    # Returning eigenvalues, eigenvectors, and projection
    return eig_val, eig_vec, projection


# In[27]:


def read_train_images():
    images = []
    with open("./sample_train.txt") as f:
            for line in f:
                images.append(line.split()[0])
    return images


# In[53]:


def find_MSE():

    # Specifying directory where files are stored
#     train_dir = os.listdir('./dataset')
    train_dir = read_train_images()
    # Initializing Squared Error list to zero matrices
#     MSE = [ np.zeros((256, 256)) for x in range(0, 10)]
    MSE = np.zeros(60)

    # going through each file
    for x in train_dir:
#         print(x)
        A = read_image(x)
        # Only taking first 60 eigenvalues and eigenvectors since rest are close to 0
        for i in range(0, 60):
            val, vec, pro = PCA_decompose(A, i)
            Ar = np.dot(vec,pro).T + np.mean(A,axis=0)
            MSE[i] = MSE[i] + np.mean(np.square(A - Ar))
            print(i, np.mean(np.square(A-Ar)))
    return MSE


# In[54]:


MSE = find_MSE()


# In[55]:


plt.plot(np.arange(60), MSE)
plt.show()

# In[60]:


# A = read_image('./dataset/000_010.jpg')
# val, vec, pro = PCA_decompose(A, 10)
# Ar = np.dot(vec,pro).T + np.mean(A,axis=0)
# MSE = np.mean(np.square(A-Ar))
# MSE
