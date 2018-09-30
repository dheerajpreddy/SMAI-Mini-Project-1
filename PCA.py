
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


def PCA_decompose(images, num_comp):

    # # Find mean and center the image
    # centered = (img - np.mean(img.T, axis=1)).T
    #
    # # Find covariance matrix
    # cov = np.cov(centered)
    #
    # # Eigen decomposition, get eigenvalues and eigenvectors
    # [eig_val, eig_vec] = np.linalg.eig(cov)
    #
    # # Getting top num_comp eigen vectors
    # if num_comp>=0 and num_comp<=256:
    #     eig_vec = eig_vec[:,range(num_comp)]
    #
    # # Finally, finding projection of the image
    # projection = np.dot(eig_vec.T, centered)
    #
    # # Returning eigenvalues, eigenvectors, and projection
    # return eig_val, eig_vec, projection

    M = np.mean(images.T, axis=1)
    # print(M)
    # center columns by subtracting column means
    C = images - M
    # print(C)
    # calculate covariance matrix of centered matrix
    V = np.matmul(C, C.T)
    # print(V)
    # eigendecomposition of covariance matrix
    values, vectors = np.linalg.eig(V)
    vectors = vectors[:,range(num_comp)]
    # print(vectors)
    # print(values)
    # project data
    P = vectors.dot(C)
    # print(P.T)
    return values, vectors, P

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
    train_dir = os.listdir('./dataset')
    # train_dir = read_train_images()
    images = None
# images = np.asarray(images)
    MSE = np.zeros(60)
    for x in train_dir:
        img = read_image('./dataset/' + x)
        img = img.flatten()
        if images is None:
            images = img
        else:
            images = np.vstack([images, img])
    # Initializing Squared Error list to zero matrices
#     MSE = [ np.zeros((256, 256)) for x in range(0, 10)]
    MSE = np.zeros(60)

    # going through each file
#     for x in train_dir:
# #         print(x)
#         A = read_image(x)
#         # Only taking first 60 eigenvalues and eigenvectors since rest are close to 0
#         for i in range(0, 60):
#             val, vec, pro = PCA_decompose(A, i)
#             Ar = np.dot(vec,pro).T + np.mean(A,axis=0)
#             MSE[i] = MSE[i] + np.mean(np.square(A - Ar))
#             print(i, np.mean(np.square(A-Ar)))
#     return MSE
    val, vec, pro = PCA_decompose(images, 32)
    Ar = np.dot(vec,pro).T + np.mean(images,axis=0)
    Ar.shape



# In[54]:


MSE = find_MSE()


# In[55]:


# plt.plot(np.arange(60), MSE)
# plt.show()

# In[60]:


# A = read_image('./dataset/000_010.jpg')
# val, vec, pro = PCA_decompose(A, 10)
# Ar = np.dot(vec,pro).T + np.mean(A,axis=0)
# MSE = np.mean(np.square(A-Ar))
# MSE
