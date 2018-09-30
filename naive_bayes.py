
# coding: utf-8

# # Question 2

# ## Importing packages

# In[1]:


import os
import sys
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ## Read Image
trainFile = sys.argv[1]
testFile = sys.argv[2]

# In[2]:


def read_image(im) :

    # Opening image as greyscale
    img = Image.open(im).convert('L')
    img.load()

    # Converting image into a numpy array
    data = np.asarray(img, dtype="int32")

    return data


# ## PCA Decompose

# In[3]:


def PCA_decompose(images, num_comp):

    # Find mean of the images
    mean = np.mean(images.T, axis=1)

    # Center the matrix by subtracting the mean
    centered = images - mean

    # Find covariance matrix
    cov = np.matmul(centered, centered.T)

    # Retrieving the eigenvalues and eigenvectors
    values, vectors = np.linalg.eig(cov)
    idx = np.argsort(values) # sorting the eigenvalues
    idx = idx[::-1]       # in ascending order
    # sorting eigenvectors according to the sorted eigenvalues
    vectors = vectors[:,idx]
    values = values[idx]
    # Reducing number of vectors based on num_comp
    vectors = vectors[:,range(num_comp)]

    # Finding projection matrix
    projection = np.dot(vectors.T, centered)

    return values, vectors, projection


# ## Calculate multivariate probability

# In[4]:


def calc_probability(x, mu, cov):
#     part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * 1)
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))


# ## Read Training Directory

# In[5]:


# Read train file
def read_train_file():
    training_directory = []
    with open(trainFile) as f:
            for line in f:
                training_directory.append(line.split())
#     while len(training_directory) < 32:
#         training_directory.append(random.choice(training_directory))
    return training_directory


# ## Combine Images into one vector

# In[6]:


def combineImages(images):
    # Getting all images and then adding them to one single numpy array
    data = None
    MSE = np.zeros(60)
    for x in images:
        img = read_image(x)
        img = img.flatten()
        if data is None:
            data = img
        else:
            data = np.vstack([data, img])
    i = 0
#     while data.shape[0] < 32:
#         data = np.vstack([data, data[np.random.choice(data.shape[0], 1, replace=False), :]])
    return data


# ## Separate Images by class

# In[7]:


def separateClasses(images):
    # Separate labels
    separated = {}
    for i in range(len(images)):
        vector = images[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector[0])
    return separated


# ## Find Mean

# In[8]:


def mean(images):
    mean = np.mean(images.T, axis=1)
    return mean


# ## Find Covariance

# In[9]:


def covariance(images):
    # Find mean of the images
    mean = np.mean(images.T, axis=1)

    # Center the matrix by subtracting the mean
    centered = images - mean

    # Find covariance matrix
    cov = np.cov(centered)

    return cov


# In[10]:


# main function
training_directory = read_train_file()
separated = separateClasses(training_directory)
dataList = []
for k, v in separated.items():
    for x in v:
        dataList.append(x)
data = combineImages(dataList)
if len(separated) < 32:
    numComp = len(separated)
else:
    numComp = 32
values, vectors, projection = PCA_decompose(data, numComp)

classes_char = [ [0, 0] for i in range(len(separated))]


# In[11]:


classes_matrices = {}
for k, v in separated.items():
#     classes_matrices.append(None)
    for x in v:
        img = read_image(x)
        img = img.flatten()
        if k not in classes_matrices:
            classes_matrices[k] = projection.dot(img.T)
        else:
            classes_matrices[k] = np.vstack([classes_matrices[k], projection.dot(img.T)])


# In[12]:


classes_char = {}
for k, v in classes_matrices.items():
    # find mean
    t = dict()
    t['mu'] = mean(v)
    centered = v - t['mu']
    # find covariance
    t['sigma'] = np.cov(centered.T)
    classes_char[k] = t
# classes_char


# In[13]:


total = 0
correct = 0
with open(testFile, 'r') as myfile:
    data=myfile.read().split('\n')

for file in data:
#     line = line.split('\n')
#     file = entire_line[0]
#     #                 label = entire_line[1]
    if file == "":
        break
#     file = file.split()[0]
    img = read_image(file)
    img = img.flatten()
    pro_img = np.matmul(projection, img.T)
    maxp = -9999999
    maxc = ""
    sump = 0
    for k, v in classes_char.items():
        v['prob'] = calc_probability(pro_img, v['mu'], v['sigma'])
        if v['prob']>maxp:
            maxp = v['prob']
            maxc = k
    print(maxc)
#                 if maxc == label:
#                     correct = correct + 1
#                 total = total + 1
# (correct/total)*100
