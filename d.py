
# coding: utf-8

# # Linear Classifier

# In[1]:


from PIL import Image
import numpy as np


# In[3]:


trainFile = 'train_sample2.txt'
testFile = 'train_sample.txt'


# In[4]:


labels = {}
images_labels = []
images = []
labelinv = {}

idx = 0

with open(trainFile, 'r') as f:
    for x in f:
        x = x.strip()
        if x:
            filePath, label = x.split(' ')
            if label not in labels:
                labels[label] = idx
                labelinv[idx] = label
                idx = idx + 1
            images_labels.append(labels[label])
            images.append(filePath)


# In[5]:


def pca(images):
    A = np.zeros((len(images),128*128*3),dtype='float64')
    for idx, img in enumerate(images):
        im = Image.open(img).resize((128, 128), Image.ANTIALIAS)
        im = np.array(im).reshape(-1)
        A[idx,:] = im/255.0
    A_mean = np.mean(A, axis=0)
    A = A - A_mean

    covM = np.matmul(A, A.T)
    # can also use eigh as it is faster and returns the eigenvalues in sorted order (it works only for symmetric matrices, A x A^T is symmetric)
    eVals, eVecs = np.linalg.eig(covM)
    idx = eVals.argsort()[::-1]
    eVals = np.real(eVals[idx])
    eVecs = np.real(eVecs[:,idx])

    eVecs = np.matmul(A.T, eVecs)
    eVecsNorm = np.sqrt(np.sum(eVecs*eVecs, axis=0))
    actualEVecs = eVecs/eVecsNorm

    return (actualEVecs, A, A_mean)


# In[6]:


eVecs, A, toAdd = pca(images)
curEVecs = eVecs[:,0:32]
A_new = np.matmul(curEVecs.T, A.T).T
A_new.shape


# In[7]:


y = np.array(images_labels)


# In[8]:


def softMax(x):
    # Subtract max(x) to compute the softmax of vector x in a numerically stable way
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis = 0)


# In[9]:


W = np.random.rand(len(labels), 32)

eta = 0.005
numIterations = 60000

for i in range(0, numIterations):
    wtx_labels = softMax(np.matmul(W, A_new.T))
    wtx_labels[y[:], np.arange(len(images))] += -1.0
    DSoftMax = (np.matmul(wtx_labels, A_new) / float(len(images))) + 0.0001 * W
    W = W - eta*(DSoftMax)


# In[10]:


wtx_labels.shape


# In[11]:


correct = 0
total = 0

with open(testFile, 'r') as f:
    for x in f:
        x = x.strip()
        if x:
            filePath = x.split()[0]
            im = Image.open(filePath).resize((128, 128), Image.ANTIALIAS)
            im = np.array(im).reshape(-1)
            im = im/255.0
            im = im - toAdd
            im = np.matmul(curEVecs.T, im).T
            print(labelinv[np.argmax(softMax(np.matmul(W, im)))])
            total = total + 1
            if labelinv[np.argmax(softMax(np.matmul(W, im)))] == filePath.split('/')[-1].split('_')[0]:
                correct = correct + 1


# In[12]:


# print(correct/float(total)*100)
#
#
# # In[ ]:
#
#
# import os
# whole_data = os.listdir('./dataset')
# total = 0
# correct = 0
#
# for file in whole_data:
#     img = read_image('./dataset/' + file)
#     img = img.flatten()
#     pro_img = np.matmul(projection, img.T)
#     print(labelinv[np.argmax(softMax(np.matmul(W, im)))])
#
#
# # In[ ]:
#
#
# len(labelinv)
#
#
# # In[20]:
#
#
# labelinv
