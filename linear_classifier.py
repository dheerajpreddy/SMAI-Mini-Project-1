import os, sys
from PIL import Image
import numpy as np


class Metadata:
    def __init__(self):
        self.class2id   = {}
        self.im2label   = []
        self.img_dir    = []
        self.label2id   = {}

m = Metadata()

trainFile = sys.argv[1]
testFile = sys.argv[2]

m.class2id = {}
m.im2label = []
m.img_dir = []
m.label2id = {}

idx = 0

with open(trainFile, 'r') as f:
    for x in f:
        x = x.strip()
        filePath, label = x.split(' ')
        if label not in m.class2id:
            m.class2id[label] = idx
            m.label2id[idx] = label
            idx = idx + 1
        m.im2label.append(m.class2id[label])
        m.img_dir.append(filePath)

def pca(images):
    A = None
    idx = 0
    while idx < len(images):
        img = Image.open(images[idx]).resize((128, 128), Image.ANTIALIAS)
        img = np.array(img).reshape(-1)
        if A is None:
            A = img/255.0
        else:
            A = np.vstack([A, img/255.0])
        idx = idx + 1
    A_mean = np.mean(A, axis=0)
    A = A - A_mean
    cov = np.cov(A)
    val, vec = np.linalg.eig(cov)

    vec = np.dot(A.T, vec)
    vecNorm = np.sqrt(np.sum((vec*vec).T, axis=1))
    vecs = vec/vecNorm
    return (vecs, A, A_mean)

vec, A, mean = pca(m.img_dir)
A_new = np.matmul(vec[:,0:32].T, A.T).T

y = np.array(m.im2label)

def find_prob(x):
    exps = np.exp(x - np.max(x))
    part1 = exps
    part2 = np.sum(part1, axis = 0)
    return part1 / part2

W = np.random.rand(len(m.class2id), 32)
eta = 0.001
numIterations = 60000
for i in range(numIterations):
    wtx = find_prob(np.matmul(W, A_new.T))
    wtx[y[:], np.arange(len(m.img_dir))] -= 1.0
    dJ = (np.matmul(wtx, A_new) / float(len(m.img_dir))) + 0.0001 * W
    W = W - eta*(dJ)


with open(testFile, 'r') as myfile:
    data=myfile.read().split('\n')

for line in data:
    if file == "":
        break
    # print(line)
    # line = line.split()[0]
    try:
        im = Image.open(line).resize((128, 128), Image.ANTIALIAS)
        im = np.array(im.resize((128,128), Image.ANTIALIAS)).reshape(-1)/float(2**8 - 1) - mean
        # im = im - mean
        im = np.dot(vec[:,0:32].T, im.T)
        print(m.label2id[np.argmax(find_prob(np.matmul(W, im)))])

    except:
        break
