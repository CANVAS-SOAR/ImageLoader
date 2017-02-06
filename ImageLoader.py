import numpy as np
import os
from PIL import Image

# @brief Class to load images for network training

class ImageLoader:
    # @brief                ImageLoader constructor
    # 
    # @param path           Path to use as working directory
    # @param trainXPath     Path to raw training data
    # @param trainYPath     Path to training data labels
    # @param testXPath      Path to test data
    # @param testYPath      Path to test data labels
    def __init__(self, imageSize=(1392,512), trainXPath="./traindata/x/", trainYPath="./traindata/y/", testXPath="./testdata/x/", testYPath="./testdata/y/"):

        self.imageSize = imageSize
        self.trainXPath = trainXPath
        self.trainYPath = trainYPath
        
        self.testXPath = testXPath
        self.testYPath = testYPath

        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        
        self.batchIndex = 0


    # @brief                Loads training data
    def loadTrainData(self):
        if os.path.isdir(self.trainXPath):
            self.trainX = loadImagesFromDir(self.imageSize, self.trainXPath)

        if os.path.isdir(self.trainYPath):
            self.trainY = loadImagesFromDir(self.imageSize, self.trainYPath)

    # @brief                Loads test data
    def loadTestData(self):
        if os.path.isdir(self.testXPath):
            self.testX = loadImagesFromDir(self.imageSize, self.testXPath)
        if os.path.isdir(self.testYPath):
            self.testY = loadImagesFromDir(self.imageSize, self.testYPath)

    # @brief                Returns a batch of size training samples
    #
    # @param size           Number of examples to include in batch
    # @param rand           If True, randomize batch
    #
    # @return batch_X       Batch of training examples
    #         batch_Y       Batch of training labels
    def getNextBatch(self, size, rand=False):
        batch_X = None;
        batch_Y = None;
        if (self.trainX is not None) and (self.trainY is not None):
            if(rand):
                ind = np.random.randint(self.trainX.shape[0], size=size)
            else:
                ind = range(self.batchIndex, self.batchIndex + size)
                self.batchIndex = (self.batchIndex + size) % self.trainX.shape[0]
            batch_X = self.trainX[[i % self.trainX.shape[0] for i in ind]]
            batch_Y = self.trainY[[i % self.trainY.shape[0] for i in ind]]
        return batch_X, batch_Y

    # @brief                Returns matrix containing test data
    #
    # @return X             Matrix containing test examples
    def getTestData(self):
        return self.testX, self.testY



# @brief                Sets working directory to path
#
# @param path           Path to become new working directory
def setPath(self, path):
    if os.path.isdir(path):
        os.chdir(path)

# @brief                Loads all images from a given directory
# 
# @param path           Path from which to load images
def loadImagesFromDir(imageSize, path="./"):
    data = None
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        data = np.stack([np.array(Image.open(os.path.join(path,f)).resize(imageSize, Image.NEAREST)) for f in files])
    return data
