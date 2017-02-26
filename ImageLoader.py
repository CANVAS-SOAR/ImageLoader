import numpy as np
import os
from PIL import Image
from glob import glob

# @brief Class to load images for network training

class ImageLoader:
    # @brief                   ImageLoader constructor
    # 
    # @param path              Path to use as working directory
    # @param trainXPattern     Pattern matching training data
    # @param trainYPattern     Pattern matching training data labels
    # @param testXPattern      Pattern matching test data
    # @param testYPattern      Pattern matching test data labels
    def __init__(self, imageSize=(1392,512), trainXPattern="./traindata/x/*", trainYPattern="./traindata/y/*", testXPattern="./testdata/x/*", testYPattern="./testdata/y/*"):

        self.imageSize = imageSize
        self.trainXPattern = trainXPattern
        self.trainYPattern = trainYPattern
        
        self.testXPattern = testXPattern
        self.testYPattern = testYPattern

        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        
        self.batchIndex = 0


    # @brief                Loads training data
    def loadTrainData(self):
        self.trainX = loadImagesFromPattern(self.imageSize, self.trainXPattern)
        self.trainY = loadImagesFromPattern(self.imageSize, self.trainYPattern)

    # @brief                Loads test data
    def loadTestData(self):
        self.testX = loadImagesFromPattern(self.imageSize, self.testXPattern)
        self.testY = loadImagesFromPattern(self.imageSize, self.testYPattern)

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

# @brief                Loads all images that match a pattern, relative to current working directory
# 
# @param imageSize      images will be resized to this size
# @param pattern        List of patterns to match filenames
def loadImagesFromPattern(imageSize, pattern="./*"):
    data = None
    files = glob(pattern)
    data = np.stack([np.array(Image.open(f).resize(imageSize, Image.NEAREST)) for f in files])
    return data

# @brief                Loads all images from a given directory
# 
# @param imageSize      Images will be resized to this size
# @param directory      Directory to load from
def loadImagesFromDir(imageSize, directory="./"):
    if(directory[-1] != "/"):
        directory = directory + "/"
    return loadImagesFromPattern(imageSize, directory+"*")
