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
    # @param cached            Whether to load all images into RAM at once
    def __init__(self, imageSize=None, trainXPattern="./traindata/x/*", trainYPattern="./traindata/y/*", testXPattern="./testdata/x/*", testYPattern="./testdata/y/*", cached=False):

        self.imageSize = imageSize
        self.trainXPattern = trainXPattern
        self.trainYPattern = trainYPattern
        
        self.testXPattern = testXPattern
        self.testYPattern = testYPattern

        self.cached=cached

        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        
        self.batchIndex = 0


    # @brief                Loads training data
    def loadTrainData(self):
        if(self.cached):
            self.trainX = loadImagesFromPattern(self.imageSize, self.trainXPattern)
            self.trainY = loadImagesFromPattern(self.imageSize, self.trainYPattern)
        else:
            self.trainX = np.array(glob(self.trainXPattern))
            self.trainY = np.array(glob(self.trainYPattern))

    # @brief                Loads test data
    def loadTestData(self):
        if(self.cached):
            self.testX = loadImagesFromPattern(self.imageSize, self.testXPattern)
            self.testY = loadImagesFromPattern(self.imageSize, self.testYPattern)
        else:
            self.testX = np.array(glob(self.testXPattern))
            self.testY = np.array(glob(self.testYPattern))

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
            if(self.cached):
                batch_X = self.trainX[[i % self.trainX.shape[0] for i in ind]]
                batch_Y = self.trainY[[i % self.trainY.shape[0] for i in ind]]
            else:
                batch_X_files = self.trainX[[i % self.trainX.shape[0] for i in ind]]
                batch_Y_files = self.trainY[[i % self.trainY.shape[0] for i in ind]]
                batch_X = loadImagesFromList(self.imageSize, batch_X_files)
                batch_Y = loadImagesFromList(self.imageSize, batch_Y_files)
        return batch_X, batch_Y

    # @brief                Returns matrix containing test data
    #
    # @param size           Max number of examples to return
    #
    # @return X             Matrix containing test examples
    # @return Y             Matrix containing test labels
    def getTestData(self, size):
        if(self.cached):
            X = self.testX[0:max(size,self.testX.shape[0])]
            Y = self.testY[0:max(size,self.testY.shape[0])]
        else:
            X = loadImagesFromList(self.imageSize, self.testX[0:min(size,self.testX.shape[0])])
            Y = loadImagesFromList(self.imageSize, self.testY[0:min(size,self.testY.shape[0])])
        self.testX = np.delete(self.testX, range(0, min(size, self.testX.shape[0])), 0)
        self.testY = np.delete(self.testY, range(0, min(size, self.testY.shape[0])), 0)
        return X, Y


# @brief                Sets working directory to path
#
# @param path           Path to become new working directory
def setPath(self, path):
    if os.path.isdir(path):
        os.chdir(path)

# @brief                Loads all images from a list of filenames
#
# @param imageSize      If not none, images will be resized to this size
# @param filenames      List of filenames
def loadImagesFromList(imageSize=None, filenames=[]):
    data = None
    if(len(filenames) != 0):
        if(imageSize is not None):
            data = np.stack([np.array(Image.open(f).resize(imageSize, Image.NEAREST)) for f in filenames])
        else:
            data = np.stack([np.array(Image.open(f)) for f in filenames])
    return data

# @brief                Loads all images that match a pattern, relative to current working directory
# 
# @param imageSize      images will be resized to this size
# @param pattern        List of patterns to match filenames
def loadImagesFromPattern(imageSize, pattern="./*"):
    files = glob(pattern)
    return loadImagesFromList(imageSize, files)

# @brief                Loads all images from a given directory
# 
# @param imageSize      Images will be resized to this size
# @param directory      Directory to load from
def loadImagesFromDir(imageSize, directory="./"):
    if(directory[-1] != "/"):
        directory = directory + "/"
    return loadImagesFromPattern(imageSize, directory+"*")

# @brief                Utility function that converts images to matrix of one-hot vectors
#
# @param images
# @param numClasses     Number of classes/length of one-hot vector
def oneHot(images, numClasses=None):
    if(numClasses is None):
        numClasses = images.max()
    return np.stack([(np.arange(numClasses) == x[:,:,None]-1).astype(int) for x in images])
