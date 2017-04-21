# ImageLoader
ImageLoader Class for Network Training

# Goal:

To create a class capable of loading test and training data.

# Setup:

`pip3 install -r requirements.txt`

# Usage: 

`from ImageLoader import ImageLoader`

# Methods:

`__init__(self, imageSize=None, trainXPattern="./traindata/x/*", trainYPattern="./traindata/y/*", testXPattern="./testdata/x/*", testYPattern="./testdata/y/*", cached=False)` : ImageLoader constructor

All parameters are optional, and are understood as follows:

`imageSize`: (width, height) tuple specifying the desired image dimensions. Images will be resized to these dimensions. If None, images will not be resized.

`trainXPattern`: Pattern matching training data. For example, the value `"./traindata/x/*.png"` will select all files ending with `.png` within the folder `./traindata/x/`

`trainYPattern`: Pattern matching training data label images

`testXPattern`: Pattern matching test data

`testYPattern`: Pattern matching test data labels

`Cached`: Boolean value, whether to load all images at once. For small datasets, it may be beneficial to set this to `True`. For large datasets, setting this to `False` will conserve memory.

`loadTrainData()`: Loads training data from specified patterns. After this function is called, data will be available via `getNextBatch()`

`loadTestData()`: Loads test data from specified patterns. After this function is called, data will be available via `getTestData()`

`getNextBatch(size, rand=False)`: Returns batch of `size` training examples. Randomized if `rand` is `True`

`getTestData()`: Returns test data matrices

# Utility functions:

`loadImagesFromList(imageSize=None, filenames=[])`: Loads images from a list of filenames. Resized to `imageSize` if not `None`

`loadImagesFromPattern(imageSize=None, pattern="./*")`: Loads all images that match a given pattern. Resized to `imageSize` if not `None`

`loadImagesFromDir(imageSize=None, directory="./")`: Loads all images from a given directory. Resized to `imageSize` if not `None`

`oneHot(images, numClasses=None)`: Converts images to matrix of one-hot vectors. May be useful for ground truth values in softmax networks. `numClasses` specifies the length of each one-hot vector. If `None`, `numClasses` is estimated by using the max value.
