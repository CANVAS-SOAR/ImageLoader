# ImageLoader
ImageLoader Class for Network Training

#Goal:

To create a class capable of loading test and training data.

Folder Layout:

Broad description of folder layouts for all projects

.../path/testdata/x/		Folder containing all raw data for testing
.../path/testdata/y/		Folder containing all testing truth values
.../path/traindata/x/		Folder containing all raw data for training
.../path/traindata/y/		Folder containing all training truth values

#Class:

ImageLoader

#Usage: 

`from ImageLoader import ImageLoader`

#Member Variables:

`trainX`: Matrix of training data
`trainY`: Matrix of training ground truth data
`TestX`: Matrix of test data
`TestY`: Matrix of test ground truth data

#Methods:

`__init__()`: Default constructor

`setPath(path)`: Sets path that user is working from, user current working directory

`loadTrainData()`: Loads training data from `./traindata/` and fills numpy matrices accessible by `getNextBatch()`

`loadTestData()`: Loads test data from `./testdata/` and fills numpy matrices accessible by `getTestData()`

`getNextBatch(size, rand=False)`: Returns batch of `size` training examples. Randomized if `rand` is `True`

`getTestData()`: Returns test data matrices
