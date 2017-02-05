# ImageLoader
ImageLoader class for network training

Data Parsing Design Layout

Goal:

To create a class capable of loading test and training data.

Folder Layout:

Broad description of folder layouts for all projects

.../path/testdata/x/		Folder containing all raw data for testing
.../path/testdata/y/		Folder containing all testing truth values
.../path/traindata/x/		Folder containing all raw data for training
.../path/traindata/y/		Folder containing all training truth values

Class:

ImageLoader

Usage: 

from imageloader import ImageLoader
	
Structure:

loader = ImageLoader(path);

Member Variables:

trainX //Matrix of training data
trainY //Matrix of training ground truth
TestX //Matrix of test data
TestY //Matrix of 

Methods:

ImageLoader() //Default constructor
ImageLoader(path) //Constructor that also sets path

setPath(path)	// set path that user is working from, user current working directory

loadTrainData() //Load data from path fill numpy matrices accessible by getNextBatch()

loadTestData()  // store test data in numpy matrices

batch_x, batch_y = getNextBatch(rand=False)      // Functionality of getting random or sequential

X, Y = getTestData() 		// return x and y test data matrices
