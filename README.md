# About the project

This is a simple object-oriented C++ implementation of neural networks. Up to now, the following layer types are implemented

* Vector/Matrix input layer
* Fully-connected layer
* Classification layer (SoftMax)
* Convolutional layer
* Pooling layer
* Flattening layer

With a few lines of code one can set up and train models for prediction, classification and image recognition problems.

# Get stated

## Dependencies

The following packages are required:

* libpng
* blas
* cblas (if not included in blas)

## Compilation

To build the library and all test examples simplfy type

```
make
```

To run the *DigitRecognition* example some external files (MNIST data files) have to be downloaded. This can be done with

```
make ressouces
```

## Usage

See the example files in the directory `/tests`
