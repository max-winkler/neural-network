#ifndef _MNIST_H_
#define _MNIST_H_

#include <string>
#include <vector>

#include "TrainingData.h"

/**
 * Class containing static method for reading MNIST data files like the 10-digit database.
 */
class MNIST
{
public:
  /**
   * Read a pair of MNIST image and label file and builds a vector of training data that can be used in 
   * NeuralNetwork::train() to train a neural network.
   *
   * @brief Read MNIST image and label file.
   *
   * @param image_file Filename of the image file (e.g. train-images-idx3-ubyte).
   * @param label_file Filename of the label file (e.g. train-labels-idx1-ubyte).
   * @param n_images Used to store the number of images that were in the data file.
   * @param width The width of one image.
   * @param height The height of one image.
   * @param n_classes Number of different classes found in the data file.
   * @param training_data Vector of training data (pair of input and labels).
   */
  static int read(const std::string& image_file, const std::string& label_file,
                  int& n_images, int& width, int& height, int& n_classes,
                  std::vector<TrainingData>& training_data);
private:
  /// Helper method to bit-wise revert an int.
  static int reverseInt (int i);
};

#endif
