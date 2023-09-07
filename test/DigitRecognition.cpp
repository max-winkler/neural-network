#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

#include "Matrix.h"
#include "TrainingData.h"
#include "NeuralNetwork.h"

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int read_training_data(const char* image_file, const char* label_file,
		       int& n_images, int& width, int& height,
		       std::vector<TrainingData>& training_data)
{
  std::ifstream image_data(image_file, std::ios::binary);
  std::ifstream label_data(label_file, std::ios::binary);

  if(!image_data.is_open() || !label_data.is_open())
    {
      std::cerr << "Unable to open file.\n";
      return -1;
    }      

  // Read meta data from image file
  uint32_t magic_number;
  
  image_data.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));  
  image_data.read(reinterpret_cast<char*>(&n_images), sizeof(n_images));
  image_data.read(reinterpret_cast<char*>(&height), sizeof(height));
  image_data.read(reinterpret_cast<char*>(&width), sizeof(width));
  
  magic_number = reverseInt(magic_number);
  n_images = reverseInt(n_images);
  height = reverseInt(height);
  width = reverseInt(width);  

  const unsigned pixels = width*height;
  
  // Read meta data from label file
  uint32_t magic_number2, n_images2;
  label_data.read(reinterpret_cast<char*>(&magic_number2), sizeof(magic_number2));
  label_data.read(reinterpret_cast<char*>(&n_images2), sizeof(n_images2));

  magic_number2 = reverseInt(magic_number2);
  n_images2 = reverseInt(n_images2);
  
  if(n_images != n_images2)
    {
      std::cerr << "ERROR: Image data and label data do not fit together.\n";
      return -1;
    }   

  // Create Training data array
  training_data.reserve(n_images);
  
  // Read pixel data of images
  unsigned char pixel_buffer[pixels];
  unsigned char label_buffer[1];
  
  for(int m=0; m<n_images; ++m)
    {
      if(image_data.peek()==EOF || label_data.peek()==EOF)
	{
	  std::cerr << "ERROR: Reached end of data file, but could not read all images.\n";
	  return -1;
	}
      
      // Read pixel data
      image_data.read((char*)pixel_buffer, pixels*sizeof(char));

      /*
      // Console output for testing only
      for(int i=0; i<height; ++i)
        {
	for(int j=0; j<width; ++j)
	  std::cout << std::setw(4) << (int)(pixel_buffer[width*i+j]);
	std::cout << std::endl;
        }
      */
      
      // Read label
      label_data.read((char*)label_buffer, sizeof(char));

      /*
      // Console output for testing only
      std::cout << "This should be number " << (int)(*label_buffer) << std::endl;
      */
      
      // Create training dataset
      Matrix x(width, height, pixel_buffer);
      Vector y(10);
      int label = (int)(*label_buffer);

      if(label < 0 || label > 9)
	{
	  std::cerr << "ERROR: Invalid label in training data detected.\n";
	  return -1;
	}
      
      y[label] = 1.;

      // TODO: Training data is copied here. Use move semantics instead
      TrainingData data(x, y);
      training_data.push_back(data);
    }
  
  image_data.close();
  label_data.close();

  return 0;
}

int main()
{
  // Read training data
  int width, height, n_images;
  std::vector<TrainingData> training_data;
  
  if(read_training_data("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte",
			n_images, width, height, training_data) != 0)    
    return -1;

  // Console output
  std::cout << "Data set:\n";
  std::cout << " images : " << n_images << std::endl;
  std::cout << " width  : " << width << std::endl;
  std::cout << " height : " << height << std::endl;

  // Create neural network
  NeuralNetwork net;
  net.addInputLayer(width, height); // input layer
  net.addFlatteningLayer();
  net.addFullyConnectedLayer(64, ActivationFunction::RELU); // hidden layer
  net.addFullyConnectedLayer(64, ActivationFunction::RELU); // hidden layer
  net.addFullyConnectedLayer(32, ActivationFunction::RELU); // hidden layer
  net.addFullyConnectedLayer(32, ActivationFunction::SIGMOID); // hidden layer
  net.addClassificationLayer(10); // output layer
  
  net.initialize();

  std::cout << net;
  
  net.train(training_data);

  // Read training data
  std::vector<TrainingData> test_data;
  int n_test;

  if(read_training_data("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte",
		    n_test, width, height, test_data) != 0)    
    return -1;  

  // Console output
  std::cout << "Training set:\n";
  std::cout << " images : " << n_images << std::endl;
  
  // Compare to test data
  int correct = 0;
  int wrong   = 0;
  for(auto data = test_data.begin(); data != test_data.end(); ++data)
    {
      Vector p = net.eval(*(data->x));
      p.indMax() == data->y.indMax() ? correct++ : wrong++;      
    }
  std::cout << "Correctly classified : " << correct << " (" << (double)correct/n_test*100 << "%)\n";
  std::cout << "Wrongly classified   : " << wrong << " (" << (double)wrong/n_test*100 << "%)\n";
  
  return 0;
}
