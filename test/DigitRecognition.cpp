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
		       int& n_images, int& width, int& height, int& n_classes,
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

  // Determine number of classes
  n_classes = 0;
  for(int m=0; m<n_images; ++m)
    {
      label_data.read((char*)label_buffer, sizeof(char));
      int label = (int)(*label_buffer);
      
      if(label > n_classes)
	n_classes = label;
    }
  n_classes++;

  std::cout << "This dataset has " << n_classes << " different classes\n";
  // Read from beginning and reread first lines
  label_data.seekg(0, label_data.beg);
  label_data.read(reinterpret_cast<char*>(&magic_number2), sizeof(magic_number2));
  label_data.read(reinterpret_cast<char*>(&n_images2), sizeof(n_images2));
  
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
      Vector y(n_classes);
      int label = (int)(*label_buffer);

      if(label < 0 || label >= n_classes)
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
  int width, height, n_images, n_classes;
  std::vector<TrainingData> training_data;
  
  char training_data_file[]  = "mnist/train-images-idx3-ubyte";
  char training_label_file[] = "mnist/train-labels-idx1-ubyte";
  char test_data_file[]      = "mnist/t10k-images-idx3-ubyte";
  char test_label_file[]     = "mnist/t10k-labels-idx1-ubyte";

  /*
  char training_data_file[]  = "mnist/emnist-letters-train-images-idx3-ubyte";
  char training_label_file[] = "mnist/emnist-letters-train-labels-idx1-ubyte";
  char test_data_file[]      = "mnist/emnist-letters-test-images-idx3-ubyte";
  char test_label_file[]     = "mnist/emnist-letters-test-labels-idx1-ubyte";
  */
  
  if(read_training_data(training_data_file, training_label_file,
			n_images, width, height, n_classes, training_data) != 0)    
    return -1;

  // Console output
  std::cout << "Data set:\n";
  std::cout << " images : " << n_images << std::endl;
  std::cout << " width  : " << width << std::endl;
  std::cout << " height : " << height << std::endl;

  // Read test data
  std::vector<TrainingData> test_data;
  int n_test, n_classes_test;

  if(read_training_data(test_data_file, test_label_file,
		    n_test, width, height, n_classes_test, test_data) != 0)    
    return -1;  

  if(n_classes != n_classes_test)
    {
      std::cerr << "ERROR: Training and test dataset have different number of classes.\n";
      return -1;
    }
  
  // Console output
  std::cout << "Training set:\n";
  std::cout << " images : " << n_images << std::endl;
  
  // Create neural network
  NeuralNetwork net;
  net.addInputLayer(width, height); // input layer
  net.addPoolingLayer(2);
  net.addFlatteningLayer();
  net.addFullyConnectedLayer(200, ActivationFunction::SIGMOID); // hidden layer
  net.addFullyConnectedLayer(80, ActivationFunction::SIGMOID); // hidden layer
  net.addClassificationLayer(n_classes); // output layer
  
  net.initialize();

  // Output neural network structure
  std::cout << net;

  OptimizationOptions options;
  options.loss_function = OptimizationOptions::LossFunction::MSE;
  options.batch_size    = 64;
  options.max_iter      = 1e4;
  options.output_every  = 10;
  options.epochs        = 10;
  
  net.train(training_data, options);

  
  // Compare to test data
  int correct = 0;
  int wrong   = 0;
  for(auto data = test_data.begin(); data != test_data.end(); ++data)
    {
      Vector p = net.eval(*(data->x));
      if(p.indMax() != data->y.indMax())
        {
	/*
	std::cout << "Wrong classification detected.\n";
	std::cout << "Image: \n" << dynamic_cast<const Matrix&>(*(data->x));
	std::cout << "This is number " << data->y.indMax()
		<< " but network predicts " << p.indMax() << ".\n";
	std::cout << p << std::endl;
	
	char a;
	std::cout << "Press enter to continue.\n";
	std::cin >> a;
	*/
	wrong++;
        }
      else
        correct++;
    }
  std::cout << "Correctly classified : " << correct << " (" << (double)correct/n_test*100 << "%)\n";
  std::cout << "Wrongly classified   : " << wrong << " (" << (double)wrong/n_test*100 << "%)\n";

  return 0;
}
