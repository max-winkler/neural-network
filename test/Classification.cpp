#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <png.h>

#include "NeuralNetwork.h"

int main()
{
  // Create neural network
  
  NeuralNetwork net;
  net.addInputLayer(2); // input layer
  net.addFullyConnectedLayer(4, ActivationFunction::TANH); // hidden layer
  net.addFullyConnectedLayer(6, ActivationFunction::TANH); // hidden layer
  net.addFullyConnectedLayer(6, ActivationFunction::TANH); // hidden layer
  net.addFullyConnectedLayer(4, ActivationFunction::SIGMOID); // hidden layer
  net.addClassificationLayer(3); // output layer
  
  net.initialize();
  
  std::cout << net; 

  // Generate data and separate into 3 classes
  const size_t sample_size = 1000;

  std::vector<TrainingData> training_data;
  training_data.reserve(sample_size);
  
  for(size_t i=0; i<sample_size; ++i)
    {
      double x = double(rand())/RAND_MAX;
      double y = double(rand())/RAND_MAX;

      size_t c;
      if(y <= pow(1-x, 3.))
        c = 0;
      else if(y >= 3*x-1)
        c = 1;
      else
        c = 2;

      Vector label = Vector(3);
      label[c] = 1.;

      training_data.push_back(TrainingData(Vector({x, y}), label));
    }

  // Train neural network
  OptimizationOptions options;
  options.loss_function = OptimizationOptions::LossFunction::MSE;
  options.batch_size    = 100;
  options.max_iter      = 1e5;
  
  net.train(training_data, options);

  // Write classification to PNG file
  FILE *fp = fopen("classification.png", "wb");
  if(!fp)
    {
      std::cerr << "Error: Could not open image file\n";
      return -1;      
    }

  const int width=512, height=512;
  
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  
  png_init_io(png, fp);

  png_set_IHDR(png, info, width, height, 8,
	     PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
	     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
	     );

  png_write_info(png, info);

  png_bytep* row_pointers = new png_bytep[height];  
  
  for(int i=0; i<height; ++i)
    {
      row_pointers[height-1-i] = new png_byte[3*width];
      png_bytep row = row_pointers[height-1-i];
      
      for(int j=0; j<width; ++j)
        {
	double x = double(j)/(width-1);
	double y = double(i)/(height-1);
	
	Vector c = net.eval(Vector({x, y}));        
	
	for(int m=0; m<3; ++m)	  
	  row[3*j + m] = png_byte(255*c[m]);
        }
    }
    
  png_write_image(png, row_pointers);
  png_write_end(png, NULL);

  if (png && info)
    png_destroy_write_struct(&png, &info);

  for(int i=0; i<height; ++i)
    delete[] row_pointers[i];
  delete[] row_pointers;

  fclose(fp);
  
  while(true)
    {
      double x, y;
      std::cout << "Evaluate classifier:\n";
      std::cout << "x =";
      std::cin >> x;
      std::cout << "y =";
      std::cin >> y;

      Vector c = net.eval(Vector({x, y}));
      std::cout << "The point (" << x << ", " << y << ") is classified as follows:\n";
      for(int i=0; i<3; ++i)
        std::cout << "  class " << i << ": " << c[i]*100 << "%\n";      
    }
  
  return 0;
}
