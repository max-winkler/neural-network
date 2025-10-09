#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <png.h>

#include "NeuralNetwork.h"
#include "Image.h"

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
  const size_t sample_size = 10000;

  std::vector<TrainingData> training_data;
  training_data.reserve(sample_size);
  
  for(size_t i=0; i<sample_size; ++i)
    {
      float x = float(rand())/RAND_MAX;
      float y = float(rand())/RAND_MAX;

      size_t c;
      if(y <= pow(1.0f-x, 3.0f))
        c = 0;
      else if(y >= 3.0f*x-1.0f)
        c = 1;
      else
        c = 2;

      Vector label = Vector(3);
      label[c] = 1.0f;
      
      training_data.push_back(TrainingData(Vector({x, y}), label));
    }
  
  // Train neural network
  OptimizationOptions options;
  options.loss_function = OptimizationOptions::LossFunction::MSE;
  options.batch_size    = 100;
  options.max_iter      = 1e5;
  options.epochs        = 10;
  
  net.train(training_data, options);
  net.save("test.xml");
  
  // Generate output image
  const int width=512, height=512;
  Tensor output(3, height, width);

  for(int i=0; i<height; ++i)
    {      
      for(int j=0; j<width; ++j)
        {
	float x = float(j)/(width-1);
	float y = float(i)/(height-1);
	
	Vector c = net.eval(Vector({x, y}));

        for (int m = 0; m < 3; ++m) {
	output(m, i, j) = c[m];
        }
      }
  }
  
  // Write image to file
  Image output_image = Image::from_rgb_tensor(output);
  output_image.write("classification.png");
  
  while(true)
    {
      float x, y;
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
