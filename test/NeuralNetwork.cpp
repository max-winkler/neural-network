#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#include "NeuralNetwork.h"

int main()
{
  // Create neural network
  Dimension layers;
  layers.push_back(2); // input layer
  layers.push_back(3); // hidden layer
  
  NeuralNetwork net(layers);

  Matrix W1(3,2);
  Vector B1(3);

  W1 = {1.2139, 0.9372, 2.3817, -0.1927, -1.2983, 3.2739};
  B1 = {0.2397, 1.9104, -4.8739};
  
  net.setParameters(0, W1, B1, ActivationFunction::SIGMOID);

  Matrix W2(1,3);
  Vector B2(1);

  W2 = {-2.3981, 1.3791};
  B2 = {1.6378};

  net.setParameters(1, W2, B2, ActivationFunction::NONE);

  std::cout << net;

  // Evaluation
  Vector x({0.8, -0.5});
  double res = net.eval(x);

  std::cout << "Evaluation: " << res << std::endl;

  // Generate training data
  srand(time(NULL));
  const size_t sample_size = 100;

  std::vector<TrainingData> trainingData;
    
  for(size_t i=0; i<sample_size; ++i)
    {
      double x = -2+4.*double(rand())/RAND_MAX;
      double y = -2+4.*double(rand())/RAND_MAX;

      double label = x*x-y*y < 1 ? 1. : 0.;

      trainingData.push_back(TrainingData({x, y}, label));
    }   

  net.train(trainingData);
  
  return 0;
}
