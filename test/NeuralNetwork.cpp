#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "NeuralNetwork.h"

int main()
{
  // Create neural network
  NeuralNetwork net;
  net.addLayer(2, ActivationFunction::NONE); // input layer
  net.addLayer(3, ActivationFunction::SIGMOID); // hidden layer
  net.addLayer(4, ActivationFunction::SIGMOID); // hidden layer
  net.addLayer(3, ActivationFunction::SIGMOID); // hidden layer
  net.initialize();
    
  std::cout << net;

  // Evaluation
  Vector x({0.8, -0.5});
  double res = net.eval(x);

  std::cout << "Evaluation: " << res << std::endl;

  // Generate training data
  srand(time(NULL));
  const size_t sample_size = 10000;

  std::vector<TrainingData> trainingData;

  std::ofstream os_training;
  os_training.open("training_data.csv");
  
  for(size_t i=0; i<sample_size; ++i)
    {
      double x = -2+4.*double(rand())/RAND_MAX;
      double y = -2+4.*double(rand())/RAND_MAX;

      double label = pow(x/1.5, 2.) + pow(y/0.8, 2.) < 1 ? 1. : 0.;

      os_training << x << ", " << y << ", " << label << std::endl;
      
      trainingData.push_back(TrainingData({x, y}, label));
    }
  os_training.close();

  // Train neural network
  net.train(trainingData, 64);

  // Plot classification function
  std::ofstream outfile;
  outfile.open("result.csv");  
  
  const size_t plot_fineness = 50;
  for(size_t i=0; i<plot_fineness; ++i)
    {
      double x = -2.+4.*double(i)/plot_fineness;
      for(size_t j=0; j<plot_fineness; ++j)
        {
	double y = -2.+4.*double(j)/plot_fineness;

	double f = net.eval(Vector{x, y});
	outfile << x << ", " << y << ", " << f << std::endl;
        }
    }
  outfile.close();
  
  return 0;
}
