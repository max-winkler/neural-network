#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "NeuralNetwork.h"

int main()
{
  // Create neural network
  Dimension layers;
  layers.push_back(2); // input layer
  layers.push_back(4); // hidden layer
  layers.push_back(3); // hidden layer
    
  NeuralNetwork net(layers);

  
  Matrix W1(4,2);
  Vector B1(4);

  W1 = {1.2139, 0.9372, 2.3817, -0.1927, -1.2983, 3.2739, 0.2837, 0.1673};
  B1 = {0.2397, 1.9104, -4.8739, -1.2379};

  net.setParameters(0, W1, B1, ActivationFunction::SIGMOID);
    
  Matrix W2(3,4);
  Vector B2(3);

  W2 = {1.2139, 0.9372, 2.3817, -0.1927, -1.2983, 3.2739, 1.3298, 2.8372, -1.2837, 1.2736, -2.1872, -0.2736};
  B2 = {0.2397, 1.9104, -4.8739};
  
  net.setParameters(1, W2, B2, ActivationFunction::SIGMOID);

  Matrix W3(1,3);
  Vector B3(1);

  W3 = {-2.3981, 1.3791, 0.2318};
  B3 = {1.6378};

  net.setParameters(2, W3, B3, ActivationFunction::NONE);
  
  
  std::cout << net;

  // Evaluation
  Vector x({0.8, -0.5});
  double res = net.eval(x);

  std::cout << "Evaluation: " << res << std::endl;

  // Generate training data
  srand(time(NULL));
  const size_t sample_size = 100;

  std::vector<TrainingData> trainingData;

  std::ofstream os_training;
  os_training.open("training_data.csv");
  
  for(size_t i=0; i<sample_size; ++i)
    {
      double x = -2+4.*double(rand())/RAND_MAX;
      double y = -2+4.*double(rand())/RAND_MAX;

      double label = x*x+y*y < 1 ? 1. : 0.;

      os_training << x << ", " << y << ", " << label << std::endl;
      
      trainingData.push_back(TrainingData({x, y}, label));
    }
  os_training.close();

  // Train neural network
  net.train(trainingData);

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
