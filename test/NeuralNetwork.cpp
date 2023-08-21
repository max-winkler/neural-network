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
  net.addInputLayer(2); // input layer
  net.addFullyConnectedLayer(4, ActivationFunction::SIGMOID); // hidden layer
  net.addFullyConnectedLayer(8, ActivationFunction::SIGMOID); // hidden layer
  net.addFullyConnectedLayer(4, ActivationFunction::SIGMOID); // hidden layer
  net.addFullyConnectedLayer(1, ActivationFunction::NONE);    // output layer
  
  net.initialize();
  
  std::cout << net;

  // Evaluation
  Vector x({0.8, -0.5});
  Vector res = net.eval(x);

  std::cout << "Evaluation: " << res << std::endl;

  // Generate training data
  srand(time(NULL));
  const size_t sample_size = 10000;

  std::vector<TrainingData> training_data;

  std::ofstream os_training;
  os_training.open("training_data.csv");
  
  for(size_t i=0; i<sample_size; ++i)
    {
      double x = -2. + 4.*double(rand())/RAND_MAX;
      double y = -2. + 4.*double(rand())/RAND_MAX;

      Vector label({0.});
      if(pow((x-0.5)/0.8, 2.) + pow(y/1.5, 2.) < 1 || (x<-0.3 && x>-1.8 && y>-1.2 && y < 1.4))
        label[0] = 1.;

      os_training << x << ", " << y << ", " << label[0] << std::endl;
      
      training_data.push_back(TrainingData(Vector({x, y}), label));
    }
  os_training.close();

  // Train neural network
  OptimizationOptions options;
  options.loss_function = OptimizationOptions::LossFunction::MSE;
  options.batch_size = 100;
  options.max_iter = 1.e5;
  options.learning_rate = 0.001;
  
  net.train(training_data, options);

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

	Vector f = net.eval(Vector{x, y});
	outfile << x << ", " << y << ", " << f[0] << std::endl;
        }
      outfile << std::endl;
    }
  outfile.close();

  // Evaluation
  size_t wrong_classified = 0;
  for(auto it = training_data.begin(); it != training_data.end(); ++it)
    {      
      double y = (net.eval(*it->x))[0] > 0.5 ? 1. : 0.;
      if(std::abs(y - it->y[0]) > 1.e-8)
        wrong_classified++;
    }
  
  std::cout << "Training sample size : " << training_data.size() << std::endl;
  std::cout << "  wrongly classified : " << wrong_classified << " ("
	  << double(wrong_classified)/training_data.size()*100 << "%)\n"; 
  
  return 0;
}
