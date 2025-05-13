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
  net.addFullyConnectedLayer(12, ActivationFunction::SIGMOID); // hidden layer
  net.addFullyConnectedLayer(8, ActivationFunction::SIGMOID); // hidden layer
  net.addFullyConnectedLayer(4, ActivationFunction::SIGMOID); // hidden layer  
  net.addFullyConnectedLayer(1, ActivationFunction::SIGMOID);    // output layer
  
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
      float x = -2.0f + 4.0f*float(rand())/RAND_MAX;
      float y = -2.0f + 4.0f*float(rand())/RAND_MAX;

      Vector label({0.0f});
      if(pow((x-0.5f)/0.8f, 2.0f) + pow(y/1.5f, 2.0f) < 1 || (x<-0.3f && x>-1.8f && y>-1.2f && y < 1.4f))
        label[0] = 1.0f;

      os_training << x << ", " << y << ", " << label[0] << std::endl;
      
      training_data.push_back(TrainingData(Vector({x, y}), label));
    }
  os_training.close();

  // Train neural network
  OptimizationOptions options;
  options.loss_function = OptimizationOptions::LossFunction::LOG;
  options.batch_size = 100;
  options.epochs = 200;
  options.max_iter = 2e5;
  options.learning_rate = 0.01;
  
  net.train(training_data, options);

  // Plot classification function
  std::ofstream outfile;
  outfile.open("result.csv");  
  
  const size_t plot_fineness = 50;
  for(size_t i=0; i<plot_fineness; ++i)
    {
      float x = -2.0f+4.0f*float(i)/plot_fineness;
      for(size_t j=0; j<plot_fineness; ++j)
        {
	float y = -2.0f+4.0f*float(j)/plot_fineness;

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
      float y = (net.eval(*it->x))[0] > 0.5f ? 1.0f : 0.0f;
      if(std::abs(y - it->y[0]) > 1.e-8)
        wrong_classified++;
    }
  
  std::cout << "Training sample size : " << training_data.size() << std::endl;
  std::cout << "  wrongly classified : " << wrong_classified << " ("
	  << float(wrong_classified)/training_data.size()*100 << "%)\n"; 
  
  return 0;
}
