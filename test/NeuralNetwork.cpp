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
  net.addLayer(2, ActivationFunction::SIGMOID); // input layer
  net.addLayer(6, ActivationFunction::SIGMOID); // hidden layer
  net.addLayer(8, ActivationFunction::SIGMOID); // hidden layer
  net.addLayer(12, ActivationFunction::SIGMOID); // hidden layer
  // net.addLayer(8, ActivationFunction::SIGMOID); // hidden layer
  net.addLayer(6, ActivationFunction::NONE); // hidden layer
  net.initialize();
  
  std::cout << net;

  // Evaluation
  Vector x({0.8, -0.5});
  double res = net.eval(x);

  std::cout << "Evaluation: " << res << std::endl;

  // Generate training data
  srand(time(NULL));
  const size_t sample_size = 10000;

  std::vector<TrainingData> training_data;

  std::ofstream os_training;
  os_training.open("training_data.csv");
  
  for(size_t i=0; i<sample_size; ++i)
    {
      double x = -2+4.*double(rand())/RAND_MAX;
      double y = -2+4.*double(rand())/RAND_MAX;

      double label = -1.;
      if(pow((x-0.5)/0.8, 2.) + pow(y/1.5, 2.) < 1 || (x<-0.5 && x>-1.8 && y>-0.8 && y < 0.8))
        label = 1.;

      os_training << x << ", " << y << ", " << label << std::endl;
      
      training_data.push_back(TrainingData({x, y}, label));
    }
  os_training.close();

  // Train neural network
  net.train(training_data, 256);

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

  // Evaluation
  size_t wrong_classified = 0;
  for(auto it = training_data.begin(); it != training_data.end(); ++it)
    {      
      double y = net.eval(it->x)>0? 1. : -1.;
      if(std::abs(y - it->y) > 1.e-8)
        wrong_classified++;
    }
  
  std::cout << "Training sample size : " << training_data.size();
  std::cout << "  wrongly classified : " << wrong_classified << " ("
	  << double(wrong_classified)/training_data.size()*100 << "%)\n"; 
  
  return 0;
}
