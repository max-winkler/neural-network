#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <mpi.h>

#include "Matrix.h"
#include "TrainingData.h"
#include "NeuralNetwork.h"
#include "MNIST.h"
#include "Image.h"

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string xml_file("digit_rec_network.xml");
  
  std::cout << "I am rank " << rank+1 << " of " << size << std::endl;

  // Read training data
  int width, height, n_images, n_classes;
  std::vector<TrainingData> training_data;
  
  char training_data_file[]  = "mnist/train-images-idx3-ubyte";
  char training_label_file[] = "mnist/train-labels-idx1-ubyte";
  char test_data_file[]      = "mnist/t10k-images-idx3-ubyte";
  char test_label_file[]     = "mnist/t10k-labels-idx1-ubyte";
  
  if (MNIST::read(training_data_file, training_label_file, n_images, width,
                  height, n_classes, training_data) != 0) {  
    std::cerr << "Unable to read MNIST files " << training_data_file << " and "
	    << training_label_file << ".\n";
    return -1;
  }
  
  // Read test data
  std::vector<TrainingData> test_data;
  int n_test, n_classes_test;
  
  if(MNIST::read(test_data_file, test_label_file,
	       n_test, width, height, n_classes_test, test_data) != 0)    
    return -1;
  
  if (n_classes != n_classes_test) {    
    std::cerr << "ERROR: Training and test dataset have different number of classes.\n";
    return -1;
  }

  // Console output
  if (rank == 0) {      

    std::cout << " training data : " << n_images << std::endl;
    std::cout << " test data     : " << n_test << std::endl;
    std::cout << " classes : " << n_classes << std::endl;
    std::cout << " width   : " << width << std::endl;
    std::cout << " height  : " << height << std::endl;
  }

  // Create neural network
  NeuralNetwork net;
  
  std::ifstream fs(xml_file.c_str());
  if (!fs.good()) {    
    std::cout << "No neural network XML file exists. Create and training new network.\n";      
      
    net.addInputLayer(width, height); // input layer
    net.addConvolutionLayer(16, 5, ActivationFunction::RELU, 1, 2);
    net.addPoolingLayer(2, 2);
    net.addConvolutionLayer(32, 5, ActivationFunction::RELU, 1, 2);
    net.addPoolingLayer(2, 2);
    net.addFlatteningLayer();
    net.addFullyConnectedLayer(50, ActivationFunction::SIGMOID);
    net.addFullyConnectedLayer(20, ActivationFunction::SIGMOID);
    net.addClassificationLayer(n_classes); // output layer
      
    net.initialize();
  } else {    
    fs.close();
    std::cout << "Loaded (pre-trained) neural network from XML file\n";
    net = NeuralNetwork::load(xml_file);
  }

  // Output neural network structure  
  if(rank == 0)    
    std::cout << net;

  // Refine optimization options
  OptimizationOptions options;
  options.loss_function = OptimizationOptions::LossFunction::LOG;
  options.batch_size    = 1024;
  options.max_iter      = 1e4;
  options.output_every  = 10;
  options.epochs        = 10;
  options.learning_rate = 0.005 /5;
  
  net.train(training_data, options);

  if (rank == 0) {    
    // Write feature maps to image file for 10 inputs
    for (int i = 0; i < 10; ++i) {      
      std::vector<std::unique_ptr<DataArray>> outputs = net.getLayerOutputs(*(test_data[i].x));
      int ctr = 0;
      for (const auto &y : outputs) {        
        if (auto tensor = dynamic_cast<Tensor *>(y.get())) {          
	Image img = Image::from_tensor(*tensor);
	    
	std::stringstream ss;
	ss << "output_" << ctr << "_" << i << ".png";
	img.write(ss.str());
        }
        ++ctr;
      }
    }
    
    // Compare to test data
    int correct = 0;
    int wrong   = 0;
    
    for (auto data = test_data.begin(); data != test_data.end(); ++data) {        
      Vector p = net.eval(*(data->x));
      if(p.indMax() != data->y.indMax())
        wrong++;	  
      else
        correct++;
    }
      
    std::cout << "Correctly classified : " << correct << " (" << (float)correct/n_test*100 << "%)\n";
    std::cout << "Wrongly classified   : " << wrong << " (" << (float)wrong/n_test*100 << "%)\n";

    net.save(xml_file.c_str());
  }
  
  MPI_Finalize();
  return 0;
}
