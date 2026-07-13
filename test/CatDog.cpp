#include <filesystem>
#include <string>
#include <vector>
#include <fstream>

#include "Activation.h"
#include "Image.h"
#include "NeuralNetwork.h"
#include "TrainingData.h"
#include "mpi.h"
#include "stb_image.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string xml_file("cat_dog_network.xml");

  std::string data_dir("/LOCAL/maxwin/cat_dog_images/");
  std::string output_dir("/LOCAL/maxwin/output/");
  std::vector<TrainingData> training_data;
  std::vector<TrainingData> test_data;

  size_t n_test = 25000 / 10;
  size_t count = 0, skipped = 0, dog = 0, cat = 0;
  
  // Iterate over directory with test images  
  for (const auto& file : fs::directory_iterator(data_dir)) {
    if (!file.is_regular_file()) continue;
    
    std::string filename = file.path().filename().string();
    
    // Skip images that are too small
    ImageInfo info = Image::get_info(file.path().string());
    if (info.width < 128 || info.height < 128) {
      if(rank == 0)
        std::cout << "Skipping too small image: " << filename << "(" << info.width
	        << "x" << info.height << ")\n";
      skipped++;
      continue;
    }

    // Read and transform image
    Image img = Image::read(file.path().string());
    img.resize(128);
    
    // Get label from filename
    std::string type = filename.substr(0, filename.find('.'));
    Vector label(2);
    if (type == "dog") {
      label[0] = 1.;
      label[1] = 0.;
    } else if (type == "cat") {
      label[0] = 0.;      
      label[1] = 1.;
    }
    else {
      continue;
    }

    if (type == "dog") {
      if (dog < n_test / 2) test_data.emplace_back(img.to_matrix(), label);
      else
        training_data.emplace_back(img.to_matrix(), label);
      dog++;
    }
    if (type == "cat") {
      if (cat < n_test / 2) test_data.emplace_back(img.to_matrix(), label);
      else
        training_data.emplace_back(img.to_matrix(), label);
      cat++;      
    }          
    count++;
    
    // if (cat == 512 && dog == 512)
    //   break;      
  }

  if (rank == 0) {
    std::cout << "The data set contains " << count << " images (" << dog
              << " dogs and " << cat << " cats) and we skip " << skipped
              << " of them.\n";
    std::cout << "Test data: " << n_test << " vs. " << test_data.size()
              << std::endl;
  }

  NeuralNetwork net;

  std::ifstream in_fs(xml_file.c_str());
  if (!in_fs.good()) {
    net.addInputLayer(128, 128);
    
    net.addConvolutionLayer(8, 5, ActivationFunction::RELU, 1, 2);
    net.addPoolingLayer(2);
    net.addConvolutionLayer(16, 5, ActivationFunction::RELU, 1, 2);
    net.addPoolingLayer(2);
    net.addConvolutionLayer(32, 5, ActivationFunction::RELU, 1, 2);
    net.addPoolingLayer(4);
    net.addConvolutionLayer(64, 5, ActivationFunction::RELU, 1, 2);
    net.addPoolingLayer(4);
    net.addFlatteningLayer();
    net.addFullyConnectedLayer(64, ActivationFunction::RELU);
    // net.addFullyConnectedLayer(20, ActivationFunction::RELU);
    // net.addFullyConnectedLayer(5, ActivationFunction::RELU);
    net.addClassificationLayer(2);

    net.initialize();
  } else {    
    in_fs.close();
    std::cout << "Loaded (pre-trained) neural network from XML file\n";
    net = NeuralNetwork::load(xml_file);

    std::ofstream of("outputs.dat");
    for (int d = 0; d < 20; ++d) {
      auto outputs = net.getLayerOutputs(*training_data[d].x);
      int i = 0;
      of << "Data " << d << "(class" << training_data[d].y.indMax() << ")\n";
      for (const auto& out : outputs) {
        of << "Layer " << i++ << ":\n";
        if (auto vec_ptr = dynamic_cast<Vector*>(out.get())) {
          of << *vec_ptr << "\n";
        }
        // 2. Wenn es eine Matrix ist, nutze deinen Matrix-<<-Operator
        else if (auto mat_ptr = dynamic_cast<Matrix*>(out.get())) {
          of << *mat_ptr << "\n";
        }
        // 3. Wenn es ein Tensor ist, nutze deinen Tensor-<<-Operator
        else if (auto ten_ptr = dynamic_cast<Tensor*>(out.get())) {
          of << *ten_ptr << "\n";
        } else {
          of << "Unbekannter DataArray-Typ!\n";
        }
      }
    }

    /*
    for (const auto& dat : training_data) {
      auto outputs = net.getLayerOutputs(*dat.x);
      std::cout << "C" << dat.y.indMax()
                << " O-1:" << dynamic_cast<Vector&>(**(outputs.end() - 2))
                << " O:" << dynamic_cast<Vector&>(**(outputs.end() - 1))
                << std::endl;


    }
    */
  }
  //MPI_Finalize();
  //return 0;

  if (rank == 0) std::cout << net;

  OptimizationOptions options;
  options.loss_function = OptimizationOptions::LOG;
  options.epochs = 10;
  options.learning_rate = 0.0001;
  options.batch_size = 512;
  options.momentum = 0.9;  

  net.train(training_data, options);

  // Compare with the first 1000 data (=both training and test data)
  if (rank == 0) {

    int correct = 0, wrong = 0;
    
    for (const auto& data: test_data) {
      Vector p = net.eval(*data.x);
      int class_p = p.indMax();
      std::cout << class_p;
      int class_y = data.y.indMax();

      if (class_p == class_y)
        correct++;
      else
        wrong++;
    }
    std::cout << std::endl;

    std::cout << "Correctly classified : " << correct << " ("
              << (float)correct / n_test * 100 << "%)\n";
    std::cout << "Wrongly classified   : " << wrong << " ("
              << (float)wrong / n_test * 100 << "%)\n";

    net.save(xml_file.c_str());
  }
  
  MPI_Finalize();
  return 0;
}
