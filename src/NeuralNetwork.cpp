#include "NeuralNetwork.h"

// Include layer classes
#include "VectorInputLayer.h"
#include "MatrixInputLayer.h"
#include "FullyConnectedLayer.h"
#include "FlatteningLayer.h"
#include "ConvolutionalLayer.h"
#include "PoolingLayer.h"

#include "Random.h"

#include <fstream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <pugixml.hpp>

NeuralNetwork::NeuralNetwork()
  : initialized(false), layers()
{
  layers.reserve(10);
}

NeuralNetwork::NeuralNetwork(size_t n_layers)
  : initialized(false), layers(n_layers)
{
  // layers.reserve(n_layers);
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& other)
  : initialized(true), layers()
{  
  for(auto layer = other.layers.begin(); layer != other.layers.end(); ++layer)    
    layers.push_back((*layer)->clone());  
}


NeuralNetwork::NeuralNetwork(NeuralNetwork&& other)
  : initialized(true), layers(std::move(other.layers))
{}

NeuralNetwork& NeuralNetwork::operator=(NeuralNetwork&& other)
{
  if (this != &other) {
    initialized = true;
    layers = std::move(other.layers);
  }
  return *this;
}


NeuralNetwork NeuralNetwork::createLike(const NeuralNetwork& net)
{
  NeuralNetwork other;
  
  for(auto layer = net.layers.begin(); layer != net.layers.end(); ++layer)    
    other.layers.push_back((*layer)->zeros_like());
  
  return other;
}


void NeuralNetwork::addInputLayer(size_t i, size_t j)
{
  if(j==0)
    layers.emplace_back(std::make_unique<VectorInputLayer>(i));
  else
    layers.emplace_back(std::make_unique<MatrixInputLayer>(i, j));
}

void NeuralNetwork::addPoolingLayer(size_t batch, size_t S)
{
  const Layer& prev_layer = *layers.back();
  
  switch(prev_layer.layer_type)
    {
    case LayerType::MATRIX_INPUT:
    case LayerType::CONVOLUTION:
    case LayerType::POOLING:
      break;
    default:
      std::cerr << "ERROR: A pooling layer can only follow a "
	      << Layer::LayerName.at(LayerType::MATRIX_INPUT)
	      << ", " << Layer::LayerName.at(LayerType::CONVOLUTION)
	      << " or " << Layer::LayerName.at(LayerType::POOLING) << std::endl;
      return;
    }

  layers.emplace_back(std::make_unique<PoolingLayer>(prev_layer.dim, batch, S, 0));
}

void NeuralNetwork::addFlatteningLayer()
{  
  const Layer& prev_layer = *layers.back();
  
  // Check if previous layer produces a matrix
  switch(prev_layer.layer_type)
    {
    case LayerType::MATRIX_INPUT:
    case LayerType::CONVOLUTION:
    case LayerType::POOLING:
      break;
    default:
      std::cerr << "ERROR: A flattening layer can only follow a "
	      << Layer::LayerName.at(LayerType::MATRIX_INPUT)
	      << ", " << Layer::LayerName.at(LayerType::CONVOLUTION)
	      << " or " << Layer::LayerName.at(LayerType::POOLING) << std::endl;
      return;
    }
  
  layers.emplace_back(std::make_unique<FlatteningLayer>(prev_layer.dim[0], prev_layer.dim[1], prev_layer.dim[2]));  
}

void NeuralNetwork::addConvolutionLayer(size_t F, size_t batch, ActivationFunction act, size_t S, size_t P)
{
  const Layer& prev_layer = *layers.back();
  
  // Check if previous layer produces a matrix
  switch(prev_layer.layer_type)
    {
    case LayerType::MATRIX_INPUT:
    case LayerType::CONVOLUTION:
    case LayerType::POOLING:
      break;
    default:
      std::cerr << "ERROR: A convolution layer can only follow a "
	      << Layer::LayerName.at(LayerType::MATRIX_INPUT)
	      << ", " << Layer::LayerName.at(LayerType::CONVOLUTION)
	      << " or " << Layer::LayerName.at(LayerType::POOLING) << std::endl;
      return;
    }  
  
  layers.emplace_back(std::make_unique<ConvolutionalLayer>(prev_layer.dim, F, batch, S, P, act));    
}

void NeuralNetwork::addFullyConnectedLayer(size_t width, ActivationFunction act)
{
  layers.emplace_back(std::make_unique<FullyConnectedLayer>(width, layers.back()->dim[0], act));
}

void NeuralNetwork::addClassificationLayer(size_t width)
{
  layers.emplace_back(std::make_unique<FullyConnectedLayer>(width, layers.back()->dim[0], ActivationFunction::SOFTMAX));
}

void NeuralNetwork::initialize()
{ 
  // Build up weight matrices and bias vectors
  auto it_pred = layers.begin();
  
  for(auto it = it_pred+1; it != layers.end(); ++it, ++it_pred)
    (*it)->initialize();
  
  initialized = true;
}

size_t NeuralNetwork::n_layers() const
{
  return layers.size();
}

Vector NeuralNetwork::eval(const DataArray& x) const
{
  // Input layer
  DataArray* x_tmp;
  switch(layers.front()->layer_type)
    {
    case LayerType::VECTOR_INPUT:
      // Create vector in first layer
      x_tmp = new Vector(dynamic_cast<const Vector&>(x));
      break;
    case LayerType::MATRIX_INPUT:
      x_tmp = new Tensor(dynamic_cast<const Tensor&>(x));
      break;      
    }
  
  // Hidden and output layers
  for(auto layer_it = layers.begin()+1; layer_it != layers.end(); ++layer_it)    
    (*layer_it)->eval(x_tmp);
  
  Vector y = dynamic_cast<Vector&>(*x_tmp);
  
  return y;
}

void NeuralNetwork::train(const std::vector<TrainingData>& data, OptimizationOptions options)
{
  // Parameters for momentum method
  const float momentum = 0.9;
  
  size_t n_data = data.size();
  
  if(options.batch_size == 0)
    options.batch_size = n_data;
      
  // Auxiliary variables reused in backpropagation
  std::vector<std::vector<DataArray*>> z(layers.size());
  std::vector<std::vector<DataArray*>> y(layers.size()+1);

  // Random number generator for shuffling batches
  Random rnd_gen = Random::create_mt19937_random_generator();
  
  // Allocate memory for auxiliary vectors
  for(size_t l=0; l<layers.size(); ++l)
    {
      z[l] = std::vector<DataArray*>(options.batch_size);
      y[l] = std::vector<DataArray*>(options.batch_size);

      for(size_t idx = 0; idx < options.batch_size; ++idx)
	{	  
	  switch(layers[l]->layer_type)
	    {
	    case MATRIX_INPUT:
	    case POOLING:
	    case CONVOLUTION:
	      // TODO: Are y and z really needed in input layers?	      
	      y[l][idx] = new Tensor(layers[l]->dim[0], layers[l]->dim[1], layers[l]->dim[2]);
	      z[l][idx] = new Tensor(layers[l]->dim[0], layers[l]->dim[1], layers[l]->dim[2]);
	      
	      break;
	    case FULLY_CONNECTED:
	    case VECTOR_INPUT:
	    case CLASSIFICATION:
	    case FLATTENING:
	      z[l][idx] = new Vector(layers[l]->dim[0]);
	      y[l][idx] = new Vector(layers[l]->dim[0]);
	      break;
	      
	    default:
	      std::cerr << "ERROR: Allocation of memory not implemented for "
		      << Layer::LayerName.at(layers[l]->layer_type) << " yet.\n";
	    }
	}
    }
  
  // Index set for training data
  std::vector<size_t> data_idx(n_data);
  std::iota(data_idx.begin(), data_idx.end(), 0);
    
  // Initialize time measurement
  std::chrono::steady_clock::time_point begin_time = std::chrono::steady_clock::now();

  // Monitor algorithm progress  
  std::ofstream os;
  os.open("training.csv");  
  
  // Save increment for momentum method
  NeuralNetwork increment = NeuralNetwork::createLike(*this);

  float grad_norm = 1.;  
  size_t i=0;
  for(size_t epoch=0; epoch<options.epochs; ++epoch)
    {
      std::cout << "**************************************************************\n";
      std::cout << "* Epoch " << epoch+1 << " of " << options.epochs << std::endl;
      std::cout << "**************************************************************\n";

      std::cout << std::setw(20) << "Batch"
		<< std::setw(20) << "Functional value"
		<< std::setw(20) << "gradient norm" << std::endl;
      
      std::shuffle(data_idx.begin(), data_idx.end(), rnd_gen.generator());
      
      for(size_t start_idx = 0;
	  start_idx < n_data-options.batch_size+1;
	  start_idx+= options.batch_size)
        {	
	  std::vector<size_t> batch_data_idx(data_idx.begin() + start_idx,
					     data_idx.begin() + start_idx + options.batch_size);

	  float f = evalFunctional(data, y, z, batch_data_idx, options);
      
	  NeuralNetwork grad_net = evalGradient(data, y, z, batch_data_idx, options);
	
	  grad_norm = grad_net.norm();

	  if(grad_norm < 1.e-8 || std::isnan(grad_norm))
	    {
	      std::cerr << "WARNING: The norm of the gradient is " << grad_norm << ", so we might be stuck in a local minimum.\n";
	      return;
	    }

	  // Console output
	  if(i++%options.output_every == 0)
	    {
	      std::cout << std::setw(13) << start_idx/options.batch_size << " / "
			<< std::setw(4) << n_data/options.batch_size
			<< std::setw(20) << f << std::setw(20) << grad_norm << std::endl;
	    }
	
	  // for testing only. remove later
	  // gradientTest(grad_net, data, data_idx, options);
	  // return;

	  // Update increment
	  increment.update_increment(momentum, grad_net, options.learning_rate);

	  // Subtract increment
	  apply_increment(increment);
        }
    }

  os.close();

  std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();

  std::cout << "Learning process finished after "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count()
	    << "[ms]" << std::endl;

  // Delete memory
  for(size_t l=0; l<layers.size(); ++l)
    for(size_t idx = 0; idx < options.batch_size; ++idx)
      {
	delete y[l][idx];
	delete z[l][idx];
      }
}

void NeuralNetwork::update_increment(float momentum, const NeuralNetwork& gradient, float step)
{
  auto layer = layers.begin();
  auto grad_layer = gradient.layers.begin();
  
  for(; layer != layers.end(); ++layer, ++grad_layer)    
    (*layer)->update_increment(momentum, **grad_layer, step);    
}

void NeuralNetwork::apply_increment(const NeuralNetwork& increment)
{
  auto layer = layers.begin();
  auto inc_layer = increment.layers.begin();
  
  for(; layer != layers.end(); ++layer, ++inc_layer)    
    (*layer)->apply_increment(**inc_layer);    
}

float NeuralNetwork::evalFunctional(const std::vector<TrainingData>& data,
				     std::vector<std::vector<DataArray*>>& y,
				     std::vector<std::vector<DataArray*>>& z,
				     const std::vector<size_t>& data_indices,
				     OptimizationOptions options) const
{
  size_t n_data = data.size();

  // Set initial layer
  switch(layers[0]->layer_type)
    {
    case VECTOR_INPUT:
      for(size_t idx=0; idx<options.batch_size; ++idx)
	{
	  dynamic_cast<Vector&>(*y[0][idx]) = dynamic_cast<Vector&>(*data[data_indices[idx]].x);
	}
      break;
      
    case MATRIX_INPUT:
      for(size_t idx=0; idx<options.batch_size; ++idx)
	{
	  dynamic_cast<Tensor&>(*y[0][idx]) = dynamic_cast<Tensor&>(*data[data_indices[idx]].x);
	}      
      break;
      
    default:
      std::cerr << "ERROR: The input layer must be of type VECTOR_INPUT or MATRIX_INPUT.\n";
      break;
    }

  size_t l = 0;
  for(auto layer = layers.begin()+1; layer != layers.end(); ++layer, ++l)
    {
      for(size_t idx=0; idx<options.batch_size; ++idx)
	(*layer)->forward_propagate(*y[l][idx], *z[l+1][idx], *y[l+1][idx]);			     
    }
  
  // Evaluate objective functional
  float f = 0.;
  for(size_t idx=0; idx<options.batch_size; ++idx)
    {
      // Get true data and prediction
      const Vector& Y = data[data_indices[idx]].y;
      const Vector& P = dynamic_cast<Vector&>(*(y[layers.size()-1][idx]));
	
      switch(options.loss_function)
	{
	case OptimizationOptions::LossFunction::MSE:
	  f += 0.5*pow(::norm(Y-P), 2.);
	  break;
	case OptimizationOptions::LossFunction::LOG:
	  // TODO: What happens for nL > 1?
	  if(P[0] <= 0. || P[0] >= 1.)
	    {
	      std::cerr << "Error: Can not evaluate log loss function for argument p=" << P[0] << std::endl;
	      std::cerr << "  to avoid this consider using the Sigmoid activation function in the output layer.\n";
	      return 0;
	    }
	  for(size_t i=0; i<layers.back()->dim[0]; ++i)
	    f -= (Y[i] * log(P[i]) + (1-Y[i])*log(1-P[i])); 
	  break;
	default:
	  std::cerr << "Error: Unknown loss function provided.\n";
	  return 0;
	}
    }
  
  return f;
}

NeuralNetwork NeuralNetwork::evalGradient(const std::vector<TrainingData>& data,
				  const std::vector<std::vector<DataArray*>>& y,
				  const std::vector<std::vector<DataArray*>>& z,
				  const std::vector<size_t>& data_indices,
				  OptimizationOptions options) const
{
  size_t n_data = data.size();

  NeuralNetwork grad_net(n_layers());
  //NeuralNetwork grad_net = NeuralNetwork::createLike(*this);
    
  std::vector<DataArray*> Dy(options.batch_size);
  
  // Set final value in backpropagation
  for(size_t idx=0; idx<options.batch_size; ++idx)
    {
      const Vector& Y = data[data_indices[idx]].y;
      const Vector& P = dynamic_cast<Vector&>(*y[layers.size()-1][idx]);

      switch(options.loss_function)
	{
	case OptimizationOptions::LossFunction::MSE:

	  Dy[idx] = new Vector(P - Y);
	  break;
	  
	case OptimizationOptions::LossFunction::LOG:
	  {
	    size_t nL = layers.back()->dim[0];
	    Dy[idx] = new Vector(nL);	  

	    for(size_t i=0; i<nL; ++i)
	      (*Dy[idx])[i] = (1.-Y[i]) / (1.-P[i]) - Y[i] / P[i];
	  }
	  break;
	}
    }

  // Backward propagation (desired version)
  for(size_t l=layers.size(); l-- >0; )
    grad_net.layers[l] = layers[l]->backward_propagate(Dy, y[l-1], z[l]);
  
  for(size_t idx=0; idx<options.batch_size; ++idx)         
    delete Dy[idx];    
  
  return grad_net;
}

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& net) 
{
  os << "Neural network with (" << (net.layers.size()) << " layers)\n\n";
  
  for(auto layer = net.layers.begin(); layer != net.layers.end(); ++ layer)
    {
      if(layer != net.layers.begin())
	os << "       ↓\n";
      
      os << "  " << **layer << std::endl;
    }
      
  return os;
}

void NeuralNetwork::save(const std::string& filename) const
{
  std::ofstream os(filename);
  save(os);
  os.close();
}

void NeuralNetwork::save(std::ostream& os) const
{ 
  pugi::xml_document doc;

  pugi::xml_node el_network = doc.append_child("network");
      
  for(auto layer = layers.begin(); layer!=layers.end(); ++layer)
    {
      pugi::xml_node el_layer = el_network.append_child("layer");

      // Dimensions string (comma-separated)
      std::ostringstream ss;
      for(int i=0; i<(*layer)->dim.size()-1; ++i)
        ss << (*layer)->dim[i] << ", ";
      ss << (*layer)->dim.back();

      // Add attributes to layer
      el_layer.append_attribute("dimension").set_value(ss.str().c_str());
      el_layer.append_attribute("type").set_value(Layer::LayerShortName.at((*layer)->layer_type).c_str());
      

      // Add layer parameters
      pugi::xml_node el_parameters = el_layer.append_child("parameters");
      for(const auto& [param,value] : (*layer)->get_parameters())        
        el_parameters.append_child(param.c_str()).text().set(value.c_str());        

      // Add layer weights
      pugi::xml_node el_weights = el_layer.append_child("weights");
      for(const auto& [weight, value] : (*layer)->get_weights())
        {
          pugi::xml_node el_weight = el_weights.append_child(weight.c_str());
          std::ostringstream ss, ss_dim;
          
          const float* data = value.first;
          std::vector<size_t> dim = value.second;

          size_t size=1;
          for(size_t i=0; i<dim.size(); ++i)
            {
              size *= dim[i];
              ss_dim << dim[i];
              if(i < dim.size()-1)
                ss_dim << ",";
            }

          // Write data
          size_t line_length = 8;
          for (size_t i = 0; i < size; ++i) {
            /*
              if (i % line_length == 0 && i != 0) {
                ss << "\n    ";
              }
            */
            ss << " " << data[i];
          }
          
          el_weight.append_attribute("dimension").set_value(ss_dim.str().c_str());
          el_weight.append_child(pugi::node_pcdata).set_value(ss.str().c_str());
        }
      
      // (*layer)->save(os);
    }
  
  doc.save(os);
}

NeuralNetwork NeuralNetwork::load(const std::string& filename)
{
  NeuralNetwork network;
  
  pugi::xml_document doc;
  pugi::xml_parse_result result = doc.load_file(filename.c_str());

  if(!result)
    {
      std::cerr << "ERROR: Can not read file " << filename << "\n";
      std::exit(EXIT_FAILURE);
    }

  pugi::xml_node root = doc.child("network");
  std::vector<size_t> prev_dim;
  
  for (pugi::xml_node el_layer : root.children("layer"))
    {      
      // Determine layer type
      LayerType layer_type = Layer::LayerTypeFromShortName.at(el_layer.attribute("type").value());
      std::cout << "Layer-Typ: " << layer_type << "\n";

      // Get output dimension
      std::string dim_str = el_layer.attribute("dimension").value();
      std::stringstream ss(dim_str);
      std::string token;

      std::vector<size_t> dim;
      while (std::getline(ss, token, ','))
        dim.push_back(static_cast<size_t>(std::stoul(token)));        
    
      // Create parameter list
      pugi::xml_node el_parameters = el_layer.child("parameters");
      std::map<std::string, std::string> parameters;
      for (pugi::xml_node el_param : el_parameters.children())
        {
          parameters[el_param.name()] = el_param.child_value();
          std::cout << "  Parameter: " << el_param.name()
                    << " = " << el_param.child_value() << "\n";
        }

      // Create weight list
      pugi::xml_node el_weights = el_layer.child("weights");
      std::map<std::string, std::pair<const float*, std::vector<size_t>>> weights;
        
      for (pugi::xml_node el_weight : el_weights.children()) {

        std::stringstream ss(el_weight.attribute("dimension").value());
        std::string token;
        std::vector<size_t> weight_dim;
        while (std::getline(ss, token, ','))
	weight_dim.push_back(static_cast<size_t>(std::stoul(token)));        
        
        std::string weight_name = el_weight.name();

        std::string weight_value;
        std::stringstream ss2(el_weight.child_value());
        std::vector<float> weight_data;
        
        while(ss2 >> weight_value)
	weight_data.push_back(std::stof(weight_value));      

        weights[weight_name] = std::make_pair(weight_data.data(), weight_dim);                
      }

      switch(layer_type)
        {
        case VECTOR_INPUT:
	std::cout << "Vector Input\n";
	break;
        case FULLY_CONNECTED:
	network.layers.emplace_back(FullyConnectedLayer::create_from_parameters(dim, prev_dim, parameters, weights));
	std::cout << "Fully Connected\n";
	break;
        default:
	std::cout << "Somethind else\n";
	break;
        }

      prev_dim = std::move(dim);
      std::cout << "\n";
    }
 
  
  return network;
}

void NeuralNetwork::gradientTest(const NeuralNetwork& grad_net,
                                 const std::vector<TrainingData>& data,
                                 const std::vector<size_t>& data_idx,
                                 OptimizationOptions options) const
{    
  // Initialize a direction
  NeuralNetwork direction = NeuralNetwork::createLike(*this);
  NeuralNetwork zero_net = NeuralNetwork::createLike(*this);  
  direction.initialize(); // fills weights with random data

  float deriv_exact = grad_net.dot(direction);

  size_t n_data = data.size();
  
  std::vector<std::vector<DataArray*>> z(layers.size());
  std::vector<std::vector<DataArray*>> y(layers.size()+1);

  // Allocate memory for auxiliary vectors
  for(size_t l=0; l<layers.size(); ++l)
    {
      z[l] = std::vector<DataArray*>(options.batch_size);
      y[l] = std::vector<DataArray*>(options.batch_size);

      for(size_t idx = 0; idx < options.batch_size; ++idx)
        {
	switch(layers[l]->layer_type)
	  {
	  case MATRIX_INPUT:
	  case POOLING:
	  case CONVOLUTION:
	    // TODO: Are y and z really needed in input layers?	      
	    y[l][idx] = new Tensor(layers[l]->dim[0], layers[l]->dim[1], layers[l]->dim[2]);
	    z[l][idx] = new Tensor(layers[l]->dim[0], layers[l]->dim[1], layers[l]->dim[2]);
	      
	    break;
	  case FULLY_CONNECTED:
	  case VECTOR_INPUT:
	  case CLASSIFICATION:
	  case FLATTENING:
	    z[l][idx] = new Vector(layers[l]->dim[0]);
	    y[l][idx] = new Vector(layers[l]->dim[0]);
	    break;
	      
	  default:
	    std::cerr << "ERROR: Allocation of memory not implemented for "
		    << Layer::LayerName.at(layers[l]->layer_type) << " yet.\n";
	  }
        }
    }
  
  float f = evalFunctional(data, y, z, data_idx, options);
  std::cout << "Value in x0: " << f << std::endl; 
  
  for(float s=1.; s>1.e-12; s*=0.5)
    {
      // NEW
      std::vector<std::vector<DataArray*>> z_s(layers.size());
      std::vector<std::vector<DataArray*>> y_s(layers.size()+1);

      // Allocate memory for auxiliary vectors
      for(size_t l=0; l<layers.size(); ++l)
        {
	z_s[l] = std::vector<DataArray*>(options.batch_size);
	y_s[l] = std::vector<DataArray*>(options.batch_size);

	for(size_t idx = 0; idx < options.batch_size; ++idx)
	  {
	    switch(layers[l]->layer_type)
	      {
	      case MATRIX_INPUT:
	      case POOLING:
	      case CONVOLUTION:
	        // TODO: Are y and z really needed in input layers?	      
	        y_s[l][idx] = new Tensor(layers[l]->dim[0], layers[l]->dim[1], layers[l]->dim[2]);
	        z_s[l][idx] = new Tensor(layers[l]->dim[0], layers[l]->dim[1], layers[l]->dim[2]);
	      
	        break;
	      case FULLY_CONNECTED:
	      case VECTOR_INPUT:
	      case CLASSIFICATION:
	      case FLATTENING:
	        z_s[l][idx] = new Vector(layers[l]->dim[0]);
	        y_s[l][idx] = new Vector(layers[l]->dim[0]);
	        break;
	      
	      default:
	        std::cerr << "ERROR: Allocation of memory not implemented for "
		        << Layer::LayerName.at(layers[l]->layer_type) << " yet.\n";
	      }
	  }
        }
      
      NeuralNetwork dir_s(direction);
      dir_s.update_increment(-s, zero_net, 0.);
      
      NeuralNetwork net_s(*this);
      net_s.apply_increment(dir_s);
  
      float f_s = net_s.evalFunctional(data, y_s, z_s, data_idx, options);
      std::cout << "Value in x+s*d: " << f_s << std::endl;
      
      float deriv_fd = (f_s - f)/s;
      
      std::cout << "  derivative by gradient: " << deriv_exact
		<< " vs. by difference quotient: " << deriv_fd
		<< " relative error: " << std::abs(deriv_exact - deriv_fd)/deriv_exact << std::endl;
      
    }

  for(size_t l=0; l<layers.size(); ++l)
    for(size_t idx = 0; idx < options.batch_size; ++idx)
      {
	delete y[l][idx];
	delete z[l][idx];
      }  
}

float NeuralNetwork::dot(const NeuralNetwork& rhs) const
{ 
  float val = 0.0f;
  
  auto layer = layers.begin();
  auto layer_rhs = rhs.layers.begin();
  
  for(; layer != layers.end(); ++layer, ++layer_rhs)
    val += (*layer)->dot(**layer_rhs);
   
  return val;
}

float NeuralNetwork::norm() const
{
  return sqrt(dot(*this));
}

OptimizationOptions::OptimizationOptions() : max_iter(1e4), batch_size(128),
				     learning_rate(1.e-2), loss_function(LossFunction::MSE),
				     output_every(1), epochs(3)				     
{}
