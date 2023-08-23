#include "NeuralNetwork.h"

#include <fstream>
#include <algorithm>
#include <numeric>
#include <chrono>

NeuralNetwork::NeuralNetwork()
  : initialized(false), layers(), rnd_gen(std::random_device()())
{
  layers.reserve(10);
}

NeuralNetwork::NeuralNetwork(NeuralNetwork&& other)
  : initialized(true), layers(std::move(other.layers)), rnd_gen(std::random_device()())
{}

NeuralNetwork NeuralNetwork::createLike(const NeuralNetwork& net)
{
  NeuralNetwork other;

  for(auto layer = net.layers.begin(); layer != net.layers.end(); ++layer)
    {
      // Create now layer with zero weights
      Layer new_layer(layer->dimension, layer->layer_type, layer->activation_function);
      
      new_layer.weight = Matrix(layer->weight.nRows(), layer->weight.nCols());
      new_layer.bias = Vector(layer->bias.length());
	
      other.layers.push_back(new_layer);
    }

  return other;
}

/*
NeuralNetwork::NeuralNetwork(std::vector<size_t> width_)
  : initialized(true), layers(width_.size()), rnd_gen(std::random_device()())
{  
  // Add output dimension
  width.push_back(1);

  params.weight.reserve(layers);
  params.bias.reserve(layers);
  params.activation.reserve(layers);
  
  for(Dimension::const_iterator it = width.begin(); it+1 != width.end(); ++it)
    {
      params.weight.push_back(Matrix(*(it+1), *it));
      params.bias.push_back(Vector(*(it+1)));
      params.activation.push_back((it+1!=width.end()) ? ActivationFunction::SIGMOID : ActivationFunction::NONE);
    }
}
*/

void NeuralNetwork::addInputLayer(size_t i, size_t j)
{
  LayerType layer_type = (j==0 ? LayerType::VECTOR_INPUT : LayerType::MATRIX_INPUT);
  Layer layer(std::pair<size_t, size_t>(i, j), layer_type, ActivationFunction::NONE);
  layers.push_back(layer);
}

void NeuralNetwork::addFullyConnectedLayer(size_t width, ActivationFunction act)
{
  Layer layer(std::pair<size_t, size_t>(width, 0), LayerType::FULLY_CONNECTED, act);
  layers.push_back(layer);
}

void NeuralNetwork::addClassificationLayer(size_t width)
{
  Layer layer(std::pair<size_t, size_t>(width, 0), LayerType::CLASSIFICATION, ActivationFunction::SOFTMAX);
  layers.push_back(layer);  
}

void NeuralNetwork::initialize()
{ 
  // Build up weight matrices and bias vectors
  auto it_pred = layers.begin();
  
  for(auto it = it_pred+1; it != layers.end(); ++it, ++it_pred)
    {
      if(it->layer_type == LayerType::FULLY_CONNECTED
	 || it->layer_type == LayerType::CLASSIFICATION)
	{
	  if(it_pred->layer_type == LayerType::FULLY_CONNECTED
	     || it_pred->layer_type == LayerType::VECTOR_INPUT
	     || it_pred->layer_type == LayerType::FLATTENING)	    
	    {
	      // Build weight matrix and bias vector
	      size_t m = it->dimension.first;
	      size_t n = it_pred->dimension.first;

	      // Initialize random matrix
	      Matrix W(m, n);
	      for(size_t i=0; i<m; ++i)
		for(size_t j=0; j<n; ++j)
		  W[i][j] = random_real(rnd_gen);
	      
	      it->weight = W;
	      it->bias = Vector(m);
	    }
	  else
	    {
	      std::cerr << "ERROR: A " << Layer::LayerName[it->layer_type]
			<< " follows a " << Layer::LayerName[it_pred->layer_type]
			<< " which is incompatible.\n";
	      return;
	    }
	  
	}
      else
	{
	  std::cerr << "ERROR: Initialization of neural network with " << Layer::LayerName[it->layer_type]
		    << " is not implemented yet.\n";
	}
    }  
  
  initialized = true;
}

/*
void NeuralNetwork::setParameters(size_t layer, const Matrix& matrix, const Vector& vector, ActivationFunction act)
{
  if(layer < 0 || layer > layers-1)
    {
      std::cerr << "Error: The layer " << layer << " is not feasible. Must be between 0 and " << layers-1 << std::endl;
      return;
    }

  if(matrix.nRows() != width[layer+1] || matrix.nCols() != width[layer])
    {
      std::cerr << "Error: The provided weight matrix has invalid size.\n";
      std::cerr << "  Given is (" << matrix.nRows() << "," << matrix.nCols() << ")"
		<< " but required is (" << width[layer+1] << "," << width[layer] << ")\n";
      return;
    }
  
  if(vector.size() != width[layer+1])
    {
      std::cerr << "Error: The provided bias vector has invalid size.\n";
      std::cerr << "  Given is (" << vector.size() 
		<< " but required is (" << width[layer+1] << "," << ")\n";
      return;
    }
  
  params.weight[layer] = matrix;
  params.bias[layer] = vector;
  params.activation[layer] = act;
}
*/

Vector NeuralNetwork::eval(const DataArray& x) const
{
  DataArray* x_tmp;
  
  for(auto layer = layers.begin(); layer != layers.end(); ++layer)
    {      
      switch(layer->layer_type)
	{
	case LayerType::VECTOR_INPUT:
	  // Create vector in first layer
	  x_tmp = new Vector(dynamic_cast<const Vector&>(x));
	  break;
	case LayerType::FULLY_CONNECTED:
	case LayerType::CLASSIFICATION:
	  {
	    Vector& x_ref = dynamic_cast<Vector&>(*x_tmp);
	    
	    dynamic_cast<Vector&>(*x_tmp) = activate(layer->weight * x_ref + layer->bias, layer->activation_function);
	  }
	  break;
	default:
	  std::cerr << "ERROR: The layer type " << layer->layer_type << " is invalid or not implemented yet.\n";	  
	}      
    }

  Vector y = dynamic_cast<Vector&>(*x_tmp);
  delete x_tmp;
  
  return y;
}

void NeuralNetwork::train(const std::vector<TrainingData>& data, OptimizationOptions options)
{
  // Parameters for momentum method
  const double momentum = 0.9;
  
  size_t n_data = data.size();
  
  if(options.batch_size == 0)
    options.batch_size = n_data;
      
  // Auxiliary variables reused in backpropagation
  std::vector<std::vector<DataArray*>> z(layers.size());
  std::vector<std::vector<DataArray*>> y(layers.size()+1);
  
  // Allocate memory for auxiliary vectors
  for(size_t l=0; l<layers.size(); ++l)
    {
      z[l] = std::vector<DataArray*>(options.batch_size);
      y[l] = std::vector<DataArray*>(options.batch_size);

      for(size_t idx = 0; idx < options.batch_size; ++idx)
	{	  
	  switch(layers[l].layer_type)
	    {	      
	    case FULLY_CONNECTED:
	    case VECTOR_INPUT:
	    case CLASSIFICATION:
	      
	      z[l][idx] = new Vector(layers[l].dimension.first);
	      y[l][idx] = new Vector(layers[l].dimension.first);
	      break;
	      
	    default:
	      std::cerr << "ERROR: Allocation of memory not implemented for this layer type yet.\n";
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
  NeuralNetwork increment;

  size_t i=0;
  double grad_norm = 1.;  
  
  // TODO: Find a better stopping criterion
  while(i++ < options.max_iter && grad_norm > 1.e-10)
    {
      std::shuffle(data_idx.begin(), data_idx.end(), rnd_gen);
      
      double f = evalFunctional(data, y, z, data_idx, options);
      
      NeuralNetwork grad_net = evalGradient(data, y, z, data_idx, options);
      
      grad_norm = grad_net.norm();

      // Console output
      if(i%1000 == 0)
        {
	  std::cout << "Iteration " << i << std::endl;
	  std::cout << "functional value : " << f << std::endl;
	  std::cout << "learning rate    : " << options.learning_rate << std::endl;
	  std::cout << "gradient norm    : " << grad_norm << std::endl;
	  std::cout << "=========================================" << std::endl;
        }
      
      os << i << ", " << f << ", " << grad_norm << std::endl;
      
      // for testing only. remove later
      //gradientTest(grad_net, data, data_idx, options);
      //return;

      // Update weights
      if(i>1)
	{
	  // increment = (-options.learning_rate)*grad_net + momentum*increment;
	  increment *= momentum;
	  increment += (-options.learning_rate)*grad_net;	  
	}
      else
	{
	  increment = (-options.learning_rate)*grad_net;
	}
      
      *this += 1.*increment;
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

double NeuralNetwork::evalFunctional(const std::vector<TrainingData>& data,
				     std::vector<std::vector<DataArray*>>& y,
				     std::vector<std::vector<DataArray*>>& z,
				     const std::vector<size_t>& data_indices,
				     OptimizationOptions options) const
{
  size_t n_data = data.size();

  // Set initial layer
  switch(layers[0].layer_type)
    {
    case VECTOR_INPUT:
      for(size_t idx=0; idx<options.batch_size; ++idx)
	{
	  size_t n = data[data_indices[idx]].x->nEntries();	    
	  dynamic_cast<Vector&>(*y[0][idx]) = Vector(n, &(data[data_indices[idx]].x->operator[](0)));
	}
      break;
    case MATRIX_INPUT:
      // TODO: Implement
      // break;
    default:
      std::cerr << "ERROR: The input layer must be of type VECTOR_INPUT or MATRIX_INPUT.\n";
      break;
    }
  
  // Forward propagation
  int l=0;
  for(auto layer = layers.begin()+1; layer != layers.end(); ++layer, ++l)
    {
       switch(layer->layer_type)
	{
	case FULLY_CONNECTED:
	case CLASSIFICATION:
	  for(size_t idx=0; idx<options.batch_size; ++idx)
	    {
	      Vector& y_prev = dynamic_cast<Vector&>(*y[l][idx]);
	      
	      dynamic_cast<Vector&>(*z[l][idx]) = layer->weight * y_prev + layer->bias;	      
	      dynamic_cast<Vector&>(*y[l+1][idx]) = activate(dynamic_cast<Vector&>(*z[l][idx]), layer->activation_function);
	    }	    
	  break;
	default:
	  std::cerr << "ERROR: Forward propagation not implemented yet for layer type "
		    << Layer::LayerName[layer->layer_type] << ".\n";
	}
    }

  // Evaluate objective functional
  double f = 0.;
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
	  f -= (Y[0] * log(P[0]) + (1-Y[0])*log(1-P[0])); 
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
  
  NeuralNetwork grad_net = NeuralNetwork::createLike(*this);
    
  std::vector<DataArray*> Dy(options.batch_size);
  std::vector<DataArray*> Dz(options.batch_size);
  
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
	    size_t nL = layers.back().dimension.first;
	    Dy[idx] = new Vector(nL);	  

	    for(size_t i=0; i<nL; ++i)
	      (*Dy[idx])[i] = (1.-Y[i]) / (1.-P[i]) - Y[i] / P[i];
	  }
	  break;
	}
      Dz[idx] = new Vector();
    }
  
  // Do backpropagation
  for(size_t l=layers.size(); l-- >1; )
    {      
      for(size_t idx=0; idx<options.batch_size; ++idx)
        {
	  const Vector& z_idx = dynamic_cast<Vector&>(*z[l-1][idx]);
	  
	  switch(layers[l].activation_function)
	    {
	    case ActivationFunction::SOFTMAX:
	      // Activation functions taking all components into account
	      dynamic_cast<Vector&>(*Dz[idx]) = dynamic_cast<Vector&>(*Dy[idx]) * DactivateCoupled(z_idx, layers[l].activation_function);
	      
	      break;

	    default:
	      // Activation functions applied component-wise
	      dynamic_cast<Vector&>(*Dz[idx]) = dynamic_cast<Vector&>(*Dy[idx]) * diag(Dactivate(z_idx, layers[l].activation_function));
	    }

	  
	  dynamic_cast<Vector&>(*Dy[idx]) = dynamic_cast<Vector&>(*Dz[idx]) * layers[l].weight;
	
	  // Gradient w.r.t. weight and bias
	  grad_net.layers[l].weight += outer(dynamic_cast<Vector&>(*Dz[idx]), dynamic_cast<Vector&>(*y[l-1][idx]));
	  grad_net.layers[l].bias += dynamic_cast<Vector&>(*Dz[idx]);
        }
    }

  // TODO: Clean up memory
  for(size_t idx=0; idx<options.batch_size; ++idx)
    {
      delete Dz[idx];
      delete Dy[idx];
    }
  
  return grad_net;
}

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& net) 
{
  os << "Neural network with (" << (net.layers.size()) << " layers)\n";
  os << "  Input layer         : " << (net.layers[0]) << "\n";
  
  for(auto layer = net.layers.begin(); layer != net.layers.end(); ++ layer)
    {
      if(layer != net.layers.begin())
	os << "       ↓\n";
      
      os << "  " << *layer << std::endl;
    }
      
  return os;
}

NeuralNetwork& NeuralNetwork::operator=(const ScaledNeuralNetwork& other)
{  
  for(auto layer = other.network->layers.begin(); layer != other.network->layers.end(); ++layer)
    {
      // Create now layer with zero weights
      Layer new_layer(layer->dimension, layer->layer_type, layer->activation_function);
      
      new_layer.weight = layer->weight;
      new_layer.bias = layer->bias;

      new_layer.weight *= other.scale;
      new_layer.bias *= other.scale;

      layers.push_back(new_layer);
    }

  return *this;
}

ScaledNeuralNetwork::ScaledNeuralNetwork(double scale ,
					 const NeuralNetwork& network)
  : scale(scale), network(&network)
{
}

ScaledNeuralNetwork operator*(double scale, const NeuralNetwork& params)
{
  return ScaledNeuralNetwork(scale, params);
}

NeuralNetwork operator+(const NeuralNetwork& lhs, const NeuralNetwork& rhs)
{
  NeuralNetwork net;

  for(auto lhs_layer = lhs.layers.begin(), rhs_layer = rhs.layers.begin();
      lhs_layer != lhs.layers.end(); ++lhs_layer, ++rhs_layer)
    {
      // Create now layer with zero weights
      Layer new_layer(lhs_layer->dimension, lhs_layer->layer_type, lhs_layer->activation_function);
      
      new_layer.weight = lhs_layer->weight + rhs_layer->weight;
      new_layer.bias = lhs_layer->bias + rhs_layer->bias;

      net.layers.push_back(new_layer);
    }

  return net;  
}

NeuralNetwork operator+(const NeuralNetwork& lhs, const ScaledNeuralNetwork& rhs)
{
  NeuralNetwork net;

  for(auto lhs_layer = lhs.layers.begin(), rhs_layer = rhs.network->layers.begin();
      lhs_layer != lhs.layers.end(); ++lhs_layer, ++rhs_layer)
    {
      // Create now layer with zero weights
      Layer new_layer(lhs_layer->dimension, lhs_layer->layer_type, lhs_layer->activation_function);
      
      new_layer.weight = rhs_layer->weight;
      new_layer.weight *= rhs.scale;
      new_layer.weight += lhs_layer->weight;

      new_layer.bias = rhs_layer->bias;
      new_layer.bias *= rhs.scale;
      new_layer.bias += lhs_layer->bias;

      net.layers.push_back(new_layer);
    }

  return net;
}
/*
NeuralNetworkParameters operator+(const ScaledNeuralNetworkParameters& lhs, const ScaledNeuralNetworkParameters& rhs)
{
  NeuralNetworkParameters params;

  size_t layers = lhs.params->weight.size();
    
  params.weight.reserve(layers);
  params.bias.reserve(layers);
  params.activation.reserve(layers);

  for(size_t l=0; l<layers; ++l)
    {
      size_t m = lhs.params->weight[l].nRows();
      size_t n = lhs.params->weight[l].nCols();
      
      params.weight.push_back(Matrix(m, n));
      params.bias.push_back(Vector(m));
      params.activation.push_back(lhs.params->activation[l]);
			    
      for(size_t i=0; i<m; ++i)
        {
	for(size_t j=0; j<n; ++j)
	  {
	    params.weight[l][i][j] = lhs.scale * lhs.params->weight[l][i][j] + rhs.scale * rhs.params->weight[l][i][j];
	  }
	params.bias[l][i] = lhs.scale * lhs.params->bias[l][i] + rhs.scale * rhs.params->bias[l][i];
        }
    }
  return params;
}
*/

void NeuralNetwork::gradientTest(const NeuralNetwork& grad_net,
				 const std::vector<TrainingData>& data,
				 const std::vector<size_t>& data_idx,
				 OptimizationOptions options) const
{  
  // Initialize a direction
  NeuralNetwork direction;
  
  direction.layers.reserve(layers.size());
 
  for(auto layer = layers.begin(); layer != layers.end(); ++layer)
    {
      size_t m = layer->weight.nRows();
      size_t n = layer->weight.nCols();
      
      Matrix weight(m, n);
      Vector bias(m);
      
      for(size_t i=0; i<m; ++i)
        {
	for(size_t j=0; j<n; ++j)
	  {
	    weight[i][j] = 1.;
	  }
	bias[i] = 1.;
        }

      Layer new_layer(layer->dimension, layer->layer_type, layer->activation_function);
      
      new_layer.weight = weight;
      new_layer.bias = bias;

      direction.layers.push_back(new_layer);
    }

  double deriv_exact = grad_net.dot(direction);

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
	  switch(layers[l].layer_type)
	    {	      
	    case FULLY_CONNECTED:
	    case VECTOR_INPUT:
	    case CLASSIFICATION:
	      
	      z[l][idx] = new Vector(layers[l].dimension.first);
	      y[l][idx] = new Vector(layers[l].dimension.first);
	      break;
	      
	    default:
	      std::cerr << "ERROR: Allocation of memory not implemented for this layer type yet.\n";
	    }
	}
    }
  
  double f = evalFunctional(data, y, z, data_idx, options);
  std::cout << "Value in x0: " << f << std::endl; 
            
  for(double s=1.; s>1.e-12; s*=0.5)
    {
      NeuralNetwork net_s = (*this) + s*direction;
  
      double f_s = net_s.evalFunctional(data, y, z, data_idx, options);
      std::cout << "Value in x+s*d: " << f_s << std::endl;
      
      double deriv_fd = (f_s - f)/s;

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

double NeuralNetwork::dot(const NeuralNetwork& rhs) const
{ 
  double val = 0.;
  
  auto layer = layers.begin()+1;
  auto layer_rhs = rhs.layers.begin()+1;
  
  for(; layer != layers.end(); ++layer, ++layer_rhs)
    val += layer->dot(*layer_rhs);
   
  return val;
}

double NeuralNetwork::norm() const
{
  return dot(*this);
}

NeuralNetwork& NeuralNetwork::operator*=(double s)
{
  for(auto layer = layers.begin(); layer != layers.end(); ++layer)
    {
      layer->weight *= s;
      layer->bias *= s;
    }

  return *this;       
}

NeuralNetwork& NeuralNetwork::operator+=(const ScaledNeuralNetwork& other)
{
  auto layer = layers.begin();
  auto layer_rhs = other.network->layers.begin();
  
  for(; layer != layers.end();
      ++layer, ++layer_rhs)
    {    
      layer->weight += other.scale * layer_rhs->weight;
      layer->bias += other.scale * layer_rhs->bias;
    }
  
  return *this;
}

OptimizationOptions::OptimizationOptions() : max_iter(1e4), batch_size(128),
					     learning_rate(1.e-2), loss_function(LossFunction::MSE)
{}
