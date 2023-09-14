#include "NeuralNetwork.h"

#include <fstream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>

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

      new_layer.S = layer->S;
      new_layer.P = layer->P;
      
      other.layers.push_back(new_layer);
    }

  return other;
}

void NeuralNetwork::addInputLayer(size_t i, size_t j)
{
  LayerType layer_type = (j==0 ? LayerType::VECTOR_INPUT : LayerType::MATRIX_INPUT);
  Layer layer(std::pair<size_t, size_t>(i, j), layer_type, ActivationFunction::NONE);
  layers.push_back(layer);
}

void NeuralNetwork::addPoolingLayer(size_t batch)
{
  const Layer& prev_layer = layers.back();

  // Check if layer matches previous layer
  switch(prev_layer.layer_type)
    {
    case LayerType::MATRIX_INPUT:
    case LayerType::CONVOLUTION:
    case LayerType::POOLING:
      break;
    default:
      std::cerr << "ERROR: A pooling layer can only follow a " << Layer::LayerName[LayerType::MATRIX_INPUT]
	      << ", " << Layer::LayerName[LayerType::CONVOLUTION]
	      << " or " << Layer::LayerName[LayerType::POOLING] << std::endl;
      return;
    }

  // Dimension of resulting matrix
  size_t m = (prev_layer.dimension.first-1) / batch + 1;
  size_t n = (prev_layer.dimension.second-1) / batch + 1;

  // Add layer
  Layer layer(std::pair<size_t, size_t>(m, n), LayerType::POOLING, ActivationFunction::NONE);
  // TODO: Extend this function to variable padding P
  layer.S = batch;
  layers.push_back(layer);
}

void NeuralNetwork::addFlatteningLayer()
{
  const Layer& prev_layer = layers.back();

  // Check if previous layer produces a matrix
  switch(prev_layer.layer_type)
    {
    case LayerType::MATRIX_INPUT:
    case LayerType::CONVOLUTION:
    case LayerType::POOLING:
      break;
    default:
      std::cerr << "ERROR: A flattening layer can only follow a " << Layer::LayerName[LayerType::MATRIX_INPUT]
		<< ", " << Layer::LayerName[LayerType::CONVOLUTION]
		<< " or " << Layer::LayerName[LayerType::POOLING] << std::endl;
      return;
    }
  // Dimension of flattened matrix
  size_t dim = prev_layer.dimension.first * prev_layer.dimension.second;
  
  Layer layer(std::pair<size_t, size_t>(dim, 0), LayerType::FLATTENING, ActivationFunction::NONE);
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
		  W[i][j] = -1.+2*random_real(rnd_gen); // TODO: Which interval range is best?
	      
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
      else if(it->layer_type == LayerType::FLATTENING
	    || it->layer_type == LayerType::POOLING)
        {
	// Nothing to be done here
        }
      else
        {
	std::cerr << "ERROR: Initialization of neural network with " << Layer::LayerName[it->layer_type]
		<< " is not implemented yet.\n";
        }
    }
  
  initialized = true;
}

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
	case LayerType::MATRIX_INPUT:
	  x_tmp = new Matrix(dynamic_cast<const Matrix&>(x));
	  break;
	case LayerType::FULLY_CONNECTED:
	case LayerType::CLASSIFICATION:
	  {
	    Vector& x_ref = dynamic_cast<Vector&>(*x_tmp);
	    
	    dynamic_cast<Vector&>(*x_tmp) = activate(layer->weight * x_ref + layer->bias, layer->activation_function);
	  }
	  break;
	case LayerType::FLATTENING:
	  {
	    // TODO: Delete old x_tmp after flattening
	    const Matrix* tmp = dynamic_cast<const Matrix*>(x_tmp);
	    x_tmp = new Vector(tmp->flatten());
	    delete tmp;
	  }
	  break;
	case LayerType::POOLING:
	  {
	    Matrix& x_ref = dynamic_cast<Matrix&>(*x_tmp);
	    x_ref = x_ref.pool(POOLING_MAX, layer->S, 0);
	  }
	  break;
	default:
	  std::cerr << "ERROR: Evaluation of neural networks involving " << Layer::LayerName[layer->layer_type]
		  << " is invalid or not implemented yet.\n";
	  return Vector();
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
	    case MATRIX_INPUT:
	    case POOLING:
	      // TODO: Are y and z really needed in input layers?
	      y[l][idx] = new Matrix(layers[l].dimension.first, layers[l].dimension.second);
	      z[l][idx] = new Matrix(layers[l].dimension.first, layers[l].dimension.second);

	      break;
	    case FULLY_CONNECTED:
	    case VECTOR_INPUT:
	    case CLASSIFICATION:
	    case FLATTENING:
	      z[l][idx] = new Vector(layers[l].dimension.first);
	      y[l][idx] = new Vector(layers[l].dimension.first);
	      break;
	      
	    default:
	      std::cerr << "ERROR: Allocation of memory not implemented for "
			<< Layer::LayerName[layers[l].layer_type] << " yet.\n";
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

  double grad_norm = 1.;  
  size_t i=0;
  for(size_t epoch=0; epoch<options.epochs; ++epoch)
    {
      std::cout << "**************************************************************\n";
      std::cout << "* Epoch " << epoch+1 << " of " << options.epochs << std::endl;
      std::cout << "**************************************************************\n";

      std::cout << std::setw(20) << "Batch" << std::setw(20) << "Functional value" << std::setw(20) << "gradient norm" << std::endl;
      
      std::shuffle(data_idx.begin(), data_idx.end(), rnd_gen);

      for(size_t start_idx=0; start_idx<n_data-options.batch_size; start_idx+=options.batch_size)
        {	
	std::vector<size_t> batch_data_idx(data_idx.begin() + start_idx,
				     data_idx.begin() + start_idx + options.batch_size);

	double f = evalFunctional(data, y, z, batch_data_idx, options);
      
	NeuralNetwork grad_net = evalGradient(data, y, z, batch_data_idx, options);
	
	grad_norm = grad_net.norm();

	// Console output
	if(i++%options.output_every == 0)
	  {
	    std::cout << std::setw(13) << start_idx/options.batch_size << " / " << std::setw(4) << n_data/options.batch_size;
	    std::cout << std::setw(20) << f << std::setw(20) << grad_norm << std::endl;
	  }
	
	// for testing only. remove later
	// gradientTest(grad_net, data, data_idx, options);
	// return;

	// Update weights
	if(epoch==0 && start_idx==0)
	  {
	    increment = (-options.learning_rate)*grad_net;
	  }
	else
	  {
	    // increment = (-options.learning_rate)*grad_net + momentum*increment;
	    increment *= momentum;
	    increment += (-options.learning_rate)*grad_net;	  
	  }
      
	*this += 1.*increment;
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
	  dynamic_cast<Vector&>(*y[0][idx]) = dynamic_cast<Vector&>(*data[data_indices[idx]].x);
	}
      break;
      
    case MATRIX_INPUT:
      for(size_t idx=0; idx<options.batch_size; ++idx)
	{
	  dynamic_cast<Matrix&>(*y[0][idx]) = dynamic_cast<Matrix&>(*data[data_indices[idx]].x);
	}      
      break;
      
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
	case FLATTENING:
	  for(size_t idx=0; idx<options.batch_size; ++idx)
	    {
	      Matrix& y_prev = dynamic_cast<Matrix&>(*y[l][idx]);
	  
	      // z is unused here
	      dynamic_cast<Vector&>(*y[l+1][idx]) = y_prev.flatten();
	    }
	  break;
	case POOLING:
	  for(size_t idx=0; idx<options.batch_size; ++idx)
	    {
	      Matrix& y_prev = dynamic_cast<Matrix&>(*y[l][idx]);
	      
	      // z is unused here
	      dynamic_cast<Matrix&>(*y[l+1][idx]) = y_prev.pool(POOLING_MAX, layer->S, layer->P);
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
  
  // Backward propagation
  for(size_t l=layers.size(); l-- >1; )
    {      
      for(size_t idx=0; idx<options.batch_size; ++idx)
        {
	switch(layers[l].layer_type)
	  {
	  case FULLY_CONNECTED:
	  case CLASSIFICATION:
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
	    break;

	  case FLATTENING:
	    {
	      DataArray* tmp = Dy[idx];
	      Dy[idx] = new Matrix(dynamic_cast<Vector&>(*Dy[idx]).reshape(layers[l].dimension.first,
							       layers[l].dimension.second));
	      delete tmp;
	    }
	    break;
	  case POOLING:
	    {
	      DataArray* tmp = Dy[idx];

	      Matrix& Dy_idx = dynamic_cast<Matrix&>(*Dy[idx]);
	      const Matrix& y_res = dynamic_cast<Matrix&>(*y[l-1][idx]);
	      
	      Dy_idx = Dy_idx.unpool(y_res, POOLING_MAX, layers[l].S, layers[l].P);
	    }	    
	    break;
	  default:
	    std::cerr << "ERROR: Backpropagation over " << Layer::LayerName[layers[l].layer_type]
		    << " not implemented yet.\n";
	    break;
	  }
        }
    }

  for(size_t idx=0; idx<options.batch_size; ++idx)
    {
      delete Dz[idx];
      delete Dy[idx];
    }
  
  return grad_net;
}

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& net) 
{
  os << "Neural network with (" << (net.layers.size()) << " layers)\n\n";
  
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

      new_layer.S = layer->S;
      new_layer.P = layer->P;

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
  return lhs + ScaledNeuralNetwork(1., rhs);
}

NeuralNetwork operator+(const NeuralNetwork& lhs, const ScaledNeuralNetwork& rhs)
{
  NeuralNetwork net;

  for(auto lhs_layer = lhs.layers.begin(), rhs_layer = rhs.network->layers.begin();
      lhs_layer != lhs.layers.end(); ++lhs_layer, ++rhs_layer)
    {
      // Create now layer with zero weights
      Layer new_layer(lhs_layer->dimension, lhs_layer->layer_type, lhs_layer->activation_function);

      new_layer.weight = lhs_layer->weight;
      new_layer.weight+= rhs.scale * rhs_layer->weight;            

      new_layer.bias = lhs_layer->bias;
      new_layer.bias+= rhs.scale * rhs_layer->bias;

      new_layer.S = lhs_layer->S;
      new_layer.P = lhs_layer->P;
      
      net.layers.push_back(new_layer);
    }

  return net;
}

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
	    case MATRIX_INPUT:
	    case POOLING:
	      // TODO: Are y and z really needed in input layers?
	      y[l][idx] = new Matrix(layers[l].dimension.first, layers[l].dimension.second);
	      z[l][idx] = new Matrix(layers[l].dimension.first, layers[l].dimension.second);

	      break;
	    case FULLY_CONNECTED:
	    case VECTOR_INPUT:
	    case CLASSIFICATION:
	    case FLATTENING:
	      z[l][idx] = new Vector(layers[l].dimension.first);
	      y[l][idx] = new Vector(layers[l].dimension.first);
	      break;
	      
	    default:
	      std::cerr << "ERROR: Allocation of memory not implemented for "
			<< Layer::LayerName[layers[l].layer_type] << " yet.\n";
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
				     learning_rate(1.e-2), loss_function(LossFunction::MSE),
				     output_every(1), epochs(3)				     
{}
