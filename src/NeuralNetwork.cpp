#include "NeuralNetwork.h"

#include "VectorInputLayer.h"
#include "FullyConnectedLayer.h"
#include "Random.h"

#include <fstream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>

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
    {
      layers.push_back(std::make_unique<VectorInputLayer>(i));
    }
  else
    {
      std::cerr << "ERROR: Implement matrix input layer first.\n";
    }
}

void NeuralNetwork::addPoolingLayer(size_t batch)
{
  /*
  const Layer prev_layer = layers.back();
  
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
  */
}

void NeuralNetwork::addFlatteningLayer()
{
  /*
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
  */
}

void NeuralNetwork::addConvolutionLayer(size_t batch, ActivationFunction act, size_t S, size_t P)
{
  /*
  const Layer& prev_layer = layers.back();
  
  // Check if previous layer produces a matrix
  switch(prev_layer.layer_type)
    {
    case LayerType::MATRIX_INPUT:
    case LayerType::CONVOLUTION:
    case LayerType::POOLING:
      break;
    default:
      std::cerr << "ERROR: A convolution layer can only follow a " << Layer::LayerName[LayerType::MATRIX_INPUT]
		<< ", " << Layer::LayerName[LayerType::CONVOLUTION]
		<< " or " << Layer::LayerName[LayerType::POOLING] << std::endl;
      return;
    }

  const size_t m1 = prev_layer.dimension.first;
  const size_t n1 = prev_layer.dimension.second;

  // TODO: This not neccesarily yields an integer. Catch this case?
  size_t m2 = (m1-batch+2*P)/S+1;
  size_t n2 = (n1-batch+2*P)/S+1;

  Layer layer(std::pair<size_t, size_t>(m2, n2), LayerType::CONVOLUTION, act);
  layer.m = batch;
  layer.S = S;
  layer.P = P;
  
  layers.push_back(layer);
  */
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
    {
      (*it)->initialize();
      /*
      switch(it->layer_type)
	{
	case FULLY_CONNECTED:
	case CLASSIFICATION:	  
	  if(it_pred->layer_type == LayerType::FULLY_CONNECTED
	     || it_pred->layer_type == LayerType::VECTOR_INPUT
	     || it_pred->layer_type == LayerType::FLATTENING)	    
	    {
	      // Get layer
	      auto layer = dynamic_cast<FullyConnectedLayer&>(*it);
		
	      // Build weight matrix and bias vector
	      size_t m = it->dim[0];
	      size_t n = it_pred->dim[0];

	      // Initialize random matrix
	      Matrix W(m, n);
	      for(size_t i=0; i<m; ++i)
		for(size_t j=0; j<n; ++j)
		  W[i][j] = -1.+2*random_real(rnd_gen);
	      
	      layer.set_weight(W);
	      layer.set_bias(Vector(m));
	    }
	  else
	    {
	      std::cerr << "ERROR: A " << Layer::LayerName[it->layer_type]
			<< " follows a " << Layer::LayerName[it_pred->layer_type]
			<< " which is incompatible.\n";
	      return;
	    }
	  break;
	  
	case FLATTENING:
	case POOLING:
	  // nothing to be done here
	  break;
	  
	case CONVOLUTION:
	  
	  {
	    Matrix W(it->m,it->m);
	    for(size_t i=0; i<it->m; ++i)
	      for(size_t j=0; j<it->m; ++j)
		W[i][j] = 0.5+random_normal(rnd_gen);

	    it->weight = W;
	    it->bias = Vector(1);
	  }
	  
	  break;
	  
	default:
	  std::cerr << "ERROR: Initialization of neural network with " << Layer::LayerName[it->layer_type]
		    << " is not implemented yet.\n";
	  break;
	}
      */
    }
  
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
      x_tmp = new Matrix(dynamic_cast<const Matrix&>(x));
      break;      
    }
  
  // Hidden and output layers
  for(auto layer_it = layers.begin()+1; layer_it != layers.end(); ++layer_it)    
    (*layer_it)->forward_propagate(*x_tmp);
  
  // OLD VERSION
  /*
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
	case LayerType::CONVOLUTION:
	  {
	    Matrix& x_ref = dynamic_cast<Matrix&>(*x_tmp);
	    x_ref = x_ref.convolve(layer->weight, layer->S, layer->P);
	    x_ref += layer->bias[0];
	    x_ref = activate(x_ref, layer->activation_function);
	  }
	  break;	  
	default:
	  std::cerr << "ERROR: Evaluation of neural networks involving " << Layer::LayerName[layer->layer_type]
		  << " is invalid or not implemented yet.\n";
	  return Vector();
	}      
    }
  */
  
  Vector y = dynamic_cast<Vector&>(*x_tmp);
  
  return y;
}

void NeuralNetwork::train(const std::vector<TrainingData>& data, OptimizationOptions options)
{
  // Parameters for momentum method
  const double momentum = 0.3;
  
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
	      /*
	      y[l][idx] = new Matrix(layers[l].dimension.first, layers[l].dimension.second);
	      z[l][idx] = new Matrix(layers[l].dimension.first, layers[l].dimension.second);
	      */
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
			<< Layer::LayerName[layers[l]->layer_type] << " yet.\n";
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

  double grad_norm = 1.;  
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

	  double f = evalFunctional(data, y, z, batch_data_idx, options);
      
	  NeuralNetwork grad_net = evalGradient(data, y, z, batch_data_idx, options);
	
	  grad_norm = grad_net.norm();

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

void NeuralNetwork::update_increment(double momentum, const NeuralNetwork& gradient, double step)
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

double NeuralNetwork::evalFunctional(const std::vector<TrainingData>& data,
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
	  dynamic_cast<Matrix&>(*y[0][idx]) = dynamic_cast<Matrix&>(*data[data_indices[idx]].x);
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
	(*layer)->eval_functional(*y[l][idx], *z[l][idx], *y[l+1][idx]);			     
    }
  
  // Forward propagation
  /*
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
	  
	case CONVOLUTION:
	  for(size_t idx=0; idx<options.batch_size; ++idx)
	    {
	      Matrix& y_prev = dynamic_cast<Matrix&>(*y[l][idx]);
	      Matrix& z_idx = dynamic_cast<Matrix&>(*z[l][idx]);
	      
	      z_idx = y_prev.convolve(layer->weight, layer->S, layer->P);
	      z_idx += layer->bias[0];
	      dynamic_cast<Matrix&>(*y[l+1][idx]) = activate(z_idx, layer->activation_function);
	    }
	  break;
	  
	default:
	  std::cerr << "ERROR: Forward propagation not implemented yet for layer type "
		  << Layer::LayerName[layer->layer_type] << ".\n";
	}
    }
  */
  
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
    {
      grad_net.layers[l] = layers[l]->backpropagate(Dy, y[l-1], z[l-1]);
    }
  
  // Backward propagation (old version)
  /*
  for(size_t l=layers.size(); l-- >1; )
    {      
      for(size_t idx=0; idx<options.batch_size; ++idx)
        {
	switch(layers[l].layer_type)
	  {
	  case FULLY_CONNECTED:
	  case CLASSIFICATION:
	    {
	      const Vector& y_idx = dynamic_cast<Vector&>(*y[l-1][idx]);
	      const Vector& z_idx = dynamic_cast<Vector&>(*z[l-1][idx]);
	      Vector& Dy_idx = dynamic_cast<Vector&>(*Dy[idx]);
	      Vector Dz_idx;
	      
	      switch(layers[l].act)
	        {
	        case ActivationFunction::SOFTMAX:
		// Activation functions taking all components into account
	          Dz_idx = Dy_idx * DactivateCoupled(z_idx, layers[l].activation_function);		  
		break;
		  
	        default:
		// Activation functions applied component-wise
		  Dz_idx = Dy_idx * diag(Dactivate(z_idx, layers[l].activation_function));
	        }
	      
	      
	      // Gradient w.r.t. weight and bias
	      grad_net.layers[l].weight += outer(Dz_idx, y_idx);
	      grad_net.layers[l].bias += Dz_idx;

	      // Gradient w.r.t. data
	      Dy_idx = Dz_idx * layers[l].weight;
	    }
	    break;

	  case FLATTENING:
	    {
	      DataArray* tmp = Dy[idx];
	      Dy[idx] = new Matrix(dynamic_cast<Vector&>(*Dy[idx]).reshape(layers[l-1].dimension.first,
							       layers[l-1].dimension.second));
	      delete tmp;
	    }
	    break;
	  case POOLING:
	    {
	      Matrix& Dy_idx = dynamic_cast<Matrix&>(*Dy[idx]);
	      const Matrix& y_res = dynamic_cast<Matrix&>(*y[l-1][idx]);
	      
	      Dy_idx = Dy_idx.unpool(y_res, POOLING_MAX, layers[l].S, layers[l].P);
	    }	    
	    break;
	  case CONVOLUTION:
	    {
	      Matrix& Dy_idx = dynamic_cast<Matrix&>(*Dy[idx]);
	      const Matrix& z_idx = dynamic_cast<Matrix&>(*z[l-1][idx]);	      
	      const Matrix& y_idx = dynamic_cast<Matrix&>(*y[l-1][idx]);
	      const Matrix Dz_idx = Dactivate(z_idx, layers[l].activation_function);
	      
	      // Gradient w.r.t. kernel matrix and bias
	      grad_net.layers[l].weight += y_idx.back_convolve(multiply(Dz_idx, Dy_idx), layers[l].S, layers[l].P);
	      grad_net.layers[l].bias[0] += Dy_idx.inner(Dz_idx);
	      
	      // Gradient w.r.t. data
	      Dy_idx = multiply(Dz_idx, Dy_idx).kron(layers[l].weight);
	    }
	    break;
	  default:
	    std::cerr << "ERROR: Backpropagation over " << Layer::LayerName[layers[l].layer_type]
		    << " not implemented yet.\n";
	    break;
	  }
        }
    }
  */
  
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

  for(auto layer = layers.begin(); layer!=layers.end(); ++layer)
    {
      os << "[ " << (*layer)->get_name() << " ]\n";
      os << "  dimension : " << (*layer)->dim[0] << '\n'; // << ", " << layer->dim[1] << '\n';
      // TODO: Reimplement this
      /*
	os << "  stride    : " << layer->S << '\n';
	os << "  padding   : " << layer->P << '\n';
	os << "  weight    : ";
      
      for(size_t i=0; i<layer->weight.nRows(); ++i)
        {
	if(i>0)
	  os << "             ";
	
	for(size_t j=0; j<layer->weight.nCols(); ++j)
	  os << std::setw(12) << layer->weight[i][j] << (j<layer->weight.nCols()-1 ? ", " : "\n");
        }
      os << "  bias      : ";
      for(size_t i=0; i<layer->bias.length(); ++i)
        os << std::setw(12) << layer->bias[i] << (i<layer->bias.length()-1 ? ", " : "\n");	  
      */
    }
  
  os.close();
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

  /*
  for(auto layer = layers.begin(); layer != layers.end(); ++layer)
    {
      const size_t m = layer->weight.nRows();
      const size_t n = layer->weight.nCols();

      const size_t bias_dim = (layer->layer_type == LayerType::CONVOLUTION ? 1 : m);
      
      Matrix weight(m, n);
      Vector bias(bias_dim);
      
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
  */
  
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
	  switch(layers[l]->layer_type)
	    {
	    case MATRIX_INPUT:
	    case POOLING:
	    case CONVOLUTION:
	      // TODO: Are y and z really needed in input layers?
	      /*
		y[l][idx] = new Matrix(layers[l].dim[0], layers[l].dimension.second);
		z[l][idx] = new Matrix(layers[l].dimension.first, layers[l].dimension.second);
	      */
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
			<< layers[l]->get_name() << " yet.\n";
	    }
	}
    }
  
  double f = evalFunctional(data, y, z, data_idx, options);
  std::cout << "Value in x0: " << f << std::endl; 
  
  for(double s=1.; s>1.e-12; s*=0.5)
    {      
      NeuralNetwork dir_s(direction);
      dir_s.update_increment(-s, zero_net, 0.);
      
      NeuralNetwork net_s(*this);
      net_s.apply_increment(dir_s);
  
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
  
  auto layer = layers.begin();
  auto layer_rhs = rhs.layers.begin();
  
  for(; layer != layers.end(); ++layer, ++layer_rhs)
    val += (*layer)->dot(**layer_rhs);
   
  return val;
}

double NeuralNetwork::norm() const
{
  return dot(*this);
}

OptimizationOptions::OptimizationOptions() : max_iter(1e4), batch_size(128),
				     learning_rate(1.e-2), loss_function(LossFunction::MSE),
				     output_every(1), epochs(3)				     
{}
