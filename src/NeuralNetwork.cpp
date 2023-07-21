#include "NeuralNetwork.h"

#include <fstream>
#include <algorithm>
#include <numeric>
#include <chrono>

NeuralNetwork::NeuralNetwork() : initialized(false), width(), layers(0), rnd_gen(std::random_device()())
{
}

NeuralNetwork::NeuralNetwork(std::vector<size_t> width_)
  : initialized(true), width(width_), layers(width_.size()), rnd_gen(std::random_device()())
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

void NeuralNetwork::addLayer(size_t width, ActivationFunction act)
{
  this->width.push_back(width);
  params.activation.push_back(act);
  ++layers;
}

void NeuralNetwork::initialize()
{
  // Add output dimension
  width.push_back(1);

  // Reserve memory for parameters
  params.weight.reserve(layers);
  params.bias.reserve(layers);
  
  // Build up weight matrices and bias vectors
  for(Dimension::const_iterator it = width.begin(); it+1 != width.end(); ++it)
    {
      size_t m=*(it+1), n=*it;

      // Initialize random matrix
      Matrix W(m, n);
      for(size_t i=0; i<m; ++i)
	for(size_t j=0; j<n; ++j)
	  W[i][j] = random_real(rnd_gen);
	  
      params.weight.push_back(W);
      params.bias.push_back(Vector(m));
    }  
  
  initialized = true;
}

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

double NeuralNetwork::eval(const Vector& x) const
{
  Vector x_tmp(x);
  for(size_t l=0; l<layers; ++l)    
    x_tmp = activate(params.weight[l] * x_tmp + params.bias[l], params.activation[l]);

  return x_tmp[0];
}

void NeuralNetwork::train(const std::vector<TrainingData>& data, size_t batch_size)
{
  size_t n_data = data.size();
  
  if(batch_size == 0)
    batch_size = n_data;
      
  // Auxiliary variables reused in backpropagation
  std::vector<std::vector<Vector>> z;
  std::vector<std::vector<Vector>> y;

  // Index set for training data
  std::vector<size_t> data_idx(n_data);
  std::iota(data_idx.begin(), data_idx.end(), 0);
    
  // Initialize time measurement
  std::chrono::steady_clock::time_point begin_time = std::chrono::steady_clock::now();
  
  std::ofstream os;
  os.open("training.csv");  
  
  double learning_rate = 1.e-2;
  double grad_norm = 1.;

  size_t i=0;
  // TODO: Find a better stopping criterion
  while(i++ < 1e4)
    {
      std::shuffle(data_idx.begin(), data_idx.end(), rnd_gen);
      
      double f = eval_functional(params, data, y, z, data_idx, batch_size);
      
      NeuralNetworkParameters grad_params = eval_gradient(params, data, y, z, data_idx, batch_size);
      NeuralNetworkParameters params_new;

      grad_norm = sqrt(grad_params.dot(grad_params));
      
      // for testing only. remove later
      //gradient_test(grad_params, data, data_idx, batch_size);
      //return;
      
      double f_new = 2*f;
      while(learning_rate > 1.e-10 && f_new > f)
        {
	  params_new = params + (-learning_rate)*grad_params;
	  f_new = eval_functional(params_new, data, y, z, data_idx, batch_size);
	  learning_rate /= 2.;
        }
      params = params_new;

      if(i%1 == 0)
        {
	std::cout << "Iteration " << i << std::endl;
	std::cout << "functional value : " << f << std::endl;
	std::cout << "learning rate    : " << 2.*learning_rate << std::endl;
	std::cout << "gradient norm    : " << grad_norm << std::endl;
	std::cout << "=========================================" << std::endl;
        }

      os << i << ", " << f << ", " << grad_norm << std::endl;
      
      learning_rate *= 4.;
    }
  os.close();

  std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();

  std::cout << "Learning process finished after "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count()
	    << "[ms]" << std::endl;
}

double NeuralNetwork::eval_functional(const NeuralNetworkParameters& params,
			        const std::vector<TrainingData>& data,
			        std::vector<std::vector<Vector>>& y,
			        std::vector<std::vector<Vector>>& z,
			        const std::vector<size_t>& data_indices,
			        size_t batch_size) const
{
  size_t n_data = data.size();

  y = std::vector<std::vector<Vector>>(layers+1);
  z = std::vector<std::vector<Vector>>(layers);
  
  // Allocate memory for auxiliary vectors
  for(size_t l=0; l<layers+1; ++l)
    {
      if(l<layers)
        z[l] = std::vector<Vector>(batch_size);
      y[l] = std::vector<Vector>(batch_size);
    }
  
  for(size_t idx=0; idx<batch_size; ++idx)
    y[0][idx] = data[data_indices[idx]].x;
    
  for(size_t l=0; l<layers; ++l)    
    for(size_t idx=0; idx<batch_size; ++idx)
      {        
        z[l][idx] = params.weight[l]*y[l][idx] + params.bias[l];
        y[l+1][idx] = activate(z[l][idx], params.activation[l]);
      }
  
  double f = 0.;
  for(size_t idx=0; idx<batch_size; ++idx)
    f += 0.5*pow(y[layers][idx][0] - data[data_indices[idx]].y, 2.);

  return f;
}

NeuralNetworkParameters NeuralNetwork::eval_gradient(const NeuralNetworkParameters& params,
					   const std::vector<TrainingData>& data,
					   const std::vector<std::vector<Vector>>& y,
					   const std::vector<std::vector<Vector>>& z,
					   const std::vector<size_t>& data_indices,
					   size_t batch_size) const
{
  size_t n_data = data.size();
  
  NeuralNetworkParameters grad_params;

  grad_params.weight = std::vector<Matrix>(layers);  
  grad_params.bias = std::vector<Vector>(layers);
  grad_params.activation = std::vector<ActivationFunction>(layers);
    
  std::vector<Vector> Dy(n_data);
  std::vector<Vector> Dz(n_data);
 
  // Initialize gradient
  for(size_t l=0; l<layers; ++l)
    {
      grad_params.weight[l] = Matrix(width[l+1], width[l]);
      grad_params.bias[l] = Vector(width[l+1]);
      grad_params.activation[l] = params.activation[l];
    }

  for(size_t idx=0; idx<batch_size; ++idx)
    Dy[idx] = y[layers][idx] - Vector({data[data_indices[idx]].y});
  
  for(size_t l=layers; l-- >0; )
    {
      for(size_t idx=0; idx<batch_size; ++idx)
        {
	Dz[idx] = Dy[idx] * DiagonalMatrix(Dactivate(z[l][idx], params.activation[l]));
	Dy[idx] = Dz[idx] * params.weight[l];	
	
	// Gradient w.r.t. weight and bias
	grad_params.weight[l] += outer(Dz[idx], y[l][idx]);	
	grad_params.bias[l] += Dz[idx];
        }
    }

  return grad_params;
}

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& net) 
{
  os << "Neural network with (" << (net.layers) << " layers)\n";
  os << "  Input dimension         : " << (net.width[0]) << "\n";

  os << "  Hidden layer dimensions : ";
  for(size_t i=1; i<net.layers-1; ++i)    
    os << net.width[i] << ", ";
  os << net.width[net.layers-1] << std::endl;
  os << "  Weights and biases      : ";
  os << net.params;
  
  return os;
}

ScaledNeuralNetworkParameters::ScaledNeuralNetworkParameters(double scale ,
						 const NeuralNetworkParameters& params)
  : scale(scale), params(&params)
{
}

ScaledNeuralNetworkParameters operator*(double scale, const NeuralNetworkParameters& params)
{
  return ScaledNeuralNetworkParameters(scale, params);
}

NeuralNetworkParameters operator+(const NeuralNetworkParameters& lhs, const ScaledNeuralNetworkParameters& rhs)
{
  NeuralNetworkParameters params;

  size_t layers = lhs.weight.size();
    
  params.weight.reserve(layers);
  params.bias.reserve(layers);
  params.activation.reserve(layers);

  for(size_t l=0; l<layers; ++l)
    {
      size_t m = lhs.weight[l].nRows();
      size_t n = lhs.weight[l].nCols();
      
      params.weight.push_back(Matrix(m, n));
      params.bias.push_back(Vector(m));
      params.activation.push_back(lhs.activation[l]);
			    
      for(size_t i=0; i<m; ++i)
        {
	for(size_t j=0; j<n; ++j)
	  {
	    params.weight[l][i][j] = lhs.weight[l][i][j] + rhs.scale * rhs.params->weight[l][i][j];
	  }
	params.bias[l][i] = lhs.bias[l][i] + rhs.scale * rhs.params->bias[l][i];
        }
    }
  return params;
}

void NeuralNetwork::gradient_test(const NeuralNetworkParameters& grad_params,
				  const std::vector<TrainingData>& data,
				  const std::vector<size_t>& data_idx,
				  size_t batch_size) const
{  
  // Initialize a direction
  NeuralNetworkParameters direction;
  
  direction.weight.reserve(layers);
  direction.bias.reserve(layers);
  direction.activation.reserve(layers);
 
  for(size_t l=0; l<layers; ++l)
    {
      size_t m = params.weight[l].nRows();
      size_t n = params.weight[l].nCols();

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
      
      direction.weight.push_back(weight);
      direction.bias.push_back(bias);
      direction.activation.push_back(params.activation[l]);
    }

  std::cout << "Neural network parameters:\n";
  std::cout << params << std::endl;

  std::cout << "Gradient parameters:\n";
  std::cout << grad_params << std::endl;

  std::cout << "Direction parameters:\n";
  std::cout << direction << std::endl;
  
  double deriv_exact = grad_params.dot(direction);

  size_t n_data = data.size();
    
  std::vector<std::vector<Vector>> z(layers);
  std::vector<std::vector<Vector>> y(layers+1);

  // Allocate memory for auxiliary vectors
  for(size_t l=0; l<layers+1; ++l)
    {
      if(l<layers)
        z[l] = std::vector<Vector>(n_data);
      y[l] = std::vector<Vector>(n_data);
    }

  double f = eval_functional(params, data, y, z, data_idx, batch_size);  
            
  for(double s=1.; s>1.e-12; s*=0.5)
    {
      NeuralNetworkParameters params_s = params + s*direction;
  
      double f_s = eval_functional(params_s, data, y, z, data_idx, batch_size);      
      double deriv_fd = (f_s - f)/s;

      std::cout << "  derivative by gradient: " << deriv_exact
	      << " vs. by difference quotient: " << deriv_fd
	      << " relative error: " << std::abs(deriv_exact - deriv_fd)/deriv_exact << std::endl;
      
    }  
}

double NeuralNetworkParameters::dot(const NeuralNetworkParameters& rhs) const
{
  double d = 0.;

  size_t layers = weight.size();
  for(size_t l=0; l<layers; ++l)
    {
      size_t m = weight[l].nRows();
      size_t n = weight[l].nCols();
      
      for(size_t i=0; i<m; ++i)
        {
	for(size_t j=0; j<n; ++j)
	  d += weight[l][i][j] * rhs.weight[l][i][j];
	
	d += bias[l][i] * rhs.bias[l][i];
        }
    }
  
  return d;
}

std::ostream& operator<<(std::ostream& os, const NeuralNetworkParameters& params)
{
  size_t layers = params.weight.size();

  for(size_t l=0; l<layers; ++l)
    {
      os << "W" << l << " =\n" << params.weight[l];
      os << "B" << l << " =\n" << params.bias[l] << std::endl << std::endl;;
      os << "Activation  (" << l << "): " << params.activation[l] << std::endl;
    }
  return os;
}
