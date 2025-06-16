#include "LayerFactory.h"

#include "VectorInputLayer.h"
#include "MatrixInputLayer.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "PoolingLayer.h"
#include "FlatteningLayer.h"

#include "Activation.h"

std::unique_ptr<Layer> LayerFactory::create(LayerType type,
				    const std::vector<size_t>& dim,
				    const std::vector<size_t>& in_dim,
				    const ParameterMap& parameters,
				    const WeightMap& weights)
{
  Layer* layer;
  
  switch(type)
    {
    case VECTOR_INPUT:
      layer = new VectorInputLayer(dim[0]);
      break;
    case MATRIX_INPUT:
      layer = new MatrixInputLayer(dim[1], dim[2]);
      break;
    case FULLY_CONNECTED:
      layer = new FullyConnectedLayer(dim[0], in_dim[0],
			        ActivationFunctionFromName.at(parameters.at("activation")));
      break;
    case CONVOLUTION:
      {
        size_t F = std::stoul(parameters.at("features"));
        size_t S = std::stoul(parameters.at("stride"));
        size_t P = std::stoul(parameters.at("padding"));
        size_t k = std::stoul(parameters.at("kernelsize"));
        ActivationFunction act = ActivationFunctionFromName.at(parameters.at("activation"));
        
        // Create layer instance
        layer = new ConvolutionalLayer(in_dim, F, k, S, P, act);
      }
      break;
    case POOLING:
      {
        size_t S = std::stoul(parameters.at("stride"));
        size_t P = std::stoul(parameters.at("padding"));
        size_t k = std::stoul(parameters.at("kernelsize"));

        layer = new PoolingLayer(in_dim, k, S, P);
      }
      break;
    case FLATTENING:
      layer = new FlatteningLayer(in_dim[0], in_dim[1], in_dim[2]);
      break;
    default:
      std::cerr << "ERROR: LayerFactory can not create layers of type " << type << std::endl;
      return std::unique_ptr<Layer>(nullptr);
    }

  layer->set_weights(weights);

  return std::unique_ptr<Layer>(layer);
}
