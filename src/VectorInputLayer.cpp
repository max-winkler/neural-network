#include "VectorInputLayer.h"

VectorInputLayer::VectorInputLayer(size_t dim)
  : Layer(std::vector(1, dim), LayerType::VECTOR_INPUT)
{  
}

