#include <iomanip>

#include "Layer.h"

const std::unordered_map<LayerType, std::string> Layer::LayerName =
  {
    {VECTOR_INPUT,    "Vector Input Layer"},
    {MATRIX_INPUT,    "Matrix Input Layer"},
    {FULLY_CONNECTED, "Fully connected Layer"},
    {CLASSIFICATION,  "Classification Layer"},
    {CONVOLUTION,     "Convolution Layer"},
    {POOLING,         "Pooling Layer"},
    {FLATTENING,      "Flattening Layer"},
    {UNKNOWN,         "Unknown Layer Type"}
  };

const std::unordered_map<LayerType, std::string> Layer::LayerShortName = {
  {VECTOR_INPUT,      "vectorinput"},
  {MATRIX_INPUT,      "matrixinput"},
  {FULLY_CONNECTED,   "fullyconnected"},
  {CLASSIFICATION,    "classification"},
  {CONVOLUTION,       "convolution"},
  {POOLING,           "pooling"},
  {FLATTENING,        "flattening"},
  {UNKNOWN,           "unknown"}
};

const std::unordered_map<std::string, LayerType> Layer::LayerTypeFromShortName = {
  {"vectorinput",     VECTOR_INPUT},
  {"matrixinput",     MATRIX_INPUT},
  {"fullyconnected",  FULLY_CONNECTED},
  {"classification",  CLASSIFICATION},
  {"convolution",     CONVOLUTION},
  {"pooling",         POOLING},
  {"flattening",      FLATTENING},
  {"unknown",         UNKNOWN}
};

Layer::Layer(std::vector<size_t> dim, LayerType layer_type)
  : dim(dim), layer_type(layer_type)
{}

std::ostream& operator<<(std::ostream& os, const Layer& layer)
{
  os << layer.get_name() << " (";
  os << layer.dim[0];

  for(auto it = layer.dim.begin()+1; it != layer.dim.end(); ++it)
    os << "x" << *it;

  os << " neurons)\n";
  
  return os;
}

std::string Layer::get_name() const
{
  return Layer::LayerName.at(this->layer_type);
}

// Virtual functions that must be overridden
/*
void Layer::eval(DataArray*& x) const {}
void Layer::forward_propagate(const DataArray& x, DataArray& z, DataArray& y) const {}
std::unique_ptr<Layer> Layer::backward_propagate(std::vector<DataArray*>&,
						 const std::vector<DataArray*>&,
						 const std::vector<DataArray*>&) const {
  return std::unique_ptr<Layer>(new Layer(std::vector<size_t>(0), LayerType::UNKNOWN));
}
std::unique_ptr<Layer> Layer::zeros_like() const
{
  return std::unique_ptr<Layer>(new Layer(std::vector<size_t>(0), LayerType::UNKNOWN));
}
std::unique_ptr<Layer> Layer::clone() const
{
  return std::unique_ptr<Layer>(new Layer(*this));
}
*/
float Layer::dot(const Layer&) const { return 0.;}
void Layer::initialize() {}  
void Layer::update_increment(float, const Layer&, float) {}
void Layer::apply_increment(const Layer&) {}

void Layer::save(std::ostream&) const {}

std::map<std::string, std::string> Layer::get_parameters() const
{
  return {};
}

std::map<std::string, std::pair<const float*, std::vector<size_t>>> Layer::get_weights() const
{
  return {};
}
