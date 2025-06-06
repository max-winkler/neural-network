#include <iomanip>

#include "FlatteningLayer.h"
#include "Tensor.h"

FlatteningLayer::FlatteningLayer(size_t dim1, size_t dim2, size_t dim3)
  : Layer(std::vector<size_t>(1, dim1*dim2*dim3), LayerType::FLATTENING),
    in_dim1(dim1), in_dim2(dim2), in_dim3(dim3)
{
}

void FlatteningLayer::eval(DataArray*& x_) const
{  
  Tensor* x = dynamic_cast<Tensor*>(x_);
  DataArray* x_new = new Vector(x->flatten());
  
  delete x_;
  x_ = x_new;
}

void FlatteningLayer::forward_propagate(const DataArray& x_, DataArray& z_, DataArray& y_) const
{
  const Tensor& x = dynamic_cast<const Tensor&>(x_);
  Vector& y = dynamic_cast<Vector&>(y_);

  y = x.flatten();
}

std::unique_ptr<Layer> FlatteningLayer::backward_propagate(std::vector<DataArray*>& DY,
							   const std::vector<DataArray*>& Y,
							   const std::vector<DataArray*>& Z) const
{
  FlatteningLayer* output = new FlatteningLayer(in_dim1, in_dim2, in_dim3);

  auto y_it = Y.begin(), z_it = Z.begin();
  auto Dy_it = DY.begin();
  
  for(;  y_it != Y.end();
      ++y_it, ++z_it, ++Dy_it)
    {
      Vector* Dy = dynamic_cast<Vector*>(*Dy_it);
      
      *Dy_it = new Tensor(Dy->reshape(in_dim1, in_dim2, in_dim3));
      delete Dy;
    }

  return std::unique_ptr<Layer>(output);
}

std::unique_ptr<Layer> FlatteningLayer::clone() const
{
  return std::unique_ptr<Layer>(new FlatteningLayer(in_dim1, in_dim2, in_dim3));
}

std::unique_ptr<Layer> FlatteningLayer::zeros_like() const
{
  return std::unique_ptr<Layer>(new FlatteningLayer(in_dim1, in_dim2, in_dim3));
}

void FlatteningLayer::save(std::ostream& os) const
{
  os << "[ " << get_name() << " ]\n";
  os << std::setw(16) << " dimension : " << dim[0] << '\n';
}

