#include <cstring>

#include "Tensor.h"

Tensor::Tensor() : DataArray(1), d(1), m(1)
{
}

Tensor::Tensor(size_t d, size_t m, size_t n)
  : DataArray(d*m*n), d(d), m(m)
{
}

Tensor::Tensor(const Tensor& T)
  : DataArray(T), d(T.d), m(T.m)
{
}

Tensor& Tensor::operator=(const Tensor& other)
{
  if(d != other.d || m != other.m || size != other.size)
    {
      delete[] data;

      d = other.d;
      m = other.m;
      size = other.size;
      
      data = new double[size];
    }
  
  memcpy(data, other.data, size*sizeof(double));
  return *this;
}

Tensor& Tensor::operator=(Tensor&& other)
{
  if(this == &other)
    std::cerr << "WARNING: Self assignment of a tensor. This might fail.\n";

  delete[] data;

  size = other.size;
  d = other.d;
  m = other.m;

  data = other.data;
  other.data = nullptr;

  return *this;
}

size_t Tensor::nChannels() const {return d; }
size_t Tensor::nRows() const {return m; }
size_t Tensor::nCols() const {return size/m/d;}

double& Tensor::operator()(size_t c, size_t i, size_t j)
{
  // n = size/d/m
  // m = m
  return data[c*size/d + i*size/(d*m) + j];
}

const double& Tensor::operator()(size_t c, size_t i, size_t j) const
{
  return data[c*size/d + i*size/(d*m) + j];
}

Tensor& Tensor::operator*=(double a)
{
  for(double* data_ptr = data; data_ptr != data+size; ++data_ptr)
    (*data_ptr) *= a;

  return *this;
}

Tensor& Tensor::operator+=(const Tensor& T)
{

  const double* data_ptr_T;
  double* data_ptr;
    
  for(data_ptr = data, data_ptr_T = T.data;
      data_ptr != data+size;
      ++data_ptr, ++data_ptr_T)
    (*data_ptr) += (*data_ptr_T);

  return *this;
}

Tensor Tensor::operator+(const Tensor& T) const
{
  if(size != T.size || d != T.d || m != T.m)
    {
      std::cerr << "WARNING: Tensors have incompatible size for summaton.\n";
      return Tensor();
    }

  
  Tensor Z(*this);
  Z += T;
  return Z;    
}
