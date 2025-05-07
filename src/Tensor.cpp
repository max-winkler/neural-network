#include <cstring>

#include "Tensor.h"
#include "Vector.h"

TensorSlice::TensorSlice(size_t m, size_t n, double* data) : m(m), n(n), data(data) {}

double& TensorSlice::operator()(size_t i, size_t j)
{
  return data[i*n + j];
}

TensorSlice& TensorSlice::operator=(const Matrix& A)
{
  if(A.nRows() != m || A.nCols() != n)
    {
      std::cerr << "ERROR: Tensor slice and matrix have incompatible dimension for assignment.\n";
      return *this;
    }

  // TODO: Could be faster with memcpy
  for(size_t i=0; i<m; ++i)
    for(size_t j=0; j<n; ++j)
      (*this)(i,j) = A(i,j);

  return *this;
}

TensorSlice& TensorSlice::operator+=(double a)
{
  for(size_t i = 0; i < m*n; ++i)
    data[i] += a;

  return *this;
}

TensorSlice& TensorSlice::operator+=(const Matrix& A)
{
  for(size_t i=0; i<m; ++i)
    for(size_t j=0; j<n; ++j)
      data[i*n+j] += A(i,j);

  return *this;
}

Tensor::Tensor() : DataArray(1), d(1), m(1) {}
Tensor::Tensor(size_t d, size_t m, size_t n) : DataArray(d*m*n), d(d), m(m) {}
Tensor::Tensor(size_t d, size_t m, size_t n, const double* x) : DataArray(d*m*n), d(d), m(m)
{
  memcpy(data, x, size*sizeof(double));
}

Tensor::Tensor(const Tensor& T)
  : DataArray(T), d(T.d), m(T.m)
{
}

Tensor::Tensor(const Matrix& A)
  : DataArray(A), d(1), m(A.nRows())
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

Tensor& Tensor::operator-=(const Tensor& T)
{
  const double* data_ptr_T;
  double* data_ptr;
    
  for(data_ptr = data, data_ptr_T = T.data;
      data_ptr != data+size;
      ++data_ptr, ++data_ptr_T)
    (*data_ptr) -= (*data_ptr_T);

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

Vector Tensor::flatten() const
{
  return Vector(size, data);
}

TensorSlice Tensor::operator[](size_t c)
{
  size_t n = size/m/d;
  return TensorSlice(m, size/m/d, data + c*m*n);
}

TensorSlice Tensor::operator[](size_t c) const
{
  size_t n = size/m/d;
  return TensorSlice(m, size/m/d, data + c*m*n);
}

ScaledTensor::ScaledTensor(double scale, const Tensor& tensor) : scale(scale), tensor(&tensor) {}

ScaledTensor operator*(double scale, const Tensor& tensor)
{
  return ScaledTensor(scale, tensor);
}
