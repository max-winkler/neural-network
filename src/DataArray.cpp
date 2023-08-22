#include "DataArray.h"

#include <cstring>

DataArray::DataArray(size_t size)
  : size(size)
{
  data = new double[size];
  std::memset(data, 0, size*sizeof(double));
}

DataArray::DataArray(const DataArray& other)
  : size(other.size)
{
  data = new double[size];
  std::memcpy(data, other.data, size*sizeof(double));
}

DataArray::~DataArray()
{
  delete[] data;
}

double& DataArray::operator[](size_t i)
{
  return data[i];
}

const double& DataArray::operator[](size_t i) const
{
  return data[i];
}

double DataArray::inner(const DataArray& B) const
{
  const DataArray& A = *this;
  double val = 0.;

  double* A_ptr = A.data;
  double* B_ptr = B.data;
  
  for(size_t i=0; i<size; ++i, ++A_ptr, ++B_ptr)
    val += (*A_ptr)*(*B_ptr);
  
  return val;
}

size_t DataArray::nEntries() const
{
  return size;
}
