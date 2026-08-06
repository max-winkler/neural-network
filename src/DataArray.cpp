#include "DataArray.h"

#include <cstring>

DataArray::DataArray() : data(nullptr), size(0) {}

DataArray::DataArray(size_t size)
  : size(size)
{
  data = new float[size];

  for(size_t i=0; i<size; ++i)
    data[i] = 0.;
  // std::memset(data, 0, size*sizeof(float));
}

DataArray::DataArray(const DataArray& other)
  : size(other.size)
{
  data = new float[size];
  std::memcpy(data, other.data, size*sizeof(float));
}

DataArray::DataArray(size_t s, const float* x) : size(s), data(new float[s]) {
  std::memcpy(data, x, size*sizeof(float));
}

DataArray::DataArray(size_t s, float* x, TakeOwnership)
    : size(s), data(x) {}

  
DataArray::~DataArray()
{
  delete[] data;
}

DataArray::DataArray(DataArray&& other) noexcept
  : size(other.size), data(other.data)
{
  other.data = nullptr;
}

DataArray& DataArray::operator=(const DataArray& other) {
  if (this == &other) return *this;

  delete[] data;
  size = other.size;

  if (size > 0)
    {
      data = new float[size];
      memcpy(data, other.data, size * sizeof(float));
  } else
    data = nullptr;

  return *this; 
}  

DataArray& DataArray::operator=(DataArray&& other) noexcept {
  if (this != &other) {
    delete[] data;
    
    data = other.data;
    size = other.size;

    other.data = nullptr;
  }
  return *this;  
}  

float& DataArray::operator[](size_t i)
{
  return data[i];
}

const float& DataArray::operator[](size_t i) const
{
  return data[i];
}

float* DataArray::values() { return data; }
const float* DataArray::values() const { return data; }

float DataArray::inner(const DataArray& B) const
{
  const DataArray& A = *this;
  float val = 0.;

  float* A_ptr = A.data;
  float* B_ptr = B.data;
  
  for(size_t i=0; i<size; ++i, ++A_ptr, ++B_ptr)
    val += (*A_ptr)*(*B_ptr);
  
  return val;
}

float sum(const DataArray& A)
{
  float val = 0.;
  for(float* data_ptr = A.data; data_ptr!=A.data+A.size; ++data_ptr)
    val += *data_ptr;
  
  return val;
}

size_t DataArray::nEntries() const
{
  return size;
}
