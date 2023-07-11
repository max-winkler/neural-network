#include <cstring>
#include <iomanip>

#include "Vector.h"

Vector::Vector(size_t n) : n(n)
{
  data = new double[n];
  memset(data, 0., n*sizeof(double));
}

Vector::Vector(const Vector& other) : n(other.n)
{
  data = new double[n];
  memcpy(data, other.data, n*sizeof(double));
}

Vector::~Vector()
{
  delete[] data;
}

size_t Vector::size() const
{
  return n;
}

double& Vector::operator[](size_t i)
{
  return data[i];
}
const double& Vector::operator[](size_t i) const
{
  return data[i];
}

Vector Vector::operator+(const Vector& other) const
{
  if(n != other.n)
    {
      std::cerr << "Error: Vector sized are incompatible for summation: (" << n << ") vs. (" << other.n << ")\n";
      return Vector(0);
    }

  Vector c(n);
  for(double *data_ptr = data, *data_ptr_other = other.data, *data_ptr_c = c.data;
      data_ptr != data+n;
      ++data_ptr, ++data_ptr_other, ++data_ptr_c)
    {
      (*data_ptr_c) = (*data_ptr)+(*data_ptr_other);
    }
  return c;
}

std::ostream& operator<<(std::ostream& os, const Vector& vector)
{
  os << "[ ";
  for(size_t i=0; i<vector.n; ++i)
    {
      os << std::setw(7) << std::left << vector.data[i] << " ";
    }
  os << "]";
  
  return os;
}
