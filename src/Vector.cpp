#include <cmath>
#include <cstring>
#include <iomanip>

#include "Vector.h"

Vector::Vector() : DataArray(1)
{
}

Vector::Vector(size_t n)
  : DataArray(n)
{}

Vector::Vector(size_t n, const double* x)
  : DataArray(n)
{
  memcpy(data, x, size*sizeof(double));  
}

Vector::Vector(const Vector& other)
  : DataArray(other)
{
}

Vector::Vector(std::initializer_list<double> val) : DataArray(val.size())
{
  size_t i=0;
  for(auto x = val.begin(); x!=val.end(); ++x, ++i)
    data[i] = *x;
}

Vector::~Vector()
{
}

Vector& Vector::operator=(const Vector& other)
{
  if(size != other.size)
    {
      delete[] data;  
      size = other.size;
      data = new double[size];
    }
  
  memcpy(data, other.data, size*sizeof(double));
  return *this;
}

Vector& Vector::operator=(Vector&& other)
{
  delete[] data;
  
  size = other.size;
  data = other.data;
  other.data = nullptr;

  return *this;
}

Vector& Vector::operator=(std::initializer_list<double> val)
{
  size = val.size();
  // TODO: Check if array large big enough?
  size_t i=0;
  for(auto x = val.begin(); x!=val.end(); ++x, ++i)
    data[i] = *x;

  return *this;
}

size_t Vector::length() const
{
  return size;
}

Matrix Vector::reshape(size_t m, size_t n) const
{
  return Matrix(m, n, data);
}

Vector Vector::operator+(const Vector& other) const
{
  if(size != other.size)
    {
      std::cerr << "Error: Vector sized are incompatible for summation: (" << size << ") vs. (" << other.size << ")\n";
      return Vector(0);
    }

  Vector c(size);
  for(double *data_ptr = data, *data_ptr_other = other.data, *data_ptr_c = c.data;
      data_ptr != data+size;
      ++data_ptr, ++data_ptr_other, ++data_ptr_c)
    {
      (*data_ptr_c) = (*data_ptr)+(*data_ptr_other);
    }
  return c;
}

Vector Vector::operator-(const Vector& other) const
{
  // TODO: Implement an axpy method and replace + and - by calls to axpy
  if(size != other.size)
    {
      std::cerr << "Error: Vector sized are incompatible for summation: (" << size << ") vs. (" << other.size << ")\n";
      return Vector(0);
    }

  Vector c(size);
  for(double *data_ptr = data, *data_ptr_other = other.data, *data_ptr_c = c.data;
      data_ptr != data+size;
      ++data_ptr, ++data_ptr_other, ++data_ptr_c)
    {
      (*data_ptr_c) = (*data_ptr)-(*data_ptr_other);
    }
  return c;
}

std::ostream& operator<<(std::ostream& os, const Vector& vector)
{
  os << "[ ";
  for(size_t i=0; i<vector.size; ++i)
    {
      os << std::setw(7) << std::left << vector.data[i] << " ";
    }
  os << "]";
  
  return os;
}

Vector Vector::operator*(const Matrix& A) const
{
  size_t m = A.nRows(), n = A.nCols();

  if(this->size != m)
    {
      std::cerr << "Error: Vector and matrix have incompatible size for multiplication.\n";
      std::cerr << "  (" << this->size << ")  vs. (" << m << "," << n << ")\n";
      return Vector(0);
    }
  
  Vector y(n);

  double* A_col;
  double* x_data;
      
  for(size_t i=0; i<n; ++i)
    {
      double val=0;
      for(A_col = A.data + i, x_data = data; x_data != data+m; x_data++, A_col+=n)
        val += (*A_col)*(*x_data);

      y[i] = val;
    }

  return y;
}

Vector Vector::operator*(const DiagonalMatrix& A) const
{
  size_t n = A.diagonal->size;

  if(this->size != n)
    {
      std::cerr << "Error: Vector and matrix have incompatible size for multiplication.\n";
      std::cerr << "  (" << this->size << ")  vs. (" << n << "," << n << ")\n";
      return Vector(0);
    }
  
  Vector y(n);
      
  for(size_t i=0; i<n; ++i)
    y[i] = (*A.diagonal)[i]*data[i];

  return y;
}

Vector& Vector::operator+=(const Vector& B)
{
  operator+=(ScaledVector(1., B));
  return *this;
}

Vector& Vector::operator+=(const ScaledVector& B)
{
  if(B.vector->size != size)
    {
      std::cerr << "Error: Vectors have incompatible dimension for summation.\n";
      std::cerr << "  (" << size << ") vs. (" << B.vector->size  << ")\n";
    }

  double* data_ptr;
  const double* B_data_ptr;

  for(data_ptr = data, B_data_ptr = B.vector->data; data_ptr != data+size; ++data_ptr, ++B_data_ptr)
    *data_ptr += B.scale * (*B_data_ptr);
  
  return *this;
}

Vector& Vector::operator*=(double a)
{
  for(double* data_ptr = data; data_ptr != data + size; ++data_ptr)
    (*data_ptr) *= a;
  return *this;
}

size_t Vector::indMax() const
{
  size_t ind = 0;
  double max = data[0];
  for(size_t l=0; l<size; ++l)
    if(data[l] > max)
      {
        max = data[l];
        ind = l;
      }

  return ind;
}

DiagonalMatrix::DiagonalMatrix(const Vector& x)
  : diagonal(&x)
{
}

Rank1Matrix::Rank1Matrix(const Vector& u, const Vector& v)
  : u(&u), v(&v)
{}

size_t Rank1Matrix::nRows() const { return u->length();}
size_t Rank1Matrix::nCols() const { return v->length();}

DiagonalMatrix diag(const Vector& x)
{
  return DiagonalMatrix(x);
}

Rank1Matrix outer(const Vector& x, const Vector& y)
{
  return Rank1Matrix(x, y);
}

double norm(const Vector& x, double p)
{
  double val = 0.;
  for(double* it = x.data; it != x.data+x.size; ++it)
    val += pow(*it, 2.);
  return sqrt(val);
}

ScaledVector::ScaledVector(double scale, const Vector& vector) : scale(scale), vector(&vector) {}

ScaledVector operator*(double scale, const Vector& vector)
{
  return ScaledVector(scale, vector);
}
