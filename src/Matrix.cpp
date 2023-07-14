#include <iomanip>
#include <cstring>

#include "Matrix.h"
#include "Vector.h"

Matrix::Matrix(size_t m, size_t n) : m(m), n(n)
{
  data = new double[m*n];
  memset(data, 0., m*n*sizeof(double));
}

Matrix::Matrix(const Matrix& other) : m(other.m), n(other.n)
{
  data = new double[m*n];
  memcpy(data, other.data, m*n*sizeof(double));
}

Matrix::~Matrix()
{
  delete[] data;
}

Matrix& Matrix::operator=(const Matrix& other)
{
  m = other.m; n = other.n;
  data = new double[m*n];
  memcpy(data, other.data, m*n*sizeof(double));
  return *this;
}

MatrixRow::MatrixRow(double* data_ptr) : data_ptr(data_ptr)
{}

Matrix& Matrix::operator=(std::initializer_list<double> val)
{
  size_t i=0;
  for(auto x = val.begin(); x!=val.end(); ++x, ++i)
    data[i] = *x;

  return *this;
}


std::pair<size_t, size_t> Matrix::size() const { return std::pair<size_t, size_t>(m, n); }
size_t Matrix::nRows() const {return m;}
size_t Matrix::nCols() const {return n;}


MatrixRow Matrix::operator[](size_t i)
{
  return MatrixRow(&(data[i*n]));
}

const MatrixRow Matrix::operator[](size_t i) const
{
  return MatrixRow(&(data[i*n]));
}

double& MatrixRow::operator[](size_t j)
{
  return data_ptr[j];
}

Vector Matrix::operator*(const Vector& b) const
{
  if(n != b.n)
    {
      std::cerr << "Matrix and vector have incompatible size: (" << m << "," << n << ") vs. (" << b.n << ")\n";
      return Vector(0);
    }

  Vector c(m);

  for(int i=0; i<m; ++i)
    {
      double val = 0.;
      for(double *data_row = &(data[i*n]), *data_vec = b.data;
	  data_row != &(data[(i+1)*n]);
	  ++data_row, ++data_vec)
	{
	  val += (*data_row)*(*data_vec);
	}
      c[i] = val;
    }
  return c;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
{
  for(size_t i=0; i<matrix.m; ++i)
    {
      os << "[ ";
      for(size_t j=0; j<matrix.n; ++j)
	os << std::left << std::setw(7) << matrix.data[i*matrix.n+j] << " ";
      os << "]\n";
    }
  return os;
}
