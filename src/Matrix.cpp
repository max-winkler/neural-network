#include <iomanip>
#include <cstring>

#include "Matrix.h"
#include "Vector.h"

Matrix::Matrix() : n(1), m(1)
{
  data = new double[1];
  data[0] = 0.;
}

Matrix::Matrix(size_t m, size_t n) : m(m), n(n)
{
  data = new double[m*n];
  memset(data, 0., m*n*sizeof(double));
}

Matrix::Matrix(size_t m, size_t n, const unsigned char* pixels) : m(m), n(n)
{
  data = new double[m*n];

  double* data_ptr;
  const unsigned char* pixel_ptr;
  
  for(data_ptr = data, pixel_ptr = pixels; data_ptr != data+m*n; ++data_ptr, ++pixel_ptr)
    *data_ptr = double(*pixel_ptr) / 255.;
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
  if(m != other.m || n != other.n)
    {
      delete[] data;
      m = other.m; n = other.n;
      data = new double[m*n];
    }
  
  memcpy(data, other.data, m*n*sizeof(double));
  return *this;
}

Matrix& Matrix::operator=(Matrix&& other)  
{
  delete[] data;
  
  n = other.n;
  m = other.m;
  
  data = other.data;
  other.data = nullptr;
  
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

Matrix& Matrix::operator*=(double a)
{
  for(double* data_ptr = data; data_ptr != data+m*n; ++data_ptr)
    (*data_ptr) *= a;
  
  return *this;  
}
  
const double& MatrixRow::operator[](size_t j) const
{
  return data_ptr[j];
}

Vector Matrix::operator*(const Vector& b) const
{
  if(n != b.n)
    {
      std::cerr << "Error: Matrix and vector have incompatible size for multiplication.\n";
      std::cerr << "  (" << m << "," << n << ") vs. (" << b.n << ")\n";
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

Matrix& Matrix::operator+=(const Matrix& B)
{
  if(B.n != n || B.m != m)
    {
      std::cerr << "Error: Matrices have incompatible dimension for summation.\n";
      std::cerr << "  (" << m << ", " << n << ") vs. (" << B.m << ", " << B.n << ")\n";
    }

  double* data_ptr;
  const double* B_data_ptr;

  for(data_ptr = data, B_data_ptr = B.data; data_ptr != data+m*n; ++data_ptr, ++B_data_ptr)
    *data_ptr += *B_data_ptr;
  
  return *this;
}

Matrix& Matrix::operator+=(const Rank1Matrix& B)
{
  if(B.n != n || B.m != m)
    {
      std::cerr << "Error: Matrices have incompatible dimension for summation.\n";
      std::cerr << "  (" << m << ", " << n << ") vs. (" << B.m << ", " << B.n << ")\n";
    }

  double* data_ptr;
  const double* u_data_ptr;
  const double* v_data_ptr;

  for(data_ptr = data, u_data_ptr = B.u->data;
      data_ptr != data+m*n; ++u_data_ptr)
    for(v_data_ptr = B.v->data; v_data_ptr != B.v->data+n; ++data_ptr, ++v_data_ptr)
      {
	*data_ptr += (*u_data_ptr)*(*v_data_ptr);
      }
  
  return *this;
}

Matrix Matrix::convolve(const Matrix& K, size_t S, size_t P) const
{
  // Size of current matrix and filter
  const size_t n1 = m;
  const size_t n2 = n;
  const size_t m = K.nRows();

  // Size of resulting matrix
  const size_t n1_new = (n1-m+2*P)/S+1;
  const size_t n2_new = (n2-m+2*P)/S;

  Matrix A(n1_new, n2_new);

  for(size_t i=-P; i<n1_new+P; i+=S)
    {
      for(size_t j=-P; j<n2_new+P; j+=S)
	{
	  for(size_t k=0; k<m; ++k)
	    {
	      for(size_t l=0; l<m; ++l)
		{
		  // Skip when matrix is accessed out of bounds (extend by zero)
		  if(i+m-k < 0 || j+m-l < 0 || i+m-k >= n1 || j+m-l >= n2)
		    continue;
		  
		  A[i/S+P][j/S+P] += K[k][l] * (*this)[i+m-k][j+m-l];
		}
	    }
	}
    }

  return A;    
}

void Matrix::write_pixels(unsigned char* pixels) const
{
  unsigned char* pixel_ptr;
  double* data_ptr;
  size_t i=0; 
  for(data_ptr = data, pixel_ptr = pixels; data_ptr != data+m*n; ++data_ptr, ++pixel_ptr, ++i)
    *pixel_ptr = (unsigned char)(255.*(*data_ptr));
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
