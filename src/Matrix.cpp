#include <iomanip>
#include <cstring>

#include "Matrix.h"
#include "Vector.h"

Matrix::Matrix() : DataArray(1), m(1)
{
}

Matrix::Matrix(size_t m, size_t n) : DataArray(m*n), m(m)
{
}

Matrix::Matrix(size_t m, size_t n, const double* x)
  : DataArray(m*n), m(m)
{
  memcpy(data, x, size*sizeof(double));  
}

Matrix::Matrix(size_t m, size_t n, const unsigned char* pixels) : DataArray(m*n), m(m)
{
  double* data_ptr;
  const unsigned char* pixel_ptr;
  
  for(data_ptr = data, pixel_ptr = pixels; data_ptr != data+m*n; ++data_ptr, ++pixel_ptr)
    *data_ptr = double(*pixel_ptr) / 255.;
}

Matrix::Matrix(const Matrix& other) : DataArray(other), m(other.m)
{
}

Matrix& Matrix::operator=(const Matrix& other)
{
  if(nRows() != other.nRows() || nCols() != other.nCols())
    {
      delete[] data;
      m = other.m; size = other.size;
      data = new double[size];
    }
  
  memcpy(data, other.data, size*sizeof(double));
  return *this;
}

Matrix& Matrix::operator=(Matrix&& other)  
{
  if(this == &other)
    std::cerr << "WARNING: Self assignment of matrix. This might fail.\n";
  
  delete[] data;
  
  size = other.size;
  m = other.m;
  
  data = other.data;
  other.data = nullptr;
  
  return *this;
}

Matrix& Matrix::operator=(std::initializer_list<double> val)
{
  if(val.size() != size)
    {
      std::cerr << "ERROR: Number of elements in initializer list do not match the dimension of the data array.\n";
      return *this;
    }
  
  size_t i=0;
  for(auto x = val.begin(); x!=val.end(); ++x, ++i)
    data[i] = *x;

  return *this;
}

MatrixRow::MatrixRow(double* data_ptr) : data_ptr(data_ptr)
{}

size_t Matrix::nRows() const {return m;}
size_t Matrix::nCols() const {return size/m;}


MatrixRow Matrix::operator[](size_t i)
{
  return MatrixRow(&(data[i*size/m]));
}

const MatrixRow Matrix::operator[](size_t i) const
{
  return MatrixRow(&(data[i*size/m]));
}

double& MatrixRow::operator[](size_t j)
{
  return data_ptr[j];
}

Matrix& Matrix::operator*=(double a)
{
  for(double* data_ptr = data; data_ptr != data+size; ++data_ptr)
    (*data_ptr) *= a;
  
  return *this;  
}
  
const double& MatrixRow::operator[](size_t j) const
{
  return data_ptr[j];
}

Matrix Matrix::operator+(const Matrix& B) const
{
  Matrix C(*this);
  C += B;
  
  return C;
}

Vector Matrix::operator*(const Vector& b) const
{
  size_t n = nCols();
  
  if(n != b.size)
    {
      std::cerr << "Error: Matrix and vector have incompatible size for multiplication.\n";
      std::cerr << "  (" << nRows() << "," << nCols() << ") vs. (" << b.size << ")\n";
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
  if(B.size != size || B.m != m)
    {
      std::cerr << "Error: Matrices have incompatible dimension for summation.\n";
      std::cerr << "  (" << nRows() << ", " << nCols() << ") vs. (" << B.nRows() << ", " << B.nCols() << ")\n";
    }

  double* data_ptr;
  const double* B_data_ptr;

  for(data_ptr = data, B_data_ptr = B.data; data_ptr != data+size; ++data_ptr, ++B_data_ptr)
    *data_ptr += *B_data_ptr;
  
  return *this;
}

Matrix& Matrix::operator+=(const Rank1Matrix& B)
{
  size_t n = nRows();
  
  if(B.nRows() != nRows() || B.nCols() != nCols())
    {
      std::cerr << "Error: Matrices have incompatible dimension for summation.\n";
      std::cerr << "  (" << m << ", " << n << ") vs. (" << B.nRows() << ", " << B.nCols() << ")\n";
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
  const size_t n1 = nRows();
  const size_t n2 = nCols();
  const size_t m = K.nRows();

  // Size of resulting matrix
  const size_t n1_new = (n1-m+2*P)/S+1;
  const size_t n2_new = (n2-m+2*P)/S+1;

  Matrix A(n1_new, n2_new);

  for(size_t i=-P; i+S+m<n1+P; i+=S)
    {
      for(size_t j=-P; j+S+m<n2+P; j+=S)
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

Matrix Matrix::pool(int type, size_t S, size_t P) const
{
  const size_t n1 = nRows();
  const size_t n2 = nCols();

  const size_t n1_new = (n1 + 2*P)/S;
  const size_t n2_new = (n2 + 2*P)/S;  
  
  Matrix A(n1_new, n2_new);

  for(size_t i=0; i < n1_new;  ++i)
    {
      for(size_t j=0; j < n2_new;  ++j)
	{
	  switch(type)
	    {
	    case POOLING_MAX:
	      {
		double max_val = 0.;
	      
		// Find maxiumum within the patch
		for(int k=0; k<S; ++k)
		  {
		    for(int l=0; l<S; ++l)
		      {
			size_t i2 = i*S-P+k;
			size_t j2 = j*S-P+l;

			if(i2 < 0 || i2 >= n1 || j2 < 0 || j2 >= n2)
			  continue;

			double cur_val = (*this)[i2][j2];
			if(cur_val > max_val)
			  max_val = cur_val;
		      }
		  }
	      
		A[i][j] = max_val;
		break;
	      }
	    default:
	      std::cerr << "ERROR: Pooling type is not implemented yet.\n";
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
  for(data_ptr = data, pixel_ptr = pixels; data_ptr != data+size; ++data_ptr, ++pixel_ptr, ++i)
    *pixel_ptr = (unsigned char)(255.*std::max(0.,std::min(1.,(*data_ptr))));
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
{
  size_t m = matrix.nRows();
  size_t n = matrix.nCols();
  
  for(size_t i=0; i<m; ++i)
    {
      os << "[ ";
      for(size_t j=0; j<n; ++j)
	os << std::left << std::setw(7) << matrix.data[i*n+j] << " ";
      os << "]\n";
    }
  return os;
}
