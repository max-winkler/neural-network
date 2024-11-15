#include <iomanip>
#include <cstring>
#include <cblas.h>
#include <cmath>

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

double& Matrix::operator()(size_t i, size_t j)
{
  return data[i*size/m + j];
}

const double& Matrix::operator()(size_t i, size_t j) const
{
  return data[i*size/m + j];
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

  cblas_dgemv(CblasRowMajor, CblasNoTrans, nRows(), nCols(), 1., data, nCols(), b.data, 1, 0., c.data, 1);
  
  /**
  for(int i=0; i<m; ++i)
    {
      double val = 0.;
      
      const double* data_row_end = &(data[(i+1)*n]);
      for(double *data_row = &(data[i*n]), *data_vec = b.data;
	  data_row != data_row_end;
	  ++data_row, ++data_vec)
	{
	  val += (*data_row)*(*data_vec);
	}
      c[i] = val;
    }
  */
  return c;
}

Matrix& Matrix::operator+=(double a)
{
  for(double* data_ptr = data; data_ptr != data+size; ++data_ptr)
    *data_ptr += a;
  
  return *this;
}
  
Matrix& Matrix::operator+=(const Matrix& B)
{
  operator+=(ScaledMatrix(1., B));
  return *this;
}

Matrix& Matrix::operator+=(const ScaledMatrix& B)
{
  if(B.matrix->size != size || B.matrix->m != m)
    {
      std::cerr << "Error: Matrices have incompatible dimension for summation.\n";
      std::cerr << "  (" << nRows() << ", " << nCols()
		<< ") vs. (" << B.matrix->nRows() << ", " << B.matrix->nCols() << ")\n";
    }
  
  double* data_ptr;
  const double* B_data_ptr;

  for(data_ptr = data, B_data_ptr = B.matrix->data;
      data_ptr != data+size;
      ++data_ptr, ++B_data_ptr)
    *data_ptr += B.scale * (*B_data_ptr);
  
  return *this;
}

Matrix& Matrix::operator+=(const Rank1Matrix& B)
{
  size_t n = nCols();
  
  if(B.nRows() != nRows() || B.nCols() != nCols())
    {
      std::cerr << "Error: Matrices have incompatible dimension for summation.\n";
      std::cerr << "  (" << m << ", " << n << ") vs. (" << B.nRows() << ", " << B.nCols() << ")\n";
    }
  
  cblas_dger(CblasRowMajor, nRows(), nCols(), 1., B.u->data, 1, B.v->data, 1, data, nCols());
  
  /*
  double* data_ptr;
  const double* u_data_ptr;
  const double* v_data_ptr;  

  for(data_ptr = data, u_data_ptr = B.u->data;
      data_ptr != data+m*n; ++u_data_ptr)
    {
      const double* v_data_end = B.v->data+n;
      for(v_data_ptr = B.v->data; v_data_ptr != v_data_end; ++data_ptr, ++v_data_ptr)
	{
	  *data_ptr += (*u_data_ptr)*(*v_data_ptr);
	}
    }
  */
  return *this;
}

Matrix multiply(const Matrix& A, const Matrix& B)
{
  const size_t m = A.nRows();
  const size_t n = A.nCols();
  
  if(m != B.nRows() || n != B.nCols())
    {
      std::cerr << "ERROR: Matrices have incompatible dimension for Hadamard product.\n";
      return Matrix();
    }

  Matrix C(m,n);

  double* data_ptr;
  const double* A_data_ptr;
  const double* B_data_ptr;

  for(data_ptr = C.data, A_data_ptr = A.data, B_data_ptr = B.data;
      data_ptr != C.data+m*n; ++data_ptr, ++A_data_ptr, ++B_data_ptr)
    *data_ptr = (*A_data_ptr)*(*B_data_ptr);

  return C; 
}

Vector Matrix::flatten() const
{
  return Vector(nRows()*nCols(), data);
}

Matrix Matrix::convolve(const Matrix& K, size_t S, size_t P) const
{
  if(P != 0)
    {
      std::cerr << "ERROR: Convolution with padding not implemented yet.\n";
      return Matrix();
    }
  
  // Size of current matrix and filter
  const size_t n1 = nRows();
  const size_t n2 = nCols();
  const size_t m = K.nRows();

  // Size of resulting matrix
  const size_t n1_new = (n1-m+2*P)/S+1;
  const size_t n2_new = (n2-m+2*P)/S+1;

  Matrix A(n1_new, n2_new);

  for(size_t i=0; i<n1_new; ++i)
    for(size_t j=0; j<n2_new; ++j)
      {
	double val = 0.;
	for(size_t k=0; k<m; ++k)
	  for(size_t l=0; l<m; ++l)
	    {	    
	      val += (*this)(i*S+k,j*S+l) * K(k,l);
	    }
	A(i,j) = val;
      }
  
  return A;    
}

Matrix Matrix::back_convolve(const Matrix& Y, size_t J, size_t P) const
{
  const size_t n1 = nRows();
  const size_t n2 = nCols();
  const size_t M = Y.nRows();

  const size_t n1_new = n1-J*(M-1);
  const size_t n2_new = n2-J*(M-1);

  Matrix K(n1_new, n2_new);

  // TODO: Implement back convolution with padding
  if(P!=0)
    {
      std::cerr << "ERROR: Back convolution with padding not implemented yet.\n";
      return Matrix();
    }

  for(size_t k=0; k<n1_new; ++k)
    for(size_t l=0; l<n2_new; ++l)
      {
	double val = 0.;
	for(size_t i=0; i<M; ++i)	
	  for(size_t j=0; j<M; ++j)
	    val += (*this)(i*J+k,j*J+l) * Y(i,j);
	K(k,l) = val;
      }

  return K;
}

Matrix Matrix::pool(int type, size_t S, size_t P) const
{
  const size_t n1 = nRows();
  const size_t n2 = nCols();

  const size_t n1_new = (n1 + -1 + 2*P)/S + 1;
  const size_t n2_new = (n2 + -1 + 2*P)/S + 1;  
  
  Matrix A(n1_new, n2_new);

  if(P!=0)
    {
      std::cerr << "ERROR: Back convolution with padding not implemented yet.\n";
      return Matrix();
    }

  for(size_t i=0; i < n1_new;  ++i)    
    for(size_t j=0; j < n2_new;  ++j)
      {
        switch(type)
	{
	case POOLING_MAX:
	  {
	    double max_val = 0.;
	      
	    // Find maxiumum within the patch
	    for(int k=0; k<S; ++k)		  
	      for(int l=0; l<S; ++l)
	        {
		size_t i2 = i*S-P+k;
		size_t j2 = j*S-P+l;
		      
		if(i2 < 0 || i2 >= n1 || j2 < 0 || j2 >= n2)
		  continue;
		      
		double cur_val = (*this)(i2,j2);
		if(cur_val > max_val)
		  max_val = cur_val;		    
	        }
		
	    A(i,j) = max_val;
	    break;
	  }
	default:
	  std::cerr << "ERROR: Pooling type is not implemented yet.\n";
	}
      }

  return A;
}

Matrix Matrix::unpool(const Matrix& A, int type, size_t S, size_t P) const
{
  const size_t n1 = nRows();
  const size_t n2 = nCols();

  const size_t n1_new = A.nRows();
  const size_t n2_new = A.nCols();
  
  Matrix B(n1_new, n2_new);
  
  // Iterate over
  for(size_t k=0; k<n1; ++k)
    for(size_t l=0; l<n2; ++l)
      {
        switch(type)
	{
	case POOLING_MAX:
	  {
	    // TODO: The following is only correct for P=0, implement padding later

	    // Determine index for maximum
	    int i_max = 0, j_max = 0;
	    double val_max = A[S*k][S*l];
	    
	    for(int i=0; i<S; ++i)
	      for(int j=0; j<S; ++j)
	        {
		  if(A(S*k+i,S*l+j) > val_max)
		  {
		    i_max = i; j_max = j;
		    val_max = A(S*k+i,S*l+j);
		  }
	        }

	    B(k*S+i_max,l*S+j_max) = (*this)(k,l);
	  }
	  break;
	default:
	  std::cerr << "ERROR: Unpooling for this type not implemented yet.\n";
	}
      }
  return B;
}

Matrix Matrix::kron(const Matrix& K, int S, int overlap) const
{
  const size_t n1 = nRows();
  const size_t n2 = nCols();
  const size_t m = K.nRows();

  // if no stride given we assume its the same like kernel size
  if(S==0) S=m;
  
  Matrix G((n1-1)*S+m, (n2-1)*S+m);

  for(size_t i=0; i<n1; ++i)
    for(size_t j=0; j<n2; ++j)
      {
	double scale = (*this)(i,j);
	for(size_t k=0; k<m; ++k)
	  for(size_t l=0; l<m; ++l)	    
	    G(i*S+k,j*S+l) += scale * K(k,l);	    
      }
  
  return G;
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

ScaledMatrix::ScaledMatrix(double scale, const Matrix& matrix) : scale(scale), matrix(&matrix) {}
ScaledMatrix operator*(double scale, const Matrix& matrix)
{
  return ScaledMatrix(scale, matrix);
}
