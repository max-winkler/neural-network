#include "LinAlg.h"

namespace linalg
{
  Matrix multiply(const MatrixView& A, const MatrixView& B)
  {
    const size_t m = A.nRows();
    const size_t n = A.nCols();
  
    if(m != B.nRows() || n != B.nCols())
      {
        std::cerr << "ERROR: Matrices have incompatible dimension for Hadamard product.\n";
        return Matrix();
      }

    Matrix C(m,n);

    float* data_ptr;
    const float* A_data_ptr;
    const float* B_data_ptr;

    for(size_t i=0; i<m; ++i)
      for(size_t j=0; j<n; ++j)
        C(i,j) = A(i,j) * B(i,j);
    
    return C; 
  }

  float dot(const MatrixView& A, const MatrixView& B)
  {
    size_t m = A.nRows(), n = A.nCols();

    if(m != B.nRows() || n != B.nCols())
      {
        std::cerr << "ERROR: Matrices have incompatible size for computation of the Frobenius "
	        << "inner product.\n";
        return 0;
      }

    float val = 0.;
    const float* A_data_ptr;
    const float* B_data_ptr;

    for(A_data_ptr = A.data, B_data_ptr = B.data;
        A_data_ptr != A.data+m*n; ++A_data_ptr, ++B_data_ptr)
      val += (*A_data_ptr)*(*B_data_ptr);

    return val;  
  }
  
  Matrix convolve(const MatrixView&  A, const MatrixView& K,
	        size_t S, size_t P, bool flip)
  {
    if(S != 1)
      {
        std::cerr << "ERROR: Convolution with stride different than 1 not implemented yet.\n";
        return Matrix();
      }
  
    // Size of current matrix and filter
    const size_t n1 = A.nRows();
    const size_t n2 = A.nCols();
    const size_t m = K.nRows();

    // Size of resulting matrix
    const size_t n1_new = (n1-m+2*P)/S+1;
    const size_t n2_new = (n2-m+2*P)/S+1;

    Matrix Z(n1_new, n2_new);

    for(size_t i=0; i<n1_new; ++i)
      for(size_t j=0; j<n2_new; ++j)
        {        
	float val = 0.;
	for(size_t k=0; k<m; ++k)
	  for(size_t l=0; l<m; ++l)	  
	    {
	      // Indices to access left-hand side
	      int x_k = static_cast<int>(i+k)-static_cast<int>(P);
	      int x_l = static_cast<int>(j+l)-static_cast<int>(P);

	      // Bound check when padding is used
	      if(x_k < 0 || x_l < 0 || x_k >= n1 || x_l >= n2)
	        continue;

	      // Indices to access right-hand side
	      int k_idx = flip ? m-1-k : k;
	      int l_idx = flip ? m-1-l : l;
	      
	      // Update entry
	      val += A(x_k,x_l) * K(k_idx, l_idx);
	    }
	Z(i,j) = val;
        }
  
    return Z;    
  }

  Matrix tensor_convolve(const Tensor& T, const Tensor& K, size_t S, size_t P, bool flip)
  {
    // TODO: Can we reuse the matrix convolve function? Transform each slice to matrix view and convolve
    // them should do the trick. Convolve should then be a free function taking MatrixView as argument.
  
    if(S != 1)
      {
        std::cerr << "ERROR: Convolution with stride not equal to 1 is not implemented yet.\n";
        return Matrix();
      }

    // Size of current matrix and filter
    const size_t d  = T.nChannels();
    const size_t n1 = T.nRows();
    const size_t n2 = T.nCols();
    const size_t m  = K.nRows();

    // Size of resulting matrix
    const size_t n1_new = (n1-m+2*P)/S+1;
    const size_t n2_new = (n2-m+2*P)/S+1;

    if(d != K.nChannels())
      {
        std::cerr << "ERROR: Number of channels of input and kernel matrix mus coincide.\n";
        return Matrix();
      }

    Matrix A(n1_new, n2_new);
    for(size_t i=0; i<n1_new; ++i)
      for(size_t j=0; j<n2_new; ++j)
        {
	float val = 0.;
	for(size_t c=0; c<d; ++c)
	  for(size_t k=0; k<m; ++k)
	    for(size_t l=0; l<m; ++l)
	      {
	        // Indices to access left-hand side
	        int x_k = static_cast<int>(i+k)-static_cast<int>(P);
	        int x_l = static_cast<int>(j+l)-static_cast<int>(P);

	        // Bound check when padding is used
	        if(x_k < 0 || x_l < 0 || x_k >= n1 || x_l >= n2)
		continue;

	        // Indices to access right-hand side
	        int k_idx = flip ? m-1-k : k;
	        int l_idx = flip ? m-1-l : l;
	    
	        val += T(c, i*S+k,j*S+l) * K(c, k_idx,l_idx);
	      }
	A(i,j) = val;
        }
  
    return A;    
  }

  Tensor tensor_convolve(const Tensor& T, const Matrix& K, size_t S, size_t P)
  {
    if(S != 1 || P != 0)
      {
        std::cerr << "ERROR: Convolution with padding or stride not equal to 1 not implemented yet.\n";
        return Tensor();
      }
  
    // Size of current matrix and filter
    const size_t d  = T.nChannels();
    const size_t n1 = T.nRows();
    const size_t n2 = T.nCols();
    const size_t m  = K.nRows();
  
    // Size of resulting matrix
    const size_t n1_new = (n1-m+2*P)/S+1;
    const size_t n2_new = (n2-m+2*P)/S+1;

    Tensor U(d, n1_new, n2_new);
    for(size_t c=0; c<d; ++c)
      for(size_t i=0; i<n1_new; ++i)
        for(size_t j=0; j<n2_new; ++j)
	{
	  float val = 0;
	
	  for(size_t k=0; k<m; ++k)
	    for(size_t l=0; l<m; ++l)
	      val += T(c, i+k, j+l) * K(k,l);

	  U(c,i,j) = val;
	}

    return U;
  }

  Matrix pool(const MatrixView& A, int type, size_t k, size_t S, size_t P)
  {
    if(S==0) S = k;
    
    const size_t n1 = A.nRows();
    const size_t n2 = A.nCols();

    const size_t n1_new = (n1 - k)/S + 1;
    const size_t n2_new = (n2 - k)/S + 1;  
  
    Matrix B(n1_new, n2_new);

    if(P!=0)
      {
        std::cerr << "ERROR: Pooling with padding not implemented yet.\n";
        return Matrix();
      }

    for(size_t i=0; i < n1_new;  ++i)    
      for(size_t j=0; j < n2_new;  ++j)
        {
	switch(type)
	  {
	  case POOLING_MAX:
	    {
	      float max_val = 0.;

          // FIXME: Old version, replace S by k...
          
	      // Find maxiumum within the patch
	      for(int k=0; k<S; ++k)		  
	        for(int l=0; l<S; ++l)
		{
		  size_t i2 = i*S-P+k;
		  size_t j2 = j*S-P+l;
		      
		  if(i2 < 0 || i2 >= n1 || j2 < 0 || j2 >= n2)
		    continue;
		      
		  float cur_val = A(i2,j2);
		  if(cur_val > max_val)
		    max_val = cur_val;		    
		}
		
	      B(i,j) = max_val;
	      break;
	    }
	  default:
	    std::cerr << "ERROR: Pooling type is not implemented yet.\n";
	  }
        }
      
    return B;
  }

  Matrix unpool(const MatrixView& A, const MatrixView& B, int type, size_t S, size_t P)
  {
    const size_t n1 = A.nRows();
    const size_t n2 = A.nCols();

    const size_t n1_new = B.nRows();
    const size_t n2_new = B.nCols();
  
    Matrix D(n1_new, n2_new);
  
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
	      float val_max = B(S*k,S*l);
	    
	      for(int i=0; i<S; ++i)
	        for(int j=0; j<S; ++j)
		{
		  if(B(S*k+i,S*l+j) > val_max)
		    {
		      i_max = i; j_max = j;
		      val_max = B(S*k+i,S*l+j);
		    }
		}

	      D(k*S+i_max,l*S+j_max) = A(k,l);
	    }
	    break;
	  default:
	    std::cerr << "ERROR: Unpooling for this type not implemented yet.\n";
	  }
        }
    return D;
  }

} // end of namespace linalg


MatrixView::MatrixView(const float* data, size_t m, size_t n) : data(data), m(m), n(n) {}
MatrixView::MatrixView(const Matrix& A) : data(A.data), m(A.m), n(A.size/A.m) {}
MatrixView::MatrixView(const TensorSlice& T) : data(T.data), m(T.m), n(T.n) {}

size_t MatrixView::nRows() const { return m; }
size_t MatrixView::nCols() const { return n; }
