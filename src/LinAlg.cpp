#include "LinAlg.h"
#include <cblas.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "DataArray.h"

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

    float* data_ptr = C.values();
    const float* A_data_ptr = A.data;
    const float* B_data_ptr = B.data;
    
    for(size_t i=0; i<m*n; ++i)
      data_ptr[i] = A_data_ptr[i]*B_data_ptr[i];
    
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

	      // TODO: This bound check might reduce the performance. Better implementation possible.
	      // Bound check when padding is used
	      if (x_k < 0 || x_l < 0 || static_cast<size_t>(x_k) >= n1 ||
		static_cast<size_t>(x_l) >= n2)                
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

  Matrix convolve_2(const MatrixView&  A, const MatrixView& K,
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
    
    Matrix im_A = im2col(A, m);
    Matrix im_K = im2col(K, m);
    Matrix Z = matmul(im_A, im_K, true);
    Z.reshape(n1_new, n2_new);
    
    return Z;    
  }
  
  
  Matrix tensor_convolve(const Tensor& T, const Tensor& K, size_t S, size_t P, bool flip)  {
  
    if(S != 1)
      {
        std::cerr << "ERROR: Convolution with stride not equal to 1 is not implemented yet.\n";
        exit(1);
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
        std::cerr << "ERROR: Number of channels of input and kernel matrix must coincide.\n";
        return Matrix();
      }

    Matrix A(n1_new, n2_new);

    for (size_t c = 0; c < d; ++c)
      A += convolve(T[c], K[c], S, P, flip);
  
    return A;    
  }

  Tensor unused_tensor_convolve(const Tensor& T, const Matrix& K, size_t S, size_t P)
  {
    if(S != 1)
      {
        std::cerr << "ERROR: Convolution with or stride not equal to 1 not implemented yet.\n";
        exit(1);
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

  Matrix pool(const MatrixView& A, size_t k, int type, size_t S, size_t P)
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
	      
	      // Find maxiumum within the patch
	      for(size_t m=0; m<k; ++m)		  
	        for(size_t n=0; n<k; ++n)
		{
		  size_t i2 = i*S+m;
		  size_t j2 = j*S+n;
		  
		  // Udate maximum value
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

  Matrix unpool(const MatrixView& A, const MatrixView& B, size_t k, int type, size_t S, size_t P)
  {
    if(S==0) S = k;
    
    const size_t n1_new = A.nRows();
    const size_t n2_new = A.nCols();

    const size_t n1 = B.nRows();
    const size_t n2 = B.nCols();

    if((n1-k) % S != 0 || (n2-k) % S != 0 || n1_new != (n1-k)/S+1 || n2_new != (n2-k)/S+1)
      {
        std::cerr << "ERRORS: Dimensions and parameters not correct for unpooling.\n";
        std::cerr << "        Debug linalg::unpool to find the mistake.\n";
        std::exit(EXIT_FAILURE);
      }
    
    Matrix D(n1, n2);
  
    // Iterate over
    for(size_t m=0; m<n1_new; ++m)
      for(size_t n=0; n<n2_new; ++n)
        {
	switch(type)
	  {
	  case POOLING_MAX:
	    {
	      // TODO: The following is only correct for P=0, implement padding later

	      // Determine index for maximum
	      int i_max = 0, j_max = 0;
	      float val_max = B(S*m,S*n);
	    
	      for(size_t i=0; i<k; ++i)
	        for(size_t j=0; j<k; ++j)
		{
		  if(B(S*m+i,S*n+j) > val_max)
		    {
		      i_max = i; j_max = j;
		      val_max = B(S*m+i,S*n+j);
		    }
		}

	      D(S*m+i_max,S*n+j_max) += A(m,n);
	    }
	    break;
	  default:
	    std::cerr << "ERROR: Unpooling for this type not implemented yet.\n";
	  }
        }
    return D;
  }

  Matrix matmul(const MatrixView& A, const MatrixView& B, bool transA,
                bool transB) {
    size_t m = !transA ? A.nRows() : A.nCols();
    size_t n = !transB ? B.nCols() : B.nRows();
    Matrix out(m, n);

    matmul_into(A, B, out.values(), transA, transB);

    return out;
  }

  void matmul_into(const MatrixView& A, const MatrixView& B, float* out_data,
                   bool transA, bool transB) {
    size_t m = !transA ? A.nRows() : A.nCols();
    size_t n = !transB ? B.nCols() : B.nRows();
    size_t k = !transA ? A.nCols() : A.nRows();

    if (k != (!transB ? B.nRows() : B.nCols())) {
      std::cerr
          << "ERROR: Matrices have incompatible dimension for multiplication\n";
      std::cerr << " A is " << A.nRows() << " x " << A.nCols()
                << (transA ? " (to be transposed)\n" : "\n");
      std::cerr << " B is " << B.nRows() << " x " << B.nCols()
                << (transB ? " (to be transposed)\n" : "\n");

      exit(-1);
    }

    cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, m, n, k, 1.0f, A.data,
                A.nCols(), B.data, B.nCols(), 0.0f, out_data, n);
  }

  Matrix im2col(const Tensor& T, size_t k, size_t P) {
    // Output dimension
    size_t orig_h = T.nRows();
    size_t orig_w = T.nCols();
    size_t out_h = orig_h + 2 * P - k + 1;
    size_t out_w = orig_w + 2 * P - k + 1;

    size_t d = T.nChannels();

    // Dimension of output matrix
    size_t out_rows = d * k * k;
    size_t out_cols = out_h * out_w;
    float* __restrict out_data = new float[out_rows * out_cols];

    const float* __restrict T_data = T.values();  // Layout: (d, orig_h, orig_w)
    const size_t plane = orig_h * orig_w;         // Stride pro Kanal

    size_t col = 0;
    for (size_t i = 0; i < out_h; ++i) {
      for (size_t j = 0; j < out_w; ++j) {
          size_t row = 0;
          for (size_t c = 0; c < d; ++c) {
          const float* __restrict A_channel = T_data + c * plane;
          for (size_t m = 0; m < k; ++m) {
            long src_i = (long)i + (long)m - (long)P;
            bool valid_i = (src_i >= 0 && src_i < (long)orig_h);
            const float* __restrict A_row =
                valid_i ? A_channel + src_i * orig_w : nullptr;
            for (size_t n = 0; n < k; ++n) {
                long src_j = (long)j + (long)n - (long)P;
                float val = 0.0f;
                if (valid_i && src_j >= 0 && src_j < (long)orig_w)
                    val = A_row[src_j];
                out_data[out_cols * row + col] = val;
                ++row;
            }
          }
          }
          ++col;
      }
    }
    return Matrix(out_rows, out_cols, out_data, TakeOwnership());
  }

  Tensor col2im(const Matrix& T_col, size_t C_in, size_t orig_h, size_t orig_w,
                size_t k, size_t P) {
    size_t out_h = orig_h + 2 * P - k + 1;
    size_t out_w = orig_w + 2 * P - k + 1;
    size_t out_cols = out_h * out_w;

    Tensor T(C_in, orig_h, orig_w);

    float* __restrict T_data = T.values();
    const float* __restrict col_data = T_col.values();

    const size_t plane = orig_h * orig_w;

    size_t col = 0;
    for (size_t i = 0; i < out_h; ++i) {
      for (size_t j = 0; j < out_w; ++j) {
        size_t row = 0;
        for (size_t c = 0; c < C_in; ++c) {
          float* __restrict dX_channel = T_data + c * plane;
          for (size_t m = 0; m < k; ++m) {
            long src_i = (long)i + (long)m - (long)P;
            bool valid_i = (src_i >= 0 && src_i < (long)orig_h);
            const float* __restrict col_row_base =
                col_data + row * out_cols + col;
            for (size_t n = 0; n < k; ++n) {
              long src_j = (long)j + (long)n - (long)P;
              if (valid_i && src_j >= 0 && src_j < (long)orig_w)
                dX_channel[src_i * orig_w + src_j] +=
                    col_row_base[n * out_cols];
              ++row;
            }
          }
        }
        ++col;
      }
    }
    return T;
  }

  Matrix im2col(const MatrixView& A, size_t k) {
    // todo: Matrix is copied in Tensor constructor. This should be avoided.
    // idea: implement a proxy class TensorView (analogous to MatrixView)
    // return im2col(Tensor(1,A.nRows(), A.nCols(), A.data), k);

    // Output dimension
    size_t out_h = A.nRows() - k + 1;
    size_t out_w = A.nCols() - k + 1;
    
    // Dimension of output matrix
    size_t out_rows = k * k;
    size_t out_cols = out_h * out_w;

    // Matrix out = Matrix::uninitialized(out_rows, out_cols);
    //float* __restrict out_data = out.values();
    float* __restrict out_data = new float[out_rows * out_cols];

    size_t col = 0;
    for (size_t i = 0; i < out_h; ++i) {
      for (size_t j = 0; j < out_w; ++j) {
        size_t row = 0;
        for (size_t m = 0; m < k; ++m) {         
	for (size_t n = 0; n < k; ++n) {
	  out_data[out_cols * row + col] = A(i + m, j + n);
	  ++row;          
	  //out(row, col) = A(i+m, j+n);
	}
        }
        ++col;                
      }
    }

    // return out;
    return Matrix(out_rows, out_cols, out_data, TakeOwnership());
  }

  Matrix kernels2row(const std::vector<Tensor>& kernels) {
    if (kernels.size() == 0) {
      std::cerr << "ERROR: The kernel list is empty.\n";
      exit(-1);
    }

    const Tensor& T = kernels[0];
    size_t tensor_size = T.nChannels() * T.nRows() * T.nCols();
    Matrix output = Matrix::uninitialized(kernels.size(), tensor_size);

    for(size_t c=0; c<kernels.size(); ++c) 
      memcpy(output.values() + c*tensor_size, kernels[c].values(), tensor_size*sizeof(float));
    
    return output;
  }
 
}  // end of namespace linalg

MatrixView::MatrixView(const float* data, size_t m, size_t n) : data(data), m(m), n(n) {}
MatrixView::MatrixView(const Matrix& A) : data(A.data), m(A.m), n(A.size/A.m) {}
MatrixView::MatrixView(const TensorSlice& T) : data(T.data), m(T.m), n(T.n) {}

size_t MatrixView::nRows() const { return m; }
size_t MatrixView::nCols() const { return n; }
