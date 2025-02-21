#ifndef _POOLING_LAYER_H_
#define _POOLING_LAYER_H_

#include "Layer.h"

/**
 * Class for pooling layers handling matrix-valued input and computing a matrix-valued output
 * using the max pooling strategy. This class is inherited from the abstract Layer class.
 */
class PoolingLayer : public Layer
{
 public:
  /**
   * Constructor creating a pooling layer instance. Given inputs are the dimension of the input layer, the stride
   * and the padding.
   *
   * @brief Create pooling layer instance
   *
   * @param in_dim1 Number of rows of the input matrix.
   * @param in_dim2 Number of columns of the input matrix.
   * @param k Width/height of the patches for the maximum pooling.
   * @param S Stride parameter.
   * @param P Padding parameter.
   */
  PoolingLayer(size_t, size_t, size_t k, size_t S=0, size_t P=0);

  /**
   * Compute the output of the layer for provided input data.
   *
   * @brief Evaluate the layer
   *
   * @param x_ Data to be read and written by the layer. Must be of type Matrix.
   */
  void eval(DataArray*&) const override;

  /**
   * Perform a forward propagation for this layer.
   *
   * @brief Forward propagation for the layer.
   *
   * @param x_ The input data array. Must be of type Matrix.
   * @param z_ This parameter is unused for pooling layers.
   * @param y_ Output data of type Matrix.
   */
  void forward_propagate(const DataArray& x, DataArray& z, DataArray& y) const override;

  /**
   * Perform a backward propagation for this layer.
   *
   * @brief Backward propagation for the layer.
   *
   * @param DY Set of data to be processed. The output is further processed by the previous layer.
   * @param Y Set of input data. Each entry must be of type Matrix.
   * @param Z This parameter is unused for pooling layers.
   */
  std::unique_ptr<Layer> backward_propagate(std::vector<DataArray*>&,
					    const std::vector<DataArray*>&,
					    const std::vector<DataArray*>&) const override;  

  /**
   * Creates a copy of the layer.
   *
   * @brief Create copy.
   */
  std::unique_ptr<Layer> clone() const override;

  /**
   * Create empty layer of the same type.
   *
   * @brief Create empty layer.
   */
  std::unique_ptr<Layer> zeros_like() const override;  

  /**
   * Save the layer to text written in the output stream
   *
   * @brief Save layer
   *
   * @param os Output stream to be written into.
   */
  void save(std::ostream&) const override;
  
 private:
  
  size_t k;
  size_t S;
  size_t P;

  size_t in_dim1;
  size_t in_dim2;

  int type;
};

#endif
