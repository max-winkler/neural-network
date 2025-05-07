#ifndef _DATA_ARRAY_H_
#define _DATA_ARRAY_H_

#include <iostream>

/**
 * Base class for all types of tensors (vectors, matrices, etc.). 
 *
 * This container stores a pointer to a data array and provides basic methods
 * that all tensors have in common.
 */
class DataArray
{
 public:
  /**
   * Standard constructor initializing a DataArray with given size. All components are set to zero.
   *
   * @param size The number of components stored in the data array (n1*n2*...*nd for tensors of stage d)
   */
  DataArray(size_t);

  /**
   * Constructor creating a copy of another DataArray.
   *
   * @param other The data array to be copied.
   */
  DataArray(const DataArray&);

  /**
   * Destructor that deallocates the internal memory used to store the DataArray.
   */
  virtual ~DataArray();

  /**
   * Element access by reference. Can be used to write single components of the array.
   *
   * @param i Index of component to be accessed.
   */
  double& operator[](size_t);

  /**
   * Element access by const reference. Can be used to read single components of the array.
   *
   * @param i Index of component to be accessed.
   */
  const double& operator[](size_t) const;

  /**
   * Standard inner product for two DataArrays with matching dimension. For vectors this 
   * corresponds to the Euklidean inner product, for matrices this is the Frobenious inner
   * product.
   *
   * @param other The data array to be multiplied with (*this)
   */ 
  double inner(const DataArray&) const;

  /**
   * Computes the sum of all entries of the data array.
   *
   * @param A Data array to be summed up.
   */
  friend double sum(const DataArray&);

  /**
   * Getter method returning the number of components of the DataArray.
   */
  size_t nEntries() const;

  // TODO: Add a method apply(std::function) for element-wise application of functions, e.g.,
  // the activation function.
 protected:
  /**
   * Pointer to the memory where the entries of the DataArray are stored
   */
  double* data;
  /**
   * Integer storing the size of the DataArray (n1*n2*...*nd for tensors of stage d).
   */
  size_t size;
};

#endif
