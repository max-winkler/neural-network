#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <png.h>
#include <string>
#include <vector>

#include "Matrix.h"
#include "Tensor.h"

/**
 * Class used to store images that can be read from a PNG file and converted into a
 * matrix or tensor to be processed by the neural network.
 */
class Image
{
 public:

  /**
   * Read PNG image from a file.
   *
   * @brief Read PNG image.
   *
   * @param filename The name of the file to be read.
   */
  static Image read(const std::string& filename);

  /**
   * Create gray-scale image from a matrix having entries between 0 (black) and 1 (white).
   *
   * @brief Create image from matrix.
   *
   * @param A The matrix representing a gray-scale image.
   */
  static Image from_matrix(const Matrix&);

  /**
   * Create a gray-scale image from a tensor having entries between 0 (black) and 1 (white).
   * The tensor slices are concatenated horizontally in the resulting image.
   *
   * @brief Create image from tensor.
   *
   * @param T The tensors whose slices represent the gray-scale images.
   */
  static Image from_tensor(const Tensor&);
  
  /**
   * Write image into a file.
   *
   * @brief Write PNG image.
   *
   * @param filename The name of the file to be written.
   */
  void write(const std::string& filename);

  /**
   * Convert image into a matrix representing a gray-scale image. The resulting matrix
   * has entries between 0 (black) and 1 (white).
   *
   * @brief Convert image to matrix.
   */
  Matrix to_matrix() const;
  
 private:

  Image() = delete;

  /**
   * Constructor creating an image from a data vector.
   *
   * @brief Create image from data vector.
   *
   * @param width The width oh the image.
   * @param height The height of the image.
   * @param data Data vector containing \p width x \ height blocks of 4 bytes (RGB and A).
   */
  Image(int width, int height, std::vector<unsigned char>&& data);

  /// Data vector
  std::vector<unsigned char> data;

  /// Image width
  size_t width;

  /// Image height
  size_t height;
};

#endif
