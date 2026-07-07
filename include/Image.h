#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <png.h>
#include <string>
#include <vector>

#include "Matrix.h"
#include "Tensor.h"

struct ImageInfo {
  ImageInfo(size_t width, size_t height) : width(width), height(height) {}  
  ImageInfo(int width, int height)
      : width(static_cast<size_t>(width)),
        height(static_cast<size_t>(height)) {}  
  size_t width;
  size_t height;  
};

/**
 * Class used to store images that can be read from a PNG file and converted into a
 * matrix or tensor to be processed by the neural network.
 */
class Image
{
 public:
  /**
   * Returns width of the image
   */
  size_t get_width() const { return width; }
  /**
   * Returns height of the image
   */
  size_t get_height() const { return height; }
  
  /**
   * Read PNG image from a file.
   *
   * @brief Read PNG image.
   *
   * @param filename The name of the file to be read.
   */
  static Image read(const std::string& filename);

  /**
   * Get the size of an image with a given file name.
   *
   * @param filename The name of the file to be read.
   */
  static ImageInfo get_info(const std::string& filename);
  
  /**
   * Create gray-scale image from a matrix having entries between 0 (black) and 1 (white).
   *
   * @brief Create image from matrix.
   *
   * @param A The matrix representing a gray-scale image.
   */
  static Image from_matrix(const Matrix&);

  /**
   * Create a gray-scale image from a tensor having entries between 0 (black)
   * and 1 (white). The tensor slices are concatenated horizontally in
   * the resulting image.
   *
   * @brief Create image from tensor.
   *
   * @param T The tensor whose slices represent the gray-scale images.
   */
  static Image from_tensor(const Tensor&);

  /**
   * Create a color image from a 3-slice tensor. The channeentries
   * (0,i,j), (1,i,j) and (2,i,j) contain the red/green/blue value of
   * the pixel in row i and column j.
   *
   * @brief Create color image from 3-slice tensor.
   *
   * @param T The tensor representing the image.
   */
  static Image from_rgb_tensor(const Tensor &);
  
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

  /**
   * Resizes the image to a square image using cropping and bilinear
   * interpolation.
   */
  void resize(size_t);
  
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


  /**
   * Read PNG image from a file.
   *
   * @brief Read PNG image.
   *
   * @param filename The name of the file to be read.
   */
  static Image read_png(const std::string& filename);


  /**
   * Read PNG image from a file.
   *
   * @brief Read JPG image.
   *
   * @param filename The name of the file to be read.
   */
  static Image read_jpg(const std::string& filename);

  /**
   * Get the size of an image with a given file name.
   *
   * @param filename The name of the file to be read.
   */
  static ImageInfo get_info_png(const std::string& filename);
  static ImageInfo get_info_jpg(const std::string& filename);
  
  /// Data vector
  std::vector<unsigned char> data;

  /// Image width
  size_t width;

  /// Image height
  size_t height;
};

#endif
