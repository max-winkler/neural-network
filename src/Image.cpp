#include "Image.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Image::Image(int width, int height, std::vector<unsigned char>&& data)
  : width(width), height(height), data(std::move(data)) {}

Image Image::read(const std::string& filename) {
  std::string extension = filename.substr(filename.find_last_of(".") + 1);

  if (extension == "png")
    return read_png(filename);
  else if (extension == "jpg" || extension == "jpeg")
    return read_jpg(filename);
  else {
    std::cerr << "Unknown image type: " << extension << std::endl;
    exit(-1);    
  }    
}  
    
Image Image::read_png(const std::string& filename)
{
  FILE *fp = fopen(filename.c_str(), "rb");

  if(!fp)
    {
      std::cerr << "ERROR: File " << filename << " not found.\n";
      exit(-1);
    }
  
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  
  png_init_io(png, fp);
  png_read_info(png, info);

  unsigned width  = png_get_image_width(png, info);
  unsigned height = png_get_image_height(png, info);
  int color_type  = png_get_color_type(png, info);
  int bit_depth   = png_get_bit_depth(png, info);
  
  if(bit_depth == 16)
    png_set_strip_16(png);

  if(color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png);

  // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
  if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png);

  if(png_get_valid(png, info, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png);

  // These color_type don't have an alpha channel then fill it with 0xff.
  if(color_type == PNG_COLOR_TYPE_RGB ||
     color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

  if(color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png);

  png_read_update_info(png, info);
  
  if(4*width != png_get_rowbytes(png,info))    
    std::cerr << "ERROR: Unexpected behavior in Image::read function.\n";      
    
  std::vector<png_byte> img_data(4*width*height);
  std::vector<png_bytep> row_pointers(height);
  
  for(unsigned y = 0; y < height; y++)
    row_pointers[y] = img_data.data() + 4*width*y;  
  
  png_read_image(png, row_pointers.data());

  fclose(fp);

  png_destroy_read_struct(&png, &info, NULL);

  return Image(width, height, std::move(img_data));
}

Image Image::read_jpg(const std::string& filename) {
  
  int width, height, channels;
  unsigned char* rgb_data = stbi_load(filename.c_str(), &width, &height, &channels, 3);

  if (!rgb_data) {
    std::cerr << "ERROR: Can not load JPG" + filename << std::endl;
  }
  
  std::vector<unsigned char> rgba_data(width * height * 4);
  
  for (int i = 0; i < width * height; ++i) {
    rgba_data[4*i + 0] = rgb_data[3*i + 0]; // R
    rgba_data[4*i + 1] = rgb_data[3*i + 1]; // G
    rgba_data[4*i + 2] = rgb_data[3*i + 2]; // B
    rgba_data[4*i + 3] = 255;               // A 
  }
  
  stbi_image_free(rgb_data);
  
  return Image(width, height, std::move(rgba_data));
}

ImageInfo Image::get_info(const std::string& filename) {
  std::string extension = filename.substr(filename.find_last_of(".") + 1);

  if (extension == "png")
    return get_info_png(filename);
  else if (extension == "jpg" || extension == "jpeg")
    return get_info_jpg(filename);
  else {
    std::cerr << "Unknown image type: " << extension << std::endl;
    exit(-1);    
  }      
}

ImageInfo Image::get_info_png(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(), "rb");

  if (!fp) {
    std::cerr << "ERROR: File " << filename << " not found.\n";
    exit(-1);    
  }

  png_structp png =
    png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);

  png_init_io(png, fp);
  png_read_info(png, info);

  int width = png_get_image_width(png, info);
  int height = png_get_image_height(png, info);

  png_destroy_read_struct(&png, &info, NULL);
  fclose(fp);
    
  return {width, height};  
}

ImageInfo Image::get_info_jpg(const std::string& filename) {
  int width = 0, height = 0, comp = 0;
  if (!stbi_info(filename.c_str(), &width, &height, &comp)) {
    std::cerr << "ERROR: Unable to read image file " << filename << std::endl;
    exit(-1);
  }
  return {width, height};
}

void Image::write(const std::string& filename)
{
  // Write image file
  FILE *fp = fopen(filename.c_str(), "wb");
  if(!fp)
    {
      std::cerr << "Error: Could not open image file\n";
      return;      
    }
  
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  
  png_init_io(png, fp);

  png_set_IHDR(png, info, width, height, 8,
               PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
               );

  png_write_info(png, info);
  
  std::vector<png_bytep> row_pointers(height);
  for (size_t i = 0; i < height; ++i) 
    row_pointers[i] = data.data() + 4 * width * i;  
  
  png_write_image(png, row_pointers.data());
  png_write_end(png, NULL);

  if (png && info)
    png_destroy_write_struct(&png, &info);

  fclose(fp);
}

Matrix Image::to_matrix() const
{
  Matrix A(height, width);
  
  for(size_t i=0; i<height; ++i)
    for(size_t j=0; j<width; ++j)
    {
      const unsigned char* pixel = &(data[4*(i*width + j)]);
      A(i,j) = (pixel[0] + pixel[1] + pixel[2]) / (255.0f * 3.0f);
    }
  
  return A;
}

Image Image::from_matrix(const Matrix& A)
{
  size_t height = A.nRows();
  size_t width = A.nCols();

  std::vector<unsigned char> img_data(4 * width * height);

  for (size_t i = 0; i < height; ++i)
    for (size_t j = 0; j < width; ++j)
    {
      float color = A(i, j);      

      unsigned char value = static_cast<unsigned char>(color * 255.0f);

      size_t index = 4 * (i * width + j);
      img_data[index + 0] = value;  // R
      img_data[index + 1] = value;  // G
      img_data[index + 2] = value;  // B
      img_data[index + 3] = 255;    // A
    }

  return Image(width, height, std::move(img_data));
}

Image Image::from_tensor(const Tensor& T)
{
  size_t channels = T.nChannels();  
  size_t height   = T.nRows();
  size_t width    = T.nCols();

  std::vector<unsigned char> img_data(4 * width * height * channels);
  for(size_t c=0; c<channels; ++c)
    for(size_t i=0; i<height; ++i)
      for(size_t j=0; j<width; ++j)
        {
          float color = std::max(std::min(1.0f, T(c,i,j)), 0.0f);
          unsigned char value = static_cast<unsigned char>(color * 255.0f);
          
          size_t index = 4*(i*width*channels + c*width + j);

          img_data[index + 0] = value; // R
          img_data[index + 1] = value; // G          
          img_data[index + 2] = value; // B
          img_data[index + 3] = 255; // A          
        }
  return Image(width*channels, height, std::move(img_data));
}

Image Image::from_rgb_tensor(const Tensor &T)
{
  size_t channels = T.nChannels();  
  size_t height   = T.nRows();
  size_t width = T.nCols();

  if (channels != 3) {
    std::cerr << "ERROR: The tensor should have 3 channels but it has "
              << channels << std::endl;
    return Image(0,0,std::vector<unsigned char>());
  }

  std::vector<unsigned char> img_data(4 * width * height);
  for (size_t i = 0; i < height; ++i)    
    for (size_t j = 0; j < width; ++j) {
      float red   = std::clamp(T(0, i, j), 0.0f, 1.0f);
      float green = std::clamp(T(1, i, j), 0.0f, 1.0f);
      float blue  = std::clamp(T(2, i, j), 0.0f, 1.0f);

      size_t index = 4 * (i * width + j);
      
      img_data[index + 0] = static_cast<unsigned char>(red * 255.0f);
      img_data[index + 1] = static_cast<unsigned char>(green * 255.0f);
      img_data[index + 2] = static_cast<unsigned char>(blue * 255.0f);
      img_data[index + 3] = static_cast<unsigned char>(255.0f);
    }

  return Image(width, height, std::move(img_data));
}

void Image::resize(size_t new_size) {

  std::vector<unsigned char> new_data(new_size * new_size * 4);
  
  // Determine index of first row/column (rectangle to square transformation)
  size_t size = std::min(width, height);
  int start_x = 0, start_y = 0;
  if (width > height) 
    start_x = (width - size) / 2;
  else
    start_y = (height - size) / 2;
  
  float ratio = static_cast<float>(size - 1) / (new_size - 1);

  for (size_t i = 0; i < new_size; ++i)
    for (size_t j = 0; j < new_size; ++j) {
      size_t xl = std::floor(ratio * j);
      size_t yl = std::floor(ratio * i);
      size_t xh = std::min(static_cast<size_t>(std::ceil(ratio * j)), size - 1);      
      size_t yh = std::min(static_cast<size_t>(std::ceil(ratio * i)), size - 1);      

      float x_weight = ratio * j - xl;
      float y_weight = ratio * i - yl;

      for (int c = 0; c < 3; ++c) {
        // Value at (xl,yl)
        const float
            lb = data[4 * ((start_y + yl) * width + (start_x + xl)) + c],
            rb = data[4 * ((start_y + yl) * width + (start_x + xh)) + c],
            lt = data[4 * ((start_y + yh) * width + (start_x + xl)) + c],
            rt = data[4 * ((start_y + yh) * width + (start_x + xh)) + c];

        const float bottom = (1 - x_weight) * lb + x_weight * rb;
        const float top = (1 - x_weight) * lt + x_weight * rt;
        const float px_value = (1 - y_weight) * bottom + y_weight * top;
        
        new_data[4 * (i * new_size + j) + c] = static_cast<unsigned char>(std::round(px_value));
      }
      new_data[4 * (i * new_size + j) + 3] = 255;
    }

  width = new_size;
  height = new_size;
  data = std::move(new_data);
}

