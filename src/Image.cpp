#include "Image.h"

#include <iostream>

Image::Image(int width, int height, std::vector<unsigned char>&& data)
  : width(width), height(height), data(std::move(data)) {}

Image Image::read(const std::string& filename)
{
  FILE *fp = fopen(filename.c_str(), "rb");

  if(!fp)
    {
      std::cerr << "ERROR: File " << filename << " not found.\n";
    }
  
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  
  png_init_io(png, fp);
  png_read_info(png, info);

  int width      = png_get_image_width(png, info);
  int height     = png_get_image_height(png, info);
  int color_type = png_get_color_type(png, info);
  int bit_depth  = png_get_bit_depth(png, info);
  
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

  // UNCLEAR: Did I get it right? We have 4 bytes per Pixel now? RGB and A?
  if(4*width != png_get_rowbytes(png,info))    
    std::cerr << "ERROR: Unexpected behavior in Image::read function.\n";      
    
  std::vector<png_byte> img_data(4*width*height);
  std::vector<png_bytep> row_pointers(height);
  
  for(int y = 0; y < height; y++)
    row_pointers[y] = img_data.data() + 4*width*y;  
  
  png_read_image(png, row_pointers.data());

  fclose(fp);

  png_destroy_read_struct(&png, &info, NULL);

  return Image(width, height, std::move(img_data));
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
          
          size_t index = (i*width*channels + c*width + j);

          img_data[index + 0] = value; // R
          img_data[index + 0] = value; // G          
          img_data[index + 0] = value; // B
          img_data[index + 0] = 255; // A          
        }
  return Image(width*channels, height, std::move(img_data));
}
