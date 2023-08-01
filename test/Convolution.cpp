#include <iostream>
#include <png.h>
#include <cstring>

#include "Matrix.h"

png_bytep* read_image_pixels(const std::string& filename, int& width, int& height, int& color_type, int& bit_depth)
{
  FILE *fp = fopen(filename.c_str(), "rb");
  
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  
  png_init_io(png, fp);
  png_read_info(png, info);

  width      = png_get_image_width(png, info);
  height     = png_get_image_height(png, info);
  color_type = png_get_color_type(png, info);
  bit_depth  = png_get_bit_depth(png, info);
  
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

  png_bytep* row_pointers = new png_bytep[height];
  for(int y = 0; y < height; y++) {
    row_pointers[y] = new png_byte[png_get_rowbytes(png,info)];
  }

  png_read_image(png, row_pointers);

  fclose(fp);

  png_destroy_read_struct(&png, &info, NULL);

  return row_pointers;
}

void write_image(const std::string& filename, int width, int height, png_bytep* row_pointers)
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
	       PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
	       PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
	       );

  png_write_info(png, info);
    
  png_write_image(png, row_pointers);
  png_write_end(png, NULL);

  if (png && info)
    png_destroy_write_struct(&png, &info);

}


int main()
{
  int width, height, color_type, bit_depth;

  // Read image file
  png_bytep* row_pointers_in = read_image_pixels("cat.png", width, height, color_type, bit_depth);

  // Convert to black/white picture
  png_byte* image_data = new png_byte[height*width];
  png_bytep* row_pointers = new png_bytep[height];    
  for(int i=0; i<height; ++i)
    {
      row_pointers[i] = &(image_data[i*width]);
      
      for(int j=0; j<width; ++j)
	row_pointers[i][j] = (row_pointers_in[i][4*j] + row_pointers_in[i][4*j+1] + row_pointers_in[i][4*j+2])/3;
    }  

  // Store image as matrix
  Matrix image(height, width, image_data);
      
  // manipulate image with filters
  Matrix K(3,3);
  K = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  K *= (1./16);
  image = image.convolve(K, 4);


  int width_new = image.nCols();
  int height_new = image.nRows();

  std::cout << "Image size after convolution: " << width_new << " x " << height_new << std::endl;
  
  delete[] image_data;
  image_data = new png_byte[height_new*width_new];

  delete[] row_pointers;
  row_pointers = new png_bytep[height_new];
  for(int i=0; i<height_new; ++i)    
    row_pointers[i] = &(image_data[i*width_new]);
    
  image.write_pixels(image_data);
  
  // Write image  
  write_image("cat_bw.png", width_new, height_new, row_pointers);
  
  return 0;
}
