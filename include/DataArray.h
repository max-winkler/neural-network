#ifndef _DATA_ARRAY_H_
#define _DATA_ARRAY_H_

#include <iostream>

class DataArray
{
 public:
  DataArray(size_t);
  DataArray(const DataArray&);

  virtual ~DataArray();
  
  double& operator[](size_t);
  const double& operator[](size_t) const;

  // Frobenious/Euklidean inner product
  double inner(const DataArray&) const;
  
  size_t nEntries() const;
  
 protected:
  double* data;
  size_t size;
};

#endif
