#ifndef _TRAINING_DATA_H_
#define _TRAINING_DATA_H_

#include "Vector.h"
#include "Matrix.h"


class TrainingData
{
 public:
  TrainingData(const Vector&, const Vector&);
  TrainingData(const Matrix&, const Vector&);
  
  TrainingData(const TrainingData&);
  TrainingData(TrainingData&&);
  
  TrainingData& operator=(const TrainingData&);
  TrainingData& operator=(TrainingData&&);
  
  ~TrainingData();
  
  DataArray* x;
  Vector y;
};

#endif
