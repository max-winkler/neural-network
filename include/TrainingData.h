#ifndef _TRAINING_DATA_H_
#define _TRAINING_DATA_H_

#include "Vector.h"

class TrainingData
{
 public:
  TrainingData(const Vector&, double);
  
  Vector x;
  double y;
};

#endif
