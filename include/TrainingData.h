#ifndef _TRAINING_DATA_H_
#define _TRAINING_DATA_H_

#include "Vector.h"

class TrainingData
{
 public:
  TrainingData(const Vector&, const Vector&);
  
  Vector x;
  Vector y;
};

#endif
