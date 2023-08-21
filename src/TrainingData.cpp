#include "TrainingData.h"

TrainingData::TrainingData(const Vector& x, const Vector& y) : x(new Vector(x)), y(y) {}

TrainingData::~TrainingData()
{
  // TODO: Delete x somehow
}
