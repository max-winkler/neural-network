#include "TrainingData.h"

TrainingData::TrainingData(const Vector& x, const Vector& y) : x(new Vector(x)), y(y) {}

TrainingData::TrainingData(const TrainingData& other) : x(new Vector(dynamic_cast<Vector&>(*other.x))), y(other.y) {}

TrainingData::TrainingData(TrainingData&& other) : x(other.x), y(std::move(other.y))
{    
  other.x = nullptr;
}

TrainingData& TrainingData::operator=(const TrainingData& other)
{
  if(x != other.x)
    {
      delete x;

      *x = *(other.x);
      y = other.y;
    }
  
  return *this;
}

TrainingData& TrainingData::operator=(TrainingData&& other)
{
  if(x != other.x)
    {
      delete x;
      x = other.x;
    }

  y = std::move(other.y);
  return *this;
}

TrainingData::~TrainingData()
{
  delete x;
}
