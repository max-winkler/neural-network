#include "TrainingData.h"

TrainingData::TrainingData(const Vector& x, const Vector& y) : x(new Vector(x)), y(y) {}
TrainingData::TrainingData(const Matrix& x, const Vector& y) : x(new Matrix(x)), y(y) {}

TrainingData::TrainingData(const TrainingData& other) : y(other.y) {

  if(dynamic_cast<Vector*>(other.x))
    x = new Vector(dynamic_cast<Vector&>(*other.x));
  else if(dynamic_cast<Matrix*>(other.x))
    x = new Matrix(dynamic_cast<Matrix&>(*other.x));
  else
    {
      std::cerr << "ERROR: Unable to copy training data as type of input is not accurate.\n";
    }
}

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
