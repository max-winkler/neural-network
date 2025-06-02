#ifndef _TRAINING_DATA_H_
#define _TRAINING_DATA_H_

#include "Vector.h"
#include "Matrix.h"

/**
 * Class used to store training data, i.e., a matrix or vector input and the 
 * corresponding labels as vector.
 */
class TrainingData
{
 public:
  /**
   * Create training data with vector input.
   *
   * @param x The input vector.
   * @param y The label vector.
   */
  TrainingData(const Vector& x, const Vector& y);

  /**
   * Create training data with matrix input.
   *
   * @param x The input vector.
   * @param y The label vector.
   */
  TrainingData(const Matrix& x, const Vector& y);

  /**
   * Copy a training datum.
   *
   * @param other The training datum to be copied.
   */
  TrainingData(const TrainingData& other);

  /**
   * Move a training datum.
   *
   * @param other The training datum to be moved.
   */  
  TrainingData(TrainingData&& other);

  /**
   * Assign training datum from another instance.
   *
   * @param other The training datum to be copied.
   */
  TrainingData& operator=(const TrainingData&);

  /**
   * Assign a training datum from another instance via move.
   *
   * @param other The training datum to be moved.
   */ 
  TrainingData& operator=(TrainingData&& other);

  /**
   * Destroy an training datum instance.
   */
  ~TrainingData();

  /// The input data of the training datum
  DataArray* x;

  /// The label vector belonging to the input data.
  Vector y;
};

#endif
