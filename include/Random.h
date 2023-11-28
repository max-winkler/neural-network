#ifndef _RANDOM_H_
#define _RANDOM_H_

#define DISTRIBUTION_UNIFORM 0
#define DISTRIBUTION_NORMAL 1
#define DISTRIBUTION_19937 2

#include <random>

class Random
{
 public:
  
  double operator()();

  std::mt19937 generator();
  
  static Random create_uniform_random_generator();
  static Random create_normal_random_generator();
  static Random create_mt19937_random_generator();
  
 private:
  // Random number generator
  std::mt19937 rnd_gen;
  std::uniform_real_distribution<> random_uniform;
  std::normal_distribution<> random_normal;

  int distribution;
};

#endif
