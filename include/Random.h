#ifndef _RANDOM_H_
#define _RANDOM_H_

#include <random>

class Random
{
 public:
  static double get_uniform();
  static double get_normal();
  
 private:
  // Random number generator
  static std::mt19937 rnd_gen;
  static std::uniform_real_distribution<> random_uniform;
  static std::normal_distribution<> random_normal;
};

#endif
