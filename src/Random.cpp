#include <iostream>

#include "Random.h"

Random Random::create_uniform_random_generator()
{
  Random random;
  random.rnd_gen = std::mt19937(std::random_device()());
  random.random_uniform = std::uniform_real_distribution<float>(0.0f, 1.0f);
  random.distribution = DISTRIBUTION_UNIFORM;
  return random;  
}

Random Random::create_normal_random_generator(float ex, float stdev)
{
  Random random;
  random.rnd_gen = std::mt19937(std::random_device()());
  random.random_normal = std::normal_distribution<float>(ex, stdev);
  random.distribution = DISTRIBUTION_NORMAL;
  return random;
}

Random Random::create_mt19937_random_generator()
{
  Random random;
  random.rnd_gen = std::mt19937(std::random_device()());
  random.distribution = DISTRIBUTION_19937;
  return random;
}


float Random::operator()()
{
  switch(distribution)
    {
    case DISTRIBUTION_UNIFORM:
      return random_uniform(rnd_gen);
    case DISTRIBUTION_NORMAL:
      return random_normal(rnd_gen);
    case DISTRIBUTION_19937:
      return rnd_gen();
    default:
      std::cerr << "ERROR: Unknown distribution provided to random number generator.\n";
    }
  return 0.0f;
}

std::mt19937 Random::generator()
{
  return rnd_gen;
}
