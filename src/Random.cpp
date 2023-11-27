#include "Random.h"

std::mt19937 Random::rnd_gen(std::random_device()());
std::uniform_real_distribution<> Random::random_uniform(0., 1.);
std::normal_distribution<> Random::random_normal(0., 1.);

double Random::get_uniform()
{
  return random_uniform(rnd_gen);
}

double Random::get_normal()
{
  return random_normal(rnd_gen);
}
