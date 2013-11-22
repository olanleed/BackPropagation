#ifndef BACKPROPAGATION_RANDOM_GENERATOR_HPP_
#define BACKPROPAGATION_RANDOM_GENERATOR_HPP_

#include <ctime>
#include <boost/random.hpp>

namespace Random {
  inline double generate(const double min = 0.0, const double max = 1.0) {
    static boost::mt19937 rng(static_cast<unsigned long>(time(0)));
    boost::random::uniform_real_distribution<> range(min, max);
  return range(rng);
  }
};
#endif //BACKPROPAGATION_RANDOM_GENERATOR_HPP_
