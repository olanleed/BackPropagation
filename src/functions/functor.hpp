#ifndef BACKPROPAGATION_SRC_FUNCTIONS_FUNCTOR_HPP
#define BACKPROPAGATION_SRC_FUNCTIONS_FUNCTOR_HPP

#include <cmath>
#include <algorithm>

class Functor {
public:
  virtual ~Functor(void) { }

  virtual double forward(const double x) const = 0;
  virtual double backward(const double x) const = 0;
};

#endif
