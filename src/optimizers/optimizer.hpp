#ifndef BACKPROPAGATION_SRC_OPTIMIZERS_OPTIMIZER_HPP
#define BACKPROPAGATION_SRC_OPTIMIZERS_OPTIMIZER_HPP

#include "../types.hpp"

class Optimizer {
public:
  virtual ~Optimizer(void) { }
  virtual void update(Matrix& gradient, const Matrix& subgradient) = 0;
};

#endif
