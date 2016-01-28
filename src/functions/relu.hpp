#ifndef BACKPROPAGATION_SRC_FUNCTIONS_RELU_HPP
#define BACKPROPAGATION_SRC_FUNCTIONS_RELU_HPP

#include "functor.hpp"

namespace functions {
  class Relu : public Functor {
  public :
    Relu(void) { }
    virtual ~Relu(void) { }

    double forward(const double x) const {
      return std::max(0.0, x);
    }

    double backward(const double x) const {
      return x > 0.0 ? 1.0 : 0.0;
    }
  };
};

#endif
