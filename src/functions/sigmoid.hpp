#ifndef BACKPROPAGATION_SRC_FUNCTIONS_SIGMOID_HPP
#define BACKPROPAGATION_SRC_FUNCTIONS_SIGMOID_HPP

#include "functor.hpp"

namespace functions {
  class Sigmoid : public Functor {
  public :
    Sigmoid(void) { }
    virtual ~ Sigmoid(void) { }

    double forward(const double x) const {
      return 1.0 / (1.0 + std::exp(-x));
    }

    double backward(const double x) const {
      const auto val = forward(x);
      return val * (1.0 - val);
    }
  };
};

#endif
