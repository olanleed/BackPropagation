#ifndef BACKPROPAGATION_SRC_FUNCTIONS_TANH_HPP
#define BACKPROPAGATION_SRC_FUNCTIONS_TANH_HPP

#include "functor.hpp"

namespace functions {
  class Tanh : public Functor {
  public :
    Tanh(void) { }
    virtual ~Tanh(void) { }

    double forward(const double x) const {
      return std::tanh(x);
    }

    double backward(const double x) const {
      const auto val = std::tanh(x);
      return 1.0 - val * val;
    }
  };
};

#endif
