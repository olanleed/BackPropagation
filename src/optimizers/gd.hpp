#ifndef BACKPROPAGATION_SRC_OPTIMIZERS_GD_HPP
#define BACKPROPAGATION_SRC_OPTIMIZERS_GD_HPP

#include "optimizer.hpp"

namespace optimizer {
  //Gradient Descent
  class GD : public Optimizer {
  private:
    const double _alpha;

  public:
    GD(const double alpha = 0.5) : _alpha(alpha) { }
    virtual ~GD(void) { }

  public :
    void update(Matrix& gradient, const Matrix& subgradient) {
      gradient -= _alpha * subgradient;
    }

  };
};

#endif
