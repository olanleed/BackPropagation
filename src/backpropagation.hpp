#ifndef BACKPROPAGATION_INCLUDE_BACKPROPAGATION_HPP_
#define BACKPROPAGATION_INCLUDE_BACKPROPAGATION_HPP_

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <cstddef>
#include <set>
#include <functional>

#include "./optimizers/optimizers.hpp"
#include "./functions/functions.hpp"
#include "types.hpp"

template <class OptT, class FuncT>
class BackPropagation {

private :
  const double epoch_;

private:
  Matrix weight_input_;
  Matrix weight_hidden_;
  Matrix diff_weight_input_;
  Matrix diff_weight_hidden_;
  Vector hidden_;

private :
  OptT optimizer_;
  const FuncT function_;

public :
  BackPropagation(const int input_layer, const int hidden_layer,
		  const int output_layer, const double epoch,
                  const OptT& optimizer, const FuncT& function);

  virtual ~BackPropagation(void);

  void fit(const std::vector< std::pair< Vector, Vector > >& data_set);
  Vector predict(const Vector& input);

private :
  Vector forward(const Vector& input);
  void backward(const Vector& answer, const Vector& input, const Vector& output);
};

#endif //BACKPROPAGATION_INCLUDE_BACKPROPAGATION_HPP_
