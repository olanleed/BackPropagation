#include "backpropagation.hpp"
#include "random_generator.hpp"

template <class OptT, class FuncT>
BackPropagation<OptT, FuncT>::BackPropagation(const int input_layer, const int hidden_layer,
                                              const int output_layer, const double epoch,
                                              const OptT& optimizer, const FuncT& function)
  : epoch_(epoch), hidden_(hidden_layer), optimizer_(optimizer), function_(function) {

  weight_input_.resize(input_layer, hidden_layer);
  weight_hidden_.resize(hidden_layer, output_layer);
  diff_weight_input_.resize(input_layer, hidden_layer);
  diff_weight_hidden_.resize(hidden_layer, output_layer);

  for (Matrix::iterator1 it = weight_input_.begin1(); it != weight_input_.end1(); ++it) {
    std::transform(it.begin(), it.end(), it.begin(), [](const double x) { return Random::generate(); });
  }

  for (Matrix::iterator1 it = weight_hidden_.begin1(); it != weight_hidden_.end1(); ++it) {
    std::transform(it.begin(), it.end(), it.begin(), [](const double x) { return Random::generate(); });
  }
}

template <class OptT, class FuncT>
BackPropagation<OptT, FuncT>::~BackPropagation(void) { }

template <class OptT, class FuncT>
void BackPropagation<OptT, FuncT>::fit(const std::vector< std::pair< Vector, Vector > >& dataset) {
  for (std::size_t i = 0; i < epoch_; ++i) {
    for (const auto& data : dataset) {
      const Vector output = forward(data.second);
      backward(data.first, data.second, output);
      optimizer_.update(weight_input_, diff_weight_input_);
      optimizer_.update(weight_hidden_, diff_weight_hidden_);
    }
  }
}

template <class OptT, class FuncT>
Vector BackPropagation<OptT, FuncT>::representation(const Vector& input) {
  const Vector hidden = prod(input, weight_input_);

  std::transform(hidden.begin(), hidden.end(), hidden_.begin(),
		 [&](const double x) { return function_.forward(x); } );

  return prod(hidden_, weight_hidden_);
}

template <class OptT, class FuncT>
Vector BackPropagation<OptT, FuncT>::predict(const Vector& input) {
  return representation(input);
}

template <class OptT, class FuncT>
Vector BackPropagation<OptT, FuncT>::forward(const Vector& input) {
  return representation(input);
}

template <class OptT, class FuncT>
void BackPropagation<OptT, FuncT>::backward(const Vector& answer, const Vector& input, const Vector& output) {
  const Vector delta = output - answer;

  Vector hidden = prod(weight_hidden_, delta);

  for (std::size_t i = 0; i < hidden.size(); ++i) {
    hidden[i] *= function_.backward(hidden_[i]);
  }

  for (std::size_t i = 0; i < diff_weight_hidden_.size1(); ++i) {
    for (std::size_t j = 0; j < diff_weight_hidden_.size2(); ++j) {
      diff_weight_hidden_(i, j) = hidden_[i] * delta[j];
    }
  }

  for (std::size_t i = 0; i < diff_weight_input_.size1(); ++i) {
    for (std::size_t j = 0; j < diff_weight_input_.size2(); ++j) {
      diff_weight_input_(i, j) = input[i] * hidden[j];
    }
  }
}

template class BackPropagation<optimizer::GD, functions::Tanh>;
template class BackPropagation<optimizer::GD, functions::Sigmoid>;
template class BackPropagation<optimizer::GD, functions::Relu>;
