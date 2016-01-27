#include "backpropagation.hpp"
#include "random_generator.hpp"

template <class OptT>
BackPropagation<OptT>::BackPropagation(const int input_layer, const int hidden_layer,
                                       const int output_layer, const double epoch,
                                       const OptT& optimizer)
  : epoch_(epoch), hidden_(hidden_layer), optimizer_(optimizer) {

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

template <class OptT>
BackPropagation<OptT>::~BackPropagation(void) { }

template <class OptT>
void BackPropagation<OptT>::fit(const std::vector< std::pair< Vector, Vector > >& dataset) {
  for (std::size_t i = 0; i < epoch_; ++i) {
    for (const auto& data : dataset) {
      const Vector output = forward(data.second);
      backward(data.first, data.second, output);
      optimizer_.update(weight_input_, diff_weight_input_);
      optimizer_.update(weight_hidden_, diff_weight_hidden_);
    }
  }
}

template <class OptT>
Vector BackPropagation<OptT>::predict(const Vector& input) {
  Vector hidden = prod(input, weight_input_);

  std::transform(hidden.begin(), hidden.end(), hidden_.begin(),
		 [](const double x) { return 1.0 / (1.0 + std::exp(-x)); } );

  return prod(hidden_, weight_hidden_);
}

template <class OptT>
Vector BackPropagation<OptT>::forward(const Vector& input) {
  const Vector hidden = prod(input, weight_input_);

  std::transform(hidden.begin(), hidden.end(), hidden_.begin(),
		 [](const double x) { return 1.0 / (1.0 + std::exp(-x)); } );

  return prod(hidden_, weight_hidden_);
}

template <class OptT>
void BackPropagation<OptT>::backward(const Vector& answer, const Vector& input, const Vector& output) {
  const Vector delta = output - answer;
  auto sigmoid = [](const double x) { return 1.0 / (1.0 + std::exp(-x)); };
  auto diff_sigmoid = [&](const double x) { return sigmoid(x) * (1.0 - sigmoid(x)); };

  Vector hidden = prod(weight_hidden_, delta);

  for (std::size_t i = 0; i < hidden.size(); ++i) {
    hidden[i] *= diff_sigmoid(hidden_[i]);
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

template class BackPropagation<optimizer::GD>;
