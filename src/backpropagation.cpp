#include "backpropagation.hpp"
#include "random_generator.hpp"

BackPropagation::BackPropagation(const int input_layer, const int hidden_layer,
				 const int output_layer, const double learn_rate,
				 const double dropout_rate, const double momentum,
				 const double weight_decay, const double max_epoch)
  : learn_rate_(learn_rate), dropout_rate_(dropout_rate), momentum_(momentum),
    weight_decay_(weight_decay), max_epoch_(max_epoch), hidden_(hidden_layer) {

  weight_input_.resize(input_layer, hidden_layer);
  weight_hidden_.resize(hidden_layer, output_layer);
  diff_weight_input_.resize(input_layer, hidden_layer);
  diff_weight_hidden_.resize(hidden_layer, output_layer);

  for (dmatrix::iterator1 it = weight_input_.begin1(); it != weight_input_.end1(); ++it) {
    std::transform(it.begin(), it.end(), it.begin(), [](const double x) { return Random::generate(); });
  }

  for (dmatrix::iterator1 it = weight_hidden_.begin1(); it != weight_hidden_.end1(); ++it) {
    std::transform(it.begin(), it.end(), it.begin(), [](const double x) { return Random::generate(); });
  }
}

BackPropagation::~BackPropagation(void) { }

void BackPropagation::train(std::vector< std::pair< dvector, dvector > > data_set) {
  for (std::size_t i = 0; i < max_epoch_; ++i) {
    for (std::size_t j = 0; j < data_set.size(); ++j) {
      forward_propagate(data_set[j].second);
      back_propagate(data_set[j].first);
      update_weight();
    }
  }
}

BackPropagation::dvector BackPropagation::predict(const dvector& input) {
  dvector hidden = prod(input, weight_input_);

  std::transform(hidden.begin(), hidden.end(), hidden_.begin(),
		 [](const double x) { return 1.0 / (1.0 + std::exp(-x)); } );

  return prod(hidden_, weight_hidden_);
}


void BackPropagation::forward_propagate(const dvector& input) {
  input_ = input;
  dvector hidden = prod(input_, weight_input_);

  std::transform(hidden.begin(), hidden.end(), hidden_.begin(),
		 [](const double x) { return 1.0 / (1.0 + std::exp(-x)); } );


  output_ = prod(hidden_, weight_hidden_);
}

void BackPropagation::back_propagate(const dvector& answer) {
  dvector delta = output_ - answer;
  auto sigmoid = [](const double x) { return 1.0 / (1.0 + std::exp(-x)); };
  auto diff_sigmoid = [&](const int x) { return sigmoid(x) * (1.0 - sigmoid(x)); };
  dvector error_hidden = prod(weight_hidden_, delta);

  for (std::size_t i = 0; i < error_hidden.size(); ++i) {
    error_hidden[i] *= diff_sigmoid(hidden_[i]);
  }

  for (std::size_t i = 0; i < diff_weight_hidden_.size1(); ++i) {
    for (std::size_t j = 0; j < diff_weight_hidden_.size2(); ++j) {
      diff_weight_hidden_(i, j) = hidden_[i] * delta[j];
    }
  }

  for (std::size_t i = 0; i < diff_weight_input_.size1(); ++i) {
    for (std::size_t j = 0; j < diff_weight_input_.size2(); ++j) {
      diff_weight_input_(i, j) = input_[i] * error_hidden[j];
    }
  }
}

void BackPropagation::update_weight(void) {
  weight_input_ -= learn_rate_ * diff_weight_input_;
  weight_hidden_ -= learn_rate_ * diff_weight_hidden_;
}
