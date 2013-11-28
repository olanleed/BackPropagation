#include "backpropagation.hpp"
#include "random_generator.hpp"

BackPropagation::BackPropagation(const int input_layer, const int hidden_layer,
				 const int output_layer, const double learn_rate,
				 const double momentum, const double weight_decay, const double max_epoch)
  : learn_rate_(learn_rate), momentum_(momentum),
    weight_decay_(weight_decay), max_epoch_(max_epoch),
    hidden_(hidden_layer), masked_hidden_(hidden_layer) {

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

void BackPropagation::train(const std::vector< std::pair< dvector, dvector > >& data_set) {
  for (std::size_t i = 0; i < max_epoch_; ++i) {
    for (std::size_t j = 0; j < 100; ++j) {
      const int index = Random::generate_int(0, data_set.size() - 1);
      const dvector mask = generate_dropout_mask(hidden_.size());
      const dvector output = forward_propagete(data_set[index].second, mask);
      back_propagate(data_set[index].first, data_set[index].second, output, mask);
      update_weight();
    }
  }
}

BackPropagation::dvector BackPropagation::predict(const dvector& input) {
  dvector hidden = prod(input, weight_input_);

  std::transform(hidden.begin(), hidden.end(), hidden_.begin(),
		 [](const double x) { return 1.0 / (1.0 + std::exp(-x)); } );

  dvector tmp = hidden_;
  for(double& x : tmp) {
    x /= 2.0;
  }

  return prod(tmp, weight_hidden_);
}

BackPropagation::dvector BackPropagation::forward_propagete(const dvector& input, const dvector& mask) {
  dvector hidden = prod(input, weight_input_);

  std::transform(hidden.begin(), hidden.end(), hidden_.begin(),
		 [](const double x) { return 1.0 / (1.0 + std::exp(-x)); } );

  for (std::size_t i = 0; i < hidden_.size(); ++i) {
    masked_hidden_[i] = hidden_[i] * mask[i];
  }

  return prod(masked_hidden_, weight_hidden_);
}

void BackPropagation::back_propagate(const dvector& answer, const dvector& input,
				     const dvector& output, const dvector& mask) {
  const dvector delta = output - answer;
  //auto sigmoid = [](const double x) { return 1.0 / (1.0 + std::exp(-x)); };
  //auto diff_sigmoid = [&](const int x) { return sigmoid(x) * (1.0 - sigmoid(x)); };
  dvector error_hidden = prod(weight_hidden_, delta);

  for (std::size_t i = 0; i < error_hidden.size(); ++i) {
    error_hidden[i] *= masked_hidden_[i] * (1.0 - masked_hidden_[i]);
  }

  for (std::size_t i = 0; i < diff_weight_hidden_.size1(); ++i) {
    for (std::size_t j = 0; j < diff_weight_hidden_.size2(); ++j) {
      diff_weight_hidden_(i, j) = masked_hidden_[i] * delta[j];
    }
  }

  for (std::size_t i = 0; i < diff_weight_input_.size1(); ++i) {
    for (std::size_t j = 0; j < diff_weight_input_.size2(); ++j) {
      diff_weight_input_(i, j) = input[i] * error_hidden[j];
    }
  }
}

void BackPropagation::update_weight(void) {
  weight_input_ -= learn_rate_ * diff_weight_input_;
  weight_hidden_ -= learn_rate_ * diff_weight_hidden_;
}

BackPropagation::dvector BackPropagation::generate_dropout_mask(const int max_size) {
  dvector mask(max_size, 1.0);

  std::set<int> results;
  while(results.size() < (max_size / 2)) {
    results.insert(Random::generate_int(0, max_size - 1));
  }

  for (const auto& result : results) {
    mask(result) = 0.0;
  }

  return mask;
}
