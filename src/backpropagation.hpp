#ifndef BACKPROPAGATION_INCLUDE_BACKPROPAGATION_HPP_
#define BACKPROPAGATION_INCLUDE_BACKPROPAGATION_HPP_

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <cstddef>
#include <set>

class BackPropagation {
public :
  using dmatrix = boost::numeric::ublas::matrix<double>;
  using dvector = boost::numeric::ublas::vector<double>;

private :
  const double learn_rate_;
  const double momentum_;
  const double weight_decay_;
  const double max_epoch_;

private:
  dmatrix weight_input_;
  dmatrix weight_hidden_;
  dmatrix diff_weight_input_;
  dmatrix diff_weight_hidden_;
  dvector hidden_;
  dvector masked_hidden_;
public :
  BackPropagation(const int input_layer, const int hidden_layer,
		  const int output_layer, const double learn_rate,
		  const double momentum, const double weight_decay, const double max_epoch);

  virtual ~BackPropagation(void);

  void train(const std::vector< std::pair< dvector, dvector > >& data_set);
  dvector predict(const dvector& input);

private :
  dvector forward_propagete(const dvector& input, const dvector& mask);
  void back_propagate(const dvector& answer, const dvector& input,
		      const dvector& output, const dvector& mask);
  void update_weight(void);
  dvector generate_dropout_mask(const int max_size);
};

#endif //BACKPROPAGATION_INCLUDE_BACKPROPAGATION_HPP_
