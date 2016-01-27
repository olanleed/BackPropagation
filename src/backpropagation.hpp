#ifndef BACKPROPAGATION_INCLUDE_BACKPROPAGATION_HPP_
#define BACKPROPAGATION_INCLUDE_BACKPROPAGATION_HPP_

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <cstddef>
#include <set>
#include <functional>

class BackPropagation {
public :
  using dmatrix = boost::numeric::ublas::matrix<double>;
  using dvector = boost::numeric::ublas::vector<double>;

private :
  const double learn_rate_;
  const double epoch_;

private:
  dmatrix weight_input_;
  dmatrix weight_hidden_;
  dmatrix diff_weight_input_;
  dmatrix diff_weight_hidden_;
  dvector hidden_;

public :
  BackPropagation(const int input_layer, const int hidden_layer,
		  const int output_layer, const double learn_rate,
		  const double epoch);

  virtual ~BackPropagation(void);

  void fit(const std::vector< std::pair< dvector, dvector > >& data_set);
  dvector predict(const dvector& input);

private :
  dvector forward(const dvector& input);
  void backward(const dvector& answer, const dvector& input, const dvector& output);
  void update_weight(void);
};

#endif //BACKPROPAGATION_INCLUDE_BACKPROPAGATION_HPP_
