#include "../src/backpropagation.hpp"
#include <iostream>

int main(void) {
  using Matrix = boost::numeric::ublas::matrix<double>;
  using Vector = boost::numeric::ublas::vector<double>;
  std::vector< std::pair<Vector, Vector> > data_set;
  Vector v1(2), v2(2), v3(2), v4(2);
  Vector a1(1), a2(1);
  a1(0) = 0, a2(0) = 1;
  v1(0) = v1(1) = 0;
  v2(0) = 0, v2(1) = 1;
  v3(0) = 1, v3(1) = 0;
  v4(0) = v4(1) = 1;

  data_set.push_back(std::make_pair(a1, v1));
  data_set.push_back(std::make_pair(a2, v2));
  data_set.push_back(std::make_pair(a2, v3));
  data_set.push_back(std::make_pair(a1, v4));

  BackPropagation bp(2, 15, 1, 0.5, 0.1, 0.1, 10000);
  bp.train(data_set);

  std::cout << bp.predict(v1)(0) << std::endl;
  std::cout << bp.predict(v2)(0) << std::endl;
  std::cout << bp.predict(v3)(0) << std::endl;
  std::cout << bp.predict(v4)(0) << std::endl;

  return 0;
}
