#include <pybind11/eigen.h>


const Eigen::VectorXd to_homogeneous_vector(
    Eigen::Ref<const Eigen::VectorXd> x);
const Eigen::MatrixXd to_homogeneous_vectors(
    Eigen::Ref<const Eigen::MatrixXd> xs);
