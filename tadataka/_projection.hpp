#include "tadataka/_types.hpp"

const Eigen::Vector3d inv_pi(const Eigen::Vector2d x, double depth);
const Eigen::MatrixXd inv_pi(
    const Eigen::Matrix<double, Eigen::Dynamic, 2> xs,
    const Eigen::VectorXd depths);
void project_vector(
    const Eigen::Ref<const Eigen::Vector3d>& p,
    Eigen::Ref<Eigen::Vector2d> u);
void project_vectors(
    const Eigen::Ref<const RowMajorMatrixXd<Eigen::Dynamic, 3>>& P,
    Eigen::Ref<RowMajorMatrixXd<Eigen::Dynamic, 2>> U);
