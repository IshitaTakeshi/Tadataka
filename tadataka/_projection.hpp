const Eigen::Vector3d inv_pi(const Eigen::Vector2d x, double depth);
const Eigen::MatrixXd inv_pi(
    const Eigen::Matrix<double, Eigen::Dynamic, 2> xs,
    const Eigen::VectorXd depths);
