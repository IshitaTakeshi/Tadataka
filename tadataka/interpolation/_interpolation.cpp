#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;


double __interpolation(const Eigen::MatrixXd &image,
                       const double cx, const double cy) {
  double lx = floor(cx);
  double ly = floor(cy);
  int lxi = (int)lx;
  int lyi = (int)ly;

  if(lx == cx && ly == cy) {
    return image(lyi, lxi);
  }

  double ux = lx + 1.0;
  double uy = ly + 1.0;
  int uxi = (int)ux;
  int uyi = (int)uy;

  if(lx == cx) {
    return (image(lyi, lxi) * (ux - cx) * (uy - cy) +
            image(uyi, lxi) * (ux - cx) * (cy - ly));
  }

  if(ly == cy) {
    return (image(lyi, lxi) * (ux - cx) * (uy - cy) +
            image(lyi, uxi) * (cx - lx) * (uy - cy));
  }

  return (image(lyi, lxi) * (ux - cx) * (uy - cy) +
          image(lyi, uxi) * (cx - lx) * (uy - cy) +
          image(uyi, lxi) * (ux - cx) * (cy - ly) +
          image(uyi, uxi) * (cx - lx) * (cy - ly));
}


double interpolation_(const Eigen::MatrixXd &image,
                      const Eigen::Matrix<double, 2, 1> &coordinate) {
  return __interpolation(image, coordinate(0), coordinate(1));
}


Eigen::VectorXd interpolation(
    const Eigen::MatrixXd &image,
    const Eigen::Matrix<double, Eigen::Dynamic, 2> &coordinates) {
  const int N = coordinates.rows();
  Eigen::VectorXd intensities(N);
  for(int i = 0; i < N; i++) {
    intensities(i) = interpolation_(image, coordinates.row(i));
  }
  return intensities;
}


PYBIND11_MODULE(_interpolation, m) {
  m.def("interpolation", &interpolation);
}
