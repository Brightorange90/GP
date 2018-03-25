#include "CovSEard.h"
#include <fstream>
using namespace std;
using namespace Eigen;
CovSEard::CovSEard(size_t d) : Cov(d) {}
size_t CovSEard::num_hyp() const { return _dim + 1; }
MatrixXd CovSEard::k(const VectorXd& hyp, const MatrixXd& x1, const MatrixXd& x2) const
{
    const VectorXd inv_lscale = (-1 * hyp.head(_dim)).array().exp();
    return exp(2 * hyp(_dim)) * (-0.5 * sdist_mm(inv_lscale.asDiagonal() * x1, inv_lscale.asDiagonal() * x2)).array().exp();
}
MatrixXd CovSEard::dk_dhyp(const VectorXd& hyp, size_t i, const MatrixXd& x1, const MatrixXd& x2, const MatrixXd& K) const
{
    if(i < _dim)
    {
        double l_inv  = exp(-1 * hyp(i));
        MatrixXd dist = sdist_mm(l_inv*x1.row(i), l_inv*x2.row(i));
        return K.cwiseProduct(dist);
    }
    else
        return 2 * K;
}
MatrixXd CovSEard::dk_dhyp(const VectorXd& hyp, size_t i, const MatrixXd& x1, const MatrixXd& x2) const
{
    MatrixXd K = k(hyp, x1, x2);
    return dk_dhyp(hyp, i, x1, x2, K);
}
MatrixXd CovSEard::dk_dx1(const VectorXd& hyp, const VectorXd& x1, const MatrixXd& x2) const
{
    const RowVectorXd K = k(hyp, x1, x2);
    return dk_dx1(hyp, x1, x2, K);
}
MatrixXd CovSEard::dk_dx1(const VectorXd& hyp, const VectorXd& x1, const MatrixXd& x2, const RowVectorXd& K) const
{
    const VectorXd inv_lscale = (-2 * hyp.head(_dim)).array().exp();
    MatrixXd dK   = inv_lscale.asDiagonal() * (x2.colwise() - x1);
    for(size_t i  = 0; i < _dim; ++i)
        dK.row(i) = K.cwiseProduct(dK.row(i));
    return dK;
}
