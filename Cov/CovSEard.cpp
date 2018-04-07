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
std::pair<VectorXd, VectorXd> CovSEard::cov_hyp_range(const MatrixXd& xs, const VectorXd& ys) const
{
    VectorXd hyps_lb = VectorXd::Constant(num_hyp(), -1 * INF);
    VectorXd hyps_ub = VectorXd::Constant(num_hyp(), 0.5 * log(0.5  * numeric_limits<double>::max()));
    // // length scale
    for(size_t i = 0; i < _dim; ++i)
    {
        // exp(-0.5 (\frac{magic_num}{exp(_hyps_lb[i]})^2) > (1.5 * \text{numeric_limits<double>::min()})
        long max_idx, min_idx;
        xs.row(i).maxCoeff(&max_idx);
        xs.row(i).minCoeff(&min_idx);
        const double distance  = xs(i, max_idx) - xs(i, min_idx);
        const double magic_num = 0.05 * distance;

        // ub1: exp(hyp[i])^2 < 0.05 * numeric_limits<double>::max()
        // ub2: true == (exp(-1e-17) == 1.0), so we set -0.5 * \frac{distance^2}{exp(hyp[i])^2} < -1e-16
        // ub2: exp(-0.5 * d^2 / l^2) > (1 - thres)
        const double thres = 1e-4;
        double ub1         = 0.5 * log(0.05 * numeric_limits<double>::max());
        double ub2         = log(distance / sqrt(-2 * log(1 - thres)));

        double lscale_lb = log(magic_num) - 0.5 * log(-2 * log(1.5 * numeric_limits<double>::min()));
        double lscale_ub = min(ub1, ub2);
        hyps_lb(i) = lscale_lb; 
        hyps_ub(i) = lscale_ub; 
    }

    // variance
    hyps_lb(_dim) = log(max(0.0, numeric_limits<double>::epsilon() * (ys.maxCoeff() - ys.minCoeff())));
    hyps_ub(_dim) = log(10 * (ys.maxCoeff() - ys.minCoeff()));
    return {hyps_lb, hyps_ub};
}
VectorXd CovSEard::default_hyp(const MatrixXd& xs, const VectorXd& ys) const
{
    VectorXd default_hyp(num_hyp());
    for(size_t i = 0; i < _dim; ++i)
        default_hyp(i) = log(stddev<RowVectorXd>(xs.row(i)));
    default_hyp(_dim) = log(stddev<VectorXd>(ys));
    return default_hyp;
}
VectorXd CovSEard::diag_k(const VectorXd& hyp, const MatrixXd& x) const 
{
    const double sf2 = exp(2 * hyp(_dim));
    return VectorXd::Constant(x.cols(), 1, sf2);
}
MatrixXd CovSEard::diag_dk_dhyp(const VectorXd& hyp, const MatrixXd& x) const
{
    VectorXd k_diag = diag_k(hyp, x);
    return diag_dk_dhyp(hyp, x, k_diag.asDiagonal());
}
MatrixXd CovSEard::diag_dk_dhyp(const VectorXd& hyp, const MatrixXd& x, const MatrixXd&) const
{
    const double sf2 = exp(2 * hyp(_dim));
    MatrixXd grad    = MatrixXd::Zero(hyp.size(), x.cols());
    grad.row(_dim)   = RowVectorXd::Constant(1, grad.cols(), 2*sf2);
    return grad;
}
MatrixXd CovSEard::diag_dk_dx1(const VectorXd&, const VectorXd& x, const RowVectorXd&) const
{
    return MatrixXd::Zero(_dim, x.cols());
}
MatrixXd CovSEard::diag_dk_dx1(const VectorXd&, const VectorXd& x) const
{
    return MatrixXd::Zero(_dim, x.cols());
}
