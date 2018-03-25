#include "CovSEiso.h"
using namespace std;
using namespace Eigen;
CovSEiso::CovSEiso(size_t d) : Cov(d) {}
size_t CovSEiso::num_hyp() const { return 2; }
MatrixXd CovSEiso::k(const VectorXd& hyp, const MatrixXd& x1, const MatrixXd& x2) const
{
    const double l   = exp(hyp(0));
    const double sf2 = exp(2*hyp(1));
    return sf2 * (-0.5 * sdist_mm(x1/l, x2/l)).array().exp();
}
MatrixXd CovSEiso::dk_dhyp(const VectorXd& hyp, size_t i, const MatrixXd& x1, const MatrixXd& x2, const MatrixXd& K) const
{
    switch(i)
    {
        case 0:
            {
                const double l = exp(hyp(0));
                MatrixXd dist  = sdist_mm(x1/l, x2/l);
                return K.cwiseProduct(dist);
            }
        case 1:
            return 2 * K;
        default:
            cerr << "Wrong index for dk_dhyp" << endl;
            exit(EXIT_FAILURE);
    }
}
MatrixXd CovSEiso::dk_dhyp(const VectorXd& hyp, size_t i, const MatrixXd& x1, const MatrixXd& x2) const
{
    MatrixXd K = k(hyp, x1, x2);
    return dk_dhyp(hyp, i, x1, x2, K);
}
MatrixXd CovSEiso::dk_dx1(const VectorXd& hyp, const VectorXd& x1, const MatrixXd& x2) const
{
    const RowVectorXd K = k(hyp, x1, x2);
    return dk_dx1(hyp, x1, x2, K);
}
MatrixXd CovSEiso::dk_dx1(const VectorXd& hyp, const VectorXd& x1, const MatrixXd& x2, const RowVectorXd& K) const
{
    const double inv_lscale = exp(-2 * hyp(0));
    MatrixXd dK   = inv_lscale * (x2.colwise() - x1);
    for(size_t i  = 0; i < _dim; ++i)
        dK.row(i) = K.cwiseProduct(dK.row(i));
    return dK;
}
std::pair<VectorXd, VectorXd> CovSEiso::cov_hyp_range(const MatrixXd& xs, const VectorXd& ys) const
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
        hyps_lb(0)       = max(lscale_lb, hyps_lb(0));
        hyps_ub(0)       = min(lscale_ub, hyps_ub(0));
    }

    // variance
    hyps_lb(_dim) = log(max(0.0, numeric_limits<double>::epsilon() * (ys.maxCoeff() - ys.minCoeff())));
    hyps_ub(_dim) = log(10 * (ys.maxCoeff() - ys.minCoeff()));
    return {hyps_lb, hyps_ub};
}
VectorXd CovSEiso::default_hyp(const MatrixXd&, const VectorXd& ys) const
{
    VectorXd default_hyp(num_hyp());
    default_hyp(0) = 0;
    default_hyp(1) = log(stddev<VectorXd>(ys));
    return default_hyp;
}
double CovSEiso::sf2(const Eigen::VectorXd& hyp) const
{
    assert((size_t)hyp.size() >= num_hyp());
    return exp(2 * hyp(1));
}
