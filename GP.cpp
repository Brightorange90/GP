#include "GP.h"
#include "util.h"
#include "def.h"
#include "MVMO.h"
#include <nlopt.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <iomanip>
#include <map>
#include <chrono>
using namespace std;
using namespace Eigen;
using namespace std::chrono;

// train_in  :: dim * num_data
// train_out :: num_data * num_spec
GP::GP(const MatrixXd& train_in, const MatrixXd& train_out)
    : _train_in(train_in),
      _train_out(train_out),
      _cov(CovSEard(train_in.rows())), 
      _num_train(train_in.cols()), 
      _noise_lb(1e-3), 
      _dim(train_in.rows()),
      _num_hyp(_cov.num_hyp() + 2),
      _hyps_lb(VectorXd(_num_hyp)), 
      _hyps_ub(VectorXd(_num_hyp)), 
      _trained(false)
{
    assert(_num_train == static_cast<size_t>(_train_out.rows()));
    _set_hyp_range();
}

void GP::add_data(const MatrixXd& x, const MatrixXd& y)
{
    assert(static_cast<size_t>(x.rows()) == _dim);
    assert(x.cols() == y.rows());

    const size_t num_added = x.cols();
    _train_in.conservativeResize(Eigen::NoChange, _num_train + num_added);
    _train_out.conservativeResize(_num_train + num_added, Eigen::NoChange);
    _train_in.middleCols(_num_train, num_added)  = x;
    _train_out.middleRows(_num_train, num_added) = y;
    _num_train += num_added;
    _trained = false;
}
size_t GP::dim()       const noexcept { return _dim; }
size_t GP::num_hyp()   const noexcept { return _num_hyp; }
size_t GP::num_train() const noexcept { return _num_train; }
bool   GP::trained()   const noexcept { return _trained; }
const  MatrixXd& GP::train_in()  const noexcept { return _train_in; }
const  VectorXd& GP::train_out() const noexcept { return _train_out; }
void GP::set_fixed(bool f) {_fixhyps = f;}
void GP::set_noise_lower_bound(double nlb) noexcept 
{ 
    if(nlb < 0)
        cerr << "Can't set noise lower bound: lower bound must be positive" << endl;
    else
    {
        _noise_lb = nlb;
        if(_noise_free)
            cerr << "Noise-free GP, ignore the noise lower bound" << endl;
        else if(_noise_lb == 0)
        {
            _noise_lb += numeric_limits<double>::epsilon();
            cerr << "Noise level can't be zero, reset it to " << _noise_lb << ", if you want zero noise, you can use set_noise_free() function" << endl;
        }
    }
}
void GP::set_noise_free(bool flag) 
{
    _noise_free = flag;
    if(_noise_free)
        _noise_lb = 0;
}
// void GP::_calcUtilGradM()
// {
//     _utilGradMatrix = MatrixXd(_num_train, _dim * _num_train);
//     for(size_t i = 0; i < _dim; ++i)
//     {
//         _utilGradMatrix.middleCols(_num_train * i, _num_train) = sdist_mm(_train_in.row(i), _train_in.row(i));
//     }
// }
// MatrixXd GP::_covSEard(const VectorXd& hyp) const noexcept
// {
//     const VectorXd inv_lscale = (-1 * hyp.head(_dim)).array().exp();
//     const MatrixXd scaled_train_in = inv_lscale.asDiagonal() * _train_in;
//     return exp(2 * hyp(_dim)) * (-0.5 * sdist_mm(scaled_train_in, scaled_train_in)).array().exp();
// }
// MatrixXd GP::_covSEard(const VectorXd& hyp, const MatrixXd& x) const noexcept
// {
//     const VectorXd inv_lscale = (-1 * hyp.head(_dim)).array().exp();
//     return exp(2 * hyp(_dim)) *
//            (-0.5 * sdist_mm(inv_lscale.asDiagonal() * x, inv_lscale.asDiagonal() * _train_in)).array().exp();
// }
VectorXd GP::get_default_hyps() const noexcept
{
    VectorXd hyp(_num_hyp);
    hyp.head(_cov.num_hyp()) = _cov.default_hyp(_train_in, _train_out);
    hyp(hyp.size()-2)        = noise_free() ? -1 * INF : max(log(_noise_lb), log(stddev<VectorXd>(_train_out) * 1e-3));
    hyp(hyp.size()-1)        = _train_out.mean();
    return hyp;
}
void GP::_init() 
{
    // _calcUtilGradM();
    _set_hyp_range();
    _trained = false;
}
double GP::_calcNegLogProb(const std::vector<double>& x, std::vector<double>& g) const
{
    // for nlopt optimization
    assert(x.size() == _noise_free ? _num_hyp - 1 : _num_hyp);
    const bool need_g    = !g.empty();
    const VectorXd eig_x = vec2hyp(x);
    VectorXd eig_g;
    double nlz = _calcNegLogProb(eig_x, eig_g, need_g);
    if(need_g)
        g = hyp2vec(eig_g);
    return nlz;
}
double GP::_calcNegLogProb(const Eigen::VectorXd& hyp) const
{
    VectorXd dummy_g;
    return _calcNegLogProb(hyp, dummy_g, false);
}
double GP::_calcNegLogProb(const Eigen::VectorXd& hyp, Eigen::VectorXd& grad) const
{
    return _calcNegLogProb(hyp, grad, true);
}
double GP::_calcNegLogProb(const VectorXd& hyp, VectorXd& g, bool calc_grad) const
{
    assert(_num_hyp == _cov.num_hyp() + 2);
    if(_noise_free)
        assert(0 == _hyp_sn2(hyp));

    const double sn2  = _hyp_sn2(hyp);
    const double mean = _hyp_mean(hyp);
    MatrixXd K        = _cov.k(hyp.head(_cov.num_hyp()), _train_in, _train_in);
    ColPivHouseholderQR<MatrixXd> K_solver = (K+sn2*MatrixXd::Identity(_num_train, _num_train)).colPivHouseholderQr();

    double negLogProb = INF;
    if(calc_grad)
        g = VectorXd::Constant(_num_hyp, 1, INF);
    if(K_solver.info() == Eigen::Success and K_solver.isInvertible())
    {
        const VectorXd train_y = _train_out.array() - mean;
        VectorXd alpha         = K_solver.solve(train_y);
        const double data_fit_term    = 0.5 * train_y.dot(alpha);
        const double model_complexity = 0.5 * K_solver.logAbsDeterminant();
        const double norm_const       = 0.5 * _num_train * log(2 * M_PI);
        negLogProb                    = data_fit_term + model_complexity + norm_const;
#ifdef MYDEBUG
        cout << "NegLogProb: " << negLogProb << ", Data fit: " << data_fit_term << ", model_complexity: " << model_complexity << ", calc_grad: " << calc_grad << endl;
#endif
        if(not isfinite(negLogProb))
            negLogProb = INF; // no NaN allowed
        else
        {
            if(calc_grad)
            {
                g           = VectorXd::Constant(_num_hyp, INF);
                MatrixXd Q  = K_solver.inverse() - alpha * alpha.transpose();
                for(size_t i = 0; i < _cov.num_hyp(); ++i)
                {
                    // if A = A' and B = B' then trace(A * B) == sum(sum(A.*B))
                    // g(i) = 0.5 * exp(-2 * hyp(i)) * QK.cwiseProduct(utilGradMatrix.middleCols(_num_train * i, _num_train)).sum();  // log length scale
                    MatrixXd dK = _cov.dk_dhyp(hyp.head(_cov.num_hyp()), i, _train_in, _train_in, K);
                    g(i) = 0.5 * (Q.cwiseProduct(dK)).sum();
                }
                g(_num_hyp-2) = sn2 * Q.trace(); // log sn
                g(_num_hyp-1) = -1*alpha.sum();  // mean
                if(! g.allFinite())
                {
#ifdef MYDEBUG
                    cerr << "Gradient is not finite: " << g.transpose() << endl;
#endif
                    negLogProb = INF;
                    g          = VectorXd::Constant(_num_hyp, INF);
                }
            }
        }
    }
    return negLogProb;
}
double GP::train()
{
    double prob = train(get_default_hyps());
    assert(_trained);
    return prob;
}
double GP::train(const VectorXd& init_hyps) 
{ 
    _init();
    const auto f = [](const vector<double>& x, vector<double>& g, void* data) -> double {
        GP* gp = reinterpret_cast<GP*>(data);
        if(gp->vec2hyp(x).array().hasNaN())
            throw nlopt::forced_stop();
        return gp->_calcNegLogProb(x, g);
    };
    _hyps = init_hyps;
    if(_noise_free)
        _hyps(_hyps.size()-2) = -1 * INF;

    double nlz = _calcNegLogProb(_hyps);
    if(not isfinite(nlz))
        _hyps = select_init_hyp(_num_hyp * 50, _hyps);

    if(_fixhyps)
    {
        _setK();
        _trained = true;
        nlz      = _calcNegLogProb(_hyps);
        return nlz;
    }

    vector<double> hyp_lb = hyp2vec(_hyps_lb);
    vector<double> hyp_ub = hyp2vec(_hyps_ub);
    vector<double> hyp0   = hyp2vec(_hyps);
    for(size_t i = 0; i < hyp0.size(); ++i)
    {
        hyp0[i] = std::max(hyp0[i], hyp_lb[i]);
        hyp0[i] = std::min(hyp0[i], hyp_ub[i]);
    }


    // make sure the starting point is valid
    vector<double> fake_g(hyp0.size(), 0);
    nlz          = f(hyp0, fake_g, this);
    VectorXd dummy_g  = vec2hyp(fake_g);
    if(_noise_free)
        dummy_g[_num_hyp-2] = 0;
#ifdef MYDEBUG
        cout << "Starting nlz: " << nlz << endl;
        _check_hyp_range(vec2hyp(hyp0));
        double rel_err        = _likelihood_gradient_checking(vec2hyp(hyp0), dummy_g);
        cout << "Relative error of gradient: " << rel_err << endl;
#endif

    nlopt::algorithm algo = nlopt::LD_SLSQP;
    size_t max_eval       = 160;

    nlopt::opt optimizer(algo, hyp0.size());
    optimizer.set_maxeval(max_eval);
    optimizer.set_min_objective(f, this);
    optimizer.set_lower_bounds(hyp_lb);
    optimizer.set_upper_bounds(hyp_ub);
    optimizer.set_ftol_abs(1e-6);
    // if(not _noise_free)
    // {
    //     optimizer.add_inequality_constraint( [](const vector<double>& hyp, vector<double>&, void*) -> double {
    //             const size_t dim = hyp.size() - 3;
    //             const double sf  = hyp[dim];
    //             const double sn  = hyp[dim + 1];
    //             return sn - sf; // variance should be greater than noise
    //             }, nullptr);
    // }
    try
    {
        auto t1 = chrono::high_resolution_clock::now();
        optimizer.optimize(hyp0, nlz);
        auto t2 = chrono::high_resolution_clock::now();
        cout << "Training time: " << duration_cast<chrono::seconds>(t2-t1).count() << " seconds" << endl;
    }
    catch(std::runtime_error& e)
    {
#ifdef MYDEBUG
        cerr << "Nlopt exception for GP training caught: " << e.what() << ", algorithm: " << optimizer.get_algorithm_name() << endl;
#endif
    }
    _hyps = vec2hyp(hyp0);

    _setK();
    _trained = true;
    nlz      = _calcNegLogProb(_hyps);
    return nlz;
}
void GP::_predict(const MatrixXd& x, bool need_g, VectorXd& y, VectorXd& s2, MatrixXd& gy, MatrixXd& gs2) const noexcept
{
    assert(_trained);
    const size_t num_test  = x.cols();
    const double sf2       = _cov.sf2(_hyps);
    const double sn2       = _hyp_sn2(_hyps);
    const double mean      = _hyp_mean(_hyps);
    const MatrixXd k_test  = _cov.k(_hyps.head(_cov.num_hyp()), x, _train_in);
    const MatrixXd kks     = _K_solver.solve(k_test.transpose());
    y  = VectorXd::Constant(num_test, 1, mean) + k_test * _invKys;
    s2 = (VectorXd::Constant(num_test, 1, sf2) - k_test.cwiseProduct(kks.transpose()).rowwise().sum()).cwiseMax(0) + VectorXd::Constant(num_test, 1, sn2);
    if(need_g)
    {
        gy  = MatrixXd::Zero(_dim, num_test);
        gs2 = MatrixXd::Zero(_dim, num_test);
        for(size_t i = 0; i < num_test; ++i)
        {
            // XXX: assume k(x, x) = sigma_f^2
            const MatrixXd grad_ktest = _cov.dk_dx1(_hyps, x.col(i), _train_in);
            gy.col(i)    = grad_ktest * _invKys;
            gs2.col(i)   = -2 * grad_ktest * kks.col(i);
        }
    }
}
void GP::_predict_y(const Eigen::MatrixXd& x,  bool need_g, Eigen::VectorXd& y,  Eigen::MatrixXd& gy)  const noexcept
{
    assert(_trained);
    const size_t num_test  = x.cols();
    const double mean      = _hyp_mean(_hyps);
    const MatrixXd k_test  = _cov.k(_hyps.head(_cov.num_hyp()), x, _train_in);  // num_test * num_train;
    y  = VectorXd::Constant(num_test, 1, mean) + k_test * _invKys;
    if(need_g)
    {
        gy  = MatrixXd::Zero(_dim, num_test);
        for(size_t i = 0; i < num_test; ++i)
        {
            const MatrixXd grad_ktest = _cov.dk_dx1(_hyps, x.col(i), _train_in);
            gy.col(i)           = grad_ktest * _invKys;
        }
    }
}
void GP::_predict_s2(const MatrixXd& x, bool need_g, VectorXd& s2, MatrixXd& gs2) const noexcept
{
    assert(_trained);
    const size_t num_test  = x.cols();
    const double sf2       = _cov.sf2(_hyps);
    const double sn2       = _hyp_sn2(_hyps);
    const MatrixXd k_test  = _cov.k(_hyps.head(_cov.num_hyp()), x, _train_in);  // num_test * num_train;
    const MatrixXd kks     = _K_solver.solve(k_test.transpose());
    s2 = (VectorXd::Constant(num_test, 1, sf2) - k_test.cwiseProduct(kks.transpose()).rowwise().sum()).cwiseMax(0) + VectorXd::Constant(num_test, 1, sn2);
    if(need_g)
    {
        gs2 = MatrixXd::Zero(_dim, num_test);
        for(size_t i = 0; i < num_test; ++i)
        {
            const MatrixXd grad_ktest = _cov.dk_dx1(_hyps, x.col(i), _train_in);
            gs2.col(i)          = -2 * grad_ktest * kks.col(i);
        }
    }
}
double GP::predict_y(const VectorXd& x) const
{
    VectorXd y; MatrixXd dummy_g;
    _predict_y(x, false, y, dummy_g);
    return y(0);
}
double GP::predict_s2(const VectorXd& x) const
{
    VectorXd s2; MatrixXd dummy_g;
    _predict_s2(x, false, s2, dummy_g);
    return s2(0);
}
double GP::predict_y_with_grad(const VectorXd& xs, VectorXd& g) const
{
    VectorXd y; MatrixXd gg;
    _predict_y(xs, true, y, gg);
    g = gg.col(0);
    return y(0);
}
double GP::predict_s2_with_grad(const VectorXd& xs, VectorXd& g) const
{
    VectorXd s2; MatrixXd gg;
    _predict_s2(xs, true, s2, gg);
    g = gg.col(0);
    return s2(0);
}
pair<double, VectorXd> GP::predict_y_with_grad(const VectorXd& x) const
{
    VectorXd y; MatrixXd gg;
    _predict_y(x, true, y, gg);
    return {y(0), gg.col(0)};
}
pair<double, VectorXd> GP::predict_s2_with_grad(const VectorXd& x) const
{
    VectorXd s2; MatrixXd gg;
    _predict_s2(x, true, s2, gg);
    return {s2(0), gg.col(0)};
}
void GP::predict(const VectorXd& x, double& y, double& s2) const
{
    VectorXd yy, ss2;
    MatrixXd dummy_gy, dummy_gs2;
    _predict(x, false, yy, ss2, dummy_gy, dummy_gs2);
    y  = yy(0);
    s2 = ss2(0);
}
void GP::predict_with_grad(const VectorXd& xs, double& y, double& s2, VectorXd& gy, VectorXd& gs2) const
{
    VectorXd yy, ss2;
    MatrixXd dummy_gy, dummy_gs2;
    _predict(xs, true, yy, ss2, dummy_gy, dummy_gs2);
    y   = yy(0);
    s2  = ss2(0);
    gy  = dummy_gy.col(0);
    gs2 = dummy_gs2.col(0);
}
std::tuple<double, double> GP::predict(const Eigen::VectorXd& xs) const
{
    double y, s2;
    predict(xs, y, s2);
    return std::tuple<double, double>(y, s2);
}
std::tuple<double, double, VectorXd, VectorXd> GP::predict_with_grad(const VectorXd& xs) const
{
    double y, s2;
    VectorXd gy; VectorXd gs2;
    predict_with_grad(xs, y, s2, gy, gs2);
    return std::tuple<double, double, VectorXd, VectorXd>(y, s2, gy, gs2);
}
VectorXd GP::batch_predict_y(const MatrixXd& xs)  const
{
    VectorXd ys;
    MatrixXd dummy_g;
    _predict_y(xs, false, ys, dummy_g);
    return ys;
}
VectorXd GP::batch_predict_s2(const MatrixXd& xs) const
{
    VectorXd s2;
    MatrixXd dummy_g;
    _predict_s2(xs, false, s2, dummy_g);
    return s2;
}
void GP::batch_predict(const MatrixXd& xs, VectorXd& y, VectorXd& s2) const
{
    MatrixXd dummy_gy, dummy_gs2;
    _predict(xs, false, y, s2, dummy_gy, dummy_gs2);
}
void GP::_setK()
{
    _invKys = VectorXd::Zero(_num_train);
    const MatrixXd EyeM = MatrixXd::Identity(_num_train, _num_train);
    const MatrixXd Kcov = _cov.k(_hyps.head(_cov.num_hyp()), _train_in, _train_in);
    double sn2          = _hyp_sn2(_hyps);
    MatrixXd K          = sn2 * EyeM + Kcov;
    bool is_SPD         = _check_SPD(K);
    while(not is_SPD)
    {
        _hyps(_num_hyp-2) = std::isinf(_hyps(_num_hyp -2)) ? log(numeric_limits<double>::epsilon()) : _hyps(_num_hyp-2) + log(sqrt(10));
        sn2               = _hyp_sn2(_hyps);
        K                 = sn2 * EyeM + Kcov;
#ifdef MYDEBUG
        cerr << "Add noise  to " << sqrt(sn2) << endl;
#endif
        is_SPD = _check_SPD(K);
    }
    _K_solver = K.colPivHouseholderQr();
    _invKys   = _K_solver.solve(static_cast<VectorXd>(_train_out.array() - _hyp_mean(_hyps)));
}
bool GP::_check_SPD(const MatrixXd& K) const
{
    // XXX: don't check symmetry
    MYASSERT(K.rows() == K.cols());
    const size_t size     = K.rows();
    const MatrixXd Eye    = MatrixXd::Identity(size, size);
    const VectorXd eigenv = K.selfadjointView<Lower>().eigenvalues();
    const ColPivHouseholderQR<MatrixXd> K_solver   = K.colPivHouseholderQr();
    const double inv_err  = (Eye - K * K_solver.solve(Eye)).cwiseAbs().mean();
    const double cond     = abs(eigenv.maxCoeff() / eigenv.minCoeff());
    const bool is_SPD     = (inv_err < 1e-4) and (eigenv.array() >= 0).all() and K_solver.info() == Eigen::Success and K_solver.isInvertible();
#ifdef MYDEBUG
    if(not is_SPD)
        cerr << "Inv_err: " << inv_err << ", cond: " << cond << ", DecompSuccess: " << (K_solver.info() == Eigen::Success) << ", isInvertible: " << K_solver.isInvertible() << endl;
#endif
    return is_SPD;
}
const VectorXd& GP::get_hyp() const noexcept { return _hyps; }
VectorXd GP::select_init_hyp(size_t max_eval, const Eigen::VectorXd& def_hyp)
{
    MVMO::MVMO_Obj calc_nlz = [&](const VectorXd& x)->double{
        VectorXd transform_x = vec2hyp(convert(x));
        if(_noise_free)
            transform_x(_num_hyp-2) = -1 * INF;
        // return transform_x(_dim) < transform_x(_dim + 1) ? INF : _calcNegLogProb(transform_x);
        return _calcNegLogProb(transform_x);
    };
    VectorXd lb            = convert(hyp2vec(_hyps_lb));
    VectorXd ub            = convert(hyp2vec(_hyps_ub));
    VectorXd initial_guess = convert(hyp2vec(def_hyp));

    MVMO optimizer(calc_nlz, lb, ub);
    optimizer.set_max_eval(max_eval);
    optimizer.set_fs_init(0.5);
    optimizer.set_fs_final(20);
    optimizer.set_archive_size(25);
    optimizer.optimize(initial_guess);
    return std::isinf(optimizer.best_y()) ? def_hyp : vec2hyp(convert(optimizer.best_x()));
}
double GP::_likelihood_gradient_checking(const VectorXd& hyp, const VectorXd& grad)
{
    const double epsi = 1e-3;
    VectorXd check_grad(_num_hyp);
    for(size_t i = 0; i < _num_hyp; ++i)
    {
        VectorXd check_hyp = hyp;
        check_hyp[i]       = hyp[i] + epsi;
        double check_nlz1  = _calcNegLogProb(check_hyp);
        check_hyp[i]       = hyp[i] - epsi;
        double check_nlz2  = _calcNegLogProb(check_hyp);
        check_grad[i]      = (check_nlz1 - check_nlz2) / (2 * epsi);
    }
    double rel_err = (grad - check_grad).norm() / grad.norm();
#ifdef MYDEBUG
    MatrixXd check_matrix(3, _num_hyp);
    check_matrix << hyp.transpose(), grad.transpose(), check_grad.transpose();
    cout << "CheckGrad:\n" << check_matrix << endl;
    cout << "Rel Difference: " << rel_err << endl;
#endif
    return rel_err;
}
void GP::_check_hyp_range(const VectorXd& hyp) const noexcept
{
    MatrixXd check_matrix(_num_hyp, 3);
    check_matrix << _hyps_lb, _hyps_ub, hyp;
    cout         << "Hyp-range: \n" << check_matrix.transpose() << endl;
}
void GP::_set_hyp_range()
{
    _hyps_lb = VectorXd::Constant(_num_hyp, -1 * INF);
    _hyps_ub = VectorXd::Constant(_num_hyp, 0.5 * log(0.5  * numeric_limits<double>::max()));

    pair<VectorXd, VectorXd> cov_range = _cov.cov_hyp_range(_train_in, _train_out);
    _hyps_lb.head(_cov.num_hyp()) = cov_range.first;
    _hyps_ub.head(_cov.num_hyp()) = cov_range.second;

    // noise
    _hyps_lb(_num_hyp-2) = log(_noise_lb);
    
    //mean
    _hyps_lb(_num_hyp - 1) = _train_out.minCoeff();
    _hyps_ub(_num_hyp - 1) = _train_out.maxCoeff();

}
Eigen::VectorXd GP::vec2hyp(const std::vector<double>& vx) const
{
    assert(vx.size() == _noise_free ? _num_hyp - 1 : _num_hyp);
    VectorXd hyp(_num_hyp);
    if(not _noise_free)
        hyp = convert(vx);
    else
    {
        for(size_t i = 0; i < _cov.num_hyp(); ++i)
            hyp(i) = vx[i];
        hyp(_num_hyp - 2) = -1 * INF;
        hyp(_num_hyp - 1) = vx.back();
    }
    return hyp;
}
std::vector<double> GP::hyp2vec(const Eigen::VectorXd& hyp) const
{
    assert((size_t)hyp.size() == _num_hyp);
    vector<double> vx(_noise_free ? _num_hyp - 1 : _num_hyp);
    for(size_t i = 0; i < _cov.num_hyp(); ++i)
        vx[i] = hyp(i);
    if(_noise_free)
        vx[_num_hyp-2] = hyp(_num_hyp-1);
    else
    {
        vx[_num_hyp-2] = hyp(_num_hyp-2);
        vx[_num_hyp-1] = hyp(_num_hyp-1);
    }
    return vx;
}
double GP::_hyp_sn2(const VectorXd& hyp) const
{
    assert((size_t)hyp.size() == _num_hyp);
    return exp(2*hyp(_num_hyp-2));
}
double GP::_hyp_mean(const Eigen::VectorXd& hyp) const
{
    assert((size_t)hyp.size() == _num_hyp);
    return hyp(_num_hyp-1);
}
