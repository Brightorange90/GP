#include "GP.h"
#include "util.h"
#include "def.h"
#include "MVMO.h"
#include <nlopt.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <utility>
#include <iomanip>
#include <fstream>
#include <map>
#include <boost/exception/diagnostic_information.hpp>
using namespace std;
using namespace Eigen;

// train_in  :: dim * num_data
// train_out :: num_data * num_spec
GP::GP(const MatrixXd& train_in, const MatrixXd& train_out)
    : _train_in(train_in),
      _train_out(train_out),
      _num_train(train_in.cols()), 
      _noise_lb(1e-3), 
      _dim(train_in.rows()),
      _num_spec(train_out.cols()),
      _num_hyp(_dim + 3),
      _hyps_lb(MatrixXd(_num_hyp, _num_spec)), 
      _hyps_ub(MatrixXd(_num_hyp, _num_spec)), 
      _trained(false)
{
    assert(_num_train == static_cast<size_t>(_train_out.rows()));
    _set_hyp_range();
}
// Append new training data
void GP::add_data(const MatrixXd& x, const MatrixXd& y)
{
    assert(static_cast<size_t>(x.rows()) == _dim);
    assert(x.cols() == y.rows());
    assert(static_cast<size_t>(y.cols()) == _num_spec);

    const size_t num_added = x.cols();
    _train_in.conservativeResize(Eigen::NoChange, _num_train + num_added);
    _train_out.conservativeResize(_num_train + num_added, Eigen::NoChange);
    _train_in.middleCols(_num_train, num_added)  = x;
    _train_out.middleRows(_num_train, num_added) = y;
    _num_train += num_added;
    _trained = false;
}
size_t GP::dim()       const noexcept { return _dim; }
size_t GP::num_spec()  const noexcept { return _num_spec; }
size_t GP::num_hyp()   const noexcept { return _num_hyp; }
size_t GP::num_train() const noexcept { return _num_train; }
bool   GP::trained()   const noexcept { return _trained; }
const  MatrixXd& GP::train_in()  const noexcept { return _train_in; }
const  MatrixXd& GP::train_out() const noexcept { return _train_out; }
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
void GP::_calcUtilGradM()
{
    _utilGradMatrix = MatrixXd(_num_train, _dim * _num_train);
    for(size_t i = 0; i < _dim; ++i)
    {
        _utilGradMatrix.middleCols(_num_train * i, _num_train) = sdist_mm(_train_in.row(i), _train_in.row(i));
    }
}
MatrixXd GP::_covSEard(const VectorXd& hyp) const noexcept
{
    const VectorXd inv_lscale = (-1 * hyp.head(_dim)).array().exp();
    const MatrixXd scaled_train_in = inv_lscale.asDiagonal() * _train_in;
    MatrixXd K = exp(2 * hyp(_dim)) * (-0.5 * sdist_mm(scaled_train_in, scaled_train_in)).array().exp();
    return K;
}
MatrixXd GP::_covSEard(const VectorXd& hyp, const MatrixXd& x) const noexcept
{
    const VectorXd inv_lscale = (-1 * hyp.head(_dim)).array().exp();
    return exp(2 * hyp(_dim)) *
           (-0.5 * sdist_mm(inv_lscale.asDiagonal() * x, inv_lscale.asDiagonal() * _train_in)).array().exp();
}
MatrixXd GP::get_default_hyps() const noexcept
{
    MatrixXd hyps(_num_hyp, _num_spec);
    for(size_t i = 0; i < _num_spec; ++i)
    {
        VectorXd hyp(_num_hyp);
        for(size_t i = 0; i < _dim; ++i)
        {
            hyp(i) = log(stddev<RowVectorXd>(_train_in.row(i)));
        }
        hyp(_dim)   = log(stddev<VectorXd>(_train_out.col(i)) / sqrt(2));
        hyp(_dim+1) = noise_free() ? -1 * INF : max(log(_noise_lb), log(stddev<VectorXd>(_train_out.col(i)) * 1e-2));
        hyp(_dim+2) = _train_out.col(i).mean();
        hyps.col(i) = hyp;
    }
    return hyps;
}
void GP::_init() 
{
    _calcUtilGradM();
    _set_hyp_range();
    _trained = false;
}
double GP::_calcNegLogProb(const std::vector<double>& x, std::vector<double>& g, size_t idx) const
{
    // for nlopt
    assert(x.size() == _noise_free ? _num_hyp - 1 : _num_hyp);
    const bool need_g    = !g.empty();
    const VectorXd eig_x = vec2hyp(x);
    VectorXd eig_g;
    double nlz = _calcNegLogProb(eig_x, eig_g, need_g, idx);
    if(need_g)
        g = hyp2vec(eig_g);
    return nlz;
}
double GP::_calcNegLogProb(const VectorXd& hyp_, VectorXd& g, bool calc_grad, size_t idx) const
{
    assert(_num_hyp == _dim + 3);
    VectorXd hyp = hyp_;
    if(_noise_free)
        assert(std::isinf(hyp(_dim + 1)));

    const double sn2      = exp(2 * hyp(_dim + 1));
    const double mean     = hyp(_dim + 2);
    MatrixXd K            = _covSEard(hyp);
    ColPivHouseholderQR<MatrixXd> K_solver = (K+sn2*MatrixXd::Identity(_num_train, _num_train)).colPivHouseholderQr();

    double negLogProb = INF;
    g                 = VectorXd::Constant(_num_hyp, INF);
    // if(K_llt.info() == Eigen::Success and K_llt.isPositive() and (K_llt.vectorD().array() > 0).all())
    if(K_solver.info() == Eigen::Success and K_solver.isInvertible())
    {
        const VectorXd train_y = _train_out.col(idx).array() - mean;
        VectorXd alpha         = K_solver.solve(train_y);
        const double data_fit_term    = 0.5 * train_y.dot(alpha);
        // const double model_complexity = K_ldlt.matrixL().toDenseMatrix().diagonal().array().log().sum();
        // const double model_complexity = 0.5 * K_solver.vectorD().array().log().sum();
        const double model_complexity = 0.5 * K_solver.logAbsDeterminant();
        const double norm_const       = 0.5 * _num_train * log(2 * M_PI);
        negLogProb                    = data_fit_term + model_complexity + norm_const;
#ifdef MYDEBUG
        cout << "NegLogProb: " << negLogProb << ", Data fit: " << data_fit_term << ", model_complexity: " << model_complexity << endl;
#endif
        if(not isfinite(negLogProb))
            negLogProb = INF; // no NaN allowed
        else
        {
            if(calc_grad)
            {
                MatrixXd Q  = K_solver.solve(MatrixXd::Identity(_num_train, _num_train)) - alpha * alpha.transpose();
                MatrixXd QK = Q.cwiseProduct(K);
                for(size_t i = 0; i < _dim; ++i)
                {
                    // trace(Q * (K .* U)) == sum(sum(Q .* K .* U), and I don't know why
                    // g(i) = 0.5 * exp(-2 * hyp(i)) * (Q * (K.cwiseProduct(_utilGradMatrix.middleCols(_num_train * i, _num_train)))).trace();
                    g(i) = 0.5 * exp(-2 * hyp(i)) * QK.cwiseProduct(_utilGradMatrix.middleCols(_num_train * i, _num_train)).sum();  // log length scale
                }
                g(_dim)   = QK.sum();        // log sf,  trace(Q * K) == sum(sum(Q .* K)), I don't know why
                g(_dim+1) = sn2 * Q.trace(); // log sn
                g(_dim+2) = -1*alpha.sum();  // mean
                if(! g.allFinite())
                {
#ifdef MYDEBUG
                    cerr << "Gradient is not finite" << endl;
                    exit(EXIT_FAILURE);
#endif
                    negLogProb = INF;
                    g          = VectorXd::Constant(_num_hyp, INF);
                }
            }
        }
    }
    return negLogProb;
}
VectorXd GP::train()
{
    VectorXd prob = train(get_default_hyps());
    assert(_trained);
    return prob;
}
VectorXd GP::train(const MatrixXd& init_hyps) 
{ 
    _init();
    const auto f = [](const vector<double>& x, vector<double>& g, void* data) -> double {
        auto* nlopt_data      = reinterpret_cast<pair<size_t, GP*>*>(data);
        const size_t spec_idx = nlopt_data->first;
        GP* gp                = nlopt_data->second;
        double nlz            = gp->_calcNegLogProb(x, g, spec_idx);
        if(gp->vec2hyp(x).array().hasNaN())
            throw nlopt::forced_stop();
        return nlz;
    };
    _hyps = init_hyps;
    if(_noise_free)
        _hyps.row(_dim + 1) = RowVectorXd::Zero(1, _num_spec).array().log();
    for(size_t i = 0; i < _num_spec; ++i)
    {
        VectorXd fg;
        double nlz = _calcNegLogProb(_hyps.col(i), fg, false, i);
        if(not isfinite(nlz))
            _hyps.col(i) = select_init_hyp(_num_hyp * 50, i, _hyps.col(i));
    }
    VectorXd nlzs(_num_spec);
    if(_fixhyps)
    {
        _setK();
        _trained = true;
        VectorXd fg;
        for(size_t i = 0; i < _num_spec; ++i)
            nlzs(i) = _calcNegLogProb(_hyps.col(i), fg, false, i);
        return nlzs;
    }
#pragma omp parallel for schedule(static)
    for(size_t i = 0; i < _num_spec; ++i)
    {
        vector<double> hyp_lb = hyp2vec(_hyps_lb.col(i));
        vector<double> hyp_ub = hyp2vec(_hyps_ub.col(i));
        vector<double> hyp0   = hyp2vec(_hyps.col(i));
        for(size_t i = 0; i < hyp0.size(); ++i)
        {
            hyp0[i] = std::max(hyp0[i], hyp_lb[i]);
            hyp0[i] = std::min(hyp0[i], hyp_ub[i]);
        }


        // make sure the starting point is valid
        pair<size_t, GP*> nlopt_data{i, this};
        vector<double> fake_g(hyp0.size(), 0);
        double nlz          = f(hyp0, fake_g, &nlopt_data);
        VectorXd eig_fakeg  = vec2hyp(fake_g);
        if(_noise_free)
            eig_fakeg[_dim + 1] = 0;
#ifdef MYDEBUG
        cout << "Starting nlz: " << nlz << endl;
        _check_hyp_range(i, vec2hyp(hyp0));
#endif
        double rel_err        = _likelihood_gradient_checking(i, vec2hyp(hyp0), eig_fakeg);
        nlopt::algorithm algo = (isfinite(rel_err) and rel_err < 0.05 and eig_fakeg.allFinite()) ? nlopt::LD_SLSQP : nlopt::LN_COBYLA;
        size_t max_eval       = algo == nlopt::LD_SLSQP ? 160 : _num_hyp * 50;
        nlopt::opt optimizer(algo, hyp0.size());
        optimizer.set_maxeval(max_eval);
        optimizer.set_min_objective(f, &nlopt_data);
        optimizer.set_lower_bounds(hyp_lb);
        optimizer.set_upper_bounds(hyp_ub);
        optimizer.set_ftol_abs(1e-6);
        if(not _noise_free)
        {
            optimizer.add_inequality_constraint( [](const vector<double>& hyp, vector<double>&, void*) -> double {
                    size_t dim = hyp.size() - 3;
                    double sf = hyp[dim];
                    double sn = hyp[dim + 1];
                    return sn - sf;
                    }, nullptr);
        }
        try
        {
            optimizer.optimize(hyp0, nlz);
        }
        catch(std::runtime_error& e)
        {
#ifdef MYDEBUG
            cerr << "Nlopt exception for GP training caught: " << e.what() << ", algorithm: " << optimizer.get_algorithm_name() << endl;
#endif
            vector<double> g;
            hyp0 = hyp2vec(select_init_hyp(_num_hyp * 50, i, vec2hyp(hyp0)));
            nlz  = _calcNegLogProb(hyp0, g, i);
        }
        _hyps.col(i) = vec2hyp(hyp0);
        nlzs(i)      = nlz;
#ifdef MYDEBUG
        if(_noise_free)
        {
            VectorXd vhyp  = _hyps.col(i);
            MatrixXd R     = exp(-2 * vhyp(_dim)) * _covSEard(_hyps.col(i));
            double mean    = vhyp(vhyp.size() - 1);
            VectorXd y     = _train_out.col(i).array() - mean;
            double tmp     = y.transpose() * R.colPivHouseholderQr().solve(y);
            double anal_sf = sqrt(tmp / _num_train);
            double sf      = exp(vhyp(_dim));

            VectorXd g, g_;
            double vnlz  = _calcNegLogProb(vhyp, g, true, i);
            vhyp(_dim)   = log(anal_sf);
            double vnlz_ = _calcNegLogProb(vhyp, g_, true, i);
            cout << "sf = " << sf << ", anal_sf = " << anal_sf << endl;
            cout << "Nlz for origin sf:  " << vnlz << endl;
            cout << "Grad for origin sf: " << g.transpose() << endl;
            cout << "Nlz for anal sf:    " << vnlz_ << endl;
            cout << "Grad for anal sf:   " << g_.transpose() << endl;
        }
#endif
    }
    _setK();
    _trained = true;
    for(size_t i = 0; i < _num_spec; ++i)
    {
        VectorXd fg;
        nlzs(i) = _calcNegLogProb(_hyps.col(i), fg, false, i);
#ifdef MYDEBUG
        cout << "Optimized nlz for spec " << i << ": " << nlzs(i) << endl;
#endif
    }
    return nlzs;
}
void GP::_setK()
{
    _K_solver.clear();
    _K_solver.reserve(_num_spec);
    _invKys = MatrixXd::Zero(_num_train, _num_spec);
    const MatrixXd EyeM = MatrixXd::Identity(_num_train, _num_train);
    for(size_t i = 0; i < _num_spec; ++i)
    {
        const MatrixXd Kcov = _covSEard(_hyps.col(i));
        double sn2          = exp(2 * _hyps(_num_hyp-2, i));
        MatrixXd K          = sn2 * EyeM + _covSEard(_hyps.col(i));
        bool is_SPD         = _check_SPD(K);
        while(not is_SPD)
        {
            _hyps(_num_hyp-2, i) = std::isinf(_hyps(_num_hyp -2, i)) ? log(numeric_limits<double>::epsilon()) : _hyps(_num_hyp-2, i) + log(sqrt(10));
            sn2                  = exp(2 * _hyps(_num_hyp-2, i));
            K                    = sn2 * EyeM + Kcov;
#ifdef MYDEBUG
            cerr << "Add noise of spec" << i << " to " << sqrt(sn2) << endl;
#endif
            is_SPD = _check_SPD(K);
            if(std::isinf(sn2))
            {
                cerr << "Fail to build GP model" << endl;
#ifdef MYDEBUG
                cerr << "i:\n" << i << endl;
                cerr << "Train x:\n" << _train_in << endl;
                cerr << "Train y:\n" << _train_out << endl;
                cerr << "hyps:\n" << _hyps << endl;
#endif
                exit(EXIT_FAILURE);
            }
        }
        _K_solver.push_back(K.colPivHouseholderQr());
        _invKys.col(i) = _K_solver[i].solve(static_cast<VectorXd>(_train_out.col(i).array() - _hyps(_dim + 2, i)));
    }
}
bool GP::_check_SPD(const MatrixXd& K) const
{
    // XXX: don't check symmetry
    MYASSERT(K.rows() == K.cols());
    const size_t size     = K.rows();
    const MatrixXd Eye    = MatrixXd::Identity(size, size);
    const VectorXd eigenv = K.selfadjointView<Lower>().eigenvalues();
    const auto K_solver   = K.colPivHouseholderQr();
    const double inv_err  = (Eye - K * K_solver.solve(Eye)).cwiseAbs().mean();
    const double cond     = abs(eigenv.maxCoeff() / eigenv.minCoeff());
    const bool is_SPD     = (inv_err < 1e-4) and (eigenv.array() >= 0).all() and K_solver.info() == Eigen::Success and K_solver.isInvertible();
#ifdef MYDEBUG
    if(not is_SPD)
        cerr << "Inv_err: " << inv_err << ", cond: " << cond << ", DecompSuccess: " << (K_solver.info() == Eigen::Success) << ", isInvertible: " << K_solver.isInvertible() << endl;
#endif
    return is_SPD;
}
const MatrixXd& GP::get_hyp() const noexcept { return _hyps; }
void GP::_predict(size_t spec_id, const MatrixXd& x, bool need_g, VectorXd& y, VectorXd& s2, MatrixXd& gy,
                  MatrixXd& gs2) const noexcept
{
    if(! _trained)
    {
        cerr << "Model not trained!" << endl;
        exit(EXIT_FAILURE);
    }
    const size_t num_test  = x.cols();
    const VectorXd& vhyp   = _hyps.col(spec_id);
    const double sf2       = exp(2 * vhyp(_dim));
    const double sn2       = exp(2 * vhyp(_dim + 1));
    const double mean      = vhyp(_dim + 2);
    const MatrixXd k_test  = _covSEard(vhyp, x);  // num_test * num_train;
    const MatrixXd kks     = _K_solver[spec_id].solve(k_test.transpose());
    y  = VectorXd::Constant(num_test, 1, mean) + k_test * _invKys.col(spec_id);
    s2 = (VectorXd::Constant(num_test, 1, sf2) - k_test.cwiseProduct(kks.transpose()).rowwise().sum()).cwiseMax(0) + VectorXd::Constant(num_test, 1, sn2);
    if(need_g)
    {
        gy  = MatrixXd::Zero(_dim, num_test);
        gs2 = MatrixXd::Zero(_dim, num_test);
        // usually when gradient is needed, num_test is one
        const MatrixXd inv_lscale = (-2 * vhyp.head(_dim)).array().exp();
        for(size_t i = 0; i < num_test; ++i)
        {
            // MatrixXd jt    = (-1 * inv_lscale * tmp_x) * k_test.row(i).asDiagonal();
            // Jt:: partial K / partial x
            MatrixXd Jt = (inv_lscale.asDiagonal() * (_train_in.colwise() - x.col(i))) * k_test.row(i).asDiagonal();
            gy.col(i)   = Jt * _invKys.col(spec_id);
            gs2.col(i)  = -2 * Jt * kks.col(i);
        }
    }
}
void GP::predict(const MatrixXd& x, MatrixXd& y, MatrixXd& s2) const noexcept
{
    assert(static_cast<size_t>(x.rows()) == _dim);
    const size_t num_pred = x.cols();
    y  = MatrixXd::Zero(num_pred, _num_spec);
    s2 = MatrixXd::Zero(num_pred, _num_spec);
    for(size_t i = 0; i < _num_spec; ++i)
    {
        VectorXd vy(num_pred, 1);
        VectorXd vs2(num_pred, 1);
        MatrixXd gy;
        MatrixXd gs2;
        _predict(i, x, false, vy, vs2, gy, gs2);
        y.col(i)  = vy;
        s2.col(i) = vs2;
    }
}
void GP::predict(size_t spec_idx, const Eigen::VectorXd&x, double& y, double& s2) const noexcept
{
    VectorXd vy, vs2;
    predict(spec_idx, x, vy, vs2);
    y  = vy(0);
    s2 = vs2(0);
}
void GP::predict(size_t spec_idx, const Eigen::MatrixXd&x, Eigen::VectorXd& y, Eigen::VectorXd& s2) const noexcept
{
    MatrixXd dummy_gy, dummy_gs2;
    _predict(spec_idx, x, false, y, s2, dummy_gy, dummy_gs2);
}
void GP::predict_with_grad(size_t spec_idx, const Eigen::VectorXd& x, double& y, double& s2, Eigen::VectorXd& grad_y, Eigen::VectorXd& grad_s2) const noexcept
{
    VectorXd vy, vs2;
    MatrixXd gy, gs2;
    _predict(spec_idx, x, true, vy, vs2, gy, gs2);
    y       = vy(0);
    s2      = vs2(0);
    grad_y  = gy.col(0);
    grad_s2 = gs2.col(0);
}

void GP::predict_with_grad(
        size_t spec_idx,    const Eigen::MatrixXd& xs,
        Eigen::VectorXd& y, Eigen::VectorXd& s2,
        Eigen::MatrixXd& grad_y, Eigen::MatrixXd& grad_s2) const noexcept
{
    size_t num_test = xs.cols();

    y       = VectorXd(num_test, 1);
    s2      = VectorXd(num_test, 1);
    grad_y  = MatrixXd(_dim, num_test);
    grad_s2 = MatrixXd(_dim, num_test);

    for(size_t i = 0; i < num_test; ++i)
    {
        double tmp_y, tmp_s2;
        VectorXd tmp_gy, tmp_gs2;
        predict_with_grad(spec_idx, xs.col(i), tmp_y, tmp_s2, tmp_gy, tmp_gs2);
        y(i)           = tmp_y;
        s2(i)          = tmp_s2;
        grad_y.col(i)  = tmp_gy;
        grad_s2.col(i) = tmp_gs2;
    }
}
VectorXd GP::select_init_hyp(size_t max_eval, size_t spec_idx, const Eigen::VectorXd& def_hyp)
{
    MVMO::MVMO_Obj calc_nlz = [&](const VectorXd& x)->double{
        VectorXd transform_x = vec2hyp(convert(x));
        VectorXd fake_g;
        if(_noise_free)
            transform_x(_dim + 1) = -1 * INF;
        return transform_x(_dim) < transform_x(_dim + 1) ? INF 
                                                         : _calcNegLogProb(transform_x, fake_g, false, spec_idx);
    };
    VectorXd lb            = convert(hyp2vec(_hyps_lb.col(spec_idx)));
    VectorXd ub            = convert(hyp2vec(_hyps_ub.col(spec_idx)));
    VectorXd initial_guess = convert(hyp2vec(def_hyp));

    MVMO optimizer(calc_nlz, lb, ub);
    optimizer.set_max_eval(max_eval);
    optimizer.set_fs_init(0.5);
    optimizer.set_fs_final(20);
    optimizer.set_archive_size(25);
    optimizer.optimize(initial_guess);
    return std::isinf(optimizer.best_y()) ? def_hyp : vec2hyp(convert(optimizer.best_x()));
//     pair<size_t, GP*> nlopt_data{spec_idx, this};

//     nlopt::opt opt(nlopt::GN_ISRES, initial_guess.size());
//     opt.set_lower_bounds(lb);
//     opt.set_upper_bounds(ub);
//     opt.set_population(2 * initial_guess.size());
//     opt.set_maxeval(max((size_t)opt.get_population() * 10, max_eval));
//     opt.set_min_objective(calc_nlz, &nlopt_data);
//     if(not _noise_free)
//     {
//         opt.add_inequality_constraint([](const vector<double>& hyp, vector<double>&, void*) -> double {
//                 size_t dim = hyp.size() - 3;
//                 double sf  = hyp[dim];
//                 double sn  = hyp[dim + 1];
//                 return sn - sf;
//                 }, nullptr);
//     }

//     double val;
//     try
//     {
//         opt.optimize(initial_guess, val);
//     }
//     catch(runtime_error& e)
//     {
// #ifdef MYDEBUG
//         cerr << "Nlopt exception in select_init_hyp: " << e.what() << endl;
// #endif
//     }
//     catch(exception& e)
//     {
// #ifdef MYDEBUG
//         cerr << "Nlopt exception in select_init_hyp: " << e.what() << endl;
//         exit(EXIT_FAILURE);
// #endif
//     }
//     return vec2hyp(initial_guess);
}
MatrixXd GP::select_init_hyp(size_t max_eval, const MatrixXd& def_hyp)
{
    MatrixXd candidate_hyp = def_hyp;
#pragma omp parallel for
    for(size_t i = 0; i < _num_spec; ++i)
    {
        candidate_hyp.col(i) = select_init_hyp(max_eval, i, def_hyp.col(i));
    }
    return candidate_hyp;
}
double GP::_likelihood_gradient_checking(size_t spec_idx, const VectorXd& hyp, const VectorXd& grad)
{
    const double epsi = 1e-3;
    VectorXd check_grad(_num_hyp);
    for(size_t i = 0; i < _num_hyp; ++i)
    {
        VectorXd fake_g;
        VectorXd check_hyp = hyp;
        check_hyp[i]       = hyp[i] + epsi;
        double check_nlz1  = _calcNegLogProb(check_hyp, fake_g, false, spec_idx);
        check_hyp[i]       = hyp[i] - epsi;
        double check_nlz2  = _calcNegLogProb(check_hyp, fake_g, false, spec_idx);
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
void GP::_check_hyp_range(size_t spec_idx, const VectorXd& hyp) const noexcept
{
    MatrixXd check_matrix(_num_hyp, 3);
    check_matrix << _hyps_lb.col(spec_idx), _hyps_ub.col(spec_idx), hyp;
    cout         << "Hyp-range: \n" << check_matrix.transpose() << endl;
}
void GP::_set_hyp_range()
{
    _hyps_lb = MatrixXd::Constant(_num_hyp, _num_spec, -1 * INF);
    _hyps_ub = MatrixXd::Constant(_num_hyp, _num_spec, 0.5 * log(0.5  * numeric_limits<double>::max()));
    // length scale
    for(size_t i = 0; i < _dim; ++i)
    {
        // exp(-0.5 (\frac{magic_num}{exp(_hyps_lb[i]})^2) > (1.5 * \text{numeric_limits<double>::min()})
        long max_idx, min_idx;
        _train_in.row(i).maxCoeff(&max_idx);
        _train_in.row(i).minCoeff(&min_idx);
        const double distance  = _train_in(i, max_idx) - _train_in(i, min_idx);
        const double magic_num = 0.05 * distance;
        double lscale_lb       = log(magic_num) - 0.5 * log(-2 * log(1.5 * numeric_limits<double>::min()));

        // ub1: exp(hyp[i])^2 < 0.05 * numeric_limits<double>::max()
        double ub1       = 0.5 * log(0.05 * numeric_limits<double>::max());
        // ub2: true == (exp(-1e-17) == 1.0), so we set -0.5 * \frac{distance^2}{exp(hyp[i])^2} < -1e-16
        // ub2: exp(-0.5 * d^2 / l^2) > (1 - thres)
        const double thres = 1e-4;
        double ub2         = log(distance / sqrt(-2 * log(1 - thres)));

        double lscale_ub = min(ub1, ub2);
        _hyps_lb.row(i) = RowVectorXd::Constant(1, _num_spec, lscale_lb);
        _hyps_ub.row(i) = RowVectorXd::Constant(1, _num_spec, lscale_ub);
    }
    //variance and mean
    const double epsi = numeric_limits<double>::epsilon();
    for(size_t i = 0; i < _num_spec; ++i)
    {
        long max_idx, min_idx;
        _train_out.col(i).maxCoeff(&max_idx);
        _train_out.col(i).minCoeff(&min_idx);
        _hyps_lb(_dim, i)     = log(max(epsi, numeric_limits<double>::epsilon() * (_train_out(max_idx, i) - _train_out(min_idx, i))));
        _hyps_ub(_dim, i)     = log(max(10*epsi, 10 * (_train_out(max_idx, i) - _train_out(min_idx, i))));
        _hyps_lb(_dim + 2, i) = _train_out(min_idx, i) - epsi;
        _hyps_ub(_dim + 2, i) = _train_out(max_idx, i) + epsi;
    }

    // // noise
    _hyps_lb.row(_dim + 1) = RowVectorXd::Constant(1, _num_spec, log(_noise_lb));
    _hyps_ub.row(_dim + 1) = _hyps_ub.row(_dim).cwiseMax(log(10 * _noise_lb));
}
Eigen::VectorXd GP::vec2hyp(const std::vector<double>& vx) const
{
    assert(vx.size() == _noise_free ? _num_hyp - 1 : _num_hyp);
    VectorXd hyp(_num_hyp);
    if(not _noise_free)
        hyp = convert(vx);
    else
    {
        for(size_t i = 0; i < _dim + 1; ++i)
            hyp(i) = vx[i];
        hyp(_dim + 1) = -1 * INF;
        hyp(_dim + 2) = vx.back();
    }
    return hyp;
}
std::vector<double> GP::hyp2vec(const Eigen::VectorXd& hyp) const
{
    assert((size_t)hyp.size() == _num_hyp);
    vector<double> vx(_noise_free ? _num_hyp - 1 : _num_hyp);
    for(size_t i = 0; i < _dim + 1; ++i)
        vx[i] = hyp(i);
    if(_noise_free)
        vx[_dim + 1] = hyp(_dim + 2);
    else
    {
        vx[_dim + 1] = hyp(_dim + 1);
        vx[_dim + 2] = hyp(_dim + 2);
    }
    return vx;
}
