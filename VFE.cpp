#include "VFE.h"
#include "MatrixSolver.h"
#include <fstream>
#include <cstdio>
#include <chrono>
using namespace std;
using namespace Eigen;
using namespace std::chrono;
VFE::VFE(const MatrixXd& train_in, const MatrixXd& train_out, GP::CovFunc cf, GP::MatrixDecomp md)
    : GP(train_in, train_out, cf, md)
{
    // XXX: can not be noise-free
    _inducing     = train_in;
    _num_inducing = train_in.cols();
    _u_solver     = _specify_matrix_solver(_md);
    _A_solver     = _specify_matrix_solver(_md);
}
VFE::~VFE()
{
}
void VFE::set_inducing(const MatrixXd& u)
{
    _inducing     = u;
    _num_inducing = u.cols();
}
void VFE::_init()
{
    this->GP::_init();
    _jitter_u = pow(1e-1*_noise_lb, 2);
}
double VFE::train(const VectorXd& _hyp)
{
    VectorXd hyp = _hyp;
    _init();
    if(_noise_free)
    {
        cerr << "VFE can't be noise free" << endl;
        set_noise_free(false);
    }
    const auto f = [](const vector<double>& x, vector<double>& g, void* data) -> double {
        VFE* vfe = reinterpret_cast<VFE*>(data);
        if(vfe->vec2hyp(x).array().hasNaN())
            throw nlopt::forced_stop();
        return vfe->GP::_calcNegLogProb(x, g);
    };
    double nlz = this->GP::_calcNegLogProb(hyp);
    if(not isfinite(nlz))
        hyp = select_init_hyp(_num_hyp * 50, hyp);
    if(_fixhyps)
    {
        _hyps = hyp;
        _setK();
        _trained = true;
        nlz      = this->GP::_calcNegLogProb(_hyps);
        return nlz;
    }
    vector<double> hyp_lb = hyp2vec(_hyps_lb);
    vector<double> hyp_ub = hyp2vec(_hyps_ub);
    vector<double> hyp0   = hyp2vec(hyp);
    for(size_t i = 0; i < hyp0.size(); ++i)
    {
        hyp0[i] = std::max(hyp0[i], hyp_lb[i]);
        hyp0[i] = std::min(hyp0[i], hyp_ub[i]);
    }
#ifdef MYDEBUG
        VectorXd dummy_g;
        _calcNegLogProb(hyp, dummy_g, true);
        cout << "Starting nlz: " << nlz << endl;
        _check_hyp_range(vec2hyp(hyp0));
        double rel_err        = _likelihood_gradient_checking(vec2hyp(hyp0), dummy_g);
        cout << "Relative error of gradient: " << rel_err << endl;
#endif
    nlopt::algorithm algo = nlopt::LD_SLSQP;
    size_t max_eval       = 150;

    nlopt::opt optimizer(algo, hyp0.size());
    optimizer.set_maxeval(max_eval);
    optimizer.set_min_objective(f, this);
    optimizer.set_lower_bounds(hyp_lb);
    optimizer.set_upper_bounds(hyp_ub);

    nlopt::result r;
    try
    {
#ifdef MYDEBUG
        auto t1 = chrono::high_resolution_clock::now();
        r = optimizer.optimize(hyp0, nlz);
        auto t2 = chrono::high_resolution_clock::now();
        cout << "Training time: " << duration_cast<chrono::seconds>(t2-t1).count() << " seconds" << endl;
#elif
        r = optimizer.optimize(hyp0, nlz);
#endif
    }
    catch(std::runtime_error& e)
    {
#ifdef MYDEBUG
        cerr << "Nlopt exception for GP training caught: " << e.what() << ", algorithm: " << optimizer.get_algorithm_name() << endl;
#endif
        hyp  = select_init_hyp(_num_hyp * 10, vec2hyp(hyp0));
        hyp0 = hyp2vec(hyp);
    }
    cout << explain_nlopt(r) << endl;
    _hyps = vec2hyp(hyp0);

    _setK();
    _trained = true;
    return this->GP::_calcNegLogProb(_hyps);
}
void VFE::_predict(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& y, Eigen::VectorXd& s2, Eigen::MatrixXd& gy, Eigen::MatrixXd& gs2) const noexcept
{
    assert(not need_g);
}
void VFE::_predict_y(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& y, Eigen::MatrixXd& gy) const noexcept
{
    const double mean     = _hyp_mean(_hyps);
    const MatrixXd K_star = _cov->k(_hyps, x, _inducing);
    y = (K_star * _alpha).array() + mean;
    if(need_g)
    {
        gy = MatrixXd(_dim, x.cols());
        for(long i = 0; i < x.cols(); ++i)
            gy.col(i) = _cov->dk_dx1(_hyps, x.col(i), _inducing) * _alpha;
    }
}
void VFE::_predict_s2(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& s2, Eigen::MatrixXd& gs2) const noexcept
{
    const VectorXd sf2    = _cov->diag_k(_hyps, x);
    const MatrixXd K_star = _cov->k(_hyps, x, _inducing);
    const MatrixXd KinvK  = _u_solver->solve(K_star.transpose()) - _A_solver->solve(K_star.transpose());
    s2                    = (sf2 - (K_star * KinvK).diagonal()).cwiseMax(0);
}
void VFE::_setK()
{
    const double sn2   = _hyp_sn2(_hyps);
    const VectorXd sf2 = _cov->diag_k(_hyps, _train_in);
    const VectorXd r   = _train_out.array() - _hyp_mean(_hyps);
    const MatrixXd Kxu = _cov->k(_hyps, _train_in, _inducing);
    const MatrixXd Kux = Kxu.transpose();
    const MatrixXd Eye = MatrixXd::Identity(_num_inducing, _num_inducing);

    MatrixXd Kuu = _cov->k(_hyps, _inducing, _inducing);
    MatrixXd A   = Kuu + Kux * Kxu / sn2;
    _u_solver->decomp(Kuu);
    _A_solver->decomp(A);

    bool SPD = _A_solver->check_SPD() and _u_solver->check_SPD();
    while(not SPD)
    {
        Kuu = Kuu + _jitter_u * Eye;
        A   = Kuu + Kux * Kxu / sn2;
#ifdef MYDEBUG
        cerr << "Add jitter to " << _jitter_u << endl;
#endif
        _u_solver->decomp(Kuu);
        _A_solver->decomp(A);
        _jitter_u *= 2;
        SPD = _A_solver->check_SPD() and _u_solver->check_SPD();
    }
    _alpha = _A_solver->solve(Kux * r) / sn2;
}
double VFE::_calcNegLogProb(const VectorXd& hyp, VectorXd& g, bool calc_grad) const
{
    return _calcNegLogProb(hyp, g, calc_grad, _jitter_u);
}
double VFE::_calcNegLogProb(const VectorXd& hyp, VectorXd& g, bool calc_grad, double jitter_u) const
{
    const double sn2     = _hyp_sn2(hyp);
    const double mean    = _hyp_mean(hyp);
    const VectorXd y     = _train_out.array() - mean;
    const MatrixXd Kuu   = _cov->k(hyp, _inducing, _inducing) + jitter_u * MatrixXd::Identity(_num_inducing, _num_inducing);
    const MatrixXd Kxu   = _cov->k(hyp, _train_in, _inducing);
    const MatrixXd Kux   = Kxu.transpose();
    const MatrixXd Kuxxu = Kux * Kxu;
    const MatrixXd A     = sn2 * Kuu + Kuxxu;

    MatrixSolver* u_solver = _specify_matrix_solver(_md);
    MatrixSolver* A_solver = _specify_matrix_solver(_md);
    u_solver->decomp(Kuu);
    A_solver->decomp(A);

    const MatrixXd Kuu_inv = u_solver->inverse();
    const MatrixXd A_inv   = A_solver->inverse();
    const VectorXd alpha   = (y - Kxu * A_solver->solve(Kux * y)) / sn2;

    const double f0                 = 0.5 * _num_train * log(2 * M_PI);
    const double f_model_complexity = 0.5 * (A_solver->log_det() - u_solver->log_det() + (_num_train - _num_inducing) * log(sn2));
    const double f_data_fit         = 0.5 * y.dot(alpha);
    const double f_trace_term       = 0.5 * (_cov->diag_k(hyp, _train_in).sum() - (Kuu_inv * Kuxxu).trace()) / sn2;
    double nlz                      = f0 + f_model_complexity + f_data_fit + f_trace_term;
    if(not (u_solver->check_SPD() and A_solver->check_SPD() and isfinite(nlz)))
    {
        nlz = INF;
        if(calc_grad)
            g = VectorXd::Zero(_num_hyp, 1);
    }

    if(calc_grad and isfinite(nlz))
    {
        g = VectorXd::Zero(_num_hyp);
        const MatrixXd AinvKux = A_inv * Kux;
        const VectorXd F       = Kux * y;
        const VectorXd AinvF   = AinvKux * y;

        const MatrixXd AinvKuxyytKxuAinv    = AinvF * AinvF.transpose();
        const MatrixXd KxuAinvKuxyytKxuAinv = Kxu   * AinvKuxyytKxuAinv;
        const MatrixXd AinvFyt              = AinvF * y.transpose();

        const MatrixXd Kuu_inv_Kux             = Kuu_inv * Kux;
        const MatrixXd Kuu_inv_Kux_Kxu_Kuu_inv = Kuu_inv_Kux * Kuu_inv_Kux.transpose();
        const MatrixXd dKnn                    = _cov->diag_dk_dhyp(hyp, _train_in).transpose();

        // if A :: N*m, B:: m*m and B = B.transpose(), C :: m * N then (A * B * C).trace() == (B * (C * A)).trace()
        for(size_t i = 0; i < _cov->num_hyp(); ++i)
        {
            MatrixXd dKuu = _cov->dk_dhyp(hyp, i, _inducing, _inducing, Kuu);
            MatrixXd dKxu = _cov->dk_dhyp(hyp, i, _train_in, _inducing, Kxu);
            MatrixXd dKux = dKxu.transpose();


            // derivative of data fit term
            double g_datafit_1 = -1 * (sn2 * dKuu.cwiseProduct(AinvKuxyytKxuAinv).sum() + 2 * dKux.cwiseProduct(KxuAinvKuxyytKxuAinv.transpose()).sum());
            double g_datafit_2 = 2 * dKux.cwiseProduct(AinvFyt).sum();
            double g_datafit   = -0.5 * (g_datafit_1 + g_datafit_2) / sn2;


            // derivative of the model_complexity term
            double d_logA             = sn2 * A_inv.cwiseProduct(dKuu).sum() + 2 * AinvKux.cwiseProduct(dKux).sum();
            double d_logKuu           = Kuu_inv.cwiseProduct(dKuu).sum();
            double g_model_complexity = 0.5 * (d_logA - d_logKuu);

            // derivatives of the trace term
            double d_diag_Kxx = dKnn.col(i).sum();
            double d_diag_Q     = 2 * dKxu.cwiseProduct(Kuu_inv_Kux.transpose()).sum() - dKuu.cwiseProduct(Kuu_inv_Kux_Kxu_Kuu_inv).sum();
            double g_trace_term = 0.5 * (d_diag_Kxx - d_diag_Q) / sn2;

            g(i) = g_datafit + g_model_complexity + g_trace_term;
        }
        // XXX: it seems that g(_num_hyp-2) == -2 * sn2 * alpha.dot(alpha)
        g(_num_hyp-2) = -1 * sn2 * alpha.dot(alpha) + (_num_train - Kxu.cwiseProduct(AinvKux.transpose()).sum()) - 2 * f_trace_term;
        g(_num_hyp-1) = -1 * alpha.sum(); // mean
    }
#ifdef MYDEBUG
    cout << "nlz = " << nlz << ", model_complexity = " << f_model_complexity << ", data_fit =" << f_data_fit
         << ", trace_term = " << f_trace_term << ", calc_grad = " << calc_grad << endl;
#endif
    delete u_solver;
    delete A_solver;
    return nlz;
}
MatrixXd VFE::_sym(const MatrixXd& m) const
{
    return 0.5 * (m + m.transpose());
}
void VFE::test_obj(const VectorXd& hyp)
{
    VectorXd grad = hyp;
    auto t1 = std::chrono::high_resolution_clock::now();
    double nlz = _calcNegLogProb(hyp, grad, true);
    auto t2 = std::chrono::high_resolution_clock::now();
    size_t span_1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    VectorXd grad_diff = hyp;
    auto t3 = std::chrono::high_resolution_clock::now();
    VectorXd dummy;
    const double epsi = 1e-6;
    for(size_t i = 0; i < _num_hyp; ++i)
    {
        VectorXd hyp1 = hyp;
        VectorXd hyp2 = hyp;
        hyp1[i]       += epsi;
        hyp2[i]       -= epsi;
        grad_diff[i]  = (this->GP::_calcNegLogProb(hyp1) - this->GP::_calcNegLogProb(hyp2)) / (2 * epsi);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    size_t span_2 = std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count();
    MatrixXd rec(_num_hyp, 3);
    rec << hyp, grad, grad_diff;
    cout << "---------------------------------------" << endl;
    cout << rec << endl;
    cout << "nlz: " << nlz << endl;
    cout << "Time grad: "      << span_1 << " ms" << endl;
    cout << "Time grad_diff: " << span_2 << " ms" << endl;
}
