#include "FITC.h"
#include "MatrixSolver.h"
#include <fstream>
#include <chrono>
using namespace std;
using namespace Eigen;
using namespace std::chrono;
FITC::FITC(const MatrixXd& train_in, const MatrixXd& train_out, GP::CovFunc cf, GP::MatrixDecomp md)
    : GP(train_in, train_out, cf, md)
{
    // XXX: can not be noise-free
    _inducing     = train_in;
    _num_inducing = train_in.cols();
    _u_solver     = _specify_matrix_solver(_md);
    _A_solver     = _specify_matrix_solver(_md);
}
FITC::~FITC()
{
    delete _u_solver;
    delete _A_solver;
}
void FITC::set_inducing(const MatrixXd& u)
{
    _inducing     = u;
    _num_inducing = u.cols();
}
double FITC::train(const VectorXd& _hyp)
{
    VectorXd hyp = _hyp;
    _init();
    if(_noise_free)
    {
        cerr << "FITC can't be noise free" << endl;
        set_noise_free(false);
    }
    const auto f = [](const vector<double>& x, vector<double>& g, void* data) -> double {
        FITC* fitc = reinterpret_cast<FITC*>(data);
        if(fitc->vec2hyp(x).array().hasNaN())
            throw nlopt::forced_stop();
        return fitc->GP::_calcNegLogProb(x, g);
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
    // optimizer.set_xtol_abs(1e-9);

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
    }
    cout << explain_nlopt(r) << endl;
    _hyps = vec2hyp(hyp0);

    _setK();
    _trained = true;
    return this->GP::_calcNegLogProb(_hyps);
}    
void FITC::_predict(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& y, Eigen::VectorXd& s2, Eigen::MatrixXd& gy, Eigen::MatrixXd& gs2) const noexcept
{
    assert(not need_g);
    const double sf2      = _cov->sf2(_hyps);
    const double sn2      = _hyp_sn2(_hyps);
    const double mean     = _hyp_mean(_hyps);
    const MatrixXd K_star = _cov->k(_hyps, x, _inducing);
    const MatrixXd KinvK  = _u_solver->solve(K_star.transpose()) - sn2 * _A_solver->solve(K_star.transpose());
    y  = (K_star * _alpha).array() + mean;
    s2 = (sf2 + sn2 - (K_star * KinvK).diagonal().array()).cwiseMax(sn2);
    if(need_g)
    {
        gy  = MatrixXd(_dim, x.cols());
        gs2 = MatrixXd(_dim, x.cols());
        for(long i = 0; i < x.cols(); ++i)
        {
            const MatrixXd grad_k = _cov->dk_dx1(_hyps, x.col(i), _inducing);
            gy.col(i)             = grad_k * _alpha;
            gs2.col(i)            = -2 * grad_k * KinvK.col(i);
        }
    }
}
void FITC::_predict_y(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& y, Eigen::MatrixXd& gy) const noexcept
{
    assert(not need_g);
    const double mean = _hyp_mean(_hyps);
    MatrixXd K_star   = _cov->k(_hyps, x, _inducing);
    y                 = (K_star * _alpha).array() + mean;
    if(need_g)
    {
        gy = MatrixXd(_dim, x.cols());
        for(long i = 0; i < x.cols(); ++i)
        {
            MatrixXd dK_star = _cov->dk_dx1(_hyps, x.col(i), _inducing);
            gy.col(i)        = dK_star * _alpha;
        }
    }
}
void FITC::_predict_s2(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& s2, Eigen::MatrixXd& gs2) const noexcept
{
    assert(not need_g);
    assert(not need_g);
    const double sf2      = _cov->sf2(_hyps);
    const double sn2      = _hyp_sn2(_hyps);
    const MatrixXd K_star = _cov->k(_hyps, x, _inducing);
    const MatrixXd KinvK  = _u_solver->solve(K_star.transpose()) - sn2 * _A_solver->solve(K_star.transpose());

    s2  = (sf2 + sn2 - (K_star * KinvK).diagonal().array()).cwiseMax(sn2);
    if(need_g)
    {
        gs2 = MatrixXd(_dim, x.cols());
        for(long i = 0; i < x.cols(); ++i)
        {
            const MatrixXd grad_k = _cov->dk_dx1(_hyps, x.col(i), _inducing);
            gs2.col(i)            = -2 * grad_k * KinvK.col(i);
        }
    }
}
void FITC::_setK()
{
    const double sn2   = _hyp_sn2(_hyps);
    const double sf2   = _cov->sf2(_hyps);
    double jitter      = 1e-6 * sn2;
    const VectorXd r   = _train_out.array() - _hyp_mean(_hyps);
    const MatrixXd Kxu = _cov->k(_hyps, _train_in, _inducing);
    const MatrixXd Kux = Kxu.transpose();
    const MatrixXd Kuu = _cov->k(_hyps, _inducing, _inducing);
    const MatrixXd Eye = MatrixXd::Identity(_num_inducing, _num_inducing);

    _u_solver->decomp(Kuu + jitter * Eye);
    MatrixXd Kuu_inv_Kux = _u_solver->solve(Kux);
    MatrixXd Kxu_Kuu_inv = Kuu_inv_Kux.transpose();
    VectorXd Gamma       = ((sf2+sn2) - (Kxu * Kuu_inv_Kux).diagonal().array()) / sn2; // Eigen seems to have already optimized (A * B).diagonal()
    VectorXd inv_Gamma   = Gamma.cwiseInverse();
    MatrixXd A           = sn2 * Kuu + Kux * inv_Gamma.asDiagonal() * Kxu;
    _A_solver->decomp(A);

    bool SPD = _A_solver->check_SPD() and _u_solver->check_SPD();
    while(not SPD)
    {
        jitter *= 2;
#ifdef MYDEBUG
        cerr << "Add jitter to " << jitter << endl;
#endif
        _u_solver->decomp(Kuu + jitter * Eye);
        Kuu_inv_Kux = _u_solver->solve(Kux);
        Kxu_Kuu_inv = Kuu_inv_Kux.transpose();
        Gamma       = ((sf2+sn2) - (Kxu * Kuu_inv_Kux).diagonal().array()) / sn2; // Eigen seems to have already optimized (A * B).diagonal()
        inv_Gamma   = Gamma.cwiseInverse();
        A           = sn2 * Kuu + Kux * inv_Gamma.asDiagonal() * Kxu;
        _A_solver->decomp(A);
    }
    _alpha = _A_solver->solve(Kux * inv_Gamma.cwiseProduct(r));
}
double FITC::_calcNegLogProb(const VectorXd& hyp, VectorXd& g, bool calc_grad) const
{
    const double sn2       = _hyp_sn2(hyp);
    const double sf2       = _cov->sf2(hyp);
    const MatrixXd Kuu     = _cov->k(hyp, _inducing, _inducing);
    const MatrixXd Kxu     = _cov->k(hyp, _train_in, _inducing);
    const MatrixXd Kux     = Kxu.transpose();
    const VectorXd train_y = _train_out.array() - _hyp_mean(hyp);

    MatrixSolver* u_solver = _specify_matrix_solver(_md);
    MatrixSolver* A_solver = _specify_matrix_solver(_md);
    u_solver->decomp(Kuu);
    const MatrixXd Kuu_inv_Kux = u_solver->solve(Kux);
    const MatrixXd Kxu_Kuu_inv = Kuu_inv_Kux.transpose();
    const VectorXd Gamma       = ((sf2+sn2) - (Kxu * Kuu_inv_Kux).diagonal().array()) / sn2; // Eigen seems to have already optimized (A * B).diagonal()
    const VectorXd inv_Gamma   = Gamma.cwiseInverse();
    const MatrixXd A           = sn2 * Kuu + Kux * inv_Gamma.asDiagonal() * Kxu;
    A_solver->decomp(A);

    // Calculate L2 = y' * (Qn + sn2 * Gamma).inv() * y
    const MatrixXd l2_tmp1 = Kux * inv_Gamma.cwiseProduct(train_y);
    const VectorXd l2_tmp4 = inv_Gamma.cwiseProduct(train_y - Kxu * A_solver->solve(l2_tmp1));
    const double data_fit  = train_y.dot(l2_tmp4) / sn2;

    // calculate L1 = log | Q + sn2 * Gamma | 
    const double model_complexity  = A_solver->log_det() - u_solver->log_det() + Gamma.array().log().sum() + (_num_train - _num_inducing) * log(sn2);
    double nlz = 0.5 * (data_fit + model_complexity + _num_train * log(2 * M_PI));
    if(not (u_solver->check_SPD() and A_solver->check_SPD() and isfinite(data_fit) and isfinite(model_complexity)))
    {
        nlz = INF;
    }
#ifdef MYDEBUG
    cout << "nlz: " << nlz << ", data_fit: " << data_fit << ", model_complexity: " << model_complexity << ", calc_grad: " << calc_grad << endl;
#endif
    if(calc_grad)
    {
        MatrixXd Kuu_inv = u_solver->inverse();
        MatrixXd A_inv   = A_solver->inverse();
        g = VectorXd::Zero(_num_hyp, 1);

        MatrixXd dKuu              = MatrixXd::Zero(_num_inducing, _num_inducing);
        MatrixXd dKux              = MatrixXd::Zero(_num_inducing, _num_train);
        MatrixXd dKxu              = MatrixXd::Zero(_num_train, _num_inducing);
        VectorXd dKn_diag          = VectorXd::Zero(_num_train, 1);
        VectorXd dGamma            = VectorXd::Zero(_num_train, 1);
        MatrixXd dA                = MatrixXd::Zero(_num_inducing, _num_inducing);
        VectorXd inv_Gamma_y       = VectorXd::Zero(_num_train, 1);
        VectorXd gamma_inv_Gamma_y = VectorXd::Zero(_num_train, 1);
        VectorXd F                 = VectorXd::Zero(_num_inducing, 1);
        VectorXd dF                = VectorXd::Zero(_num_inducing, 1);

        // const MatrixXd Kt             = Kuu_inv_Kux.transpose();
        const MatrixXd Kux_invGamma   = Kux * inv_Gamma.asDiagonal();
        const MatrixXd inv_Gamma_Kxu  = Kux_invGamma.transpose();

        // length scales and variance
        for(size_t i = 0; i < _cov->num_hyp(); ++i)
        {
            dKuu     = _cov->dk_dhyp(hyp, i, _inducing, _inducing, Kuu);
            dKxu     = _cov->dk_dhyp(hyp, i, _train_in, _inducing, Kxu);
            dKux     = dKxu.transpose();
            dKn_diag = VectorXd::Constant(_num_train, 1, (i == _cov->num_hyp() - 1) ? 2 * _cov->sf2(hyp) : 0);

            // XXX: bottleneck, O(Nm^2) complexity
            dGamma = (dKn_diag  - Kxu_Kuu_inv.cwiseProduct(2 * dKxu - Kxu_Kuu_inv * dKuu).rowwise().sum()) / sn2;
            dA     = sn2 * dKuu + 2 * _sym(Kux_invGamma * dKxu) - Kux_invGamma  * dGamma.asDiagonal() * inv_Gamma_Kxu;

            // O(2m^2+m) = O(m^2)
            double dl1 = (A_inv.cwiseProduct(dA).sum() - Kuu_inv.cwiseProduct(dKuu).sum() + inv_Gamma.cwiseProduct(dGamma).sum());

            inv_Gamma_y       = inv_Gamma.cwiseProduct(train_y);
            gamma_inv_Gamma_y = inv_Gamma.cwiseProduct(dGamma.cwiseProduct(inv_Gamma_y));
            F                 = Kux  * inv_Gamma_y;
            dF                = dKux * inv_Gamma_y - Kux * gamma_inv_Gamma_y;

            double dl2_1 = -1  * train_y.dot(gamma_inv_Gamma_y);
            double dl2_2 = -1  * F.dot(A_inv * (dA * (A_inv * F))) + 2 * F.dot(A_inv * dF);
            g(i)         = 0.5 * (dl1 + (dl2_1 - dl2_2) / sn2);
        }

        // noise and mean
        double dl_noise_1 = 0.5  * (inv_Gamma.sum() - (inv_Gamma_Kxu * A_inv * Kux_invGamma).trace()) / sn2;
        double dl_noise_2 = -0.5 * l2_tmp4.squaredNorm() / pow(sn2, 2);
        g(_num_hyp-2)     = (dl_noise_1 + dl_noise_2) * (2 * sn2);
        g(_num_hyp-1)     = -1 * (l2_tmp4 / sn2).sum();
    }
    delete u_solver;
    delete A_solver;
    return nlz;
}
MatrixXd FITC::_sym(const MatrixXd& m) const
{
    return 0.5 * (m + m.transpose());
}
void FITC::test_obj(const VectorXd& hyp)
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
