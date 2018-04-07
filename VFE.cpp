#include "VFE.h"
#include "MatrixSolver.h"
#include <fstream>
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
}
VFE::~VFE()
{
}
void VFE::set_inducing(const MatrixXd& u)
{
    _inducing     = u;
    _num_inducing = u.cols();
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
void VFE::_predict(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& y, Eigen::VectorXd& s2, Eigen::MatrixXd& gy, Eigen::MatrixXd& gs2) const noexcept
{
}
void VFE::_predict_y(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& y, Eigen::MatrixXd& gy) const noexcept
{
}
void VFE::_predict_s2(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& s2, Eigen::MatrixXd& gs2) const noexcept
{
}
void VFE::_setK()
{
}
double VFE::_calcNegLogProb(const VectorXd& hyp, VectorXd& g, bool calc_grad) const
{
}
MatrixXd VFE::_sym(const MatrixXd& m) const
{
    return 0.5 * (m + m.transpose());
}
void VFE::test_obj(const VectorXd& hyp)
{
}
