#pragma once
#include "Eigen/Dense"
#include <string>
#include <map>
#include <vector>

// TODO: To test
// log likelihood within error tolerence
// gradient of log likelihood within error tolerence
// GP prediction
// gradient of GP prediction

class GP
{
    Eigen::MatrixXd _train_in;
    Eigen::VectorXd _train_out;
    size_t _num_train;
    double _noise_lb; // lower bound of the signal noise
    const size_t _dim;
    const size_t _num_hyp;

    // hyp: lscale, sf, sn, mean
    // lscale, sf, sn: log transformation
    // mean: no log transformation
    Eigen::VectorXd _hyps;           // (dim + 3) * _num_spec
    Eigen::VectorXd _hyps_lb;
    Eigen::VectorXd _hyps_ub;

    Eigen::MatrixXd _utilGradMatrix; // utility matrix to calculate gradient of likelihood with SEard kernel
    std::vector<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>> _K_solver;
    Eigen::MatrixXd _invKys;
    bool _fixhyps    = false; // use fixed hyperparameters, do not train
    bool _noise_free = false;
    bool _trained;

    // Initializing
    void _init();
    void _calcUtilGradM();
    void _setK(); // precompute cholK, invKy after training
    Eigen::MatrixXd _covSEard(const Eigen::VectorXd& hyp) const noexcept;
    Eigen::MatrixXd _covSEard(const Eigen::VectorXd& hyp, const Eigen::MatrixXd& x) const noexcept;  

    double _calcNegLogProb(const std::vector<double>&, std::vector<double>& g) const;
    double _calcNegLogProb(const Eigen::VectorXd& hyp, Eigen::VectorXd& g, bool calc_grad) const;

    void _predict(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& y, Eigen::VectorXd& s2, Eigen::MatrixXd& gy, Eigen::MatrixXd& gs2) const noexcept;

    Eigen::VectorXd select_init_hyp(size_t num_lscale, const Eigen::VectorXd& def_hyp);

    void _set_hyp_range();
    void _check_hyp_range(size_t, const Eigen::VectorXd&) const noexcept;
    double _likelihood_gradient_checking(size_t, const Eigen::VectorXd& hyp, const Eigen::VectorXd& grad);
    bool _check_SPD(const Eigen::MatrixXd& m) const;

    Eigen::VectorXd     vec2hyp(const std::vector<double>& vx) const;
    std::vector<double> hyp2vec(const Eigen::VectorXd& vx) const;
public:
    GP(const Eigen::MatrixXd& train_in, const Eigen::MatrixXd& train_out);
    void add_data(const Eigen::MatrixXd& train_in, const Eigen::MatrixXd& train_out);

    size_t dim()       const noexcept;
    size_t num_spec()  const noexcept;
    size_t num_hyp()   const noexcept;
    size_t num_train() const noexcept;
    Eigen::VectorXd get_default_hyps() const noexcept;
    const Eigen::MatrixXd& train_in()  const noexcept;
    const Eigen::MatrixXd& train_out() const noexcept;
    bool trained() const noexcept;
    bool noise_free() const { return _noise_free; }

    void set_noise_lower_bound(double) noexcept;
    void set_fixed(bool);
    void set_noise_free(bool flag);

    // training, return vector of negative log likelihood
    Eigen::VectorXd train(const Eigen::VectorXd& init_hyps);
    Eigen::VectorXd train(); // use default_hyp as init_hyps

    const Eigen::VectorXd& get_hyp() const noexcept;

    // prediction
    double predict_y(const Eigen::VectorXd& xs) const;
    double predict_s2(const Eigen::VectorXd& xs) const;
    double predict_y_with_grad(const Eigen::VectorXd& xs, Eigen::VectorXd& g) const;
    double predict_s2_with_grad(const Eigen::VectorXd& xs, Eigen::VectorXd& g) const;
    std::pair<double, Eigen::VectorXd> predict_y_with_grad(const Eigen::VectorXd& xs) const;
    std::pair<double, Eigen::VectorXd> predict_s2_with_grad(const Eigen::VectorXd& xs) const;

    void predict(const Eigen::VectorXd& xs, double& y, double& s2) const;
    void predict_with_grad(const Eigen::VectorXd& xs, double& y, double& s2, Eigen::VectorXd& gy, double& gs2);
    std::tuple<double, double> predict(const Eigen::VectorXd& xs) const;
    std::tuple<double, double, Eigen::VectorXd, Eigen::VectorXd> predict_with_grad(const Eigen::VectorXd& xs);

    // void predict(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, Eigen::MatrixXd& s2) const noexcept;
    // void predict(size_t, const Eigen::VectorXd&x, double& y, double& s2) const noexcept;
    // void predict(size_t, const Eigen::MatrixXd&x, Eigen::VectorXd& y, Eigen::VectorXd& s2) const noexcept;
    // void predict_with_grad(size_t, const Eigen::VectorXd& x, double& y, double& s2, Eigen::VectorXd& grad_y, Eigen::VectorXd&  grad_s2) const noexcept;
    // void predict_with_grad(size_t, const Eigen::MatrixXd& x, Eigen::VectorXd& y, Eigen::VectorXd& s2, Eigen::MatrixXd& grad_y, Eigen::MatrixXd&  grad_s2) const noexcept;
    
    Eigen::MatrixXd select_init_hyp(size_t, const Eigen::MatrixXd&);
    Eigen::VectorXd sample_GP_1D(double lb, double ub);
};
