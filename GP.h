#pragma once
#include "Eigen/Dense"
#include "CovSEard.h"
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

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> _K_solver;
    Eigen::VectorXd _invKys;
    bool _fixhyps    = false; // use fixed hyperparameters, do not train
    bool _noise_free = false;
    bool _trained;

    // Covariance functions
    // Eigen::MatrixXd _utilGradMatrix; // utility matrix to calculate gradient of likelihood with SEard kernel
    // void _calcUtilGradM();
    // Eigen::MatrixXd _covSEard(const Eigen::VectorXd& hyp) const noexcept;
    // Eigen::MatrixXd _covSEard(const Eigen::VectorXd& hyp, const Eigen::MatrixXd& x) const noexcept;  
    CovSEard _cov;

    // Initializing
    void _init();
    void _setK(); // precompute cholK, invKy after training

    double _calcNegLogProb(const std::vector<double>&, std::vector<double>& g) const;
    double _calcNegLogProb(const Eigen::VectorXd& hyp, Eigen::VectorXd& g, bool calc_grad) const;
    double _calcNegLogProb(const Eigen::VectorXd& hyp) const;
    double _calcNegLogProb(const Eigen::VectorXd& hyp, Eigen::VectorXd&) const;

    // predicting y has complexity of O(N) while predicting s2 needs O(N^2) complexity
    // somethimes we only need the y prediction, calling predict_y would be faster
    void _predict(const Eigen::MatrixXd& x,    bool need_g, Eigen::VectorXd& y, Eigen::VectorXd& s2, Eigen::MatrixXd& gy, Eigen::MatrixXd& gs2) const noexcept;
    void _predict_y(const Eigen::MatrixXd& x,  bool need_g, Eigen::VectorXd& y,  Eigen::MatrixXd& gy)  const noexcept;
    void _predict_s2(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& s2, Eigen::MatrixXd& gs2) const noexcept;

    void _set_hyp_range();
    void _check_hyp_range(const Eigen::VectorXd&) const noexcept;
    bool _check_SPD(const Eigen::MatrixXd& m) const;
    double _likelihood_gradient_checking(const Eigen::VectorXd& hyp, const Eigen::VectorXd& grad);

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
    const Eigen::VectorXd& train_out() const noexcept;
    bool trained() const noexcept;
    bool noise_free() const { return _noise_free; }

    void set_noise_lower_bound(double) noexcept;
    void set_fixed(bool);
    void set_noise_free(bool flag);

    // training, return negative log likelihood
    double train(const Eigen::VectorXd& init_hyps);
    double train(); // use default_hyp as init_hyps

    const Eigen::VectorXd& get_hyp() const noexcept;

    // prediction
    double predict_y(const Eigen::VectorXd& xs) const;
    double predict_s2(const Eigen::VectorXd& xs) const;
    double predict_y_with_grad(const Eigen::VectorXd& xs, Eigen::VectorXd& g) const;
    double predict_s2_with_grad(const Eigen::VectorXd& xs, Eigen::VectorXd& g) const;
    std::pair<double, Eigen::VectorXd> predict_y_with_grad(const Eigen::VectorXd& xs) const;
    std::pair<double, Eigen::VectorXd> predict_s2_with_grad(const Eigen::VectorXd& xs) const;

    void predict(const Eigen::VectorXd& xs, double& y, double& s2) const;
    void predict_with_grad(const Eigen::VectorXd& xs, double& y, double& s2, Eigen::VectorXd& gy, Eigen::VectorXd& gs2) const;
    std::tuple<double, double> predict(const Eigen::VectorXd& xs) const;
    std::tuple<double, double, Eigen::VectorXd, Eigen::VectorXd> predict_with_grad(const Eigen::VectorXd& xs) const;

    Eigen::VectorXd batch_predict_y(const  Eigen::MatrixXd& xs)  const;
    Eigen::VectorXd batch_predict_s2(const Eigen::MatrixXd& xs) const;
    void batch_predict(const Eigen::MatrixXd& xs, Eigen::VectorXd& y, Eigen::VectorXd& s2)    const;

    Eigen::VectorXd select_init_hyp(size_t max_eval, const Eigen::VectorXd& def_hyp);
};
