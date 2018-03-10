#pragma once
#include "Eigen/Dense"
#include <string>
#include <map>
#include <vector>
class GP
{
    Eigen::MatrixXd _train_in;
    Eigen::MatrixXd _train_out;
    size_t _num_train;
    double _noise_lb; // lower bound of the signal noise
    const size_t _dim;
    const size_t _num_spec;
    const size_t _num_hyp;

    // hyp: lscale, sf, sn, mean
    // lscale, sf, sn: log transformation
    // mean: no log transformation
    Eigen::MatrixXd _hyps;           // (dim + 3) * _num_spec
    Eigen::MatrixXd _hyps_lb;
    Eigen::MatrixXd _hyps_ub;

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
    double _calcNegLogProb(const std::vector<double>&, std::vector<double>& g, size_t idx) const;
    double _calcNegLogProb(const Eigen::VectorXd& hyp, Eigen::VectorXd& g, bool calc_grad, size_t idx) const;
    void _predict(size_t spec_id, const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& y, Eigen::VectorXd& s2, Eigen::MatrixXd& gy, Eigen::MatrixXd& gs2) const noexcept;

    Eigen::VectorXd select_init_hyp(size_t num_lscale, size_t spec_idx, const Eigen::VectorXd& def_hyp);

    void _set_hyp_range();
    void _check_hyp_range(size_t, const Eigen::VectorXd&) const noexcept;
    double _likelihood_gradient_checking(size_t, const Eigen::VectorXd& hyp, const Eigen::VectorXd& grad);
    bool _check_SPD(const Eigen::MatrixXd& m) const;
    Eigen::VectorXd vec2hyp(const std::vector<double>& vx) const;
    std::vector<double> hyp2vec(const Eigen::VectorXd& vx) const;
public:
    friend void testgp();
    GP(const Eigen::MatrixXd& train_in, const Eigen::MatrixXd& train_out);
    void add_data(const Eigen::MatrixXd& train_in, const Eigen::MatrixXd& train_out);

    size_t dim()       const noexcept;
    size_t num_spec()  const noexcept;
    size_t num_hyp()   const noexcept;
    size_t num_train() const noexcept;
    Eigen::MatrixXd get_default_hyps() const noexcept;
    const Eigen::MatrixXd& train_in()  const noexcept;
    const Eigen::MatrixXd& train_out() const noexcept;
    bool trained() const noexcept;
    bool noise_free() const { return _noise_free; }

    void set_noise_lower_bound(double) noexcept;
    void set_fixed(bool);
    void set_noise_free(bool flag);



    // training, return vector of negative log likelihood
    Eigen::VectorXd train(const Eigen::MatrixXd& init_hyps);
    Eigen::VectorXd train(); // use default_hyp as init_hyps

    const Eigen::MatrixXd& get_hyp() const noexcept;

    // prediction
    void predict(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, Eigen::MatrixXd& s2) const noexcept;
    void predict(size_t, const Eigen::VectorXd&x, double& y, double& s2) const noexcept;
    void predict(size_t, const Eigen::MatrixXd&x, Eigen::VectorXd& y, Eigen::VectorXd& s2) const noexcept;
    void predict_with_grad(size_t, const Eigen::VectorXd& x, double& y, double& s2, Eigen::VectorXd& grad_y, Eigen::VectorXd&  grad_s2) const noexcept;
    void predict_with_grad(size_t, const Eigen::MatrixXd& x, Eigen::VectorXd& y, Eigen::VectorXd& s2, Eigen::MatrixXd& grad_y, Eigen::MatrixXd&  grad_s2) const noexcept;
    

    Eigen::MatrixXd select_init_hyp(size_t, const Eigen::MatrixXd&);

};
