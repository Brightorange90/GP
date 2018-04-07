#pragma once
#include "GP.h"
class FITC : public GP
{
protected:
    Eigen::MatrixXd _sym(const Eigen::MatrixXd&) const;
    Eigen::MatrixXd _inducing;
    size_t _num_inducing;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> _inducing_solver;
    
    // inherited from GP
    double _calcNegLogProb(const Eigen::VectorXd& hyp, Eigen::VectorXd& g, bool calc_grad) const;
    void   _predict(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& y, Eigen::VectorXd& s2, Eigen::MatrixXd& gy, Eigen::MatrixXd& gs2) const noexcept;
    void   _predict_y(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& y, Eigen::MatrixXd& gy) const noexcept;
    void   _predict_s2(const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& s2, Eigen::MatrixXd& gs2) const noexcept;
    void   _setK();  // precompute cholK, invKy after training

    MatrixSolver*   _u_solver;
    MatrixSolver*   _A_solver;
    Eigen::VectorXd _alpha;

public:
    FITC(const Eigen::MatrixXd& train_in, const Eigen::MatrixXd& train_out, CovFunc cf = COV_SE_ARD, GP::MatrixDecomp md = QR);
    ~FITC();
    double train(const Eigen::VectorXd& init_hyps);
    void set_inducing(const Eigen::MatrixXd& u);
    void test_obj(const Eigen::VectorXd& hyp);
};
