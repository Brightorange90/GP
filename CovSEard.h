#pragma once
#include "Cov.h"
class CovSEard : public Cov
{
public:
    CovSEard(size_t d);
    size_t num_hyp() const;
    Eigen::MatrixXd k(const Eigen::VectorXd& hyp, const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) const;
    Eigen::MatrixXd dk_dhyp(const Eigen::VectorXd& hyp, size_t idx, const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) const;
    Eigen::MatrixXd dk_dhyp(const Eigen::VectorXd& hyp, size_t idx, const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2, const Eigen::MatrixXd& K) const;
    Eigen::MatrixXd dk_dx1(const Eigen::VectorXd&  hyp, const Eigen::VectorXd& x1, const Eigen::MatrixXd& x2) const;
    Eigen::MatrixXd dk_dx1(const Eigen::VectorXd& hyp, const Eigen::VectorXd& x1, const Eigen::MatrixXd& x2, const Eigen::RowVectorXd& K) const;
};
