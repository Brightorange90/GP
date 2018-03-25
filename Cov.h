#pragma once
#include <Eigen/Dense>
#include "def.h"
#include "util.h"
class Cov
{
protected:
    size_t _dim;

public:
    Cov(size_t d) : _dim(d){}
    virtual size_t num_hyp() const = 0;
    virtual Eigen::MatrixXd k(const Eigen::VectorXd& hyp, const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) const = 0;
    virtual Eigen::MatrixXd dk_dhyp(const Eigen::VectorXd& hyp, size_t idx, const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) const = 0;
    virtual Eigen::MatrixXd dk_dhyp(const Eigen::VectorXd& hyp, size_t idx, const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2, const Eigen::MatrixXd& K) const = 0;
    virtual Eigen::MatrixXd dk_dx1(const Eigen::VectorXd& hyp, const Eigen::VectorXd& x1, const Eigen::MatrixXd& x2) const = 0;
    virtual Eigen::MatrixXd dk_dx1(const Eigen::VectorXd& hyp, const Eigen::VectorXd& x1, const Eigen::MatrixXd& x2, const Eigen::RowVectorXd& K) const = 0;
};
