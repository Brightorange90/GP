#pragma once
#include <Eigen/Dense>
class MatrixSolver
{
public:
    MatrixSolver(){};
    virtual ~MatrixSolver(){};
    virtual void decomp(const Eigen::MatrixXd&) = 0;
    virtual bool check_SPD() const = 0;
    virtual double log_det() const = 0;
    virtual Eigen::MatrixXd solve(const Eigen::MatrixXd& ys) const = 0;
    virtual Eigen::MatrixXd inverse() const = 0;
};
class MatrixSolverLLT : public MatrixSolver
{
    Eigen::LLT<Eigen::MatrixXd> _solver;
public:
    MatrixSolverLLT();
    void   decomp(const Eigen::MatrixXd&);
    bool   check_SPD() const;
    double log_det() const;
    Eigen::MatrixXd solve(const Eigen::MatrixXd& ys) const;
    Eigen::MatrixXd inverse() const;
};
class MatrixSolverQR : public MatrixSolver
{
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> _solver;
public:
    MatrixSolverQR();
    void   decomp(const Eigen::MatrixXd&);
    bool   check_SPD() const;
    double log_det() const;
    Eigen::MatrixXd solve(const Eigen::MatrixXd& ys) const;
    Eigen::MatrixXd inverse() const;
};
