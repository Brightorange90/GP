#include "MatrixSolver.h"
#include <iostream>
using namespace std;
using namespace Eigen;

MatrixSolverLLT::MatrixSolverLLT()
    : MatrixSolver()
{}
void MatrixSolverLLT::decomp(const MatrixXd& x)
{
    _solver.compute(x);
}
MatrixXd MatrixSolverLLT::solve(const MatrixXd& ys) const
{
    return _solver.solve(ys);
}
bool MatrixSolverLLT::check_SPD() const
{
    return _solver.info() == Eigen::Success;
}
double MatrixSolverLLT::log_det() const
{
    return 2*_solver.matrixL().toDenseMatrix().diagonal().array().log().sum();
}
MatrixXd MatrixSolverLLT::inverse() const
{
    return _solver.solve(MatrixXd::Identity(_solver.rows(), _solver.cols()));
}


MatrixSolverQR::MatrixSolverQR()
    : MatrixSolver(), _solver()
{}
void MatrixSolverQR::decomp(const MatrixXd& x)
{
    _solver.compute(x);
}
MatrixXd MatrixSolverQR::solve(const MatrixXd& ys) const
{
    return _solver.solve(ys);
}
bool MatrixSolverQR::check_SPD() const
{
    return (_solver.info() == Eigen::Success) and (_solver.isInvertible());
}
double MatrixSolverQR::log_det() const
{
    return _solver.logAbsDeterminant();
}
MatrixXd MatrixSolverQR::inverse() const
{
    return _solver.inverse();
}
