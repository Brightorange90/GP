#include "util.h"
#include "def.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unsupported/Eigen/SpecialFunctions>

using namespace Eigen;
using namespace std;

Eigen::RowVectorXd split_line(const string& line)
{
    // split one line by ' '
    RowVectorXd v;
    vector<double> vec;

    stringstream s(line);
    string tok;
    while (s >> tok)
    {
        vec.push_back(stod(tok));
    }
    v = Map<VectorXd>(vec.data(), vec.size());
    return v;
}
MatrixXd read_matrix(string path)
{
    ifstream mfile(path);
    MYASSERT(mfile.is_open() && !mfile.fail());

    string line;
    vector<RowVectorXd> tmp_container;
    while (!mfile.eof() && getline(mfile, line))
    {
        VectorXd v = split_line(line);
        tmp_container.push_back(v);
    }
    mfile.close();
    if (!tmp_container.empty())
    {
        const size_t dim = tmp_container[0].size();
        MatrixXd m(tmp_container.size(), dim);
        for (size_t i = 0; i < tmp_container.size(); ++i)
        {
            if ((size_t)tmp_container[i].size() != dim)
            {
                cerr << "Invalid vector read from file, dim not match" << endl;
                exit(EXIT_FAILURE);
            }
            m.row(i) = tmp_container[i];
        }
        return m;
    }
    else
    {
        cerr << "Empty matrix in file " << path << endl;
        exit(EXIT_FAILURE);
    }
}
double sdist_vv(const VectorXd& v1, const VectorXd& v2) { return (v1 - v2).squaredNorm(); }
VectorXd sdist_vm(const VectorXd& v, const MatrixXd& m) { return (m.colwise() - v).colwise().squaredNorm(); }
MatrixXd sdist_mm(const MatrixXd& X, const MatrixXd& Y)
{
    // const VectorXd XX = X.colwise().squaredNorm().transpose();
    // const RowVectorXd YY = Y.colwise().squaredNorm();
    // return (-1 * (((2 * X.transpose() * Y).colwise() - XX).rowwise() - YY)).cwiseMax(0);

    MatrixXd dists(X.cols(), Y.cols());
    for(size_t i = 0; i < (size_t)Y.cols(); ++i)
        dists.col(i) = sdist_vm(Y.col(i), X);
    return dists;
}

VectorXd std2eig(const vector<double>& vec) { return Eigen::Map<const VectorXd>(&vec[0], vec.size()); }
vector<double> eig2std(const VectorXd& vec) { return vector<double>(vec.data(), vec.data() + vec.size()); }
Eigen::VectorXd     convert(const std::vector<double>& x) { return std2eig(x); }
std::vector<double> convert(const Eigen::VectorXd& x) { return eig2std(x); }

string explain_nlopt(nlopt::result r)
{
    switch (r)
    {
#define C(A)               \
    case nlopt::result::A: \
        return #A
        C(FAILURE);
        C(INVALID_ARGS);
        C(OUT_OF_MEMORY);
        C(ROUNDOFF_LIMITED);
        C(FORCED_STOP);
        C(SUCCESS);
        C(STOPVAL_REACHED);
        C(FTOL_REACHED);
        C(XTOL_REACHED);
        C(MAXEVAL_REACHED);
        C(MAXTIME_REACHED);
#undef C
        default:
            return "Unknown result";
    }
}
double normpdf(double x) { return exp(-0.5 * pow(x, 2)) / sqrt(2 * M_PIl); }
double normcdf(double x) { return 0.5 * erfc(-x / M_SQRT2l); }
MatrixXd normpdf(const MatrixXd& xs)
{
    return (-0.5 * xs.array().square()).exp() / sqrt(2 * M_PI);
}
MatrixXd normcdf(const MatrixXd& xs)
{
    return 0.5 * (-1 * xs / M_SQRT2).array().erfc();
}
// Approximation of log(normcdf(x)) and its gradient
VectorXd logphi(const VectorXd& xs)
{
    VectorXd lp;
    VectorXd dlp;
    logphi(xs, lp, dlp);
    return lp;
}
void logphi(const VectorXd& x, VectorXd& lp, VectorXd& dlp)
{
    lp  = VectorXd::Zero(x.size());
    dlp = VectorXd::Zero(x.size());
    for(long i = 0; i < x.size(); ++i)
        logphi(x(i), lp(i), dlp(i));
}
double logphi(double x)
{
    double  lp;
    double  dlp;
    logphi(x, lp, dlp);
    return lp;
}
void logphi(double x, double& lp, double& dlp)
{
    // Translated from logphi.m of gpml matlab toolbox
    if (pow(x, 2) < 0.0492)
    {
        const double lp0 = -1 * x / sqrt(2 * M_PI);
        Matrix<double, 14, 1> c;
        c << 0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032, -0.0045563339802, 0.00556964649138,
            0.00125993961762116, -0.01621575378835404, 0.02629651521057465, -0.001829764677455021, 2 * (1 - M_PI / 3),
            (4 - M_PI) / 3, 1, 1;
        double f = 0;
        for (long i = 0; i < c.size(); ++i)
            f = lp0 * (c(i) + f);
        lp  = -2 * f - M_LN2;
        dlp = exp(-1 * pow(x, 2) / 2 - lp) / sqrt(2 * M_PI);
    }
    else if (x < -11.3137)
    {
        Matrix<double, 5, 1> r;
        Matrix<double, 6, 1> q;
        r << 1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441, 7.409740605964741794425,
            2.9788656263939928886;
        double num = 0.5641895835477550741;
        for (long i = 0; i < r.size(); ++i)
            num = -1 * x * num / M_SQRT2 + r(i);
        q << 2.260528520767326969592, 9.3960340162350541504, 12.048951927855129036034, 17.081440747466004316,
            9.608965327192787870698, 3.3690752069827527677;
        double den = 1.0;
        for (long i = 0; i < q.size(); ++i)
            den = -1 * x * den / M_SQRT2 + q(i);
        lp  = log(0.5 * num / den) - 0.5 * pow(x, 2);
        dlp = fabs(den / num) * sqrt(M_2_PI);
    }
    else
    {
        lp  = log(0.5 * erfc(-1 * x / M_SQRT2));
        dlp = exp(-1 * pow(x, 2) / 2 - lp) / sqrt(2 * M_PI);
    }
}
double violation(const RowVectorXd& xs) 
{
    return xs.size() == 1 ? 0 : xs.tail(xs.size() - 1).cwiseMax(0).sum();
}
bool is_feas(const RowVectorXd& xs)
{
    return violation(xs) <= 0;
}
bool better(const RowVectorXd& rec1, const RowVectorXd& rec2)
{
    // Apply feasibility rule
    const double fom1 = rec1(0);
    const double fom2 = rec2(0);
    const double cv1  = violation(rec1);
    const double cv2  = violation(rec2);
    if (cv1 > 0 && cv2 > 0)
        return cv1 < cv2;
    else if (cv1 > 0 && cv2 <= 0)
        return false;
    else if (cv1 <= 0 && cv2 > 0)
        return true;
    else
        return fom1 < fom2;
}
RowVectorXd find_best(const MatrixXd& ys)
{
    vector<RowVectorXd> vec_ys;
    vec_ys.reserve(ys.rows());
    for(long i = 0; i < ys.rows(); ++i)
        vec_ys.push_back(ys.row(i));
    auto min_it = std::min_element(vec_ys.begin(), vec_ys.end(), better);
    return *min_it;
}
void find_best(const MatrixXd& xs, const MatrixXd& ys, VectorXd& best_x, RowVectorXd& best_y)
{
    MYASSERT(xs.cols() == ys.rows());
    MYASSERT(xs.cols() >= 1);
    best_x = xs.col(0);
    best_y = ys.row(0);
    for(long i = 1; i < xs.cols(); ++i)
    {
        if(better(ys.row(i), best_y))
        {
            best_y = ys.row(i);
            best_x = xs.col(i);
        }
    }
}

Eigen::MatrixXd rand_matrix(size_t num_col, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                            std::mt19937_64& eig)
{
    const size_t dim = lb.size();
    MYASSERT((size_t)ub.size() == dim);
    MYASSERT((lb.array() <= ub.array()).all());
    MatrixXd m(dim, num_col);
    uniform_real_distribution<double> distr(-1, 1);
    const VectorXd _a = 0.5 * (ub - lb);
    const VectorXd _b = 0.5 * (ub + lb);
    for(size_t i = 0; i < num_col; ++i)
        for(size_t j = 0; j < dim; ++j)
            m(j, i) = distr(eig);
    m = _a.replicate(1, num_col).cwiseProduct(m).colwise() + _b;
    return m;
}
