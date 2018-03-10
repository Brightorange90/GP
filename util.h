#pragma once
#include "def.h"
#include "Eigen/Dense"
#include <string>
#include <vector>
#include <nlopt.hpp>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
template <typename NumType>
NumType with_default(const std::map<std::string, double>& mp, std::string key, NumType default_v)
{
    if(mp.find(key) != mp.end())
        return static_cast<NumType>(mp.find(key)->second);
    else
        return default_v;
}

template <typename NumType>
NumType get_required(const std::map<std::string, double>& mp, std::string key)
{
    if(mp.find(key) != mp.end())
        return static_cast<NumType>(mp.find(key)->second);
    else
    {
        std::cerr << "Option " << key << " is required" << std::endl;
        exit(EXIT_FAILURE);
    }
}

template<typename VecType> double stdvar(const VecType& v)
{
    // v.sum() function gives different result for same vector in two program
    // the error is about less than 1e-10 and doesn't affect the optimization result much
    // however, that laeds to a failure of regression test, 
    // so I use manually calculated mean
    double vsum = 0;
    for(int i = 0; i < v.size(); ++i)
        vsum += v(i);
    double mean = vsum / (v.size());
    return (v - mean * VecType::Ones(v.size())).squaredNorm() / (v.size()-1);
}
template<typename VecType> double stddev(const VecType& v)
{
    return sqrt(stdvar<VecType>(v));
}
template<typename T>
T fromMaybe(std::map<std::string, double> option, std::string name, T defV)
{
    static_assert(std::is_arithmetic<T>::value, "Number required in fromMaybe");
    auto p = option.find(name);
    return p == option.end() ? defV : static_cast<T>(p->second);
}
// VecType should works for: std::vecotr, Eigen::VectorXd, Eigen::RowVectorXd
template <typename VecType>
std::vector<size_t> top_largest(const VecType& v, size_t n)
{
    if(n > v.size())
    {
        std::cerr << "n > v.size() in nth_index" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<size_t> indexs(v.size());
    for (size_t i = 0; i < indexs.size(); ++i) indexs[i] = i;
    std::nth_element(indexs.begin(), indexs.begin() + n - 1, indexs.end(),
                     [&](size_t i1, size_t i2) -> bool { return v[i1] > v[i2]; });
    return indexs;
}
Eigen::MatrixXd read_matrix(std::string path);
Eigen::VectorXd     std2eig(const std::vector<double>&);
std::vector<double> eig2std(const Eigen::VectorXd&);
Eigen::VectorXd     convert(const std::vector<double>&);
std::vector<double> convert(const Eigen::VectorXd&);
double sdist_vv(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2);
Eigen::VectorXd sdist_vm(const Eigen::VectorXd& v, const Eigen::MatrixXd& m);
Eigen::MatrixXd sdist_mm(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2);
std::string explain_nlopt(nlopt::result);

double normpdf(double x);
Eigen::MatrixXd normpdf(const Eigen::MatrixXd& x);
double normcdf(double x);
Eigen::MatrixXd normcdf(const Eigen::MatrixXd& x);

double logphi(double x);
void   logphi(double x, double& lp, double& dlp);
void   logphi(const Eigen::VectorXd& x, Eigen::VectorXd& lp, Eigen::VectorXd& dlp);
Eigen::VectorXd logphi(const Eigen::VectorXd& xs);

// compare different result
double violation(const Eigen::RowVectorXd& rec);
bool   is_feas(const Eigen::RowVectorXd& rec);
bool   better(const Eigen::RowVectorXd& r1, const Eigen::RowVectorXd& r2);
Eigen::RowVectorXd find_best(const Eigen::MatrixXd& ys);
void find_best(const Eigen::MatrixXd& xs, const Eigen::MatrixXd& ys, Eigen::VectorXd& best_x,
                             Eigen::RowVectorXd& best_y);

Eigen::MatrixXd rand_matrix(size_t num_col, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                            std::mt19937_64& eig = engine);

double cond(const Eigen::MatrixXd& m);
