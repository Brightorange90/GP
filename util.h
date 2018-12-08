/**
 * @file util.h
 * @author Wenlong Lyu
 */
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

/** A look-up table with default value */
template <typename NumType>
NumType with_default(const std::map<std::string, double>& mp, std::string key, NumType default_v)
{
    if(mp.find(key) != mp.end())
        return static_cast<NumType>(mp.find(key)->second);
    else
        return default_v;
}

/** 
 * Given a look-up table `mp` and a string `key`, lookup `key` in the map, if
 * `key` is found, cast the value to `NumType` and return, otherwise, report
 * error.
 */
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

/** Calculate variation of a vector */
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

/** Calculate standard deviation of a vector */
template<typename VecType> double stddev(const VecType& v)
{
    return sqrt(stdvar<VecType>(v));
}

/** Loop-up `name` in map `option`, if the key is not found, return `defV` */
template<typename T>
T fromMaybe(std::map<std::string, double> option, std::string name, T defV)
{
    static_assert(std::is_arithmetic<T>::value, "Number required in fromMaybe");
    auto p = option.find(name);
    return p == option.end() ? defV : static_cast<T>(p->second);
}

/** 
 * Return top-`n` largest values of vector `v`
 * VecType should works for: std::vecotr, Eigen::VectorXd, Eigen::RowVectorXd 
 *
 * \warning `n` should be less than the length of `v`
 * */
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
Eigen::MatrixXd read_matrix(std::string path); /**< Read a matrix from a file */
Eigen::VectorXd     std2eig(const std::vector<double>&); /**< Convert `std::vector<double>` to `Eigen::VectorXd` */
std::vector<double> eig2std(const Eigen::VectorXd&);     /**< Convert `Eigen::VectorXd` to `std::vector<double>` */
Eigen::VectorXd     convert(const std::vector<double>&); /**< @see Eigen::VectorXd     std2eig(const std::vector<double>&) */
std::vector<double> convert(const Eigen::VectorXd&);     /**< @see std::vector<double> eig2std(const Eigen::VectorXd&) */

/** Return the distance between two vectors */
double sdist_vv(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2);

/** 
 * Return a distance vector between vector `v` and matrix `m`. Each column of
 * `m` represents a vector
 * */
Eigen::VectorXd sdist_vm(const Eigen::VectorXd& v, const Eigen::MatrixXd& m);

/** 
 * Return a distance matrix between matrix `m1` and matrix `m2. Each column of
 * the matrices represents a vector`. 
 * */
Eigen::MatrixXd sdist_mm(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2);

/** Return a string that explains a nlopt::result value */
std::string explain_nlopt(nlopt::result);

/** Return the value of the PDF function of the standard normal distribution */
double normpdf(double x);

/** Return the value of the PDF function of the standard normal distribution */
Eigen::MatrixXd normpdf(const Eigen::MatrixXd& x);

/** Return the value of the CDF function of the standard normal distribution */
double normcdf(double x);

/** Return the value of the CDF function of the standard normal distribution */
Eigen::MatrixXd normcdf(const Eigen::MatrixXd& x);

/** Calculate logarithm of the CDF function of the standard normal distribution */
double logphi(double x);

/** Calculate logarithm of the CDF function of the standard normal distribution */
void   logphi(double x, double& lp, double& dlp);

/** Calculate logarithm of the CDF function of the standard normal distribution */
void   logphi(const Eigen::VectorXd& x, Eigen::VectorXd& lp, Eigen::VectorXd& dlp);

/** Calculate logarithm of the CDF function of the standard normal distribution */
Eigen::VectorXd logphi(const Eigen::VectorXd& xs);

/** Return the constraint violation of `rec` */
double violation(const Eigen::RowVectorXd& rec);

/**
 * Return true if `rec` is a feasible solution
 */
bool   is_feas(const Eigen::RowVectorXd& rec);

/**
 * Compare two solutions, return true if `r1` is better than `r2`
 */
bool   better(const Eigen::RowVectorXd& r1, const Eigen::RowVectorXd& r2);

/**
 * Given some function outputs, return the best output vector
 */
Eigen::RowVectorXd find_best(const Eigen::MatrixXd& ys);

/**
 * Given some inputs and outputs, find the best output and the corresponding best input, store them in `best_x` and `best_y`
 */
void find_best(const Eigen::MatrixXd& xs, const Eigen::MatrixXd& ys, Eigen::VectorXd& best_x,
                             Eigen::RowVectorXd& best_y);

/**
 * Generate a random matrix, each column is within [lb, ub]
 */
Eigen::MatrixXd rand_matrix(size_t num_col, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                            std::mt19937_64& eig = engine);

double cond(const Eigen::MatrixXd& m); /**< Return condition number of a matrix */
