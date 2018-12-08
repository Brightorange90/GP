/**
 * @file GP.h
 * @author Wenlong Lyu
 *
 * The Gaussian process class
 */
#pragma once
#include "Eigen/Dense"
#include <string>
#include <map>
#include <vector>

/** 
 * The class for Gaussian process regression 
 *
 * The GP class accepts two matrices as the training set, one matrix is the
 * training inputs, with dimension of `_dim * _num_train`; the other matrix is
 * the training outputs, with dimension of `_num_train * _num_spec`.
 * `_num_spec` is the number of targets, these targets share same training
 * inputs. However, this is **not** a multi-task GP class, each target would be
 * trained separately.
 **/
class GP
{
    Eigen::MatrixXd _train_in;  /**< The input of the training set,  dimension: dim * num_train*/

    /** 
     * The output of the training set, dimension: num_train * num_spec
     *
     * `num_spec` Gaussian process models would be trained, these GP models
     * have shared input features, but with different targets. For example, the
     * input features can be the design parameters of a circuit, while the
     * targets can be multiple circuit performances.
     * */
    Eigen::MatrixXd _train_out; 

    size_t _num_train;          /**< Size of the training set */
    double _noise_lb;           /**< Lower bound of the signal noise */
    const size_t _dim;          /**< Input dimension */
    const size_t _num_spec;     /**< Number of cared targets */
    const size_t _num_hyp;      /**< Number of hyper-parameters */

    // hyp: lscale, sf, sn, mean
    // lscale, sf, sn: log transformation
    // mean: no log transformation
    Eigen::MatrixXd _hyps;    /**< Currently optimized hyperparameters for all specs, dimension:  (dim + 3) * _num_spec */
    Eigen::MatrixXd _hyps_lb; /**< Lower bound of the hyperparameters, used for GP training */
    Eigen::MatrixXd _hyps_ub; /**< Upper bound of the hyperparameters, used for GP training */

    /** 
     * Utility matrix to calculate gradient of likelihood with SEard kernel
     *
     * This is a `_num_train * (_num_train * _dim)` matrix, it stores the
     * distance matrix between training inputs **in each dimension**
     * */
    Eigen::MatrixXd _utilGradMatrix; 

    /**
     * QR decomposed covariance matrix. After the GP training, QR decomposition
     * is performed to the covariance matrix, and the result is stored in
     * `_K_solver`
     * */
    std::vector<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>> _K_solver; 

    /**
     * Pre-calculated `inv(K) * train_out` to accelerate the inference
     *
     * For each target, `inv(K) * train_out` would be a `_num_train * 1` vector, so `_invKys` is a `_num_train * _num_spec` matrix
     */
    Eigen::MatrixXd _invKys;
    bool _fixhyps    = false; /**< use fixed hyperparameters, do not train */
    bool _noise_free = false; /**< If `_noise_free` is set to `true`, then noise woul not be considered */
    bool _trained;            /**< Set to `true` after GP model is trained */

    /**
     *Initialization, this function is called before training
     *
     * Calculate _utilGradMatrix, set ranges of hyperparameters, and set `_trained` to false.
     */
    void _init();
    void _calcUtilGradM(); /**< Calculate _utilGradMatrix */
    void _setK();          /**< precompute `_invKys`, `_K_solver` after training */
    Eigen::MatrixXd _covSEard(const Eigen::VectorXd& hyp) const noexcept; /**< Calculate Gaussian kernel */
    Eigen::MatrixXd _covSEard(const Eigen::VectorXd& hyp, const Eigen::MatrixXd& x) const noexcept;  /**< Calculate the Gaussian kernel */

    /**
     * Calculate the negative log likelihood one target , `idx` is used to
     * specify the which target. If vector `g` is not empty, then the gradient
     * is calculated and stored in `g`, otherwise gradient is ignored.
     *
     * This would be the loss function for GP training.
     *
     * @param [in] hyp The hyperparameter
     * @param [in, out] g The gradient of the netagive log likelihood, if an
     *                    empty vector is passed to the function, then gradient wouldn't be
     *                    calculated. 
     * @param [in] idx Specify the target, the negative log likelihood for that target is to be calculated.
     */
    double _calcNegLogProb(const std::vector<double>& hyp, std::vector<double>& g, size_t idx) const;    

    /**
     * Calculate the negative log likelihood one target , `idx` is used to
     * specify the which target. 
     *
     * If `calc_grad` is `true`, then gradient would be calculated and stored in `g`
     *
     * This would be the loss function for GP training.
     *
     * @see double _calcNegLogProb(const std::vector<double>& hyp, std::vector<double>& g, size_t idx) const
     *
     * @param [in] hyp The hyperparameter
     * @param [out] g If `calc_grad` is true, then gradient would be calculated and stored in `g`
     * @param [in] calc_grad Specify whether or not to calculate gradient
     * @param [in] idx Specify the target
     */
    double _calcNegLogProb(const Eigen::VectorXd& hyp, Eigen::VectorXd& g, bool calc_grad, size_t idx) const;

    /**
     * GP prediction for **one target**. Calculate the predictive mean and variance given some inputs
     *
     * @param [in] spec_id  Specify which target is to be predicted.
     * @param [in] x        The inputs to be predicted, each column of `x` is one input vector.
     * @param [in] need_g   Specify whether or not to calculate the gradients of predictions.
     * @param [out] y       The values of predictive mean, `y(i)` is the prediction for `x.col(i)`
     * @param [out] s2      The values of predictive variance, `s2(i)` is the prediction for `x.col(i)`
     * @param [out] gy      The gradients of the predictive mean, a `_dim * x.cols()` matrix. If `need_g` is false, then `gy` should be viewed as undefined
     * @param [out] gs2     The gradients of the predictive variance, a `_dim * x.cols()` matrix. If `need_g` is false, then `gs2` should be viewed as undefined
     */
    void _predict(size_t spec_id, const Eigen::MatrixXd& x, bool need_g, Eigen::VectorXd& y, Eigen::VectorXd& s2, Eigen::MatrixXd& gy, Eigen::MatrixXd& gs2) const noexcept;

    /** 
     * Perform global optimization to select hyperparameters for one target. The target is specified by `spec_idx`
     *
     * @param [in] max_eval Maximum loss function evaluations for the global optimization
     * @param [in] spec_idx Specify the target
     * @param [in] def_hyp An initial suggested vector of hyperparameters
     * */
    Eigen::VectorXd select_init_hyp(size_t max_eval, size_t spec_idx, const Eigen::VectorXd& def_hyp);

    void _set_hyp_range(); /**< Calculate the ranges of hyperparameters */
    void _check_hyp_range(size_t, const Eigen::VectorXd&) const noexcept; 
    double _likelihood_gradient_checking(size_t, const Eigen::VectorXd& hyp, const Eigen::VectorXd& grad); /**< Use finite difference method to check the analytical gradient of negative GP log likelihood, only used for debug */
    bool _check_SPD(const Eigen::MatrixXd& m) const; /**< Check whether or not a matrix is semi definite positive */

    /** 
     * Convert std::vector<double> to Eigen::VectorXd type hyperparameters 
     *
     * If `_noise_free` is true, then the length of `vx` should be `_num_hyp -
     * 1`, and a zero-noise value would be inserted to the converted
     * `Eigen::VectorXd` typed hyperparameters
     */
    Eigen::VectorXd vec2hyp(const std::vector<double>& vx) const; 

    /**
     * Reverse conversion of `vec2hyp`
     *
     * @see Eigen::VectorXd vec2hyp(const std::vector<double>& vx) const
     */
    std::vector<double> hyp2vec(const Eigen::VectorXd& vx) const; 
public:
    /** 
     * Constructor of the GP class 
     *
     * @param [in] train_in The inputs of the training set, a `_dim * _num_train` matrix
     * @param [in] train_out The outputs of the training set, a `_num_train * _num_spec` matrix
     * */
    GP(const Eigen::MatrixXd& train_in, const Eigen::MatrixXd& train_out);

    /** Append some new training data to the training set.  */
    void add_data(const Eigen::MatrixXd& train_in, const Eigen::MatrixXd& train_out);

    size_t dim()       const noexcept;  /**< \return Input dimension */
    size_t num_spec()  const noexcept;  /**< \return Number of targets */
    size_t num_hyp()   const noexcept;  /**< \return Number of hyperparameters **for each target** */
    size_t num_train() const noexcept;  /**< \return Number of training point */
    Eigen::MatrixXd get_default_hyps() const noexcept; /**< \return The default hyper-parameters */
    const Eigen::MatrixXd& train_in()  const noexcept; /**< \return `_train_in` */
    const Eigen::MatrixXd& train_out() const noexcept; /**< \return `_train_out` */
    bool trained() const noexcept;                  /**< \return `_trained` */
    bool noise_free() const { return _noise_free; } /**< \return `_noise_free` */

    void set_noise_lower_bound(double) noexcept;/**< Set `_noise_lb` */
    void set_fixed(bool);                       /**< Set `_fixhyps` */
    void set_noise_free(bool flag);             /**< Set `_noise_free` */



    /** 
     * GP training, with user-supplied initial hyperparameters, 
     * \return vector of negative log likelihood
     */
    Eigen::VectorXd train(const Eigen::MatrixXd& init_hyps);

    /** 
     * GP training, with default initial hyperparameters, 
     * \return vector of negative log likelihood
     */
    Eigen::VectorXd train(); // use default_hyp as init_hyps

    /** Return the current hyperparameters */
    const Eigen::MatrixXd& get_hyp() const noexcept;

    /** 
     * GP prediction for all the targets, do not calculate gradient 
     *
     * @param [in]  x The input, each column is one input vector
     * @param [out] y The mean values of GP prediction, y.row(i) would be the predictions for x.col(i)
     * @param [out] s2 The uncertainty values of GP prediction, s2.row(i) would be the predictions for x.col(i)
     * */
    void predict(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, Eigen::MatrixXd& s2) const noexcept;

    /** 
     * GP prediction for one of the targets, and support only one input vector 
     *
     * @param [in] spec_id Specify which target is to be predicted
     * @param [in] x The input vector
     * @param [out] y The mean value of the GP prediction for `x`
     * @param [out] s2 The uncertainty value of the GP prediction for `x`
     * */
    void predict(size_t spec_id, const Eigen::VectorXd&x, double& y, double& s2) const noexcept;

    /** 
     * GP prediction for one of the targets, give predictions for a batch of inputs
     *
     * @param [in] spec_id Specify which target is to be predicted
     * @param [in] x The set of input vectors, each column is one input vector
     * @param [out] y The mean values of GP prediction, y(i) would be the prediction for x.col(i)
     * @param [out] s2 The uncertainty values of GP prediction, s2(i) would be the prediction for x.col(i)
     * */
    void predict(size_t spec_id, const Eigen::MatrixXd&x, Eigen::VectorXd& y, Eigen::VectorXd& s2) const noexcept;

    /** GP prediction, calculate the gradient*/
    void predict_with_grad(size_t, const Eigen::VectorXd& x, double& y, double& s2, Eigen::VectorXd& grad_y, Eigen::VectorXd&  grad_s2) const noexcept;

    /** GP prediction, calculate the gradient*/
    void predict_with_grad(size_t, const Eigen::MatrixXd& x, Eigen::VectorXd& y, Eigen::VectorXd& s2, Eigen::MatrixXd& grad_y, Eigen::MatrixXd&  grad_s2) const noexcept;
    
    /** 
     * Perform global optimization to select hyperparameters for all the
     * targets, so `_num_spec` runs of global optimization would be performed
     * 
     * @see Eigen::VectorXd select_init_hyp(size_t max_eval, size_t spec_idx, const Eigen::VectorXd& def_hyp)
     * */
    Eigen::MatrixXd select_init_hyp(size_t, const Eigen::MatrixXd&);

};
