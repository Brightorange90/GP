#include<Eigen/Dense>
#include<chrono>
#include<fstream>
#include "GP.h"
#include "util.h"
using namespace std;
using namespace Eigen;
using namespace std::chrono;
int main(int arg_num, char** args)
{
    if(arg_num < 2)
    {
        cerr << "Usage: pengbo num_train" << endl;
        exit(EXIT_FAILURE);
    }
    const MatrixXd train_x = read_matrix("train_x").transpose();
    const VectorXd train_y = read_matrix("train_y");
    const MatrixXd test_x  = read_matrix("test_x").transpose();
    const VectorXd test_y  = read_matrix("test_y");
    long num_train = atoi(args[1]);
    if(num_train > train_x.cols())
    {
        cerr << "Num train > " << train_x.cols() << endl;
        exit(EXIT_FAILURE);
    }

    GP gp(train_x.leftCols(num_train), train_y.topRows(num_train));
    cout << "Start training" << endl;
    double nlz = gp.train(gp.get_default_hyps());
    cout << "Negative log likelihood: " << nlz << endl;
    cout << "Optimized hyperparameters:\n" << gp.get_hyp() << endl;

    long num_test = test_x.cols();
    long dim      = test_x.rows();
    VectorXd predy(num_test);
    VectorXd preds2(num_test);
    MatrixXd gpy(dim, num_test);
    MatrixXd gps2(dim, num_test);

    auto t1 = chrono::high_resolution_clock::now();
    VectorXd diff_gy(dim);
    VectorXd diff_gs2(dim);
    for(long i = 0; i < num_test; ++i)
    {
        double py, ps2;
        VectorXd gy, gs2;
        std::tie(py, ps2, gy, gs2) = gp.predict_with_grad(test_x.col(i));
        predy(i)    = py;
        preds2(i)   = ps2;
        gpy.col(i)  = gy;
        gps2.col(i) = gs2;

        if(i == 0)
        {
            for(long j = 0; j < dim; ++j)
            {
                VectorXd point = test_x.col(i);
                point[j] += 1e-3;
                double new_py  = gp.predict_y(point);
                double new_ps2 = gp.predict_s2(point);
                diff_gy[j]  = (new_py  - py)  / 1e-3;
                diff_gs2[j] = (new_ps2 - ps2) / 1e-3;
            }
        }
    }
    auto t2 = chrono::high_resolution_clock::now();
    size_t span = chrono::duration_cast<chrono::seconds>(t2-t1).count();
    cout << "Time for prediction: " << span << " seconds" << endl;

    ofstream f("pred.m");
    f << "predy = [\n" <<  predy << "\n];" << endl;
    f << "preds2 = [\n" << preds2 << "\n];" << endl;
    f << "diff_gy = [\n" << diff_gy << "\n];" << endl;
    f << "diff_gs2 = [\n" << diff_gs2 << "\n];" << endl;
    f << "g_predy = [\n" << gpy << "\n];" << endl;
    f << "g_preds2 = [\n" << gps2 << "\n];" << endl;
    f.close();

    // VectorXd predy  = gp.batch_predict_y(test_x);
    // VectorXd preds2 = gp.batch_predict_s2(test_x);

    // MatrixXd rec(predy.size(), 3);
    // rec << test_y, predy, preds2.array().sqrt();
    // cout << rec << endl;
    return 0;
}
