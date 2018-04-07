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
    const MatrixXd train_x = read_matrix("train_x").transpose();
    const VectorXd train_y = read_matrix("train_y");
    const MatrixXd test_x  = read_matrix("test_x").transpose();
    long num_train = train_x.cols();
    if(arg_num > 1)
        num_train = atoi(args[1]);

    if(num_train > train_x.cols())
    {
        cerr << "Num train > " << train_x.cols() << endl;
        exit(EXIT_FAILURE);
    }

    GP gp(train_x.leftCols(num_train), train_y.topRows(num_train), GP::CovFunc::COV_SE_ARD, GP::MatrixDecomp::QR);
    double nlz = gp.train();
    cout << "Negative log likelihood: " << nlz << endl;
    cout << "Optimized hyperparameters:\n" << gp.get_hyp() << endl;
    
    VectorXd predy, preds2;
    gp.batch_predict(test_x, predy, preds2);
    
    MatrixXd rec(test_x.cols(), 2);
    rec << predy, preds2.array().sqrt();
    ofstream f("pred");
    f << rec << endl;
    f.close();

    return EXIT_FAILURE;
}
