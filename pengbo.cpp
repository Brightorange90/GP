#include<Eigen/Dense>
#include<chrono>
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
    return 0;
}
