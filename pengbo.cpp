#include<Eigen/Dense>
#include<chrono>
#include "GP.h"
#include "util.h"
using namespace std;
using namespace Eigen;
using namespace std::chrono;
int main()
{
    const MatrixXd train_x = read_matrix("train_x").transpose();
    const VectorXd train_y = read_matrix("train_y");
    const MatrixXd test_x  = read_matrix("test_x").transpose();
    const VectorXd test_y  = read_matrix("test_y");

    GP gp(train_x, train_y);
    cout << "Start training" << endl;
    double nlz = gp.train();
    cout << "Negative log likelihood: " << nlz << endl;
    return 0;
}
