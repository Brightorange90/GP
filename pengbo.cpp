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
    auto t1 = std::chrono::high_resolution_clock::now();
    double nlz = gp.train();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto training_time = std::chrono::duration_cast<std::chrono::seconds>(t2-t1);
    cout << "Time of training: " << training_time.count() << endl;
    cout << "Negative log likelihood: " << nlz << endl;
    return 0;
}
