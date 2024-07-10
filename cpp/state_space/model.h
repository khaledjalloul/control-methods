#include <Eigen/Dense>

using namespace Eigen;

#ifndef STATE_SPACE_DECLARATION
#define STATE_SPACE_DECLARATION

class LinearStateSpaceModel
{
private:
    MatrixXd A, B, C;
    int nx, nu;

public:
    LinearStateSpaceModel(MatrixXd A, MatrixXd B, MatrixXd C = MatrixXd::Zero(0, 0));

    MatrixXd x_next(VectorXd x, VectorXd u);

    MatrixXd y(VectorXd x);

    void find_steady_state(VectorXd desired_x_ss, VectorXd desired_u_ss = VectorXd::Zero(0));
};

#endif