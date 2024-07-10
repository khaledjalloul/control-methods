#include <iostream>
#include <Eigen/Dense>

#ifndef STATE_SPACE_MODEL
#define STATE_SPACE_MODEL

using namespace std;
using namespace Eigen;

typedef Eigen::Vector2d X;
typedef float U;

class LinearStateSpaceModel
{
private:
    MatrixXd A, B, C;
    int nx, nu;

public:
    float possibleInputs[20];

    LinearStateSpaceModel(MatrixXd A, MatrixXd B);

    MatrixXd xNext(X x, U u);

    void findSteadyState(X desired_x_ss, U desired_u_ss = 0);

    float getReward(X x, U u, X xss, U uss);
};

#endif