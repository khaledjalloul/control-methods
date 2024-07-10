#include <iostream>
#include "model.h"

using namespace std;

int main()
{
    Matrix2d A;
    Matrix2d B;

    A << 1, 0.5,
        -0.2, 1;
    B << 1, 0.3,
        -0.1, 1;

    LinearStateSpaceModel model(A, B);

    Vector2d x(1, 2);
    Vector2d u(1, 1);
    cout << model.x_next(x, u) << endl;

    return 0;
}