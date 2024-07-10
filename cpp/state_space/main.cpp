#include "qLearning.h"

int main()
{
    Matrix2d A;
    MatrixXd B(2, 1);

    A << 1, 0,
        0, 1;
    B << 1, 1;

    int T = 15;

    LinearStateSpaceModel model(A, B);
    QLearning qLearning(model);

    X x0(-3, -3);
    X xss(10, 10);
    U uss = 0;

    qLearning.train(x0, xss, uss, 1000, T);

    VectorXd Q = qLearning.getQAtState(x0);
    cout << Q << endl;

    for (int i = 0; i < T; i++)
    {
        VectorXd Q = qLearning.getQAtState(x0);
        int uI = qLearning.sampleInput(x0);
        U u = model.possibleInputs[uI];
        float cost = model.getReward(x0, u, xss, uss);

        cout << "x = (" << x0(0) << ", " << x0(1) << "); u = " << u << "; cost = " << cost << endl;
        x0 = model.xNext(x0, u);
    }

    return 0;
}
