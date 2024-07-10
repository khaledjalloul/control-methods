#include "qLearning.h"

using namespace std;
using namespace Eigen;

IOFormat VectorFormat(0, 0, "", " ");

QLearning::QLearning(LinearStateSpaceModel model) : model(model)
{
    numInputOptions = sizeof(model.possibleInputs) / sizeof(*model.possibleInputs);
    weights = VectorXd::Zero(nf);
}

VectorXd QLearning::getFeatures(X x, int uI)
{
    VectorXd features = VectorXd::Zero(nf);

    U u = model.possibleInputs[uI];

    features.segment(uI * 8, 2) = x;
    features(uI * 8 + 2) = u;
    features.segment(uI * 8 + 3, 2) = x.array().pow(2);
    features(uI * 8 + 5) = pow(u, 2);
    features.segment(uI * 8 + 6, 2) = x * u;

    features(8 * 20) = 1;

    return features;
}

VectorXd QLearning::getQAtState(X x)
{
    VectorXd Q = VectorXd::Zero(numInputOptions);

    for (int i = 0; i < numInputOptions; i++)
    {
        U u = model.possibleInputs[i];
        VectorXd features = getFeatures(x, i);
        Q(i) = features.dot(weights);
    }

    return Q;
}

int QLearning::sampleInput(X x, float eps, int iter, int numIters)
{
    if ((rand() / double(RAND_MAX)) <= eps - ((double)iter / numIters))
        return rand() % numInputOptions;

    VectorXd Q = getQAtState(x);

    if (Q.minCoeff() == 0)
        return rand() % numInputOptions;

    int argmax;
    Q.minCoeff(&argmax);
    return argmax;
}

void QLearning::train(X x0, X xss, U uss, int numEps, int maxStepsPerEpisode)
{
    weights = VectorXd::Zero(nf);

    for (int i = 0; i < numEps; i++)
    {
        cout << "Training: " << ((float)i / numEps) * 100 << "%" << endl;

        X x = x0;

        for (int j = 0; j < maxStepsPerEpisode; j++)
        {
            int uI = sampleInput(x, epsilon, i, numEps);
            U u = model.possibleInputs[uI];

            float reward = model.getReward(x, u, xss, uss);

            X xNext = model.xNext(x, u);

            VectorXd features = getFeatures(x, uI);
            double Q = features.dot(weights);

            VectorXd nextQ = getQAtState(xNext);
            double maxQ = nextQ.minCoeff();

            cout << weights.format(VectorFormat) << endl;
            cout << features.format(VectorFormat) << endl;
            cout << x(0) << " " << x(1) << " " << reward << " " << Q << " " << maxQ << endl;
            cout << "----" << endl;

            weights = weights + RL_LearningRate * (reward + gamma * maxQ - Q) * features;
            // weights = weights + RL_LearningRate * gamma * features;
            weights = weights / (weights.norm() + 0.001);

            x = xNext;
        }
    }
}
