
#include "model.h"

#ifndef Q_LEARNING
#define Q_LEARNING

#include "qLearning.h"

class QLearning
{
private:
    LinearStateSpaceModel model;
    int nf = 8 * 20 + 1, numInputOptions;
    float gamma = 0.9, epsilon = 0.9, RL_LearningRate = 0.5;

    VectorXd weights;

public:
    QLearning(LinearStateSpaceModel model);

    VectorXd getFeatures(X x, int uI);

    VectorXd getQAtState(X x);

    int sampleInput(X x, float eps = 0, int iter = 0, int numIters = 0);

    void train(X x0, X xss, U uss, int numEps, int maxStepsPerEpisode);
};

#endif
