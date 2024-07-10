#include "ticTacToe.h"

using namespace std;
using namespace Eigen;

IOFormat VectorFormat(0, 0, "", "");

TicTacToe::TicTacToe()
{
    this->initialState = Matrix3i::Zero();
}

int TicTacToe::stateToIndex(TTTState state)
{
    stringstream stateStr;
    stateStr << state.format(VectorFormat);
    return stoi(stateStr.str(), nullptr, 3);
}

TTTState TicTacToe::indexToState(int index)
{
    MatrixXi state = MatrixXi::Zero(1, 9);

    for (int i = 0; i < 9; i++)
    {
        int cur = index / (int)pow(3, 8 - i);
        state(0, i) = cur;
        index = index - cur * pow(3, 8 - i);
    }

    return state.reshaped<RowMajor>(3, 3);
}

std::tuple<int, Eigen::VectorXd> TicTacToe::getRewardAndPTrans(int stateIndex, TTTAction u)
{
    TTTState state = indexToState(stateIndex);

    int reward = 0;
    VectorXd PTrans((int)pow(3, 9));

    if (isDone(state))
        PTrans(stateIndex) = 1;

    else if ((state.array() == oppSide).count() > (state.array() == side).count())
    {
        int i = u / 3, j = u % 3;

        if (state(i, j) != 0)
        {
            PTrans(stateIndex) = 1;
            reward = -1;
        }
        else
        {
            TTTState newState = state;
            newState(i, j) = side;
            PTrans(stateToIndex(newState)) = 1;
        }
    }
    else
    {
        Matrix3i blanks = (state.array() == 0).cast<int>();
        int blanksCount = blanks.count();
        float blanksProb = blanksCount > 0 ? (1. / blanksCount) : 0;

        for (int bI = 0; bI < 3; bI++)
        {
            for (int bJ = 0; bJ < 3; bJ++)
            {
                if (blanks(bI, bJ))
                {
                    TTTState blankState = state;
                    blankState(bI, bJ) = oppSide;
                    int blankStateIndex = stateToIndex(blankState);
                    PTrans(blankStateIndex) = blanksProb;
                }
            }
        }
    }

    for (int i = 0; i < 3; i++)
    {
        Vector3i row = state.row(i);
        Vector3i col = state.col(i);

        if ((row.array() == oppSide).all() || (col.array() == oppSide).all())
            return {-10, PTrans};

        if ((row.array() == side).all() || (col.array() == side).all())
            return {1, PTrans};
    }

    Vector3i diag = state.diagonal();
    if ((diag.array() == oppSide).all())
        return {-10, PTrans};
    if ((diag.array() == side).all())
        return {1, PTrans};

    Vector3i diag2 = state.rowwise().reverse().diagonal();
    if ((diag2.array() == oppSide).all())
        return {-10, PTrans};
    if ((diag2.array() == side).all())
        return {1, PTrans};

    return {reward, PTrans};
}

bool TicTacToe::isDone(TTTState state)
{
    if ((state.array() != 0).all())
        return true;

    for (int i = 0; i < 3; i++)
    {
        if (state(i, 0) != 0 && (state.row(i).array() == state(i, 0)).all())
            return true;

        if (state(0, i) != 0 && (state.col(i).array() == state(0, i)).all())
            return true;
    }

    if (state(0, 0) != 0 && (state.diagonal().array() == state(0, 0)).all())
        return true;

    if (state(0, 2) != 0 && (state.rowwise().reverse().diagonal().array() == state(0, 2)).all())
        return true;

    return false;
}

TTTState TicTacToe::play(TTTState oldState, TTTAction u)
{
    int side = 1, oppSide = 2;
    TTTState state = oldState;

    if (isDone(state))
        return oldState;

    if ((state.array() == oppSide).count() > (state.array() == side).count())
    {
        int i = u / 3, j = u % 3;

        if (state(i, j) == 0)
            state(i, j) = side;
    }

    else
    {
        int i = 0, j = 0;

        do
        {
            TTTAction u = TTTAction(rand() % 9);
            i = u / 3, j = u % 3;
        } while (state(i, j) != 0);

        state(i, j) = oppSide;
    }

    return state;
}

string TicTacToe::displayGame(TTTState state)
{
    string out = "";

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            string character = "";
            switch (state(i, j))
            {
            case TTTStateOptions::X:
                character = "X";
                break;

            case TTTStateOptions::O:
                character = "O";
                break;

            default:
                character = "_";
                break;
            }

            out.append(character + " ");
        }
        if (i < 2)
            out.append("\n");
    }

    return out;
}

TTTState TicTacToe::userPlay(TTTState oldState)
{
    TTTState state = oldState;
    string uStr;
    int u, i, j;

    do
    {
        cout << "Position (1 to 9): ";
        getline(cin, uStr);
        if (uStr.length() == 0 || !isdigit(uStr[0]))
        {
            u = -1;
            continue;
        }
        u = uStr[0] - '0' - 1;
        i = u / 3, j = u % 3;

    } while (u < 0 || u > 8 || state(i, j) != 0);

    state(i, j) = oppSide;
    return state;
}

void TicTacToe::playMatch(int numEps, MatrixXd policy)
{
    for (int i = 0; i < numEps; i++)
    {
        TTTState state = Matrix3i::Zero();

        while (!isDone(state))
        {
            state = userPlay(state);
            cout << displayGame(state) << endl;
            cout << "-------" << endl;

            if (isDone(state))
                break;

            int stateIndex = stateToIndex(state);
            TTTAction u;
            policy.row(stateIndex).maxCoeff(&u);

            state = play(state, u);
            cout << displayGame(state) << endl;
            cout << "-------" << endl;
        }

        cout << "DONE" << endl;
    }
}