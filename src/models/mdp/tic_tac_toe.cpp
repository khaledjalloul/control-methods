#include "tic_tac_toe.hpp"

Eigen::IOFormat VectorFormat(0, 0, "", "");

TicTacToe::TicTacToe() {}

int TicTacToe::state_to_index(TTTState x)
{
    std::stringstream state_str;
    state_str << x.format(VectorFormat);
    return stoi(state_str.str(), nullptr, 3);
}

TTTState TicTacToe::index_to_state(int x_index)
{
    Eigen::MatrixXi x = Eigen::MatrixXi::Zero(1, 9);

    for (int i = 0; i < 9; i++)
    {
        int cur = x_index / (int)pow(3, 8 - i);
        x(0, i) = cur;
        x_index = x_index - cur * pow(3, 8 - i);
    }

    return x.reshaped<Eigen::RowMajor>(3, 3);
}

std::tuple<int, Eigen::VectorXd> TicTacToe::get_reward_and_trans_prob(int x_index, TTTAction u, TTTState x_goal)
{
    TTTState x = index_to_state(x_index);

    int reward = 0;
    Eigen::VectorXd trans_prob((int)pow(3, 9));

    if (is_done(x))
        trans_prob(x_index) = 1;

    else if ((x.array() == opp_side_).count() > (x.array() == side_).count())
    {
        int i = u / 3, j = u % 3;

        if (x(i, j) != 0)
        {
            trans_prob(x_index) = 1;
            reward = -1;
        }
        else
        {
            TTTState x_new = x;
            x_new(i, j) = side_;
            trans_prob(state_to_index(x_new)) = 1;
        }
    }
    else
    {
        Eigen::Matrix3i blanks = (x.array() == 0).cast<int>();
        int blanks_count = blanks.count();
        float blanks_prob = blanks_count > 0 ? (1. / blanks_count) : 0;

        for (int bI = 0; bI < 3; bI++)
        {
            for (int bJ = 0; bJ < 3; bJ++)
            {
                if (blanks(bI, bJ))
                {
                    TTTState blankState = x;
                    blankState(bI, bJ) = opp_side_;
                    int blankStateIndex = state_to_index(blankState);
                    trans_prob(blankStateIndex) = blanks_prob;
                }
            }
        }
    }

    for (int i = 0; i < 3; i++)
    {
        auto row = x.row(i);
        auto col = x.col(i);

        if ((row.array() == opp_side_).all() || (col.array() == opp_side_).all())
            return {-10, trans_prob};

        if ((row.array() == side_).all() || (col.array() == side_).all())
            return {1, trans_prob};
    }

    auto diag = x.diagonal();
    if ((diag.array() == opp_side_).all())
        return {-10, trans_prob};
    if ((diag.array() == side_).all())
        return {1, trans_prob};

    auto diag2 = x.rowwise().reverse().diagonal();
    if ((diag2.array() == opp_side_).all())
        return {-10, trans_prob};
    if ((diag2.array() == side_).all())
        return {1, trans_prob};

    return {reward, trans_prob};
}

bool TicTacToe::is_done(TTTState x, TTTState x_goal)
{
    if ((x.array() != 0).all())
        return true;

    for (int i = 0; i < 3; i++)
    {
        if (x(i, 0) != 0 && (x.row(i).array() == x(i, 0)).all())
            return true;

        if (x(0, i) != 0 && (x.col(i).array() == x(0, i)).all())
            return true;
    }

    if (x(0, 0) != 0 && (x.diagonal().array() == x(0, 0)).all())
        return true;

    if (x(0, 2) != 0 && (x.rowwise().reverse().diagonal().array() == x(0, 2)).all())
        return true;

    return false;
}

TTTState TicTacToe::play(TTTState x_old, TTTAction u)
{
    int side_ = 1, opp_side_ = 2;
    TTTState x = x_old;

    if (is_done(x))
        return x_old;

    if ((x.array() == opp_side_).count() > (x.array() == side_).count())
    {
        int i = u / 3, j = u % 3;

        if (x(i, j) == 0)
            x(i, j) = side_;
    }

    else
    {
        int i = 0, j = 0;

        do
        {
            TTTAction u = TTTAction(rand() % 9);
            i = u / 3, j = u % 3;
        } while (x(i, j) != 0);

        x(i, j) = opp_side_;
    }

    return x;
}

std::string TicTacToe::display_game(TTTState x)
{
    std::string out = "";

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::string character = "";
            switch (x(i, j))
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

TTTState TicTacToe::user_play(TTTState x_old)
{
    TTTState x = x_old;
    std::string u_str;
    int u, i, j;

    do
    {
        std::cout << "Position (1 to 9): ";
        std::getline(std::cin, u_str);
        if (u_str.length() == 0 || !isdigit(u_str[0]))
        {
            u = -1;
            continue;
        }
        u = u_str[0] - '0' - 1;
        i = u / 3, j = u % 3;

    } while (u < 0 || u > 8 || x(i, j) != 0);

    x(i, j) = opp_side_;
    return x;
}

void TicTacToe::play_match(int num_eps, Eigen::MatrixXd policy)
{
    for (int i = 0; i < num_eps; i++)
    {
        TTTState x = Eigen::Matrix3i::Zero();

        while (!is_done(x))
        {
            x = user_play(x);
            std::cout << display_game(x) << std::endl;
            std::cout << "-------" << std::endl;

            if (is_done(x))
                break;

            int x_index = state_to_index(x);
            TTTAction u;
            policy.row(x_index).maxCoeff(&u);

            x = play(x, u);
            std::cout << display_game(x) << std::endl;
            std::cout << "-------" << std::endl;
        }

        std::cout << "DONE" << std::endl;
    }
}