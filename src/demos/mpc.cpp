#include "demos.hpp"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

void plot(int T, Matrix x_sim, Matrix u_sim)
{
    std::vector<double> x1_sim(x_sim.col(0).data(), x_sim.col(0).data() + x_sim.col(0).size());
    std::vector<double> x2_sim(x_sim.col(1).data(), x_sim.col(1).data() + x_sim.col(1).size());
    std::vector<double> u1_sim(u_sim.col(0).data(), u_sim.col(0).data() + u_sim.col(0).size());
    std::vector<double> u2_sim(u_sim.col(1).data(), u_sim.col(1).data() + u_sim.col(1).size());

    std::vector<int> x_x_axis, u_x_axis;
    for (int i = 0; i < T + 1; i++)
    {
        x_x_axis.push_back(i);
        if (i != T)
        {
            u_x_axis.push_back(i);
        }
    }

    plt::subplot2grid(2, 1, 0, 0);
    plt::title("x");
    plt::named_plot("x1", x_x_axis, x1_sim, "b-");
    plt::named_plot("x2", x_x_axis, x2_sim, "r-");
    plt::legend();

    plt::subplot2grid(2, 1, 1, 0);
    plt::title("y");
    plt::named_plot("u1", u_x_axis, u1_sim, "b-");
    plt::named_plot("u2", u_x_axis, u2_sim, "r-");
    plt::legend();

    plt::show();
}

void mpc_demo()
{
    Eigen::Matrix2d A{{1, 0.5},
                      {-0.2, 1}};
    Eigen::Matrix2d B{{1, 0.3},
                      {-0.1, 1}};
    auto model = std::make_shared<LTIStateSpaceModel>(A, B);

    int K = 10;
    MPC mpc(model, K);

    Eigen::Vector2d x0(-11, 9);
    Eigen::Vector2d desired_x_ss(3, 7);
    SteadyState stead_state = model->find_steady_state(desired_x_ss);
    Eigen::Vector2d x_ss = stead_state.x_ss;
    Eigen::Vector2d u_ss = stead_state.u_ss;

    int T = 20;
    Matrix x_sim = Matrix::Zero(model->nx, T + 1);
    Matrix u_sim = Matrix::Zero(model->nu, T);
    x_sim.col(0) = x0;

    for (int i = 0; i < T; i++)
    {
        Eigen::Vector2d u = mpc.step(x_sim.col(i), x_ss, u_ss);
        u_sim.col(i) = u;
        x_sim.col(i + 1) = model->x_next(x_sim.col(i), u);
    }

    plot(T, x_sim.transpose(), u_sim.transpose());
}