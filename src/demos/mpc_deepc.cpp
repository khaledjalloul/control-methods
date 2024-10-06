#include "demos.hpp"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

void plot(int T, Matrix x_mpc, Matrix u_mpc, Matrix x_deepc, Matrix u_deepc)
{
    std::vector<double> x_mpc1(x_mpc.col(0).data(), x_mpc.col(0).data() + x_mpc.col(0).size());
    std::vector<double> x_mpc2(x_mpc.col(1).data(), x_mpc.col(1).data() + x_mpc.col(1).size());
    std::vector<double> u_mpc1(u_mpc.col(0).data(), u_mpc.col(0).data() + u_mpc.col(0).size());
    std::vector<double> u_mpc2(u_mpc.col(1).data(), u_mpc.col(1).data() + u_mpc.col(1).size());

    std::vector<double> x_deepc1(x_deepc.col(0).data(), x_deepc.col(0).data() + x_deepc.col(0).size());
    std::vector<double> x_deepc2(x_deepc.col(1).data(), x_deepc.col(1).data() + x_deepc.col(1).size());
    std::vector<double> u_deepc1(u_deepc.col(0).data(), u_deepc.col(0).data() + u_deepc.col(0).size());
    std::vector<double> u_deepc2(u_deepc.col(1).data(), u_deepc.col(1).data() + u_deepc.col(1).size());

    std::vector<int> x_x_axis, u_x_axis;
    for (int i = 0; i < T + 1; i++)
    {
        x_x_axis.push_back(i);
        if (i != T)
        {
            u_x_axis.push_back(i);
        }
    }

    plt::subplot2grid(2, 2, 0, 0);
    plt::title("u (MPC)");
    plt::named_plot("u1", u_x_axis, u_mpc1, "b-");
    plt::named_plot("u2", u_x_axis, u_mpc2, "r-");
    plt::legend();

    plt::subplot2grid(2, 2, 0, 1);
    plt::title("x (MPC)");
    plt::named_plot("x1", x_x_axis, x_mpc1, "b-");
    plt::named_plot("x2", x_x_axis, x_mpc2, "r-");
    plt::legend();

    plt::subplot2grid(2, 2, 1, 0);
    plt::title("u (DeePC)");
    plt::named_plot("u1", u_x_axis, u_deepc1, "b-");
    plt::named_plot("u2", u_x_axis, u_deepc2, "r-");
    plt::legend();

    plt::subplot2grid(2, 2, 1, 1);
    plt::title("x (DeePC)");
    plt::named_plot("x1", x_x_axis, x_deepc1, "b-");
    plt::named_plot("x2", x_x_axis, x_deepc2, "r-");
    plt::legend();

    plt::show();
}

void mpc_deepc_demo()
{
    Eigen::Matrix2d A{{1, 0.5},
                      {-0.2, 1}};
    Eigen::Matrix2d B{{1, 0.3},
                      {-0.1, 1}};
    auto model = std::make_shared<LTIStateSpaceModel>(A, B);

    int K = 10;
    MPC mpc(model, K);
    DeePC deepc(model);

    Eigen::Vector2d x0(-11, 9);
    Eigen::Vector2d desired_x_ss(3, 7);
    SteadyState stead_state = model->find_steady_state(desired_x_ss);
    Eigen::Vector2d x_ss = stead_state.x_ss;
    Eigen::Vector2d u_ss = stead_state.u_ss;

    int T = 20;
    Matrix x_mpc = Matrix::Zero(model->nx, T + 1);
    Matrix x_deepc = Matrix::Zero(model->nx, T + 1);
    Matrix u_mpc = Matrix::Zero(model->nu, T);
    Matrix u_deepc = Matrix::Zero(model->nu, T);
    x_mpc.col(0) = x0;
    x_deepc.col(0) = x0;

    for (int i = 0; i < T; i++)
    {
        u_mpc.col(i) = mpc.step(x_mpc.col(i), x_ss, u_ss);
        u_deepc.col(i) = deepc.step(x_deepc.col(i), x_ss, u_ss, u_deepc, model->y(x_deepc), i);

        x_mpc.col(i + 1) = model->x_next(x_mpc.col(i), u_mpc.col(i));
        x_deepc.col(i + 1) = model->x_next(x_deepc.col(i), u_deepc.col(i));
    }

    plot(T, x_mpc.transpose(), u_mpc.transpose(), x_deepc.transpose(), u_deepc.transpose());
}