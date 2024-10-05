#include "deepc.hpp"

DeePC::DeePC(const std::shared_ptr<LTIStateSpaceModel> &model) : model_(std::move(model))
{
    T_prev_ = 6;
    T_fut_ = 15;
    L_ = T_prev_ + T_fut_;
    T_ = 100;

    Q_ = Matrix::Identity(model_->nx * T_fut_, model_->nx * T_fut_);
    R_ = Matrix::Identity(model_->nu * T_fut_, model_->nu * T_fut_) * 0.1;
    slack_cost_ = 1000;

    Matrix u_sim = 2 * Matrix::Random(model_->nu, T_);
    Matrix x_sim = Matrix::Zero((model_->nx, T_ + 1));

    for (int i = 0; i < T_; i++)
    {
        Vector u = u_sim.col(i);
        x_sim.col(i + 1) = model_->x_next(x_sim.col(i), u);
    }

    num_hankel_columns_ = T_ - L_ + 1;

    Hu_ = Matrix::Zero((model_->nu * L_, num_hankel_columns_));
    Hx_ = Matrix::Zero((model_->nx * L_, num_hankel_columns_));

    for (int i = 0; i < num_hankel_columns_; i++)
    {
        Hu_.col(i) = u_sim.col(model_->nu + L_);
        Hx_.col(i) = x_sim.col(model_->nx + L_);
    }
}

Vector DeePC::step(Vector x0, Vector x_ss, Vector u_ss, Matrix u_past, Matrix y_past_)
{
}