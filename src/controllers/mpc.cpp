#include "mpc.hpp"

MPC::MPC(const std::shared_ptr<LTIStateSpaceModel> &model, int K)
    : model_(std::move(model)), K_(K)
{
    nx_ = model_->nx;
    nu_ = model_->nu;

    Q_ = Matrix::Identity(nx_, nx_);
    R_ = Matrix::Identity(nu_, nu_) * 0.3;
    slack_cost_ = 300;
    num_vars_ = (K_ + 1) * nx_ + K_ * nu_;
    num_constraints_ = (K_ + 1) * nx_ + nx_ + nu_;

    solver_.settings()->setVerbosity(false);
    solver_.data()->setNumberOfVariables(num_vars_);
    solver_.data()->setNumberOfConstraints(num_constraints_);
}

Vector MPC::step(Vector x0, Vector x_ss, Vector u_ss)
{
    Matrix H = Matrix::Zero(num_vars_, num_vars_);
    Vector g = Vector::Zero(num_vars_);
    Matrix dynamics_constr_A = Matrix::Zero(num_constraints_, num_vars_);
    Vector dynamics_constr_b = Vector::Zero(num_constraints_);

    /* For K = 2:
    H: 0 0 0 0 0 -- x0 no cost
       0 Q 0 0 0 -- x1
       0 0 Q 0 0 -- x2 redundant cost
       0 0 0 R 0 -- u0
       0 0 0 0 R -- u1 redundant cost
    A: I  0  0  0  0  -- x0 = x0
       -A I  0  -B 0  -- x1 = A * x0 + B * u0
       0  -A I  0  -B -- x2 = A * x1 + B * u1
       0  0  I  0  0  -- x2 = x_ss
       0  0  0  0  I  -- u1 = u_ss
    b: x0
       0
       0
       x_ss
       u_ss
    */

    dynamics_constr_A.block(0, 0, nx_, nx_) = Matrix::Identity(nx_, nx_);
    dynamics_constr_b.head(nx_) = x0;

    for (int i = 0; i < K_; i++)
    {
        H.block((i + 1) * nx_, (i + 1) * nx_, nx_, nx_) = Q_;
        H.block((K_ + 1) * nx_ + i * nu_, (K_ + 1) * nx_ + i * nu_, nu_, nu_) = R_;

        g.segment((i + 1) * nx_, nx_) = -Q_ * x_ss;
        g.segment((K_ + 1) * nx_ + i * nu_, nu_) = -R_ * u_ss;

        dynamics_constr_A.block((i + 1) * nx_, (i + 1) * nx_, nx_, nx_) = Matrix::Identity(nx_, nx_);
        dynamics_constr_A.block((i + 1) * nx_, i * nx_, nx_, nx_) = -model_->A;
        dynamics_constr_A.block((i + 1) * nx_, (K_ + 1) * nx_ + i * nu_, nx_, nu_) = -model_->B;
    }

    dynamics_constr_A.block((K_ + 1) * nx_, K_ * nx_, nx_, nx_) = Matrix::Identity(nx_, nx_);
    dynamics_constr_A.block((K_ + 2) * nx_, (K_ + 1) * nx_ + (K_ - 1) * nu_, nu_, nu_) = Matrix::Identity(nx_, nx_);
    dynamics_constr_b.segment((K_ + 1) * nx_, nx_) = x_ss;
    dynamics_constr_b.tail(nu_) = u_ss;

    solver_.data()->setHessianMatrix((Eigen::SparseMatrix<double>)H.sparseView());
    solver_.data()->setGradient(g);
    solver_.data()->setLinearConstraintsMatrix((Eigen::SparseMatrix<double>)dynamics_constr_A.sparseView());
    solver_.data()->setBounds(dynamics_constr_b, dynamics_constr_b);

    solver_.initSolver();
    solver_.solveProblem();

    if (solver_.getStatus() == OsqpEigen::Status::Solved)
    {
        Vector sol = solver_.getSolution();
        return sol.segment((K_ + 1) * nx_, nu_);
    }
    return Vector::Zero(nu_);
}