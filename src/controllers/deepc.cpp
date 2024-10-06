#include "deepc.hpp"

DeePC::DeePC(const std::shared_ptr<LTIStateSpaceModel> &model) : model_(std::move(model))
{
    ny_ = model_->ny;
    nu_ = model_->nu;

    T_prev_ = 6;
    T_fut_ = 15;
    L_ = T_prev_ + T_fut_;
    int T_traj = 100;

    Q_ = Matrix::Identity(ny_, ny_);
    R_ = Matrix::Identity(nu_, nu_) * 0.1;

    Matrix u_sim = 2 * Matrix::Random(nu_, T_traj);
    Matrix x_sim = Matrix::Zero(model_->nx, T_traj + 1);
    Matrix y_sim = Matrix::Zero(ny_, T_traj);

    for (int i = 0; i < T_traj; i++)
    {
        Vector u = u_sim.col(i);
        x_sim.col(i + 1) = model_->x_next(x_sim.col(i), u);
        y_sim.col(i) = model->y<Vector>(x_sim.col(i));
    }

    num_hankel_columns_ = T_traj - L_ + 1;

    Hu_ = Matrix::Zero(nu_ * L_, num_hankel_columns_);
    Hy_ = Matrix::Zero(ny_ * L_, num_hankel_columns_);

    for (int i = 0; i < num_hankel_columns_; i++)
    {
        Hu_.col(i) = u_sim.block(0, i, nu_, L_).reshaped(nu_ * L_, 1);
        Hy_.col(i) = y_sim.block(0, i, ny_, L_).reshaped(ny_ * L_, 1);
    }

    num_vars_ = num_hankel_columns_ + (ny_ + nu_) * T_fut_;
    num_constraints_ = (ny_ + nu_) * L_ + nu_ * T_fut_;

    solver_.settings()->setVerbosity(false);
    solver_.data()->setNumberOfVariables(num_vars_);
    solver_.data()->setNumberOfConstraints(num_constraints_);
}

Vector DeePC::step(Vector x0, Vector x_ss, Vector u_ss, Matrix u_past, Matrix y_past, int current_index)
{
    solver_.clearSolver();
    solver_.data()->clearHessianMatrix();
    solver_.data()->clearLinearConstraintsMatrix();

    /* Cost (3g 2y 2u):
    H: 0 0 0 0 0 0 0 -- g no cost
       0 0 0 0 0 0 0
       0 0 0 0 0 0 0
       0 0 0 Q 0 0 0 -- y costs
       0 0 0 0 Q 0 0
       0 0 0 0 0 R 0 -- u costs
       0 0 0 0 0 0 R
    */

    Matrix H = Matrix::Zero(num_vars_, num_vars_);
    Vector g = Vector::Zero(num_vars_);
    Vector y_ref = model_->y(x_ss);

    for (int i = 0; i < T_fut_; i++)
    {
        H.block(num_hankel_columns_ + i * ny_, num_hankel_columns_ + i * ny_, ny_, ny_) = Q_;
        H.block(num_hankel_columns_ + T_fut_ * ny_ + i * nu_, num_hankel_columns_ + T_fut_ * ny_ + i * nu_, nu_, nu_) = R_;

        g.segment(num_hankel_columns_ + i * ny_, ny_) = -Q_ * y_ref;
        g.segment(num_hankel_columns_ + T_fut_ * ny_ + i * nu_, nu_) = -R_ * u_ss;
    }

    /* Constraints:
    A: Y_p 0  0
       Y_f -I 0
       U_p 0  0
       U_f 0  -I
    b: y_past
       0
       u_past
       0
    */

    Matrix Y_p = Hy_.topRows(ny_ * T_prev_);
    Matrix Y_f = Hy_.bottomRows(ny_ * T_fut_);
    Matrix U_p = Hu_.topRows(nu_ * T_prev_);
    Matrix U_f = Hu_.bottomRows(nu_ * T_fut_);

    Matrix u_past_block = Matrix::Zero(nu_, T_prev_);
    Matrix y_past_block = Matrix::Zero(ny_, T_prev_);

    for (int i = 0; i < T_prev_; i++)
    {
        int past_index = current_index - T_prev_ + i;
        if (past_index >= 0)
        {
            u_past_block.col(i) = u_past.col(past_index);
            y_past_block.col(i) = y_past.col(past_index);
        }
    }

    Vector u_past_vec = u_past_block.reshaped(nu_ * T_prev_, 1);
    Vector y_past_vec = y_past_block.reshaped(ny_ * T_prev_, 1);

    Matrix dynamics_constr_A = Matrix::Zero(num_constraints_, num_vars_);
    Vector dynamics_constr_u = Vector::Zero(num_constraints_);
    Vector dynamics_constr_l = Vector::Zero(num_constraints_);

    // y data constraints
    dynamics_constr_A.block(0, 0, ny_ * T_prev_, num_hankel_columns_) = Y_p;
    dynamics_constr_A.block(ny_ * T_prev_, 0, ny_ * T_fut_, num_hankel_columns_) = Y_f;
    dynamics_constr_A.block(ny_ * T_prev_, num_hankel_columns_, ny_ * T_fut_, ny_ * T_fut_) = -Matrix::Identity(ny_ * T_fut_, ny_ * T_fut_);

    dynamics_constr_l.head(ny_ * T_prev_) = y_past_vec;
    dynamics_constr_u.head(ny_ * T_prev_) = y_past_vec;

    // u data constraints
    dynamics_constr_A.block(ny_ * L_, 0, nu_ * T_prev_, num_hankel_columns_) = U_p;
    dynamics_constr_A.block(ny_ * L_ + nu_ * T_prev_, 0, nu_ * T_fut_, num_hankel_columns_) = U_f;
    dynamics_constr_A.block(ny_ * L_ + nu_ * T_prev_, num_hankel_columns_ + ny_ * T_fut_, nu_ * T_fut_, nu_ * T_fut_) = -Matrix::Identity(nu_ * T_fut_, nu_ * T_fut_);

    dynamics_constr_l.segment(ny_ * L_, nu_ * T_prev_) = u_past_vec;
    dynamics_constr_u.segment(ny_ * L_, nu_ * T_prev_) = u_past_vec;

    // Additional constraints
    dynamics_constr_A.block((ny_ + nu_) * L_, num_hankel_columns_ + ny_ * T_fut_, nu_ * T_fut_, nu_ * T_fut_) = Matrix::Identity(nu_ * T_fut_, nu_ * T_fut_);
    dynamics_constr_l.segment((ny_ + nu_) * L_, nu_ * T_fut_) = Vector::Constant(nu_ * T_fut_, -3);
    dynamics_constr_u.segment((ny_ + nu_) * L_, nu_ * T_fut_) = Vector::Constant(nu_ * T_fut_, 3);

    // Solver Configuration
    solver_.data()->setHessianMatrix((Eigen::SparseMatrix<double>)H.sparseView());
    solver_.data()->setGradient(g);
    solver_.data()->setLinearConstraintsMatrix((Eigen::SparseMatrix<double>)dynamics_constr_A.sparseView());
    solver_.data()->setBounds(dynamics_constr_l, dynamics_constr_u);

    solver_.initSolver();
    solver_.solveProblem();

    if (solver_.getStatus() == OsqpEigen::Status::Solved)
    {
        Vector sol = solver_.getSolution();
        return sol.segment(num_hankel_columns_ + T_fut_ * ny_, nu_);
    }
    return Vector::Zero(nu_);
}