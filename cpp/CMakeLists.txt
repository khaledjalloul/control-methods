cmake_minimum_required(VERSION 3.5)

project(control_methods)

find_package(Eigen3 REQUIRED)

set (MDP_SOURCES
    markov_decision_process/grid.cpp
    markov_decision_process/ticTacToe.cpp
    markov_decision_process/mdp.cpp
    markov_decision_process/main.cpp
    )

add_executable(mdp ${MDP_SOURCES})

set (SS_SOURCES
    state_space/model.cpp
    state_space/qLearning.cpp
    state_space/main.cpp
    )

add_executable(state_space ${SS_SOURCES})