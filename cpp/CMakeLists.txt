cmake_minimum_required(VERSION 3.5)
project(control_methods)

find_package(Eigen3 REQUIRED)

set(MODELS
    src/models/state_space/state_space.hpp
    src/models/state_space/state_space.cpp
    src/models/mdp/mdp.hpp
    src/models/mdp/grid.hpp
    src/models/mdp/grid.cpp
    src/models/mdp/tic_tac_toe.hpp
    src/models/mdp/tic_tac_toe.cpp
)

set(CONTROLLERS
    src/controllers/q_learning.hpp
    src/controllers/q_learning.cpp
)

add_library(${PROJECT_NAME} ${MODELS} ${CONTROLLERS})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

add_executable(main src/main.cpp)
target_link_libraries(main ${PROJECT_NAME})