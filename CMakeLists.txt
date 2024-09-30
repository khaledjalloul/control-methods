cmake_minimum_required(VERSION 3.5)
project(control_methods)

find_package(Eigen3 REQUIRED)

set(MODELS
    src/models/state_space/state_space.cpp
    src/models/mdp/grid.cpp
    src/models/mdp/tic_tac_toe.cpp
)

set(CONTROLLERS
    src/controllers/q_learning.cpp
    src/controllers/policy_value_iteration.cpp
)

add_library(${PROJECT_NAME} ${MODELS} ${CONTROLLERS})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

add_executable(${PROJECT_NAME}_main src/main.cpp)
target_link_libraries(${PROJECT_NAME}_main ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(TARGETS ${PROJECT_NAME}_main DESTINATION bin)