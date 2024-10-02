cmake_minimum_required(VERSION 3.5)
project(control_methods)

set(PACKAGE_DEPENDENCIES
    OsqpEigen
)

foreach(PKG ${PACKAGE_DEPENDENCIES})
    find_package(${PKG} REQUIRED)
endforeach()

set(MODELS
    src/models/state_space/state_space.cpp
    src/models/mdp/grid.cpp
    src/models/mdp/tic_tac_toe.cpp
)

set(CONTROLLERS
    src/controllers/q_learning.cpp
    src/controllers/policy_value_iteration.cpp
    src/controllers/mpc.cpp
)

add_library(${PROJECT_NAME} ${MODELS} ${CONTROLLERS})
target_link_libraries(${PROJECT_NAME} OsqpEigen::OsqpEigen)
include_directories(${PROJECT_NAME} src/models/mdp/ src/models/state_space/)

add_executable(${PROJECT_NAME}_main src/main.cpp)
target_link_libraries(${PROJECT_NAME}_main ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME}_main DESTINATION bin/${PROJECT_NAME})