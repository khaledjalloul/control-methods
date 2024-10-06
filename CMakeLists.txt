cmake_minimum_required(VERSION 3.5)
project(control_methods)

set(PACKAGE_DEPENDENCIES
    OsqpEigen
)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
foreach(PKG ${PACKAGE_DEPENDENCIES})
    find_package(${PKG} REQUIRED)
endforeach()

set(MODELS
    src/models/state_space/state_space.cpp
    src/models/mdp/grid.cpp
    src/models/mdp/tic_tac_toe.cpp
)

set(CONTROLLERS
    src/controllers/mpc.cpp
    src/controllers/deepc.cpp
)

set(DEMOS
    src/demos/grid.cpp
    src/demos/tic_tac_toe.cpp
    src/demos/policy_value_iteration.cpp
    src/demos/mpc_deepc.cpp
)

add_library(${PROJECT_NAME}
    ${MODELS}
    ${CONTROLLERS}
    ${DEMOS}
)
target_link_libraries(${PROJECT_NAME}
    OsqpEigen::OsqpEigen
    ${Python3_LIBRARIES}
)
include_directories(${PROJECT_NAME}
    src/models/mdp/
    src/models/state_space/
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_main src/main.cpp)
target_link_libraries(${PROJECT_NAME}_main ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME}_main DESTINATION bin/${PROJECT_NAME})