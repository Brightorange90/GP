cmake_minimum_required(VERSION 3.2.1)
project(gp)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -march=native") # Suppress one warning from when compiling Eigen headers

# debug macro
option(MYDEBUG "Debug macro for optimizer" OFF)
if(MYDEBUG)
    message(STATUS "Open debug marcro for GP")
    add_definitions(-DMYDEBUG)
    add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_NAN)
    if(DEFINED RAND_SEED)
        message(STATUS "Rand seed is ${RAND_SEED}")
        add_definitions(-DDEBUG_RAND_SEED=${RAND_SEED})
    else()
        add_definitions(-DDEBUG_RAND_SEED=3)
    endif()
else()
    message(STATUS "Close debug marcro")
    remove_definitions(-DMYDEBUG)
    remove_definitions(-DDEBUG_RAND_SEED)
endif()

add_subdirectory(MVMO)
include_directories(MVMO)

# GP library
set(LIBSRC util.cpp def.cpp GP.cpp FITC.cpp VFE.cpp MatrixSolver.cpp Cov/CovSEard.cpp Cov/CovSEiso.cpp)
set(LIB GP)
set(test mygp.cpp)
set(test_fitc test_fitc.cpp)
set(test_vfe test_vfe.cpp)
add_library(${LIB} ${LIBSRC})
add_executable(gp   ${test})
add_executable(fitc ${test_fitc})
add_executable(vfe ${test_vfe})
target_link_libraries(${LIB} mvmo)
target_link_libraries(gp   ${LIB})
target_link_libraries(fitc ${LIB})
target_link_libraries(vfe ${LIB})

set_property(TARGET ${LIB} gp fitc vfe PROPERTY CXX_STANDARD 11)

if(NLOPT_PATH)
    message(STATUS "NLOPT_PATH: ${NLOPT_PATH}")
    include_directories("${NLOPT_PATH}/include")
    find_library(NLOPT NAMES nlopt PATHS ${NLOPT_PATH}/lib NO_DEFAULT_PATH)
else()
    find_library(NLOPT NAMES nlopt)
endif()
if(NLOPT)
    message(STATUS "Nlopt library: ${NLOPT}")
    target_link_libraries(${LIB} ${NLOPT})
else()
    message(FATAL_ERROR "NLOPT not found")
endif()

message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
# install GP into a seperate directory
install(TARGETS ${LIB} 
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
install(FILES def.h util.h GP.h
    DESTINATION inc
    PERMISSIONS OWNER_READ GROUP_READ)
