/** 
 * @file def.h 
 * @author Wenlong Lyu
 *
 * Some basic definitions used in GP
 */
#pragma once
#include <cassert>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <functional>
/** 
 * The random seed used in debug and release mode
 * In debug mode, the random seed can be passed through CMake options
 */
extern const size_t rand_seed;

/**
 * Random number generator using `rand_seed` as the seed
 */
extern std::mt19937_64 engine;

/** Macro for infinity */
#define INF std::numeric_limits<double>::infinity()

/** Used to print TODO messages */
#define TODO                                         \
    {                                                \
        std::cerr << "TODO assertion" << endl;       \
        std::cerr << "\tFile: " << __FILE__ << endl; \
        std::cerr << "\tLine: " << __LINE__ << endl; \
        std::cerr << "\tFunc: " << __func__ << endl; \
        exit(EXIT_FAILURE);                          \
    }

/** Assertion with more detailed messages */
#ifdef MYDEBUG
#define MYASSERT(x)                                           \
    if (!(x))                                                 \
    {                                                         \
        std::cerr << "Assertion failed: " << #x << std::endl; \
        std::cerr << "\tFile: " << __FILE__ << endl;          \
        std::cerr << "\tLine: " << __LINE__ << endl;          \
        std::cerr << "\tFunc: " << __func__ << endl;          \
        exit(EXIT_FAILURE);                                   \
    }
#else
#define MYASSERT(x) assert(x)
#endif
