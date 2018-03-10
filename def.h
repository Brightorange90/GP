#pragma once
#include <cassert>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <functional>
// The random seed used in debug and release mode
extern const size_t rand_seed;
extern std::mt19937_64 engine;

#define INF std::numeric_limits<double>::infinity()

#define TODO                                         \
    {                                                \
        std::cerr << "TODO assertion" << endl;       \
        std::cerr << "\tFile: " << __FILE__ << endl; \
        std::cerr << "\tLine: " << __LINE__ << endl; \
        std::cerr << "\tFunc: " << __func__ << endl; \
        exit(EXIT_FAILURE);                          \
    }

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
