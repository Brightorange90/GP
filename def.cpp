#include "def.h"
using namespace std;

#ifndef DEBUG_RAND_SEED
#define DEBUG_RAND_SEED 0 // this value is usally supplied by CMake commane line argument
#endif

// define rand_seed, instead of just use mt19937_64 engine(0)
// so that we can print the seed for debug
#ifdef MYDEBUG
const size_t rand_seed = DEBUG_RAND_SEED;
#else
const size_t rand_seed = std::random_device{}();
#endif

mt19937_64 engine(rand_seed);
