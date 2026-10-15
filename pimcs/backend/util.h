#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <time.h>


// threads.h doesn't exist on macOS because reasons...
#define thread_local _Thread_local

typedef int64_t  int64;
typedef uint64_t uint64;
typedef uint32_t bool32;


#define MAYBE_UNUSED(v) (void)(v);
#define CLEAR_LINE "\r\e[2K"


#define swap(a, b) ({                                                  \
	static_assert(__builtin_types_compatible_p(typeof(a), typeof(b)),  \
		          "types of arguments to swap must match.");           \
	typeof(a) tmp = a; a = b; b = tmp;                                 \
})


#define fill(begin, end, value)  do { for (__auto_type _p = &(begin)[0]; _p < &(end)[0]; ++_p) *_p = (value); } while (0)


double get_time_from_os() {
	struct timespec timestamp;
	clock_gettime(CLOCK_MONOTONIC, &timestamp);

	return timestamp.tv_sec + 1e-9 * timestamp.tv_nsec;
}


double safe_sqrt(double x) {
	if (x < 0) return 0;
	return sqrt(x);
}


// ============================================================================================================
// Pseudo-random number generator
// ============================================================================================================
// source: Marsaglia, Xorshift RNGs, 2003
//    doi: 10.18637/jss.v008.i14
//
// xorshift*: high-quality, fast non-cryptographic 64-bit PRNG
//
// Note: _random_state = 0 is invalid as rand64() will always yield 0
//
// ============================================================================================================

thread_local uint64_t _random_state;


void init_random_state(uint64_t seed) {
	_random_state = ~seed;
	assert(_random_state && "random_state cannot be zero!");
}


uint64_t rand64(void) {
	_random_state ^= _random_state >> 12;
	_random_state ^= _random_state << 25;
	_random_state ^= _random_state >> 27;
	return _random_state * 0x2545f4914f6cdd1d;
}


double random_uniform(void) {
	return rand64() * 0x1p-64; // scale by 2**-64 to map to [0,1]
}


// Generates a random complex variable from a gaussian distribution with mean = 0
double complex random_complex_gaussian(double variance) {
	double two_pi = 8.0 * atan(1.0); // @optimisation: this will be computed at compile time

	double radius = sqrt(-2.0 * log(random_uniform()));
	double angle = two_pi * random_uniform();

	return sqrt(0.5 * variance) * radius * ( cos(angle) + I*sin(angle) );
}
