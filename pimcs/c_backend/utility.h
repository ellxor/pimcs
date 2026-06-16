#pragma once
#include <complex.h>
#include <stdatomic.h>
#include <time.h>

int min(int a, int b) { return (a < b) ? a : b; }
int max(int a, int b) { return (a > b) ? a : b; }

float cnormf(complex float c) {
	float r = crealf(c);
	float i = cimagf(c);
	return r*r + i*i;
}

double get_time_from_os() {
	struct timespec timestamp;
	clock_gettime(CLOCK_REALTIME, &timestamp);

	return timestamp.tv_sec + 1.0e-9 * timestamp.tv_nsec;
}

