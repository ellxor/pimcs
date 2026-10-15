#include <complex.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <sys/stat.h>

#include "util.h"


#ifdef EMBED_FUNCTION_TABLES
EMBED_FUNCTION_TABLES;
#endif


#ifdef DIAGONAL_STATE
#define PHOTON_BEGIN(wave)     (wave)->excitations - ((N + two_m) / 2)
#define PHOTON_END(wave)       PHOTON_BEGIN(wave)
#define PHOTON_SHIFT		   0
#else
#define PHOTON_BEGIN(...)	   0
#define PHOTON_END(...)        PHOTON_TRUNCATION
#define PHOTON_SHIFT           1
#endif


#define SPIN_SHIFT             (PHOTON_TRUNCATION + 1)


#define JUMP_PHOTON_ANNIHILATION			0
// sigma z
#define JUMP_DEPHASING_SAME_SPIN			1
#define JUMP_DEPHASING_LOWER_SPIN			2
#define JUMP_DEPHASING_UPPER_SPIN			3
// sigma -
#define JUMP_EMISSION_SAME_SPIN				4
#define JUMP_EMISSION_LOWER_SPIN			5
#define JUMP_EMISSION_UPPER_SPIN			6
// sigma +
#define JUMP_PUMPING_SAME_SPIN				7
#define JUMP_PUMPING_LOWER_SPIN				8
#define JUMP_PUMPING_UPPER_SPIN				9
// a^dagger sigma -
#define JUMP_CAVITY_EMISSION_SAME_SPIN		10
#define JUMP_CAVITY_EMISSION_LOWER_SPIN		11
#define JUMP_CAVITY_EMISSION_UPPER_SPIN		12
// a sigma +
#define JUMP_CAVITY_ABSORPTION_SAME_SPIN	13
#define JUMP_CAVITY_ABSORPTION_LOWER_SPIN	14
#define JUMP_CAVITY_ABSORPTION_UPPER_SPIN	15

#define JUMP_COUNT							16
#define EFFECTIVE_HAMILTONIAN				JUMP_COUNT // if index runs to end of table, i.e. no jump chosen


struct WaveVector {
	int64 two_j_sector;
	int64 excitations; // parameter used if using diagonal state optimisation (only single a per spin-m value)
	size_t time_slice; // index used to read into global function array table

	double complex alpha; // parameter used for displacement method
	double complex annihilation;

	double complex coeffs[(N + 1 + 2*SPIN_WIDTH) * (PHOTON_TRUNCATION + 1)];
};


// Tables for storing average expectation values of all simulations
double complex expectation[THREAD_COUNT][OUTPUT_COUNT][OBSERVABLES_COUNT] = {0};


double cnorm(double complex z) {
	return creal(z * conj(z));
}


size_t begin(struct WaveVector *wave) {
	return ((N - wave->two_j_sector) / 2 + SPIN_WIDTH) * (PHOTON_TRUNCATION + 1);
}


size_t end(struct WaveVector *wave) {
	return ((N + wave->two_j_sector) / 2 + SPIN_WIDTH) * (PHOTON_TRUNCATION + 1);
}


// Compute Effective Hamiltonian efficiently by using shifts rather than a matrix.
// This term is also multiplied by -i dt.
void compute_effective_hamiltonian_step(struct WaveVector *src, struct WaveVector *dst, double dt) {
	memset(dst->coeffs, 0, sizeof dst->coeffs);
	dst->two_j_sector = src->two_j_sector;
	dst->time_slice = src->time_slice;
	dst->excitations = src->excitations;
	dst->alpha = src->alpha;
	dst->annihilation = src->annihilation;

	size_t index = begin(src);

	for (int64 two_m = -src->two_j_sector; two_m <= src->two_j_sector; two_m += 2) {
		for (int64 a = PHOTON_BEGIN(src); a <= PHOTON_END(src); a += 1, index += 1) {
			if (cnorm(src->coeffs[index]) == 0) continue; // nothing to be done...

			int64 excited = (N + two_m) / 2;
			int64 ground = N - excited;

			double m = two_m / 2.0;
			int64 jpm = (src->two_j_sector + two_m) / 2;
			int64 jmm = (src->two_j_sector - two_m) / 2;

			// First the T dagger T jump terms:
			dst->coeffs[index] -= src->coeffs[index] * (dt / 2.0) * (
				PHOTON_LOSS_RATE * a +
				LOCAL_EMISSION_RATE * excited +
				LOCAL_PUMPING_RATE * ground +
				COLLECTIVE_DEPHASING_RATE * m * m +
				COLLECTIVE_EMISSION_RATE * (jmm + 1) * jpm +
				COLLECTIVE_PUMPING_RATE * (jpm + 1) * jmm +
				CAVITY_EMISSION_RATE * excited * (a + 1) +
				CAVITY_ABSORPTION_RATE * ground * a
			);

			double complex coeff = -I * dt * src->coeffs[index];
			HAMILTONIAN_CODE;

#ifdef DISPLACEMENT
			double complex shift = src->alpha * PHOTON_LOSS_RATE * dt / 2.0;

			if (a < PHOTON_TRUNCATION) dst->coeffs[index + PHOTON_SHIFT] -= src->coeffs[index] * shift * sqrt(a + 1);
			if (a > 0)                 dst->coeffs[index - PHOTON_SHIFT] += src->coeffs[index] * conj(shift) * sqrt(a);

			// Correction term from Quantum State Diffusion
			if (a > 0) dst->coeffs[index - PHOTON_SHIFT] += conj(src->annihilation) * src->coeffs[index] * sqrt(a) * PHOTON_LOSS_RATE * dt;
#endif
		}
	}
}


void evolve_under_effective_hamiltonian(struct WaveVector *wave, double dt) {
	// In this case of an exponential and linear Hamiltonian, the Runge-Kutta method
	// is identical to a Taylor series expansion, so this is performed for efficiency.
	static thread_local struct WaveVector _a, _b; // create two temporary wave vectors as a double-buffering technique.
	struct WaveVector *a = wave, *b = &_b; // these are controlled by pointers which are cheap to swap.

	double factor = 1.0;

	for (int step = 1;; step += 1) {
		// The branch in the argument should be eliminated by loop unrolling
		compute_effective_hamiltonian_step(a, b, dt); // b now contains -i Heff dt a
		factor /= step;

		// accumulate Taylor series expansion
		size_t index = begin(wave);
		double norm = 0;

		for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
			for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1, index += 1) {
				wave->coeffs[index] += factor * b->coeffs[index];
				norm += cnorm(factor * b->coeffs[index]);
			}
		}

		if (norm < HAMILTONIAN_TOLERANCE) break;

		if (step == 1) a = &_a; // a is temporarily set to &wave for first iteration to avoid a copy
		swap(a, b); // perform double-buffering
	}

#ifdef DISPLACEMENT
	// Quantum State Diffusion term
	double complex dxi = random_complex_gaussian(dt);

	size_t index = begin(wave);

	for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
		for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1, index += 1) {
			if (a) wave->coeffs[index - PHOTON_SHIFT] += wave->coeffs[index] * sqrt(PHOTON_LOSS_RATE * a) * dxi;
		}
	}
#endif
}

#ifdef DISPLACEMENT

void linear_displacement_step(struct WaveVector *src, struct WaveVector *dst, double complex shift) {
	memset(dst->coeffs, 0, sizeof dst->coeffs);
	dst->two_j_sector = src->two_j_sector;
	dst->excitations = src->excitations;
	dst->time_slice = src->time_slice;
	dst->alpha = src->alpha;

	size_t index = begin(src);

	// D dagger
	for (int64 two_m = -src->two_j_sector; two_m <= src->two_j_sector; two_m += 2) {
		for (int64 a = PHOTON_BEGIN(src); a <= PHOTON_END(src); a += 1, index += 1) {
			if (a < PHOTON_TRUNCATION) dst->coeffs[index + PHOTON_SHIFT] -= src->coeffs[index] * shift * sqrt(a + 1);
			if (a > 0)                 dst->coeffs[index - PHOTON_SHIFT] += src->coeffs[index] * conj(shift) * sqrt(a);
		}
	}
}


void displace(struct WaveVector *wave, double complex shift) {
	// In this case of an exponential and linear Hamiltonian, the Runge-Kutta method
	// is identical to a Taylor series expansion, so this is performed for efficiency.
	static thread_local struct WaveVector _a, _b; // create two temporary wave vectors as a double-buffering technique.
	struct WaveVector *a = wave, *b = &_b; // these are controlled by pointers which are cheap to swap.

	double factor = 1.0;
	int64 displacement_expansion_length = 50;

	for (int step = 1; step < displacement_expansion_length; step += 1) {
		// The branch in the argument should be eliminated by loop unrolling
		linear_displacement_step(a, b, shift); // b now contains -i Heff dt a
		factor /= step;

		// accumulate Taylor series expansion
		size_t index = begin(wave);

		for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
			for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1, index += 1) {
				wave->coeffs[index] += factor * b->coeffs[index];
			}
		}

		if (step == 1) a = &_a; // a is temporarily set to &wave for first iteration to avoid a copy
		swap(a, b); // perform double-buffering
	}

	wave->alpha += shift;
}

#endif


// ============================================================================================================
// Collective J  operator factors for changing symmetry sectors
// ============================================================================================================
// source: [TODO: add original source]
//    doi: ...
//
// source: Bastin and Martin, Permutationally invariant processes in open multiqudit systems, 2025
//    doi: 10.1088/1751-8121/addfc1
//
// independently verified by Lloyd, Keeling and Ziolkowska (2026)
//
// Note: the same_spin and lower_spin factors are undefined at J = 0, however, in this case, M = 0 is
//       the only allowed state, therefore all factors M, (J+M) and (J-M) are zero, so transitions of
//       same and lower J are forbidden - which can be confirmed by computing the factors for a basis, e.g. Jz,
//       J+ and J-. Therefore, for J = 0, these factors simply return 0.
// ============================================================================================================

double  compute_same_spin_factor(double J) { return (J == 0) ? 0 : (N/2.0 + 1) / (2*J*(J + 1)); }
double compute_lower_spin_factor(double J) { return (J == 0) ? 0 : (N/2.0 + J + 1) / (2*J*(2*J + 1)); }
double compute_upper_spin_factor(double J) { return (N/2.0 - J) / (2*(J + 1)*(2*J + 1)); }

// ============================================================================================================


// prod_j=1^N exp(-i theta/2 sigma_x) |up_j>
void coherent_state(struct WaveVector *wave, double angle) {
	memset(wave->coeffs, 0, sizeof wave->coeffs);

	double cos_angle = cos(0.5 * angle);
	double sin_angle = sin(0.5 * angle);

	size_t index = begin(wave);
	size_t last_sector = end(wave);

	if (fabs(cos_angle) < 1e-10) { wave->coeffs[index]       = 1; return; } // ground state
	if (fabs(sin_angle) < 1e-10) { wave->coeffs[last_sector] = 1; return; } // excited state

	// Compute probabilities in log space otherwise the combinatorics blows up
	double log_cos = log(cos_angle);
	double log_sin = log(sin_angle);

	double complex phase[4] = { 1.0, -I, -1.0, I }; // -i ** phase

	for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
		int64 excited = (N + two_m) / 2;
		int64 ground = N - excited;

		double log_choose = lgamma(N + 1) - lgamma(excited + 1) - lgamma(ground + 1);
		double log_probability = 0.5 * log_choose + excited * log_cos + ground * log_sin;
		double probability = exp(log_probability);

		wave->coeffs[index] = probability * phase[ground % 4];
		index += SPIN_SHIFT;
	}
}


void photon_annihilation(struct WaveVector *wave) {
	size_t index = begin(wave);

	for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
		for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1, index += 1) {
#ifdef DIAGONAL_STATE
			wave->coeffs[index] *= safe_sqrt(a);
#else
			wave->coeffs[index - PHOTON_SHIFT] += wave->coeffs[index] * sqrt(a);
			wave->coeffs[index] = 0;
#endif
		}
	}

	wave->excitations -= 1;
}


void photon_creation(struct WaveVector *wave) {
	size_t index = begin(wave);

	for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
		index += SPIN_SHIFT;

		for (int64 a = PHOTON_END(wave); a >= PHOTON_BEGIN(wave); a -= 1) {
			index -= 1;
			wave->coeffs[index] = wave->coeffs[index - PHOTON_SHIFT] * safe_sqrt(a + 1 - PHOTON_SHIFT);
		}

		index += SPIN_SHIFT;
	}

	wave->excitations += 1;
}


atomic_int thread_sync_counter = 0;
atomic_int thread_done_counter = 0;

double start_real_time = 0;
uint64_t seed_salt = 0;


void run_simulation(size_t simulation_index, size_t thread_index) {
	init_random_state(simulation_index ^ seed_salt);

	static thread_local struct WaveVector _wave;
	struct WaveVector *wave = &_wave;

	wave->two_j_sector = N;
	wave->excitations = SB_ANGLE == 0 ? N : 0;
	wave->time_slice = 0;
	wave->alpha = 0;

	coherent_state(wave, SB_ANGLE);

	double J = wave->two_j_sector / 2.0;
	double  same_spin_factor = compute_same_spin_factor(J);
	double lower_spin_factor = compute_lower_spin_factor(J);
	double upper_spin_factor = compute_upper_spin_factor(J);

	double next_write = 0;
	size_t write_index = 0;

	for (double time = 0; time <= SIMULATION_TIME;) {
		double jump_table[JUMP_COUNT] = {};
		wave->annihilation = 0;

		size_t index = begin(wave);

		for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
			for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1, index += 1) {
				double norm = cnorm(wave->coeffs[index]);
				if (!norm) continue;

				double m = two_m / 2.0;
				int64 jpm = (wave->two_j_sector + two_m) / 2;
				int64 jmm = (wave->two_j_sector - two_m) / 2;

				jump_table[JUMP_PHOTON_ANNIHILATION]  += norm * a;
				jump_table[JUMP_DEPHASING_SAME_SPIN]  += norm * m * m;
				jump_table[JUMP_DEPHASING_LOWER_SPIN] += norm * jmm * jpm;
				jump_table[JUMP_DEPHASING_UPPER_SPIN] += norm * (jmm + 1) * (jpm + 1);
				jump_table[JUMP_EMISSION_SAME_SPIN]   += norm * (jmm + 1) * jpm;
				jump_table[JUMP_EMISSION_LOWER_SPIN]  += norm * (jpm - 1) * jpm;
				jump_table[JUMP_EMISSION_UPPER_SPIN]  += norm * (jmm + 1) * (jmm + 2);
				jump_table[JUMP_PUMPING_SAME_SPIN]    += norm * (jpm + 1) * jmm;
				jump_table[JUMP_PUMPING_LOWER_SPIN]   += norm * (jmm - 1) * jmm;
				jump_table[JUMP_PUMPING_UPPER_SPIN]   += norm * (jpm + 1) * (jpm + 2);

				jump_table[JUMP_CAVITY_EMISSION_SAME_SPIN]    += norm * (jmm + 1) * jpm * (a + 1);
				jump_table[JUMP_CAVITY_EMISSION_LOWER_SPIN]   += norm * (jpm - 1) * jpm * (a + 1);
				jump_table[JUMP_CAVITY_EMISSION_UPPER_SPIN]   += norm * (jmm + 1) * (jmm + 2) * (a + 1);
				jump_table[JUMP_CAVITY_ABSORPTION_SAME_SPIN]  += norm * (jpm + 1) * jmm * a;
				jump_table[JUMP_CAVITY_ABSORPTION_LOWER_SPIN] += norm * (jmm - 1) * jmm * a;
				jump_table[JUMP_CAVITY_ABSORPTION_UPPER_SPIN] += norm * (jpm + 1) * (jpm + 2) * a;

				if (a) wave->annihilation += conj(wave->coeffs[index - PHOTON_SHIFT]) * wave->coeffs[index] * sqrt(a);
			}
		}

		// Output expectation values
		if (time >= next_write) {
			size_t begin_index = begin(wave);
			size_t end_index = end(wave) + 1;
			size_t total_range = sizeof wave->coeffs / sizeof *wave->coeffs;

			// for expectation operators, we make sure that the region outside the valid Hilbert space
			// is zeroed as OBSERVABLES code may try to read into this range.
			fill(wave->coeffs, wave->coeffs + begin_index, 0);
			fill(wave->coeffs + end_index, wave->coeffs + total_range, 0);

			size_t index = begin_index;

			for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
				for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1, index += 1) {
					if (!cnorm(wave->coeffs[index])) continue;

					double m = two_m / 2.0;
					int64 jpm = (wave->two_j_sector + two_m) / 2;
					int64 jmm = (wave->two_j_sector - two_m) / 2;

					MAYBE_UNUSED(m);
					MAYBE_UNUSED(jpm);
					MAYBE_UNUSED(jmm);

					OBSERVABLES_CODE;
				}
			}

			write_index += 1;
			next_write = SIMULATION_TIME * (double)write_index / (OUTPUT_COUNT - 1);

			wave->time_slice = write_index; // @todo: cleanup - for now use write index as read index for function tables
		}

		// Scale jump table by process rates.
		jump_table[JUMP_PHOTON_ANNIHILATION]  *= PHOTON_LOSS_RATE;
		jump_table[JUMP_DEPHASING_SAME_SPIN]  *= LOCAL_DEPHASING_RATE * same_spin_factor + COLLECTIVE_DEPHASING_RATE;
		jump_table[JUMP_DEPHASING_LOWER_SPIN] *= LOCAL_DEPHASING_RATE * lower_spin_factor;
		jump_table[JUMP_DEPHASING_UPPER_SPIN] *= LOCAL_DEPHASING_RATE * upper_spin_factor;
		jump_table[JUMP_EMISSION_SAME_SPIN]   *= LOCAL_EMISSION_RATE  * same_spin_factor + COLLECTIVE_EMISSION_RATE;
		jump_table[JUMP_EMISSION_LOWER_SPIN]  *= LOCAL_EMISSION_RATE  * lower_spin_factor;
		jump_table[JUMP_EMISSION_UPPER_SPIN]  *= LOCAL_EMISSION_RATE  * upper_spin_factor;
		jump_table[JUMP_PUMPING_SAME_SPIN]    *= LOCAL_PUMPING_RATE   * same_spin_factor + COLLECTIVE_PUMPING_RATE;
		jump_table[JUMP_PUMPING_LOWER_SPIN]   *= LOCAL_PUMPING_RATE   * lower_spin_factor;
		jump_table[JUMP_PUMPING_UPPER_SPIN]   *= LOCAL_PUMPING_RATE   * upper_spin_factor;

		jump_table[JUMP_CAVITY_EMISSION_SAME_SPIN]    *= CAVITY_EMISSION_RATE   * same_spin_factor;
		jump_table[JUMP_CAVITY_EMISSION_LOWER_SPIN]   *= CAVITY_EMISSION_RATE   * lower_spin_factor;
		jump_table[JUMP_CAVITY_EMISSION_UPPER_SPIN]   *= CAVITY_EMISSION_RATE   * upper_spin_factor;
		jump_table[JUMP_CAVITY_ABSORPTION_SAME_SPIN]  *= CAVITY_ABSORPTION_RATE * same_spin_factor;
		jump_table[JUMP_CAVITY_ABSORPTION_LOWER_SPIN] *= CAVITY_ABSORPTION_RATE * lower_spin_factor;
		jump_table[JUMP_CAVITY_ABSORPTION_UPPER_SPIN] *= CAVITY_ABSORPTION_RATE * upper_spin_factor;

		double jump_table_weight = 0;
		for (int i = 0; i < JUMP_COUNT; i += 1) jump_table_weight += jump_table[i];

		double dt = JUMP_TOLERANCE / jump_table_weight;

		if (dt > JUMP_TOLERANCE)    dt = JUMP_TOLERANCE;
		if (time + dt > next_write) dt = next_write - time;

		double r = random_uniform();
		double total_probability = 0;
		size_t choice = 0;

		for (int i = 0; i < JUMP_COUNT; i += 1) {
			double probability = jump_table[i] * dt;
			total_probability += probability;

			if (r < total_probability) {
				break;
			}

			choice += 1;
		}

		// if we get a cavity jump
		bool32 create = JUMP_CAVITY_EMISSION_SAME_SPIN <= choice && choice <= JUMP_CAVITY_EMISSION_UPPER_SPIN;
		bool32 annihilate = JUMP_CAVITY_ABSORPTION_SAME_SPIN <= choice && choice <= JUMP_CAVITY_ABSORPTION_UPPER_SPIN;

		if (create) photon_creation(wave);
		if (annihilate) photon_annihilation(wave);
		if (create || annihilate) choice -= 6; // map back onto non-cavity jump

		int64 old_two_j_sector = wave->two_j_sector;

		switch (choice) {
		case JUMP_PHOTON_ANNIHILATION: {
#ifdef DISPLACEMENT
			evolve_under_effective_hamiltonian(wave, dt); // this jump is covered by QSD
#else
			photon_annihilation(wave);
#endif
			break;
		}

		case JUMP_DEPHASING_SAME_SPIN: {
			size_t index = begin(wave);

			for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
				for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1) {
					wave->coeffs[index] *= two_m;
					index += 1;
				}
			}

			break;
		}

		case JUMP_DEPHASING_LOWER_SPIN: {
			size_t index = begin(wave);

			for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
				for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1) {
					wave->coeffs[index] *= sqrt((wave->two_j_sector - two_m)*(wave->two_j_sector + two_m));
					index += 1;
				}
			}

			wave->two_j_sector -= 2;
			break;
		}

		case JUMP_DEPHASING_UPPER_SPIN: {
			size_t index = begin(wave);

			for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
				for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1) {
					wave->coeffs[index] *= sqrt((wave->two_j_sector - two_m + 2)*(wave->two_j_sector + two_m + 2));
					index += 1;
				}
			}

			wave->two_j_sector += 2;

			// zero out new region
			size_t start = begin(wave);
			fill(wave->coeffs + start, wave->coeffs + start + SPIN_SHIFT, 0);
			fill(wave->coeffs + index, wave->coeffs + index + SPIN_SHIFT, 0);
			break;
		}

		case JUMP_EMISSION_SAME_SPIN: {
			size_t index = begin(wave);

			for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
				for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1) {
					wave->coeffs[index - SPIN_SHIFT] = sqrt((wave->two_j_sector - two_m + 2)*(wave->two_j_sector + two_m)) * wave->coeffs[index];
					index += 1;
				}
			}

			fill(wave->coeffs + index - SPIN_SHIFT, wave->coeffs + index, 0);
			wave->excitations -= 1;
			break;
		}

		case JUMP_EMISSION_LOWER_SPIN: {
			size_t index = begin(wave);

			for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
				for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1) {
					wave->coeffs[index - SPIN_SHIFT] = sqrt((wave->two_j_sector + two_m - 2)*(wave->two_j_sector + two_m)) * wave->coeffs[index];
					index += 1;
				}
			}

			wave->two_j_sector -= 2;
			wave->excitations -= 1;
			break;
		}

		case JUMP_EMISSION_UPPER_SPIN: {
			size_t index = begin(wave);

			for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
				for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1) {
					wave->coeffs[index - SPIN_SHIFT] = sqrt((wave->two_j_sector - two_m + 2)*(wave->two_j_sector - two_m + 4)) * wave->coeffs[index];
					index += 1;
				}
			}

			fill(wave->coeffs + index - SPIN_SHIFT, wave->coeffs + index + SPIN_SHIFT, 0);
			wave->two_j_sector += 2;
			wave->excitations -= 1;
			break;
		}

		// pumping operators iterate in reverse
		case JUMP_PUMPING_SAME_SPIN: {
			size_t index = end(wave);

			for (int64 two_m = wave->two_j_sector; two_m >= -wave->two_j_sector; two_m -= 2) {
				for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1) {
					wave->coeffs[index + SPIN_SHIFT] = sqrt((wave->two_j_sector + two_m + 2)*(wave->two_j_sector - two_m)) * wave->coeffs[index];
					index += 1;
				}

				index -= 2 * SPIN_SHIFT;
			}

			wave->excitations += 1;

			size_t start = begin(wave);
			fill(wave->coeffs + start, wave->coeffs + start + SPIN_SHIFT, 0);
			break;
		}

		case JUMP_PUMPING_LOWER_SPIN: {
			size_t index = end(wave);

			for (int64 two_m = wave->two_j_sector; two_m >= -wave->two_j_sector; two_m -= 2) {
				for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1) {
					wave->coeffs[index + SPIN_SHIFT] = sqrt((wave->two_j_sector - two_m - 2)*(wave->two_j_sector - two_m)) * wave->coeffs[index];
					index += 1;
				}

				index -= 2 * SPIN_SHIFT;
			}

			wave->two_j_sector -= 2;
			wave->excitations += 1;
			break;
		}

		case JUMP_PUMPING_UPPER_SPIN: {
			size_t index = end(wave);

			for (int64 two_m = wave->two_j_sector; two_m >= -wave->two_j_sector; two_m -= 2) {
				for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1) {
					wave->coeffs[index + SPIN_SHIFT] = sqrt((wave->two_j_sector + two_m + 2)*(wave->two_j_sector + two_m + 4)) * wave->coeffs[index];
					index += 1;
				}

				index -= 2 * SPIN_SHIFT;
			}

			wave->two_j_sector += 2;
			wave->excitations += 1;

			size_t start = begin(wave);
			fill(wave->coeffs + start, wave->coeffs + start + 2 * SPIN_SHIFT, 0);
			break;
		}

		case EFFECTIVE_HAMILTONIAN:
			evolve_under_effective_hamiltonian(wave, dt);
			break;
		}

		if (old_two_j_sector != wave->two_j_sector) {
			double J = wave->two_j_sector / 2.0;
			same_spin_factor = compute_same_spin_factor(J);
			lower_spin_factor = compute_lower_spin_factor(J);
			upper_spin_factor = compute_upper_spin_factor(J);
		}

		double norm = 0;
		index = begin(wave);

		for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
			for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1, index += 1) {
				norm += cnorm(wave->coeffs[index]);
			}
		}

		double renormalize = 1.0 / sqrt(norm);
		double complex annihilation = 0;
		MAYBE_UNUSED(annihilation);

		index = begin(wave);

		for (int64 two_m = -wave->two_j_sector; two_m <= wave->two_j_sector; two_m += 2) {
			for (int64 a = PHOTON_BEGIN(wave); a <= PHOTON_END(wave); a += 1, index += 1) {
				wave->coeffs[index] *= renormalize;
				if (a > 0) annihilation += conj(wave->coeffs[index - PHOTON_SHIFT]) * wave->coeffs[index] * sqrt(a);
			}
		}

#ifdef DISPLACEMENT
		if (cnorm(annihilation) > 0.01) displace(wave, annihilation);
#endif

		time += dt;
	}

	int jobs_completed = 1 + atomic_fetch_add(&thread_done_counter, 1);

	double current_real_time = get_time_from_os();
	double average_time_per_trajectory = (current_real_time - start_real_time) / jobs_completed;

	printf(CLEAR_LINE "Simulation %d/%d complete (%.3f s per trajectory)",
		jobs_completed, TRAJECTORY_COUNT, average_time_per_trajectory);
	fflush(stdout); // make sure text is actually written!
}


void *thread_handler(void *arg) {
	size_t thread_index = (size_t) arg;
	size_t simulation_index;

	while ((simulation_index = atomic_fetch_add(&thread_sync_counter, 1)) < TRAJECTORY_COUNT) {
		run_simulation(simulation_index, thread_index);
	}

	return 0;
}


int main() {
	// @todo: code cleanup
	// use the current time to seed the random number generator
	uint64_t salt = 0;
	double t = get_time_from_os();

	static_assert(sizeof(double) == sizeof(uint64_t), "sizeof(double) != sizeof(uint64_t)");
	memcpy(&salt, &t, sizeof(double));

	init_random_state(salt);
	for (int i = 0; i < 20; i += 1) rand64();
	seed_salt = rand64();

	pthread_t thread_handles[THREAD_COUNT] ={0};
	start_real_time = get_time_from_os();

	for (uintptr_t i = 0; i < THREAD_COUNT; i += 1) pthread_create(&thread_handles[i], 0, thread_handler, (void *)i);
	for (uintptr_t i = 0; i < THREAD_COUNT; i += 1) pthread_join(thread_handles[i], 0);

	printf("\n"); // new line after logging in each thread

	// @todo: maybe we should expose results of individual trajectories
	FILE *log = fopen(OUTPUT_FILE, "wb");

	for (int i = 0; i < OUTPUT_COUNT; i += 1) {
		for (int op = 0; op < OBSERVABLES_COUNT; op += 1) {
			double complex expectation_value = 0;

			for (int thread = 0; thread < THREAD_COUNT; thread += 1) {
				expectation_value += expectation[thread][i][op] / TRAJECTORY_COUNT;
			}

			fprintf(log, "%.8g\t%.8g\t", creal(expectation_value), cimag(expectation_value));
		}

		fprintf(log, "\n");
	}

	fclose(log);
}
