#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>

#include "compat.h" // tmporary fix

#include "config.h"
#include "random.h"
#include "utility.h"

enum Options {
	OUTPUT_COUNT = 300,
	MOLMER_REPEATS = 0,
	SAVE_TRAJECTORY = true,

};


#include "tmpconfig.h"

enum JumpType {
	JUMP_COLLECTIVE_SPIN_DEPHASING,
	JUMP_COLLECTIVE_SPIN_LOSS,
	JUMP_COLLECTIVE_SPIN_GAIN,

 	JUMP_SPIN_DEPHASING_SAME_J,
 	JUMP_SPIN_DEPHASING_LOWER_J,
 	JUMP_SPIN_DEPHASING_UPPER_J,

	JUMP_SPIN_LOSS_SAME_J,
	JUMP_SPIN_LOSS_LOWER_J,
	JUMP_SPIN_LOSS_UPPER_J,

 	JUMP_SPIN_GAIN_SAME_J,
 	JUMP_SPIN_GAIN_LOWER_J,
 	JUMP_SPIN_GAIN_UPPER_J,

	JUMP_SPIN_LOSS_PHOTON_GAIN_SAME_J,
	JUMP_SPIN_LOSS_PHOTON_GAIN_LOWER_J,
	JUMP_SPIN_LOSS_PHOTON_GAIN_UPPER_J,

 	JUMP_SPIN_GAIN_PHOTON_LOSS_SAME_J,
 	JUMP_SPIN_GAIN_PHOTON_LOSS_LOWER_J,
 	JUMP_SPIN_GAIN_PHOTON_LOSS_UPPER_J,

	JUMP_PHOTON_LOSS,
	JUMP_COUNT,

	EFFECTIVE_HAMILTONIAN = JUMP_COUNT,
};

// Young Tableau of wavefunction:
//
// [1111111111][222222]
// [2222222]
//
// GT Pattern:
//
// (a b)
// ( c )
//
// Constraints:
// a + b = N
// a >= c >= b
// 
// Populations:
// spin_down = c
// spin_up = N - c

typedef complex float WaveVector[NumberOfEmitters + 1][CavityTruncation];

struct TrajectoryState {
	int row1, row2; // Young diagram shape
 	int rowa, rowb; // truncated diagram (with tolerance to ignore low norm states)
	float time_step;

	// displacement transform parameters
	complex float alpha;
	complex float gjx_expect;
	complex float ann_expect;
	complex float ttc_factor;

	complex float abs_expect; // term from a sigma+ jump
	complex float ems_expect; // term from a† sigma- jump

	WaveVector *wave;
};


struct WaveVectorAllocation {
	complex float __padding1[PaddingWidth*CavityTruncation];
	WaveVector wave;
	complex float __padding2[PaddingWidth*CavityTruncation];
};


// include after type definitions
#include "tmp.h"


// PI jump weightings
float precompute_e_factor(int r1, int r2) { return (r1 == r2) ? 0 : (float)(r1 + r2 + 2) / ((r1 - r2)*(r1 - r2 + 2)); }
float precompute_f_factor(int r1, int r2) { return (r1 == r2) ? 0 : (float)(r1 + 1) / ((r1 - r2) * (r1 - r2 + 1)); }
float precompute_g_factor(int r1, int r2) { return (float)(r2) / ((r1 - r2 + 1)*(r1 - r2 + 2)); }


// Select a jump (or effective Hamiltonian) at random using probabilities in the jump table.
//
int select_random_jump(float jump_table[]) {
	float r = random_uniform();
	float seen_probability = 0;

	for (int choice = 0; choice < JUMP_COUNT; ++choice) {
		seen_probability += jump_table[choice];
		if (r < seen_probability) return choice;
	}

	// If no jump has been selected then we evolve using the effective Hamiltonian.
	return EFFECTIVE_HAMILTONIAN;
}


// Compute linear effective Hamiltonian step
//
void linear_hamiltonian_integration_step(WaveVector dest, WaveVector source, struct TrajectoryState *state) {
	memset(dest, 0, sizeof(WaveVector));

	for (int n = state->rowb; n <= state->rowa; ++n) { // number of spin down
		for (int a = 0; a < CavityTruncation; ++a) {
			int jpm = state->row1 - n; // J+M
			int jmm = n - state->row2; // J-M
			float m2 = (NumberOfEmitters - 2*n);

			// Jump dagger jump terms from effective Hamiltonian
			dest[n][a] -= source[n][a] * state->time_step/2 * (
				config.PhotonLossRate * a +
				config.DephasingRate * NumberOfEmitters +
				config.EmissionRate * (NumberOfEmitters - n) +
				config.PumpingRate * n +
				config.CollectiveDephasingRate * m2*m2 +
				config.CollectiveEmissionRate * (jmm + 1)*(jpm) +
				config.CollectivePumpingRate * (jpm + 1)*(jmm) +
				config.CavityEmissionRate * (jmm + 1)*(jpm) * (a + 1) +
				config.CavityAbsorptionRate * (jpm + 1)*(jmm) * a
			);

			if (UseDisplacement) { // extra QSD terms
				dest[n][a] -= source[n][a] * state->time_step/2 * config.PhotonLossRate * cnormf(state->ann_expect);
				dest[n][a-1] -= source[n][a] * state->time_step * config.PhotonLossRate * conjf(state->ann_expect) * sqrtf(a);
			}

			hamiltonian_term(dest, source, state, n, a);
		}
	}
}


// Higher order exponetial evolution with Wiener fluctuation for QSD
//
void evolve_under_H_eff(WaveVector wave, struct TrajectoryState *state) {
	// In this case of an exponential and linear Hamiltonian, the Runge-Kutta method
	// is identical to a Taylor series expansion, so this is performed directly for efficiency.

	static thread_local struct WaveVectorAllocation _a, _b; // Create two temporary vectors using double-buffering technique.
	auto a = wave; // Controlled by pointers which are cheap to swap.
	auto b = _b.wave;

	int factorial = 1;

	for (int i = 1; i <= config.RungeKuttaPoly; ++i) {
		linear_hamiltonian_integration_step(b, a, state); // |b> = -i Heff dt |a>
		factorial *= i;

		// accumulate Taylor series expansion
		float factor = 1.0f / factorial;

		for (int n = state->rowb; n <= state->rowa; ++n) {
			for (int a = 0; a < CavityTruncation; ++a) {
				wave[n][a] += factor * b[n][a];
			}
		}

		if (i == 1) a = _a.wave; // a is temporarily set to wave for first iteration to avoid a copy
		swap(a, b); // perform double-buffering
	}

	// Generate Wiener fluctuation for quantum state diffusion
	if (UseDisplacement) {
		complex float xi = random_complex_gaussian(state->time_step);

		for (int n = state->rowb; n <= state->rowa; ++n) {
			for (int a = 0; a < CavityTruncation; ++a) {
				if (a) wave[n][a - 1] += xi * wave[n][a] * sqrtf(config.PhotonLossRate * a); // quantum state diffusion term
				wave[n][a] -= wave[n][a] * state->ann_expect * xi; // and correction term
			}
		}
	}
}


// Implementation of Quantum Jumps:
// The coefficients of jumps are ignored as the states will be renormalised afterwards.
// The phase is also ignored as it doesn't determine the dynamics (Lindblad jumps are invariant under phase rotations).

void jump_spin_dephasing_same_j(WaveVector wave, struct TrajectoryState *state) {
	for (int n = state->rowb; n <= state->rowa; ++n) {
		for (int a = 0; a < CavityTruncation; ++a) {
			wave[n][a] *= 0.5f * (NumberOfEmitters - 2*n);
		}
	}
}


void jump_spin_dephasing_lower_j(WaveVector wave, struct TrajectoryState *state) {
	for (int n = state->rowb; n <= state->rowa; ++n) {
		for (int a = 0; a < CavityTruncation; ++a) {
			int jpm = state->row1 - n; // J + M
			int jmm = n - state->row2; // J - M

			wave[n][a] *= sqrtf(jmm*jpm);
		}
	}

	state->row1 -= 1;
	state->row2 += 1;
}


void jump_spin_dephasing_upper_j(WaveVector wave, struct TrajectoryState *state) {
	for (int n = state->rowb; n <= state->rowa; ++n) {
		for (int a = 0; a < CavityTruncation; ++a) {
			int jpm = state->row1 - n; // J + M
			int jmm = n - state->row2; // J - M

			wave[n][a] *= sqrtf((jmm + 1)*(jpm + 1));
		}
	}

	state->row1 += 1;
	state->row2 -= 1;
}


void jump_spin_loss_same_j(WaveVector wave, struct TrajectoryState *state) {
	for (int n = state->rowa; n >= state->rowb; --n) { // iterate backwards due to overlap
		for (int a = 0; a < CavityTruncation; ++a) {
			int jpm = state->row1 - n; // J + M
			int jmm = n - state->row2; // J - M

			wave[n + 1][a] = sqrtf((jmm + 1)*jpm) * wave[n][a];
		}
	}

	state->rowa += 1;
	state->rowb += 1;
}


void jump_spin_loss_lower_j(WaveVector wave, struct TrajectoryState *state) {
	for (int n = state->rowa; n >= state->rowb; --n) { // iterate backwards due to overlap
		for (int a = 0; a < CavityTruncation; ++a) {
			int jpm = state->row1 - n; // J + M
			int jmm = n - state->row2; // J - M

			wave[n + 1][a] = sqrtf((jpm - 1)*(jpm)) * wave[n][a];
		}
	}

	state->row1 -= 1;
	state->row2 += 1;
	state->rowa += 1;
	state->rowb += 1;
}


void jump_spin_loss_upper_j(WaveVector wave, struct TrajectoryState *state) {
	for (int n = state->rowa; n >= state->rowb; --n) { // iterate backwards due to overlap
		for (int a = 0; a < CavityTruncation; ++a) {
			int jpm = state->row1 - n; // J + M
			int jmm = n - state->row2; // J - M

			wave[n + 1][a] = sqrtf((jmm + 1)*(jmm + 2)) * wave[n][a];
		}
	}

	state->row1 += 1;
	state->row2 -= 1;
	state->rowa += 1;
	state->rowb += 1;
}


void jump_spin_gain_same_j(WaveVector wave, struct TrajectoryState *state) {
	for (int n = state->rowb + 1; n <= state->rowa; ++n) {
		for (int a = 0; a < CavityTruncation; ++a) {
			int jpm = state->row1 - n; // J + M
			int jmm = n - state->row2; // J - M

			wave[n - 1][a] = sqrtf((jpm + 1)*jmm) * wave[n][a];
		}
	}

	state->rowa -= 1;
	state->rowb -= 1;
}


void jump_spin_gain_lower_j(WaveVector wave, struct TrajectoryState *state) {
	for (int n = state->rowb + 2; n <= state->rowa; ++n) {
		for (int a = 0; a < CavityTruncation; ++a) {
			int jpm = state->row1 - n; // J + M
			int jmm = n - state->row2; // J - M

			wave[n - 1][a] = sqrtf((jmm - 1)*(jmm)) * wave[n][a];
		}
	}

	state->row1 -= 1;
	state->row2 += 1;
	state->rowa -= 1;
	state->rowb -= 1;
}


void jump_spin_gain_upper_j(WaveVector wave, struct TrajectoryState *state) {
	for (int n = state->rowb; n <= state->rowa; ++n) {
		for (int a = 0; a < CavityTruncation; ++a) {
			int jpm = state->row1 - n; // J + M
			int jmm = n - state->row2; // J - M

			wave[n - 1][a] = sqrtf((jpm + 1)*(jpm + 2)) * wave[n][a];
		}
	}

	state->row1 += 1;
	state->row2 -= 1;
	state->rowa -= 1;
	state->rowb -= 1;
}


void jump_photon_loss(WaveVector wave, struct TrajectoryState *state) {
	for (int n = state->rowb; n <= state->rowa; ++n) {
		for (int a = 1; a < CavityTruncation; ++a) {
			wave[n][a - 1] = sqrtf(a) * wave[n][a];
		}
		wave[n][CavityTruncation - 1] = 0;
	}
}


void jump_photon_gain(WaveVector wave, struct TrajectoryState *state) {
	for (int n = state->rowb; n <= state->rowa; ++n) {
		for (int a = CavityTruncation - 1; a > 0; --a) {
			wave[n][a] = sqrtf(a) * wave[n][a - 1];
		}
		wave[n][0] = 0;
	}
}


float normalize_state(WaveVector wave, struct TrajectoryState *state) {
	float norm = 0;

	for (int n = state->rowb; n <= state->rowa; ++n) {
		for (int a = 0; a < CavityTruncation; ++a) {
			norm += cnormf(wave[n][a]);
		}
	}

	float scale = 1.0f / sqrtf(norm);

	for (int n = state->rowb; n <= state->rowa; ++n) {
		for (int a = 0; a < CavityTruncation; ++a) {
			wave[n][a] *= scale;
		}
	}

	return norm;
}


thread_local int simulation_index;
complex float *initial_state;


struct TrajectoryState simulate_trajectory(float total_time, struct TrajectoryState *initial, complex float *expectation) {
	static thread_local struct WaveVectorAllocation wave_alloc;

	struct TrajectoryState state = {
		.row1 = NumberOfEmitters, // all spin down (maximal J sector)
		.row2 = 0,

		.rowa = NumberOfEmitters,
		.rowb = 0,

		.alpha = 0,
		.wave = &wave_alloc.wave,
	};

	if (initial) memcpy(&state, initial, sizeof state);
	else {
		memset(*state.wave, 0, sizeof *state.wave);

		for (int n = state.rowb; n <= state.rowa; ++n) {
			(*state.wave)[n][0] = initial_state[n];
		}
	}

	auto wave = *state.wave;

	float e_factor = precompute_e_factor(state.row1, state.row2);
	float f_factor = precompute_f_factor(state.row1, state.row2);
	float g_factor = precompute_g_factor(state.row1, state.row2);

	float time = 0;

	float tick_size = total_time / OUTPUT_COUNT;
	float next_write = 0; 

 	FILE *log = nullptr;

 	if (SAVE_TRAJECTORY && !expectation) {
   		char filename[100];
   		sprintf(filename, "trajectory-%d.txt", simulation_index);
   		log = fopen(filename, "wb");
 	}

	while (time < total_time) {
 		if (log && time + state.time_step > next_write) {
			complex float expectation[ExpectationOps] = {0};
			compute_expectation_values(wave, &state, expectation);

			fprintf(log, "%f", time);

			for (int op = 0; op < ExpectationOps; ++op) {
				fprintf(log, "\t%g\t%g", crealf(expectation[op]), cimagf(expectation[op]));
			}

			fprintf(log, "\n");
 			next_write += tick_size;
 		}

		float jump_table[JUMP_COUNT] = {0};
		state.gjx_expect = 0;
		state.ann_expect = 0;
		state.abs_expect = 0;
		state.ems_expect = 0;

		for (int n = state.rowb; n <= state.rowa; ++n) {
			for (int a = 0; a < CavityTruncation; ++a) {
				float norm = cnormf(wave[n][a]);

				int jpm = state.row1 - n; // J+M
				int jmm = n - state.row2; // J-M
				float m = 0.5f * (NumberOfEmitters - 2*n);

				jump_table[JUMP_SPIN_DEPHASING_SAME_J]  += norm * m * m;
				jump_table[JUMP_SPIN_DEPHASING_LOWER_J] += norm * jmm*jpm;
				jump_table[JUMP_SPIN_DEPHASING_UPPER_J] += norm * (jmm + 1)*(jpm + 1);

				jump_table[JUMP_SPIN_LOSS_SAME_J]  += norm * (jmm + 1)*(jpm);
				jump_table[JUMP_SPIN_LOSS_LOWER_J] += norm * (jpm - 1)*(jpm);
				jump_table[JUMP_SPIN_LOSS_UPPER_J] += norm * (jmm + 1)*(jmm + 2);

				jump_table[JUMP_SPIN_GAIN_SAME_J]  += norm * (jpm + 1)*(jmm);
				jump_table[JUMP_SPIN_GAIN_LOWER_J] += norm * (jmm - 1)*(jmm);
				jump_table[JUMP_SPIN_GAIN_UPPER_J] += norm * (jpm + 1)*(jpm + 2);

				jump_table[JUMP_SPIN_LOSS_PHOTON_GAIN_SAME_J]  += norm * (jmm + 1)*(jpm) * (a + 1);
				jump_table[JUMP_SPIN_LOSS_PHOTON_GAIN_LOWER_J] += norm * (jpm - 1)*(jpm) * (a + 1);
				jump_table[JUMP_SPIN_LOSS_PHOTON_GAIN_UPPER_J] += norm * (jmm + 1)*(jmm + 2) * (a + 1);

 				jump_table[JUMP_SPIN_GAIN_PHOTON_LOSS_SAME_J]  += norm * (jpm + 1)*(jmm) * a;
 				jump_table[JUMP_SPIN_GAIN_PHOTON_LOSS_LOWER_J] += norm * (jmm - 1)*(jmm) * a;
 				jump_table[JUMP_SPIN_GAIN_PHOTON_LOSS_UPPER_J] += norm * (jpm + 1)*(jpm + 2) * a;

				jump_table[JUMP_PHOTON_LOSS] += norm * a;

				if (UseDisplacement) {
					expectation_term(wave, &state, n, a);
				}

				if (a) {
					// inner product of annihilation operator
					complex float annihilation_inner = conjf(wave[n][a-1]) * wave[n][a] * sqrtf(a);

					state.ann_expect += annihilation_inner;
					state.abs_expect += annihilation_inner * n; // spin down
					state.ems_expect += annihilation_inner * (NumberOfEmitters - n); // spin up
				}
			}
		}

		// scale jump probabilities by loss rates
		jump_table[JUMP_COLLECTIVE_SPIN_DEPHASING] = jump_table[JUMP_SPIN_DEPHASING_SAME_J] * config.CollectiveDephasingRate;
		jump_table[JUMP_COLLECTIVE_SPIN_LOSS] = jump_table[JUMP_SPIN_LOSS_SAME_J] * config.CollectiveEmissionRate;
		jump_table[JUMP_COLLECTIVE_SPIN_GAIN] = jump_table[JUMP_SPIN_GAIN_SAME_J] * config.CollectivePumpingRate;
		jump_table[JUMP_SPIN_DEPHASING_SAME_J]  *= 4*config.DephasingRate * e_factor;
		jump_table[JUMP_SPIN_DEPHASING_LOWER_J] *= 4*config.DephasingRate * f_factor;
		jump_table[JUMP_SPIN_DEPHASING_UPPER_J] *= 4*config.DephasingRate * g_factor;
		jump_table[JUMP_SPIN_LOSS_SAME_J]  *= config.EmissionRate * e_factor;
		jump_table[JUMP_SPIN_LOSS_LOWER_J] *= config.EmissionRate * f_factor;
		jump_table[JUMP_SPIN_LOSS_UPPER_J] *= config.EmissionRate * g_factor;
		jump_table[JUMP_SPIN_GAIN_SAME_J]  *= config.PumpingRate * e_factor;
		jump_table[JUMP_SPIN_GAIN_LOWER_J] *= config.PumpingRate * f_factor;
		jump_table[JUMP_SPIN_GAIN_UPPER_J] *= config.PumpingRate * g_factor;
		jump_table[JUMP_SPIN_LOSS_PHOTON_GAIN_SAME_J]  *= config.CavityEmissionRate * e_factor;
		jump_table[JUMP_SPIN_LOSS_PHOTON_GAIN_LOWER_J] *= config.CavityEmissionRate * f_factor;
		jump_table[JUMP_SPIN_LOSS_PHOTON_GAIN_UPPER_J] *= config.CavityEmissionRate * g_factor;
 		jump_table[JUMP_SPIN_GAIN_PHOTON_LOSS_SAME_J]  *= config.CavityAbsorptionRate * e_factor;
 		jump_table[JUMP_SPIN_GAIN_PHOTON_LOSS_LOWER_J] *= config.CavityAbsorptionRate * f_factor;
 		jump_table[JUMP_SPIN_GAIN_PHOTON_LOSS_UPPER_J] *= config.CavityAbsorptionRate * g_factor;
		jump_table[JUMP_PHOTON_LOSS] *= config.PhotonLossRate;

		float max_factor = 1.0f; // min of 1 to guarantee max dt of tolerance
		for (int i = 0; i < JUMP_COUNT; ++i) max_factor = fmaxf(max_factor, jump_table[i]);

		state.time_step = config.JumpTolerance / max_factor;

		if (UseDisplacement) {
			complex float alpha_dot =
				-(I*config.PhotonEnergy + config.PhotonLossRate/2) * state.alpha
				-config.CavityAbsorptionRate/2 * state.abs_expect
				+config.CavityEmissionRate/2 * state.ems_expect
				-I * state.gjx_expect;

			if (expectation && time + state.time_step >= next_write) {
				*(expectation++) += conjf( state.ann_expect + state.alpha + (next_write - time) * alpha_dot ) * state.ttc_factor;
				next_write += tick_size;
			}

			state.alpha += alpha_dot * state.time_step;
		}

		time += state.time_step;

		// scale jump probabilities by dt
		for (int i = 0; i < JUMP_COUNT; ++i) jump_table[i] *= state.time_step;

		int choice = select_random_jump(jump_table);
		int row1_copy = state.row1;

		switch (choice) {
			case JUMP_COLLECTIVE_SPIN_DEPHASING: jump_spin_dephasing_same_j(wave, &state); break;
			case JUMP_COLLECTIVE_SPIN_LOSS: jump_spin_loss_same_j(wave, &state); break;
			case JUMP_COLLECTIVE_SPIN_GAIN: jump_spin_gain_same_j(wave, &state); break;

			case JUMP_SPIN_DEPHASING_SAME_J:  jump_spin_dephasing_same_j(wave, &state);  break;
			case JUMP_SPIN_DEPHASING_LOWER_J: jump_spin_dephasing_lower_j(wave, &state); break;
			case JUMP_SPIN_DEPHASING_UPPER_J: jump_spin_dephasing_upper_j(wave, &state); break;

			case JUMP_SPIN_LOSS_SAME_J:  jump_spin_loss_same_j(wave, &state);  break;
			case JUMP_SPIN_LOSS_LOWER_J: jump_spin_loss_lower_j(wave, &state); break;
			case JUMP_SPIN_LOSS_UPPER_J: jump_spin_loss_upper_j(wave, &state); break;

			case JUMP_SPIN_GAIN_SAME_J:  jump_spin_gain_same_j(wave, &state);  break;
			case JUMP_SPIN_GAIN_LOWER_J: jump_spin_gain_lower_j(wave, &state); break;
			case JUMP_SPIN_GAIN_UPPER_J: jump_spin_gain_upper_j(wave, &state); break;

			case JUMP_SPIN_LOSS_PHOTON_GAIN_SAME_J:
				jump_spin_loss_same_j(wave, &state); 
				jump_photon_gain(wave, &state);
				break;
			case JUMP_SPIN_LOSS_PHOTON_GAIN_LOWER_J:
				jump_spin_loss_lower_j(wave, &state);
				jump_photon_gain(wave, &state);
				break;
			case JUMP_SPIN_LOSS_PHOTON_GAIN_UPPER_J:
				jump_spin_loss_upper_j(wave, &state);
				jump_photon_gain(wave, &state);
				break;

 			case JUMP_SPIN_GAIN_PHOTON_LOSS_SAME_J:
				jump_spin_gain_same_j(wave, &state);
				jump_photon_loss(wave, &state);
				break;
 			case JUMP_SPIN_GAIN_PHOTON_LOSS_LOWER_J:
				jump_spin_gain_lower_j(wave, &state);
				jump_photon_loss(wave, &state);
				break;
 			case JUMP_SPIN_GAIN_PHOTON_LOSS_UPPER_J:
				jump_spin_gain_upper_j(wave, &state);
				jump_photon_loss(wave, &state);
				break;

			case JUMP_PHOTON_LOSS:
				if (UseDisplacement); // fallthough
				else { jump_photon_loss(wave, &state); break; }

			case EFFECTIVE_HAMILTONIAN:  evolve_under_H_eff(wave, &state);     break;
		}

		if (state.rowa > state.row1) state.rowa = state.row1;
		if (state.rowb < state.row2) state.rowb = state.row2;

		if (state.row1 != row1_copy) {
			e_factor = precompute_e_factor(state.row1, state.row2);
			f_factor = precompute_f_factor(state.row1, state.row2);
			g_factor = precompute_g_factor(state.row1, state.row2);
		}

		normalize_state(wave, &state);

		// expand bounds of spin-space as necessary
		float min_inner = 0, min_outer = 0;
		float max_inner = 0, max_outer = 0;

		for (int a = 0; a < CavityTruncation; ++a) {
			min_outer += cnormf((*state.wave)[state.rowb][a]);
			max_outer += cnormf((*state.wave)[state.rowa][a]);
			min_inner += cnormf((*state.wave)[state.rowb + 1][a]);
			max_inner += cnormf((*state.wave)[state.rowa - 1][a]);
		}

		if ((min_inner + min_outer) < config.ShrinkTolerance) state.rowb += 1;
		if ((max_inner + max_outer) < config.ShrinkTolerance) state.rowa -= 1;

		if (min_outer > config.ShrinkTolerance && state.rowb > state.row2) {
			state.rowb -= 1;
			memset((*state.wave)[state.rowb], 0, sizeof (*state.wave)[0]);
		}

		if (max_outer > config.ShrinkTolerance && state.rowa < state.row1) {
			state.rowa += 1;
			memset((*state.wave)[state.rowa], 0, sizeof (*state.wave)[0]);
		}

	}

   	if (log) fclose(log);
	return state;
}


/*
// Two time correlation as described by Molmer et al.
//
void two_time_correlation(complex float *expectation) {
	static thread_local struct WaveVectorAllocation second_wave;

	struct TrajectoryState trajectory = simulate_trajectory(STEADY_STATE_TIME, nullptr, nullptr);
	auto steady_state_wave = *trajectory.wave;

	trajectory.wave = &second_wave.wave;
	auto wave = *trajectory.wave;

	// split wave function into 4 states, and then evolve trajectories from there...
	//  in this case hat(a) = hat(b) + alpha

	complex float factor = 1;

	for (int split = 0; split < 4; ++split) { // cycle through 1, i, -1, -i
		for (int repeat = 0; repeat < MOLMER_REPEATS; ++repeat) {
			memcpy(wave, steady_state_wave, sizeof(WaveVector));

			for (int n = trajectory.row2; n <= trajectory.row1; ++n) {
				for (int a = 0; a < CavityTruncation; ++a) {
					if (a) wave[n][a - 1] += factor * sqrtf(a) * wave[n][a];
					wave[n][a] *= (1 + factor * trajectory.alpha);
				}
			}

			float norm = normalize_state(wave, &trajectory);
			trajectory.ttc_factor = conjf(factor) * norm;

			simulate_trajectory(CORRELATION_TIME, &trajectory, expectation);
		}

		factor *= I;
	}
}
*/


// Multithreading code.
//
atomic(int) thread_id;
atomic(int) thread_pool;
atomic(int) threads_complete;
atomic(int) total_millis;

#define CLEAR_LINE "\r\x1b[2K"

void *thread_worker(void *output) {
	auto expectation = (complex float *)output;

	int id = atomic_fetch_add(&thread_id, 1);
	int next;

	set_random_seed(id);

	while ((next = atomic_fetch_add(&thread_pool, -1)) > 0) {
		double begin = get_time_from_os();
		simulation_index = next;

		simulate_trajectory(config.TimeSpan, nullptr, nullptr);
		double end = get_time_from_os();

		int millis = (int)(1000.0f * (end - begin));
		float total_seconds = (atomic_fetch_add(&total_millis, millis) + millis) / 1000.0f;

		int complete = atomic_fetch_add(&threads_complete, 1) + 1;
		float average_seconds = total_seconds / complete;

		fprintf(stderr, CLEAR_LINE "Trajectory [%d/%d] completed, average time: %.3f seconds.", complete, config.TrajectoryCount, average_seconds);
	}

	return expectation;
}


void run_trajectories(complex float *inital_state_data) {
	// static complex float expectation[THREAD_COUNT][OUTPUT_COUNT + 1];
	thread_id = 0;
	thread_pool = config.TrajectoryCount;
	threads_complete = 0;
	initial_state = inital_state_data;
	total_millis = 0;

	pthread_t threads[ThreadCount];
	fprintf(stderr, "Running backend with UseDisplacement = %s\n", UseDisplacement ? "Yes" : "No");

	for (int i = 0; i < ThreadCount; ++i) pthread_create(&threads[i], nullptr, thread_worker, nullptr); //expectation[i]);
	for (int i = 0; i < ThreadCount; ++i) pthread_join(threads[i], nullptr);
	fprintf(stderr, "\n");

/*
	if (!MOLMER_REPEATS) return 0;

	char filepath[100];
	sprintf(filepath, "log-%d-%g.txt", N, OPTION_UP);

	auto log = fopen(filepath, "wb");

	for (int i = 0; i <= OUTPUT_COUNT; ++i) {
		complex float average = 0;
		for (int t = 0; t < THREAD_COUNT; ++t) average += expectation[t][i];
		average /= (4 * MOLMER_REPEATS * TRAJECTORY_COUNT);

		fprintf(log, "%g+%gj\n", crealf(average), cimagf(average));
	}

	fclose(log);
*/
}

