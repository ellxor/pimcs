inline void hamiltonian_term(WaveVector dest, WaveVector source, struct TrajectoryState *state, int n, int a) {
	complex float coeff = I * state->time_step * source[n][a];
	float m = 0.5f * (config.NumberOfEmitters - 2*n);
	int jpm = state->row1 - n;
	int jmm = n - state->row2;

	dest[n][a] -= coeff * config.PhotonEnergy * (a + cnormf(state->alpha));
	dest[n + 0][a] -= coeff * (1.0f + I*0.0f) * (m + 0);
	dest[n + -1][a]     -= coeff * (0.04024922359499621f + I*0.0f) * sqrtf((jpm + 1 - 0) * (jmm + 0)) * state->alpha;
	dest[n + -1][a - 1] -= coeff * (0.04024922359499621f + I*0.0f) * sqrtf((jpm + 1 - 0) * (jmm + 0)) * sqrtf(a);
	dest[n + 1][a]     -= coeff * (0.04024922359499621f + I*0.0f) * sqrtf((jmm + 1 + 0) * (jpm - 0)) * state->alpha;
	dest[n + 1][a - 1] -= coeff * (0.04024922359499621f + I*0.0f) * sqrtf((jmm + 1 + 0) * (jpm - 0)) * sqrtf(a);
	dest[n + -1][a]     -= coeff * (0.04024922359499621f + I*0.0f) * sqrtf((jpm + 1 - 0) * (jmm + 0)) * conjf(state->alpha);
	dest[n + -1][a + 1] -= coeff * (0.04024922359499621f + I*0.0f) * sqrtf((jpm + 1 - 0) * (jmm + 0)) * sqrtf((a + 1) % config.CavityTruncation);
	dest[n + 1][a]     -= coeff * (0.04024922359499621f + I*0.0f) * sqrtf((jmm + 1 + 0) * (jpm - 0)) * conjf(state->alpha);
	dest[n + 1][a + 1] -= coeff * (0.04024922359499621f + I*0.0f) * sqrtf((jmm + 1 + 0) * (jpm - 0)) * sqrtf((a + 1) % config.CavityTruncation);
}

inline void expectation_term(WaveVector wave, struct TrajectoryState *state, int n, int a) {
	float m = 0.5f * (config.NumberOfEmitters - 2*n);
	int jpm = state->row1 - n;
	int jmm = n - state->row2;

	if ((n + -1) >= state->rowb) state->gjx_expect += conjf(wave[n + -1][a]) * wave[n][a] * (0.04024922359499621f + I*0.0f) * sqrtf((jpm + 1 - 0) * (jmm + 0));
	if ((n + 1) <= state->rowa) state->gjx_expect += conjf(wave[n + 1][a]) * wave[n][a] * (0.04024922359499621f + I*0.0f) * sqrtf((jmm + 1 + 0) * (jpm - 0));
}

inline void compute_expectation_values(WaveVector wave, struct TrajectoryState *state, complex float *expect) {
	for (int n = state->rowb; n <= state->rowa; ++n) {
		for (int a = 0; a < config.CavityTruncation; ++a) {
			float m = 0.5f * (config.NumberOfEmitters - 2*n);
			int jpm = state->row1 - n;
			int jmm = n - state->row2;

			if ((n + 0) <= state->rowa && (a + 0) < config.CavityTruncation) expect[0] += conjf(wave[n + 0][a + 0]) * wave[n][a] * sqrtf(a + 0) * sqrtf(a + 1 + -1);
			if ((n + 0) <= state->rowa && (a + 1) < config.CavityTruncation) expect[0] += conjf(wave[n + 0][a + 1]) * wave[n][a] * state->alpha * sqrtf(a + 1 + 0);
			if ((n + 0) <= state->rowa && (a + -1) >= 0) expect[0] += conjf(wave[n + 0][a + -1]) * wave[n][a] * sqrtf(a + 0) * conjf(state->alpha);
			if ((n + 0) <= state->rowa && (a + 0) < config.CavityTruncation) expect[0] += conjf(wave[n + 0][a + 0]) * wave[n][a] * state->alpha * conjf(state->alpha);
			if ((n + 0) <= state->rowa && (a + 0) < config.CavityTruncation) expect[1] += conjf(wave[n + 0][a + 0]) * wave[n][a] * (m + 0);
		}
	}
}

