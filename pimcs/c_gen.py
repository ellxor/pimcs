import ctypes, os, random
from .dicke import Dicke
from .operators import *


def ops_to_factor(ops) -> tuple[int, int, str]:
    if len(ops) == 0:
        return 0, 0, "1"

    spin_index = 0
    boson_index = 0
    weights = []

    for op in reversed(ops):
        match op:
            case PIOperatorKind.Jz:
                weights.append(f"(m + {spin_index})")
            case PIOperatorKind.Jp:
                weights.append(f"sqrtf((jpm + 1 - {spin_index}) * (jmm + {spin_index}))")
                spin_index -= 1
            case PIOperatorKind.Jm:
                weights.append(f"sqrtf((jmm + 1 + {spin_index}) * (jpm - {spin_index}))")
                spin_index += 1
            case PIOperatorKind.A:
                weights.append(f"sqrtf(a + {boson_index})")
                boson_index -= 1
            case PIOperatorKind.Ad:
                weights.append(f"sqrtf(a + 1 + {boson_index})")
                boson_index += 1
            case PIOperatorKind.Ap:
                weights.append(f"state->alpha")
            case PIOperatorKind.As:
                weights.append(f"conjf(state->alpha)")

    return spin_index, boson_index, " * ".join(weights)



def generate_hamiltonian_term(terms):
    max_spin_index = 0
    max_boson_index = 0
    string_builder = ""

    # function definition and terms needed for z,± basis, and photon-energy term (always included)
    string_builder += (
        "void hamiltonian_term(WaveVector dest, WaveVector source, struct TrajectoryState *state, int64 n, int64 a) {\n"
        "\tcomplex float coeff = I * state->time_step * source[n][a];\n"
        "\tfloat m = 0.5f * (NumberOfEmitters - 2*n);\n"
        "\tint64 jpm = state->row1 - n;\n"
        "\tint64 jmm = n - state->row2;\n"
        "\tsize_t tindex = (size_t)(TsLength * (state->time - config.StartTime) / (config.EndTime - config.StartTime));\n"
	"\tif (tindex >= TsLength) tindex = TsLength - 1;\n\n"
    )

    tfuncs = []
    tid = 0

    for coeff, spins, bosons, tfactor in terms:
        spin_index, boson_index, factor = ops_to_factor(spins + bosons)

        max_spin_index = max(max_spin_index, spin_index)
        max_boson_index = max(max_boson_index, boson_index)
        cond = f"if (a + {boson_index} < CavityTruncation) " if boson_index > 0 else ""

        tf_string = "1"
        if len(tfactor) > 1:
            tfuncs.append(tfactor)
            tf_string = f"tfunc[{tid}][tindex]"
            tid += 1

        string_builder += f"\t{cond}dest[n + {spin_index}][a + {boson_index}] -= coeff * ({coeff.real}f + I*{coeff.imag}f) * {factor} * {tf_string};\n"

    string_builder += "}\n\n" # terminate function
    return string_builder, max_spin_index, max_boson_index, tfuncs



def generate_expectation_values(expect) -> str:
    string_builder = ""

    # function definition, loop over states and terms needed for z,± basis
    string_builder += (
        "void compute_expectation_values(WaveVector wave, struct TrajectoryState *state, complex float *expect) {\n"
        "\tfor (int64 n = state->rowb; n <= state->rowa; ++n) {\n"
        "\t\tfor (int64 a = state->mina; a <= state->maxa; ++a) {\n"
        "\t\t\tfloat m = 0.5f * (NumberOfEmitters - 2*n);\n"
        "\t\t\tint64 jpm = state->row1 - n;\n"
        "\t\t\tint64 jmm = n - state->row2;\n"
    )      
 
    for i, op in enumerate(expect):
        collected = to_sum_of_products(op, 0)

        for coeff, spin, boson, tfactor in collected:
            assert len(tfactor) == 1, "observables are not currently time-dependent"

            spin_index, boson_index, factor = ops_to_factor(spin + boson)
            cond = " && ".join([
                f"(n + {spin_index}) " + (">= state->rowb" if spin_index < 0 else "<= state->rowa"),
                f"(a + {boson_index}) " + (">= 0" if boson_index < 0 else "< CavityTruncation"),
            ])
            string_builder += f"\t\t\tif ({cond}) expect[{i}] += ({coeff.real}f + I*{coeff.imag}f) * conjf(wave[n + {spin_index}][a + {boson_index}]) * wave[n][a] * {factor};\n"

    string_builder += "\t\t}\n\t}\n}\n\n"
    return string_builder


def generate_alpha_eom(H):
    terms = to_sum_of_products(H, None)
    string_builder = ""

    # for all terms:
    # - use normal ordering in expansion
    # - then for each term (a†)^k -> k (a†)^(k - 1) [+ a...]
    # - and the a... terms all commute so the other term disappears [A,BC] = [A,B]C + B[A,C]

    string_builder += (
        "complex float compute_alpha_eom(WaveVector wave, struct TrajectoryState *state) {\n"
        "\tcomplex float expect = 0;\n\n"
        "\tfor (int64 n = state->rowb; n <= state->rowa; ++n) {\n"
        "\t\tfor (int64 a = state->mina; a <= state->maxa; ++a) {\n"
        "\t\t\tfloat m = 0.5f * (NumberOfEmitters - 2*n);\n"
        "\t\t\tint64 jpm = state->row1 - n;\n"
        "\t\t\tint64 jmm = n - state->row2;\n"
    )      
 
    for coeff, spin, boson, tfactor in terms:
        if len(tfactor) > 1:
            raise ValueError(f"Time-dependent Hamiltonian are not supported with displacement yet!")

        if len(boson) > 2:
            raise ValueError(f"This Hamiltonian is not yet supported with displacement enabled (only up to quadratic in boson mode)!")

        match boson:
            case [PIOperatorKind.A, PIOperatorKind.Ad] | [PIOperatorKind.Ad, PIOperatorKind.A]:
                assert len(spin) == 0, "not yet implemented!"
                commutator = "state->alpha" # a
            case [PIOperatorKind.Ad, PIOperatorKind.Ad]:
                assert False, "not yet implemented!"
                commutator = "2 * conjf(state->alpha)" #  2a†
            case [PIOperatorKind.Ad]:
                commutator = "1"
            case _:
                continue

        spin_index, _, factor = ops_to_factor(spin)
        boson_index = 0 # TODO: fix with above

        cond = " && ".join([
            f"(n + {spin_index}) " + (">= state->rowb" if spin_index < 0 else "<= state->rowa"),
            f"(a + {boson_index}) " + (">= 0" if boson_index < 0 else "< CavityTruncation"),
        ])

        string_builder += f"\t\t\tif ({cond}) expect += ({coeff.real}f + I*{coeff.imag}f) * conjf(wave[n + {spin_index}][a + {boson_index}]) * wave[n][a] * {factor} * {commutator};\n"

    string_builder += "\t\t}\n\t}\n\n\treturn expect;\n}\n\n"
    return string_builder


def generate_backend_code(H, expect, tlist, displace: bool):
    extra = ""

    if displace:
        extra = generate_alpha_eom(H)
        H = H.displace()
        expect = [op.displace() for op in expect]

    collected = to_sum_of_products(H, tlist)

    string_builder, max_spin_index, max_boson_index, tfuncs = generate_hamiltonian_term(collected)
    string_builder += generate_expectation_values(expect)
    string_builder += extra

    return string_builder, max_spin_index, max_boson_index, tfuncs


def generate_config(system: Dicke, boson_dim: int, tspan: [float], e_count: int, ntraj: int,
                    ncpu: int, jtol: float, stol: float, spin_width: int, boson_width: int, output_count: int, rkpoly: int, ts: int, displace: bool) -> str:
    string_builder = ""

    # constant integral values used for array lengths
    string_builder += "enum {\n";
    string_builder += f"\tNumberOfEmitters = {system.N},\n"
    string_builder += f"\tCavityTruncation = {boson_dim},\n"
    string_builder += f"\tExpectationOps   = {e_count},\n"
    string_builder += f"\tThreadCount      = {ncpu},\n"
    string_builder += f"\tSpinWidth        = {spin_width},\n"
    string_builder += f"\tBosonWidth       = {boson_width},\n"
    string_builder += f"\tOutputCount      = {output_count},\n"
    string_builder += f"\tTsLength         = {ts},\n"
    string_builder += f"\tUseDisplacement  = {int(displace)},\n"
    string_builder += "};\n\n"

    string_builder += "static const struct Config config = {\n";

    string_builder += f"\t.PhotonLossRate          = {system.cavity_loss},\n"
    string_builder += f"\t.DephasingRate           = {system.dephasing},\n"
    string_builder += f"\t.EmissionRate            = {system.emission},\n"
    string_builder += f"\t.PumpingRate             = {system.pumping},\n"
    string_builder += f"\t.CollectiveDephasingRate = {system.collective_dephasing},\n"
    string_builder += f"\t.CollectiveEmissionRate  = {system.collective_emission},\n"
    string_builder += f"\t.CollectivePumpingRate   = {system.collective_pumping},\n"
    string_builder += f"\t.CavityEmissionRate      = {system.cavity_emission},\n"
    string_builder += f"\t.CavityAbsorptionRate    = {system.cavity_absorption},\n"

    string_builder += f"\t.StartTime       = {tspan[0]},\n"
    string_builder += f"\t.EndTime         = {tspan[-1]},\n"
    string_builder += f"\t.TrajectoryCount = {ntraj},\n"
    string_builder += f"\t.RungeKuttaPoly  = {rkpoly},\n"
    string_builder += f"\t.JumpTolerance   = {jtol},\n"
    string_builder += f"\t.ShrinkTolerance = {stol},\n"

    string_builder += "};\n"
    return string_builder



def build_executable():
    assert os.system("cc -c -std=c11 -pthread -fPIC -Werror -O3 -march=native -ffast-math pimcs/c_backend/main.c") == 0

    hash_id = random.randint(0, 2**64 - 1)
    output = f"./main-{hash_id:x}.so"

    assert os.system(f"cc -fPIC -shared -o {output} main.o -lm -pthread") == 0
    return output, hash_id

