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
                weights.append(f"sqrt((jpm + 1 - {spin_index}) * (jmm + {spin_index}))")
                spin_index -= 1
            case PIOperatorKind.Jm:
                weights.append(f"sqrt((jmm + 1 + {spin_index}) * (jpm - {spin_index}))")
                spin_index += 1
            case PIOperatorKind.A:
                weights.append(f"sqrt(a + {boson_index})")
                boson_index -= 1
            case PIOperatorKind.Ad:
                weights.append(f"sqrt(a + 1 + {boson_index})")
                boson_index += 1
            case PIOperatorKind.Ap:
                weights.append(f"state->alpha")
            case PIOperatorKind.As:
                weights.append(f"conj(state->alpha)")

    return spin_index, boson_index, " * ".join(weights)



def generate_expectation_values(expect) -> str:
    string_builder = ""

    # function definition, loop over states and terms needed for z,± basis
    string_builder += (
        "void compute_expectation_values(struct TrajectoryState *state, complex double *expect, int64 n, int64 a) {\n"
        "\tdouble m = 0.5f * (NumberOfEmitters - 2*n);\n"
        "\tint64 jpm = state->row1 - n;\n"
        "\tint64 jmm = n - state->row2;\n"
    )      
 
    for i, op in enumerate(expect):
        collected = to_sum_of_products(op, 0)

        for coeff, spin, boson, tfactor in collected:
            assert len(tfactor) == 1, "observables are not currently time-dependent"

            spin_index, boson_index, factor = ops_to_factor(spin + boson)
            if spin_index == 0 and boson_index == 0:
                string_builder += f"\texpect[{i}] += ({coeff.real} + I*{coeff.imag}) * {factor};\n"

    string_builder += "}\n\n"
    return string_builder


def generate_backend_code(H, expect, tlist, displace: bool, two_time_correlation: bool):
    assert not two_time_correlation, "two time correlations are not supported for diagonal optimization"
    string_builder = generate_expectation_values(expect)
    return string_builder, 0, 0, []


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
    hash_id = random.randint(0, 2**64 - 1)
    hash_str = f"{hash_id:x}"

    assert os.system(f"cc -c -o main-{hash_str}.o -std=c11 -pthread -fPIC -O3 -march=native -ffast-math pimcs/c_backend/main_diagonal_heff.c") == 0
    output = f"./main-{hash_str}.so"

    assert os.system(f"cc -fPIC -shared -o {output} main-{hash_str}.o -lm -pthread") == 0
    return output, hash_id

