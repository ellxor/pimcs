import ctypes, os, random
from .operators import *
from .dicke import Dicke


def generate_hamiltonian_term(terms) -> str:
    max_index = 0
    string_builder = ""

    # function definition and terms needed for z,± basis, and photon-energy term (always included)
    string_builder += (
        "void hamiltonian_term(WaveVector dest, WaveVector source, struct TrajectoryState *state, int n, int a) {\n"
        "\tcomplex float coeff = I * state->time_step * source[n][a];\n"
        "\tfloat m = 0.5f * (NumberOfEmitters - 2*n);\n"
        "\tint jpm = state->row1 - n;\n"
        "\tint jmm = n - state->row2;\n\n"
    )

    for coeff, spins, bosons in terms:
        spin_index, photon_index,  factor = ops_to_factor(spins + bosons)
        cond = f"if (a + {photon_index} < CavityTruncation) " if photon_index > 0 else ""

        string_builder += f"\t{cond}dest[n + {spin_index}][a + {photon_index}] -= coeff * ({coeff.real}f + I*{coeff.imag}f) * {factor};\n"
        max_index = max(max_index, photon_index)

    string_builder += "}\n\n" # terminate function
    padding = max_index + 1

    return string_builder, padding



def generate_expectation_values(expect) -> str:
    string_builder = ""

    # function definition, loop over states and terms needed for z,± basis
    string_builder += (
        "void compute_expectation_values(WaveVector wave, struct TrajectoryState *state, complex float *expect) {\n"
        "\tfor (int n = state->rowb; n <= state->rowa; ++n) {\n"
        "\t\tfor (int a = state->mina; a <= state->maxa; ++a) {\n"
        "\t\t\tfloat m = 0.5f * (NumberOfEmitters - 2*n);\n"
        "\t\t\tint jpm = state->row1 - n;\n"
        "\t\t\tint jmm = n - state->row2;\n\n"
    )      
    
    for i, op in enumerate(expect):
        collected = to_sum_of_products(op)

        for coeff, spin, boson in collected:
            spin_index, boson_index, factor = ops_to_factor(spin + boson)
            cond = " && ".join([
                f"(n + {spin_index}) " + (">= state->rowb" if spin_index < 0 else "<= state->rowa"),
                f"(a + {boson_index}) " + (">= 0" if boson_index < 0 else "< CavityTruncation"),
            ])
            string_builder += f"\t\t\tif ({cond}) expect[{i}] += conjf(wave[n + {spin_index}][a + {boson_index}]) * wave[n][a] * {factor};\n"

    string_builder += "\t\t}\n\t}\n}\n\n"
    return string_builder



def generate_backend_code(H, expect, displace: bool) -> tuple[float, str]:
    collected = to_sum_of_products(H)

    string_builder, padding = generate_hamiltonian_term(collected)
    string_builder += generate_expectation_values(expect)

    return padding, string_builder



def generate_config(system: Dicke, boson_dim: int, tspan: [float], e_count: int, ntraj: int,
                    ncpu: int, jtol: float, stol: float, padding: int, disable_displ: bool, output_count: int) -> str:
    string_builder = ""

    # constant integral values used for array lengths
    string_builder += "enum {\n";
    string_builder += f"\tNumberOfEmitters = {system.N},\n"
    string_builder += f"\tCavityTruncation = {boson_dim},\n"
    string_builder += f"\tExpectationOps   = {e_count},\n"
    string_builder += f"\tThreadCount      = {ncpu},\n"
    string_builder += f"\tPaddingWidth     = {padding},\n"
    string_builder += f"\tOutputCount      = {output_count},\n"
    string_builder += "};\n\n"

    string_builder += "static const struct Config config = {\n";

    #string_builder += f"\t.NumberOfEmitters = {system.N},\n"
    #string_builder += f"\t.CavityTruncation = {boson_dim},\n"
    #string_builder += f"\t.ExpectationOps   = {e_count},\n"
    string_builder += f"\t.TimeSpan         = {tspan[-1]}f,\n"

    string_builder += f"\t.PhotonLossRate          = (float){system.cavity_loss},\n"
    string_builder += f"\t.DephasingRate           = (float){system.dephasing},\n"
    string_builder += f"\t.EmissionRate            = (float){system.emission},\n"
    string_builder += f"\t.PumpingRate             = (float){system.pumping},\n"
    string_builder += f"\t.CollectiveDephasingRate = (float){system.collective_dephasing},\n"
    string_builder += f"\t.CollectiveEmissionRate  = (float){system.collective_emission},\n"
    string_builder += f"\t.CollectivePumpingRate   = (float){system.collective_pumping},\n"
    string_builder += f"\t.CavityEmissionRate      = (float){system.cavity_emission},\n"
    string_builder += f"\t.CavityAbsorptionRate    = (float){system.cavity_absorption},\n"

    string_builder += f"\t.TrajectoryCount = {ntraj},\n"
    string_builder += f"\t.RungeKuttaPoly  = {4},\n"
    string_builder += f"\t.JumpTolerance   = {jtol}f,\n"
    string_builder += f"\t.ShrinkTolerance = {stol}f,\n"

    string_builder += "};\n"
    return string_builder



def build_executable():
    assert os.system("cc -c -std=c11 -pthread -fPIC -O3 -march=native -ffast-math pimcs/c_backend/main.c") == 0

    hash_id = random.randint(0, 2**64 - 1)
    output = f"./main-{hash_id:x}.so"

    assert os.system(f"cc -fPIC -shared -o {output} main.o -lm -pthread") == 0
    return ctypes.CDLL(output), hash_id

