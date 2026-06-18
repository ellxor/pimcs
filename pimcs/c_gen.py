from pimcs.operators import *
from pimcs.dicke import Dicke
import ctypes, os, random


def generate_hamiltonian_term(bare_terms, linear_terms, linear_dagger_terms) -> str:
    max_index = 0
    string_builder = ""

    # function definition and terms needed for z,± basis, and photon-energy term (always included)
    string_builder += (
        "void hamiltonian_term(WaveVector dest, WaveVector source, struct TrajectoryState *state, int n, int a) {\n"
        "\tcomplex float coeff = I * state->time_step * source[n][a];\n"
        "\tfloat m = 0.5f * (NumberOfEmitters - 2*n);\n"
        "\tint jpm = state->row1 - n;\n"
        "\tint jmm = n - state->row2;\n\n"
        "\tdest[n][a] -= coeff * config.PhotonEnergy * (a + cnormf(state->alpha));\n"
    )

    for coeff, spins in bare_terms:
        index, _,  factor = ops_to_factor(spins)
        string_builder += f"\tdest[n + {index}][a] -= coeff * ({coeff.real}f + I*{coeff.imag}f) * {factor};\n"
        max_index = max(max_index, index)
    
    for coeff, spins in linear_terms:
        index, _, factor = ops_to_factor(spins)
        string_builder += f"\tdest[n + {index}][a]     -= coeff * ({coeff.real}f + I*{coeff.imag}f) * {factor} * state->alpha;\n"
        string_builder += f"\tdest[n + {index}][a - 1] -= coeff * ({coeff.real}f + I*{coeff.imag}f) * {factor} * sqrtf(a);\n"
        max_index = max(max_index, index)

    for coeff, spins in linear_dagger_terms:
        index, _, factor = ops_to_factor(spins)
        string_builder += f"\tdest[n + {index}][a]     -= coeff * ({coeff.real}f + I*{coeff.imag}f) * {factor} * conjf(state->alpha);\n"
        string_builder += f"\tdest[n + {index}][a + 1] -= coeff * ({coeff.real}f + I*{coeff.imag}f) * {factor} * sqrtf((a + 1) % CavityTruncation);\n"
        max_index = max(max_index, index)

    string_builder += f"\tdest[n][a - 1] += coeff * conjf(state->gjx_expect) * sqrtf(a);\n"
    string_builder += f"\tdest[n][a + 1] += coeff * state->gjx_expect * sqrtf(a + 1);\n"
    string_builder += "}\n\n" # terminate function
    padding = max_index + 1

    return string_builder, padding



def generate_equation_of_motion_term(linear_dagger_terms) -> str:
    string_builder = ""

    # function definition and terms needed for z,± basis
    string_builder += (
        "void expectation_term(WaveVector wave, struct TrajectoryState *state, int n, int a) {\n"
        "\tfloat m = 0.5f * (NumberOfEmitters - 2*n);\n"
        "\tint jpm = state->row1 - n;\n"
        "\tint jmm = n - state->row2;\n\n"
    )

    for coeff, spin in linear_dagger_terms:
        index, _, factor = ops_to_factor(spin)
        cond = f"(n + {index}) " + (">= state->rowb" if index < 0 else "<= state->rowa")
        string_builder += f"\tif ({cond}) state->gjx_expect += conjf(wave[n + {index}][a]) * wave[n][a] * ({coeff.real}f + I*{coeff.imag}f) * {factor};\n"

    string_builder += "}\n\n" # terminate function
    return string_builder



def generate_expectation_values(expect, displace: bool) -> str:
    string_builder = ""

    # function definition, loop over states and terms needed for z,± basis
    string_builder += (
        "void compute_expectation_values(WaveVector wave, struct TrajectoryState *state, complex float *expect) {\n"
        "\tfor (int n = state->rowb; n <= state->rowa; ++n) {\n"
        "\t\tfor (int a = 0; a < CavityTruncation; ++a) {\n"
        "\t\t\tfloat m = 0.5f * (NumberOfEmitters - 2*n);\n"
        "\t\t\tint jpm = state->row1 - n;\n"
        "\t\t\tint jmm = n - state->row2;\n\n"
    )      
    
    for i, op in enumerate(expect):
        collected = to_sum_of_products(op.displace())

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

    bare_terms = []
    linear_terms = []
    linear_dagger_terms = []
    boson_energy = 0

    for coeff, spins, bosons in collected:
        assert len(bosons) <= 2, "Hamiltonian is not yet supported!"
    
        if len(bosons) == 2:
            assert len(spins) == 0, "Hamiltonian is not yet supported!"
            assert bosons[0] == PIOperatorKind.Ad and bosons[1] == PIOperatorKind.A, "Hamiltonian is not yet supported!"

            assert isclose(coeff.imag, 0), "Energy of bosonic mode must be real"
            boson_energy = float(coeff.real)
    
        elif len(bosons) == 1:
            if bosons[0] == PIOperatorKind.A:
                linear_terms.append((coeff, spins))
            else:
                linear_dagger_terms.append((coeff, spins))
    
        else:
            bare_terms.append((coeff, spins)) 
    
    
    code, padding = generate_hamiltonian_term(bare_terms, linear_terms, linear_dagger_terms)

    string_builder = ""
    string_builder += code
    string_builder += generate_equation_of_motion_term(linear_dagger_terms)
    string_builder += generate_expectation_values(expect, displace)

    return boson_energy, padding, string_builder



def generate_config(system: Dicke, boson_dim: int, tspan: [float], e_count: int, ntraj: int,
                    ncpu: int, boson_energy: float, jtol: float, padding: int, disable_displ: bool, output_count: int) -> str:
    displacement_flag = "false" if disable_displ else "true"
    string_builder = ""

    # constant integral values used for array lengths
    string_builder += "enum {\n";
    string_builder += f"\tNumberOfEmitters = {system.N},\n"
    string_builder += f"\tCavityTruncation = {boson_dim},\n"
    string_builder += f"\tExpectationOps   = {e_count},\n"
    string_builder += f"\tThreadCount      = {ncpu},\n"
    string_builder += f"\tPaddingWidth     = {padding},\n"
    string_builder += f"\tUseDisplacement  = {displacement_flag},\n"
    string_builder += f"\tOutputCount      = {output_count},\n"
    string_builder += "};\n\n"

    string_builder += "static const struct Config config = {\n";

    #string_builder += f"\t.NumberOfEmitters = {system.N},\n"
    #string_builder += f"\t.CavityTruncation = {boson_dim},\n"
    #string_builder += f"\t.ExpectationOps   = {e_count},\n"
    string_builder += f"\t.TimeSpan         = {tspan[-1]}f,\n"

    string_builder += f"\t.PhotonEnergy            = (float){boson_energy},\n"
    string_builder += f"\t.PhotonLossRate          = (float){system.cavity_loss},\n"
    string_builder += f"\t.DephasingRate           = (float){system.dephasing},\n"
    string_builder += f"\t.EmissionRate            = (float){system.emission},\n"
    string_builder += f"\t.PumpingRate             = (float){system.pumping},\n"
    string_builder += f"\t.CollectiveDephasingRate = (float){system.collective_dephasing},\n"
    string_builder += f"\t.CollectiveEmissionRate  = (float){system.collective_emission},\n"
    string_builder += f"\t.CollectivePumpingRate   = (float){system.collective_pumping},\n"
    string_builder += f"\t.CavityEmissionRate      = (float){system.cavity_emission},\n"
    string_builder += f"\t.CavityAbsorptionRate    = (float){system.cavity_absorption},\n"

    # string_builder += f"\t.ThreadCount     = {ncpu},\n"
    string_builder += f"\t.TrajectoryCount = {ntraj},\n"
    string_builder += f"\t.RungeKuttaPoly  = {4},\n"
    string_builder += f"\t.JumpTolerance   = {jtol}f,\n"
    string_builder += f"\t.ShrinkTolerance = {1e-20}f,\n"

    string_builder += "};\n"
    return string_builder


def build_executable():
    assert os.system("cc -c -std=c11 -pthread -fPIC -O3 -march=native -ffast-math pimcs/c_backend/main.c") == 0

    hash_id = random.randint(0, 2**64 - 1)
    output = f"./main-{hash_id:x}.so"

    assert os.system(f"cc -fPIC -shared -o {output} main.o -lm -pthread") == 0
    return ctypes.CDLL(output), hash_id

