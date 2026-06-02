import numpy as np

from spin_frustration_updated import Bond, SpinSystem
from spin_frustration_main import (
    inspect_scan_point,
    plot_coupling_scan,
    plot_kahn_spin_population_scan,
    print_ground_state_report,
    print_spectrum_summary,
)


##______________________________________________________________________________###

two_site_system = SpinSystem(
    spins=[1/2, 1/2],
    bonds=[Bond(0, 1, J=-1.0)],
)

triangle_half_spin = SpinSystem(
    spins=[1/2, 1/2, 1/2],
    #spins=[1, 1, 1],
    bonds=[
        Bond(0, 1, J=500),
        Bond(1, 2, J=500),
        Bond(2, 0, J=500),
    ],
)

square_half_spin = SpinSystem(
    spins=[1/2, 1/2, 1/2, 1/2],
    bonds=[
        Bond(0, 1, J=500),
        Bond(1, 2, J=500),
        Bond(2, 3, J=500),
        Bond(3, 0, J=500),
    ],
)

square_with_diag_half_spin = SpinSystem(
    spins=[1/2, 1/2, 1/2, 1/2],
    bonds=[
        Bond(0, 1, J=100.0),
        Bond(1, 2, J=100.0),
        Bond(2, 3, J=100.0),
        Bond(3, 0, J=100.0),
        Bond(0, 2, J=100.0),
        Bond(1, 3, J=100.0),
    ],
)

triangle_with_DM_half_spin = SpinSystem(
    spins=[1/2, 1/2, 1/2],
    bonds=[
        Bond(0, 1, J=500.0, D=(0.0, 0.0, 50)),
        Bond(1, 2, J=500.0, D=(0.0, 0.0, 50)),
        Bond(2, 0, J=500.0, D=(0.0, 0.0, 50)),
    ],
)


##______________________________________________________________________________###


# ----------------------------
# Useful execution shortcuts
# ----------------------------

def run_ground_state_diagnostics(system: SpinSystem = square_half_spin) -> None:
    print_spectrum_summary(system, n_states=min(8, system.total_dim))
    print_ground_state_report(system)


def run_scan_point_example(
    system: SpinSystem = triangle_with_DM_half_spin,
    varied_bond=(0, 1),
    Jprime: float = 12.0,
) -> None:
    inspect_scan_point(system, bond=varied_bond, J=Jprime)


def run_frustration_scan_plot(system: SpinSystem = square_half_spin) -> None:
    # A wider sweep makes projector vs energy differences visible for this model.
    j_values = np.linspace(200,800, 400)
    plot_coupling_scan(
        system=system,
        bond=(1, 2),
        #j_values=j_values,
        frustration_definition="energy",
       # compare_with="energy",
        mode="global_metric",
        title="Projector vs energy-based frustration",
        x_mode="ratio",
        reference_bond=(0, 1),
        show_reference_bond_only=False,
        scan_values=j_values,
        scan_param="J",
        xlabel="J'/J",
        ylabel="Frustration",
        
    )

def run_frustration_scan_plot_scanned_only(system: SpinSystem = triangle_half_spin) -> None:
    j_values = np.linspace(1.0, 20.0, 120)
    plot_coupling_scan(
        system=system,
        bond=(0, 1),
        frustration_definition="projector",
        mode="scanned_only",
        title="Scanned bond frustration only",
        x_mode="ratio",
        reference_bond=(1, 2),
        scan_values=j_values,
        scan_param="J",
        xlabel="J'/J",
        ylabel="Frustration",
        
    )

def run_dm_scan_plot(system: SpinSystem = triangle_with_DM_half_spin) -> None:
    d_values = np.linspace(50, 1300, 1000)
    plot_coupling_scan(
        system=system,
        bond=(0, 1),
        scan_values=d_values,
        scan_param="Dz",
        frustration_definition="projector",
        compare_with="energy",
        mode="scanned_and_global",
        title="Projector vs energy frustration while scanning Dz",
    )


def run_dm_scan_plot_dx(system: SpinSystem = triangle_with_DM_half_spin) -> None:
    d_values = np.linspace(-5.0, 5.0, 120)
    plot_coupling_scan(
        system=system,
        bond=(1, 2),
        scan_values=d_values,
        scan_param="Dx",
        frustration_definition="projector",
        mode="global_metric",
        title="Global frustration metric while scanning Dx",
    )


def run_dm_ratio_scan_plot(system: SpinSystem = triangle_with_DM_half_spin) -> None:
    d_values = np.linspace(-12.0, 12.0, 160)
    plot_coupling_scan(
        system=system,
        bond=(0, 1),
        scan_values=d_values,
        scan_param="Dz",
        x_mode="ratio",
        reference_bond=(1, 2),
        frustration_definition="projector",
        mode="scanned_and_global",
        compare_with="energy",
        title="Global frustration vs D'/D",
    )


def run_kahn_population_plot(system: SpinSystem = square_half_spin) -> None:
    j_values = np.linspace(200, 800, 400)
    plot_kahn_spin_population_scan(
        system=system,
        bond=(1, 2),
        j_values=j_values,
        selected_sites=[0, 1, 2],
        reference_bond=(0, 1), 
        temperature=4.2,
        g_factor=2,
        k_B_unit="cm-1",
        title="Kahn site populations vs J'/J",
        xlabel="J'/J",
        ylabel="Spin populations",
    )


def is_system_frustrated(
    system: SpinSystem,
    *,
    tol: float = 1e-10,
    return_gap: bool = False,
    return_energies: bool = False,
):
    """
    Lower-bound energetic criterion:
        frustrated <=> E0 > sum_<ij> e_ij^min

    Parameters
    ----------
    system
        Spin system to classify.
    tol
        Numerical tolerance applied to the strict inequality.
    return_gap
        If True, return (is_frustrated, gap) where
        gap = E0 - sum_<ij> e_ij^min.
    return_energies
        If True, also return (ground_state_energy, expected_min_energy), where
        expected_min_energy = sum_<ij> e_ij^min.

    Returns
    -------
    bool
        By default, only the frustration flag.
    tuple
        If return_gap=True and/or return_energies=True, returns a tuple that
        starts with the frustration flag and then includes the requested values.
    """
    gs = system.ground_state_manifold(eig_tol=tol)

    expected_min_energy = 0.0
    for bond in system.bonds:
        evals = np.linalg.eigvalsh(system.local_bond_hamiltonian(bond))
        expected_min_energy += float(np.real_if_close(evals[0]))

    gap = float(gs.energy - expected_min_energy)
    if abs(gap) < tol:
        gap = 0.0

    frustrated = bool(gap > tol)

    if return_gap and return_energies:
        return frustrated, gap, float(gs.energy), float(expected_min_energy)
    if return_gap:
        return frustrated, gap
    if return_energies:
        return frustrated, float(gs.energy), float(expected_min_energy)
    return frustrated


def run_lower_bound_definition_tests() -> None:
    # Two-site dimer can realize the local bond minimum exactly.
    dimer = SpinSystem(
        spins=[1 / 2, 1 / 2],
        bonds=[Bond(0, 1, J=1.0)],
    )
    dimer_gap = dimer.energy_lower_bound_gap()
    assert abs(dimer_gap) < 1e-10, f"Expected near-zero gap, got {dimer_gap}"
    assert not dimer.is_frustrated_lower_bound(), "Dimer should not be frustrated"

    # Two disconnected AFM dimers can each be minimized simultaneously.
    disconnected_dimers = SpinSystem(
        spins=[1 / 2, 1 / 2, 1 / 2, 1 / 2],
        bonds=[
            Bond(0, 1, J=1.0),
            Bond(2, 3, J=1.0),
        ],
    )
    disconnected_gap = disconnected_dimers.energy_lower_bound_gap()
    assert abs(disconnected_gap) < 1e-10, (
        f"Expected near-zero gap for disconnected dimers, got {disconnected_gap}"
    )
    assert not disconnected_dimers.is_frustrated_lower_bound(), (
        "Disconnected dimers should not be frustrated"
    )

    # AFM triangle cannot minimize all pairwise bonds at once.
    afm_triangle = SpinSystem(
        spins=[1 / 2, 1 / 2, 1 / 2],
        bonds=[
            Bond(0, 1, J=1.0),
            Bond(1, 2, J=1.0),
            Bond(2, 0, J=1.0),
        ],
    )
    triangle_gap = afm_triangle.energy_lower_bound_gap()
    assert triangle_gap > 1e-6, f"Expected positive frustration gap, got {triangle_gap}"
    assert afm_triangle.is_frustrated_lower_bound(), "AFM triangle should be frustrated"

    print("Lower-bound frustration tests passed.")


if __name__ == "__main__":

    #run_ground_state_diagnostics()
    # run_scan_point_example()
    run_frustration_scan_plot()
    #run_dm_scan_plot()
    #run_dm_scan_plot_dx()
    #run_dm_ratio_scan_plot()
    #run_kahn_population_plot()
   #run_lower_bound_definition_tests()
    #print(is_system_frustrated(square_half_spin, return_gap=True, return_energies=True)) 
    #pass
    print("testrun complete.")
