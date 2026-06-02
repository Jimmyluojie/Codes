from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from spin_frustration_updated import Bond, CouplingScanResult, SpinSystem


K_B_BY_ENERGY_UNIT = {
    "reduced": 1.0,
    "j": 1.380649e-23,
    "joule": 1.380649e-23,
    "ev": 8.617333262e-5,
    "mev": 8.617333262e-2,
    "cm^-1": 6.950348005e-1,
    "cm-1": 6.950348005e-1,
    "hartree": 3.166811563e-6,
    "eh": 3.166811563e-6,
}

# Change this one string to switch default thermal scaling everywhere in Kahn helpers.
DEFAULT_K_B_UNIT = "reduced"


def resolve_k_b(k_B: Optional[float], *, energy_unit: str = DEFAULT_K_B_UNIT) -> float:
    """
    Resolve a Boltzmann constant from an explicit value or from an energy-unit key.

    Parameters
    ----------
    k_B : float or None
        Explicit Boltzmann constant. If provided, this value is used directly.
    energy_unit : str
        Unit label for Hamiltonian couplings J, D, ... used when ``k_B`` is None.
        Supported values include: reduced, J/joule, eV, meV, cm^-1, Hartree.
    """
    if k_B is not None:
        value = float(k_B)
        if value <= 0:
            raise ValueError("k_B must be positive.")
        return value

    unit_key = str(energy_unit).strip().lower()
    if unit_key not in K_B_BY_ENERGY_UNIT:
        known = ", ".join(sorted(K_B_BY_ENERGY_UNIT.keys()))
        raise ValueError(
            f"Unknown energy_unit '{energy_unit}'. Supported values: {known}."
        )
    return float(K_B_BY_ENERGY_UNIT[unit_key])


def _apply_publication_style(ax, *, major_labelsize: int = 16) -> None:
    """Apply a clean black-on-white style suitable for publication plots."""
    bg = "white"
    fig = ax.figure
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1.2,
        colors="black",
        labelsize=major_labelsize,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=1.0,
        colors="black",
    )

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def plot_coupling_scan(
    scan=None,
    *,
    system=None,
    bond=None,
    j_values=None,
    scan_values=None,
    scan_param="J",
    x_mode="raw",
    reference_bond=None,
    show_reference_bond_only=False,
    reference_J=None,
    reference_D=None,
    frustration_definition="projector",
    compare_with=None,
    mode="per_bond",
    xlabel=None,
    ylabel=None,
    title=None,
    savepath=None,
    show=True,
    use_ms_blocks=None,
    eig_tol=1e-10,
    ms_tol=1e-12,
):
    """
    Plot a coupling scan for one frustration definition, or compare two definitions.

    You can either pass an already computed ``scan`` or pass ``system``, ``bond``,
    and ``j_values``/``scan_values`` so the function builds the scan internally.

    Parameters
    ----------
    scan : CouplingScanResult or None
        Precomputed scan.
    system, bond, j_values, scan_values, scan_param
        Used only when ``scan`` is None.
    frustration_definition : {"projector", "energy"}
        Definition used for the main scan when the scan is built internally.
    compare_with : {"projector", "energy"} or None
        If given, compute a second scan and overlay it for comparison.
    mode : str
        What to plot:
            - "per_bond"

            - "global_total"
            - "global_metric"
            - "energy"
            - "all"
            - "scanned_and_global"
            - "scanned_vs_rest"
            - "scanned_only"
            - "compare_definitions"
    show_reference_bond_only : bool
        If True, restrict per-bond plotting to the selected ``reference_bond``.
        This applies to modes that include per-bond curves.
    """
    def _make_scan(definition):
        if scan is not None:
            if definition != scan.definition:
                values_in = scan_values if scan_values is not None else j_values
                if system is None or bond is None or values_in is None:
                    raise ValueError(
                        "To compare with another definition when a precomputed scan "
                        "is passed, also provide system, bond, and j_values/scan_values."
                    )
                return system.scan_bond_coupling(
                    bond=bond,
                    j_values=j_values,
                    scan_values=scan_values,
                    scan_param=scan_param,
                    frustration_definition=definition,
                    use_ms_blocks=use_ms_blocks,
                    eig_tol=eig_tol,
                    ms_tol=ms_tol,
                )
            return scan

        values_in = scan_values if scan_values is not None else j_values
        if system is None or bond is None or values_in is None:
            raise ValueError(
                "Either pass scan=... or pass system=..., bond=..., and j_values/scan_values=..."
            )

        return system.scan_bond_coupling(
            bond=bond,
            j_values=j_values,
            scan_values=scan_values,
            scan_param=scan_param,
            frustration_definition=definition,
            use_ms_blocks=use_ms_blocks,
            eig_tol=eig_tol,
            ms_tol=ms_tol,
        )

    primary = _make_scan(frustration_definition)
    secondary = None
    if compare_with is not None:
        compare_with = str(compare_with).strip().lower()
        if compare_with == primary.definition:
            secondary = None
        else:
            secondary = _make_scan(compare_with)

    x_mode = str(x_mode).strip().lower()
    if x_mode not in {"raw", "ratio"}:
        raise ValueError("x_mode must be 'raw' or 'ratio'.")

    x = np.asarray(primary.j_values, dtype=float)
    if x_mode == "ratio":
        if system is None:
            raise ValueError(
                "x_mode='ratio' requires system=... so a reference value can be resolved."
            )
        ref_value = _resolve_scan_reference_value(
            system,
            varied_bond=primary.bond_index,
            scan_param=getattr(primary, "scan_param", "J"),
            reference_bond=reference_bond,
            reference_J=reference_J,
            reference_D=reference_D,
        )
        x = x / ref_value

    reference_key = None
    if reference_bond is not None:
        if isinstance(reference_bond, Bond):
            reference_key = (int(reference_bond.i), int(reference_bond.j))
        elif isinstance(reference_bond, tuple) and len(reference_bond) == 2:
            reference_key = (int(reference_bond[0]), int(reference_bond[1]))
        elif isinstance(reference_bond, int):
            if system is not None:
                _, b = system._resolve_bond_reference(reference_bond)
                reference_key = (int(b.i), int(b.j))
            elif scan is not None:
                keys = list(scan.per_bond.keys())
                if not (0 <= reference_bond < len(keys)):
                    raise IndexError("reference_bond index out of range for provided scan.")
                reference_key = tuple(keys[reference_bond])
            else:
                raise ValueError(
                    "reference_bond as an index requires system=... or scan=..."
                )
        else:
            raise TypeError("reference_bond must be a bond index, ordered pair (i, j), or Bond object.")

    if show_reference_bond_only and reference_key is None:
        raise ValueError("show_reference_bond_only=True requires reference_bond=...")

    if xlabel is None:
        i, j = primary.scanned_bond
        p = getattr(primary, "scan_param", "J")
        if x_mode == "raw":
            if p == "J":
                xlabel = rf"$J_{{{i}{j}}}$"
            elif p == "Dx":
                xlabel = rf"$D^x_{{{i}{j}}}$"
            elif p == "Dy":
                xlabel = rf"$D^y_{{{i}{j}}}$"
            elif p == "Dz":
                xlabel = rf"$D^z_{{{i}{j}}}$"
            else:
                xlabel = rf"${p}_{{{i}{j}}}$"
        else:
            if p == "J":
                xlabel = rf"$J'_{{{i}{j}}}/J$"
            elif p == "Dx":
                xlabel = rf"$D'^x_{{{i}{j}}}/D^x$"
            elif p == "Dy":
                xlabel = rf"$D'^y_{{{i}{j}}}/D^y$"
            elif p == "Dz":
                xlabel = rf"$D'^z_{{{i}{j}}}/D^z$"
            else:
                xlabel = rf"${p}'/{p}$"

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=120)

    _apply_publication_style(ax)

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#17becf",
        "#9467bd",
        "#f2b600",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]
    markers = ["s", "o", "^", "x", "*", "v", "D", "P", "<", ">"]

    # Distinct visual identities when comparing frustration definitions.
    definition_styles = {
        "projector": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "energy": {"color": "#d62728", "marker": "^", "linestyle": "--"},
    }

    def _plot_single(scan_obj, linestyle=None, alpha=1.0):
        nonlocal ylabel
        style = definition_styles.get(
            scan_obj.definition,
            {"color": colors[0], "marker": "o", "linestyle": "-"},
        )
        ls = style["linestyle"] if linestyle is None else linestyle
        color_shift = 0 if scan_obj.definition == "projector" else (len(colors) // 2)
        method_tag = f" ({scan_obj.definition}-based)" if secondary is not None else ""

        def _per_bond_items():
            if not show_reference_bond_only:
                return list(scan_obj.per_bond.items())
            if reference_key not in scan_obj.per_bond:
                raise ValueError(
                    f"reference_bond {reference_key} is not present in the scanned system bonds."
                )
            return [(reference_key, scan_obj.per_bond[reference_key])]

        if mode == "per_bond":
            for idx, (bond_key, values) in enumerate(_per_bond_items()):
                marker = markers[idx % len(markers)]
                ax.plot(
                    x, values,
                    linestyle=ls,
                    color=colors[(idx + color_shift) % len(colors)],
                    marker=marker,
                    linewidth=1.4,
                    markersize=4.0,
                    alpha=alpha,
                    markeredgewidth=1.0 if marker == "x" else 0.0,
                    label=f"{bond_key}{method_tag}",
                )
            if ylabel is None:
                ylabel = "Partial frustration"

        elif mode == "global_total":
            ax.plot(
                x, scan_obj.global_total,
                linestyle=ls,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.6,
                markersize=4.0,
                alpha=alpha,
                label=f"{scan_obj.definition} total{method_tag}",
            )
            if ylabel is None:
                ylabel = "Total frustration"

        elif mode == "global_metric":
            ax.plot(
                x, scan_obj.global_metric, ls,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4.2,
                alpha=alpha,
                label=f"{'energy-based frustration' if scan_obj.definition == 'energy' else scan_obj.definition + ' global'}{method_tag}",
            )
            if ylabel is None:
                ylabel = "Global frustration metric"

        elif mode == "energy":
            ax.plot(
                x, scan_obj.ground_energies,
                linestyle=ls,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.5,
                markersize=4.0,
                alpha=alpha,
                label=f"Ground-state energy{method_tag}",
            )
            if ylabel is None:
                ylabel = "Ground-state energy"

        elif mode == "all":
            for idx, (bond_key, values) in enumerate(_per_bond_items()):
                marker = markers[idx % len(markers)]
                ax.plot(
                    x, values,
                    linestyle=ls,
                    color=colors[(idx + color_shift) % len(colors)],
                    marker=marker,
                    linewidth=1.3,
                    markersize=3.8,
                    alpha=alpha,
                    markeredgewidth=1.0 if marker == "x" else 0.0,
                    label=f"{bond_key}{method_tag}",
                )
            ax.plot(
                x, scan_obj.global_metric,
                linestyle=ls,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4.2,
                alpha=alpha,
                label=f"{('energy-based frustration' if scan_obj.definition == 'energy' else scan_obj.definition + ' global')}{method_tag}",
            )
            if ylabel is None:
                ylabel = "Frustration"

        elif mode == "scanned_and_global":
            scanned_key = tuple(scan_obj.scanned_bond)
            if scanned_key in scan_obj.per_bond:
                ax.plot(
                    x, scan_obj.per_bond[scanned_key], ls,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.6,
                    markersize=4.2,
                    alpha=alpha,
                    label=f"Bond {scanned_key} [{scan_obj.definition}]{method_tag}",
                )
            ax.plot(
                x, scan_obj.global_metric, ls,
                color=style["color"],
                marker="s",
                linewidth=1.8,
                markersize=4.2,
                alpha=alpha,
                label=f"{('energy-based frustration' if scan_obj.definition == 'energy' else scan_obj.definition + ' global')}{method_tag}",
            )
            if ylabel is None:
                ylabel = "Frustration"

        elif mode == "scanned_vs_rest":
            scanned_key = tuple(scan_obj.scanned_bond)
            for idx, (bond_key, values) in enumerate(_per_bond_items()):
                marker = markers[idx % len(markers)]
                is_scanned = (bond_key == scanned_key)
                ax.plot(
                    x, values,
                    linestyle=ls,
                    color=colors[(idx + color_shift) % len(colors)],
                    marker=marker,
                    linewidth=1.6 if is_scanned else 1.3,
                    markersize=4.2 if is_scanned else 3.8,
                    alpha=alpha,
                    markeredgewidth=1.0 if marker == "x" else 0.0,
                    label=f"Bond {bond_key}" + (" [scanned]" if is_scanned else "") + method_tag,
                )
            if ylabel is None:
                ylabel = "Partial frustration"

        elif mode == "scanned_only":
            selected_key = reference_key if show_reference_bond_only else tuple(scan_obj.scanned_bond)
            if selected_key in scan_obj.per_bond:
                values = scan_obj.per_bond[selected_key]
                ax.plot(
                    x, values,
                    linestyle=ls,
                    color=style["color"],
                    marker=markers[0],
                    linewidth=1.8,
                    markersize=4.5,
                    alpha=alpha,
                    label=f"Bond {selected_key}{method_tag}",
                )
            if ylabel is None:
                ylabel = "Reference bond frustration" if show_reference_bond_only else "Scanned bond frustration"

        elif mode == "compare_definitions":
            ax.plot(
                x, scan_obj.global_metric, ls,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4.2,
                alpha=alpha,
                label=f"{'energy-based frustration' if scan_obj.definition == 'energy' else scan_obj.definition + ' global metric'}{method_tag}",
            )
            if ylabel is None:
                ylabel = "Global frustration metric"

        else:
            raise ValueError(
                "mode must be one of: "
                "'per_bond', 'global_total', 'global_metric', "
                "'energy', 'all', 'scanned_and_global', 'scanned_vs_rest', "
                "'scanned_only', 'compare_definitions'."
            )

    _plot_single(primary, linestyle=None, alpha=1.0)
    if secondary is not None:
        _plot_single(secondary, linestyle=None, alpha=0.95)

    if title is None:
        if secondary is None:
            title = f"{primary.definition.capitalize()} frustration scan"
        else:
            title = f"Comparison: {primary.definition} vs {secondary.definition}"

    ax.set_xlabel(xlabel, fontsize=18, color="black")
    ax.set_ylabel(ylabel, fontsize=18, color="black")
    ax.set_title(title, fontsize=18, color="black")

    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        fontsize=16,
    )

    ax.grid(False)
    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def _resolve_reference_coupling_for_bond_pop(system, varied_bond, reference_bond=None, reference_J=None, tol=1e-12):
    """
    Determine the reference coupling J used to plot the ratio J'/J.

    Priority:
    1) explicit ``reference_J``
    2) the coupling on ``reference_bond``
    3) automatic inference from the unscanned bonds, if they all share the same J
    """
    if reference_J is not None:
        reference_J = float(reference_J)
        if abs(reference_J) < tol:
            raise ValueError("reference_J must be non-zero to build J'/J.")
        return reference_J

    if reference_bond is not None:
        _, bond_obj = system._resolve_bond_reference(reference_bond)
        if abs(bond_obj.J) < tol:
            raise ValueError("The reference bond has J = 0, so J'/J is undefined.")
        return float(bond_obj.J)

    varied_index, _ = system._resolve_bond_reference(varied_bond)
    remaining_J = [float(b.J) for idx, b in enumerate(system.bonds) if idx != varied_index]
    if not remaining_J:
        raise ValueError(
            "Cannot infer a reference J when the system contains only the scanned bond. "
            "Pass reference_J=... explicitly."
        )

    unique = []
    for J in remaining_J:
        if not any(abs(J - u) < tol for u in unique):
            unique.append(J)

    if len(unique) != 1:
        raise ValueError(
            "Could not infer a unique reference J from the unscanned bonds. "
            "Pass reference_bond=... or reference_J=... explicitly."
        )
    if abs(unique[0]) < tol:
        raise ValueError("The inferred reference J is zero, so J'/J is undefined.")
    return unique[0]


def _resolve_scan_reference_value(
    system,
    *,
    varied_bond,
    scan_param,
    reference_bond=None,
    reference_J=None,
    reference_D=None,
    tol=1e-12,
):
    """
    Resolve the denominator used for ratio x-axis in scans.

    Supports scan_param in {"J", "Dx", "Dy", "Dz"} and returns the matching
    reference value from explicit input, a chosen reference bond, or automatic
    inference from unscanned bonds when possible.
    """
    scan_param = str(scan_param).strip()
    if scan_param not in {"J", "Dx", "Dy", "Dz"}:
        raise ValueError("scan_param must be one of: 'J', 'Dx', 'Dy', 'Dz'.")

    if scan_param == "J":
        return _resolve_reference_coupling_for_bond_pop(
            system,
            varied_bond,
            reference_bond=reference_bond,
            reference_J=reference_J,
            tol=tol,
        )

    comp = {"Dx": 0, "Dy": 1, "Dz": 2}[scan_param]
    comp_label = {"Dx": "D^x", "Dy": "D^y", "Dz": "D^z"}[scan_param]

    if reference_D is not None:
        ref = float(reference_D)
        if abs(ref) < tol:
            raise ValueError(f"reference_D must be non-zero to build {comp_label}'/{comp_label}.")
        return ref

    if reference_bond is not None:
        _, b = system._resolve_bond_reference(reference_bond)
        ref = float(b.D[comp])
        if abs(ref) < tol:
            raise ValueError(f"The reference bond has {comp_label}=0, ratio is undefined.")
        return ref

    varied_index, _ = system._resolve_bond_reference(varied_bond)
    remaining = [float(b.D[comp]) for idx, b in enumerate(system.bonds) if idx != varied_index]
    if not remaining:
        raise ValueError(
            f"Cannot infer a reference {comp_label} when only the scanned bond exists. "
            "Pass reference_D=... explicitly."
        )

    unique = []
    for val in remaining:
        if not any(abs(val - u) < tol for u in unique):
            unique.append(val)
    if len(unique) != 1:
        raise ValueError(
            f"Could not infer a unique reference {comp_label} from unscanned bonds. "
            "Pass reference_bond=... or reference_D=... explicitly."
        )
    if abs(unique[0]) < tol:
        raise ValueError(f"The inferred reference {comp_label} is zero, ratio is undefined.")
    return unique[0]


def _resolve_scan_bond_indices(system, bonds):
    """
    Normalize a bond selection into a list of bond indices in ``system.bonds``.
    """
    if bonds is None:
        return []

    out = []
    for bond in bonds:
        idx, _ = system._resolve_bond_reference(bond)
        if idx not in out:
            out.append(idx)
    return out


def _equilateral_triangle_scan_indices(system, anchor_bond):
    """
    Infer the 3 undirected edges of the unique triangle containing ``anchor_bond``.

    For each undirected pair, exactly one ordered bond must exist in the system.
    """
    _, anchor = system._resolve_bond_reference(anchor_bond)
    i, j = int(anchor.i), int(anchor.j)

    undirected_neighbors = {site: set() for site in range(system.n_sites)}
    for b in system.bonds:
        a, c = int(b.i), int(b.j)
        undirected_neighbors[a].add(c)
        undirected_neighbors[c].add(a)

    common = sorted(undirected_neighbors[i].intersection(undirected_neighbors[j]))
    if len(common) != 1:
        raise ValueError(
            "Could not infer a unique equilateral-triangle partner site from anchor_bond. "
            "Pass vary_bonds explicitly instead."
        )
    k = common[0]

    tri_pairs = [
        tuple(sorted((i, j))),
        tuple(sorted((j, k))),
        tuple(sorted((k, i))),
    ]

    index_by_pair = {}
    for idx, b in enumerate(system.bonds):
        pair = tuple(sorted((int(b.i), int(b.j))))
        if pair in tri_pairs:
            if pair in index_by_pair:
                raise ValueError(
                    f"Multiple ordered bonds match undirected pair {pair}. "
                    "Use vary_bonds explicitly for this model."
                )
            index_by_pair[pair] = idx

    missing = [pair for pair in tri_pairs if pair not in index_by_pair]
    if missing:
        raise ValueError(
            "The inferred triangle is incomplete in the bond list for pairs: "
            f"{missing}. Use vary_bonds explicitly."
        )

    return [index_by_pair[pair] for pair in tri_pairs]


def plot_frustration_vs_d_over_j(
    *,
    system,
    d_values,
    anchor_bond,
    vary_bonds=None,
    d_component="z",
    symmetry_constraint=None,
    reference_bond=None,
    reference_J=None,
    frustration_definition="projector",
    mode="scanned_and_global",
    xlabel=None,
    ylabel=None,
    title=None,
    savepath=None,
    show=True,
    use_ms_blocks=None,
    eig_tol=1e-10,
    ms_tol=1e-12,
):
    """
    Plot frustration versus D/J while varying one or several DMI bond components.

    Parameters
    ----------
    system : SpinSystem
        Input spin system.
    d_values : sequence of float
        Values assigned to the selected D-component.
    anchor_bond
        Bond used as the displayed "scanned" bond and as default reference-J lookup.
    vary_bonds : sequence or None
        Bonds whose D-component is varied together. If None, only ``anchor_bond``
        is varied unless ``symmetry_constraint='equilateral_triangle'``.
    d_component : {'x', 'y', 'z'}
        Which D component is scanned.
    symmetry_constraint : {None, 'equilateral_triangle'}
        If 'equilateral_triangle', infer the 3 triangle edges that contain
        ``anchor_bond`` and vary them together with identical values.
    """
    comp = str(d_component).strip().lower()
    if comp not in {"x", "y", "z"}:
        raise ValueError("d_component must be one of: 'x', 'y', 'z'.")
    comp_idx = {"x": 0, "y": 1, "z": 2}[comp]

    d_values = np.asarray(d_values, dtype=float)
    if d_values.ndim != 1 or d_values.size == 0:
        raise ValueError("d_values must be a non-empty 1D sequence.")

    anchor_index, anchor_obj = system._resolve_bond_reference(anchor_bond)
    if vary_bonds is None:
        if symmetry_constraint == "equilateral_triangle":
            varied_indices = _equilateral_triangle_scan_indices(system, anchor_bond)
        else:
            varied_indices = [anchor_index]
    else:
        varied_indices = _resolve_scan_bond_indices(system, vary_bonds)
        if not varied_indices:
            varied_indices = [anchor_index]

    j_ref = _resolve_reference_coupling_for_bond_pop(
        system,
        anchor_bond,
        reference_bond=reference_bond,
        reference_J=reference_J,
    )
    ratios = d_values / j_ref

    energies = np.empty(d_values.size, dtype=float)
    degeneracies = np.empty(d_values.size, dtype=int)
    global_average = np.empty(d_values.size, dtype=float)
    global_total = np.empty(d_values.size, dtype=float)
    global_metric = np.empty(d_values.size, dtype=float)
    per_bond_series = {
        (b.i, b.j): np.empty(d_values.size, dtype=float)
        for b in system.bonds
    }

    for k, dval in enumerate(d_values):
        new_bonds = list(system.bonds)
        for idx in varied_indices:
            old = new_bonds[idx]
            dvec = list(map(float, old.D))
            dvec[comp_idx] = float(dval)
            new_bonds[idx] = Bond(i=old.i, j=old.j, J=float(old.J), D=tuple(dvec))

        scanned = SpinSystem(spins=system.spins, bonds=new_bonds)
        fr = scanned.frustration_by_definition(
            definition=frustration_definition,
            use_ms_blocks=use_ms_blocks,
            eig_tol=eig_tol,
            ms_tol=ms_tol,
        )
        energies[k] = fr.ground_energy
        degeneracies[k] = fr.degeneracy
        global_average[k] = fr.average
        global_total[k] = fr.total
        global_metric[k] = fr.global_metric
        for bond_key, value in fr.per_bond.items():
            per_bond_series[bond_key][k] = value

    scan_like = CouplingScanResult(
        scanned_bond=(anchor_obj.i, anchor_obj.j),
        bond_index=anchor_index,
        scan_param=f"D{comp}",
        j_values=ratios,
        ground_energies=energies,
        degeneracies=degeneracies,
        global_average=global_average,
        global_total=global_total,
        global_metric=global_metric,
        per_bond=per_bond_series,
        definition=str(frustration_definition).strip().lower(),
    )

    if xlabel is None:
        comp_tex = {"x": "D^x", "y": "D^y", "z": "D^z"}[comp]
        xlabel = rf"${comp_tex}/J$"

    if title is None:
        varied_keys = [
            (int(system.bonds[idx].i), int(system.bonds[idx].j))
            for idx in varied_indices
        ]
        title = (
            f"{scan_like.definition.capitalize()} frustration vs D/J "
            f"(varied bonds: {varied_keys})"
        )

    fig, ax = plot_coupling_scan(
        scan=scan_like,
        system=system,
        mode=mode,
        x_mode="raw",
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        savepath=savepath,
        show=show,
    )

    return fig, ax, {
        "ratios": ratios,
        "d_values": d_values,
        "scan": scan_like,
        "reference_J": float(j_ref),
        "varied_bond_indices": tuple(varied_indices),
        "symmetry_constraint": symmetry_constraint,
        "d_component": comp,
    }


def _resolve_selected_bond_keys(system, bonds):
    """
    Normalize a bond selection into a list of ordered bond keys ``(i, j)``.
    """
    if bonds is None:
        return [(b.i, b.j) for b in system.bonds]

    keys = []
    for bond in bonds:
        _, bond_obj = system._resolve_bond_reference(bond)
        key = (bond_obj.i, bond_obj.j)
        if key not in keys:
            keys.append(key)
    return keys


def compute_bond_population_scan(
    *,
    system,
    varied_bond,
    j_values,
    selected_bonds=None,
    reference_bond=None,
    reference_J=None,
    use_ms_blocks=None,
    eig_tol=1e-10,
    ms_tol=1e-12,
):
    """
    Compute projector-based bond populations for a scan where one bond is varied.

    The population of bond ``(i, j)`` is defined here as

        p_ij = Tr(P_ij rho_GS) = 1 - f_ij,

    where ``P_ij`` is the projector onto the local ground-state manifold of that
    bond Hamiltonian and ``rho_GS`` is the equal-weight mixture over the exact
    ground-state manifold. This is the natural bond analogue of the quantity that
    already appears in the projector-based frustration definition.

    Returns
    -------
    ratios : ndarray
        The x-axis values ``J'/J``.
    populations : dict
        Selected bond populations as arrays versus ``J'/J``.
    scan : CouplingScanResult
        The underlying projector-based coupling scan.
    reference_value : float
        The reference coupling ``J`` used in the ratio.
    """
    scan = system.scan_bond_coupling(
        bond=varied_bond,
        j_values=j_values,
        frustration_definition="projector",
        use_ms_blocks=use_ms_blocks,
        eig_tol=eig_tol,
        ms_tol=ms_tol,
    )

    reference_value = _resolve_reference_coupling_for_bond_pop(
        system,
        varied_bond,
        reference_bond=reference_bond,
        reference_J=reference_J,
    )
    ratios = np.asarray(scan.j_values, dtype=float) / reference_value

    selected_keys = _resolve_selected_bond_keys(system, selected_bonds)
    populations = {
        bond_key: 1.0 - np.asarray(scan.per_bond[bond_key], dtype=float)
        for bond_key in selected_keys
    }
    return ratios, populations, scan, reference_value


def plot_bond_population_scan(
    *,
    system,
    varied_bond,
    j_values,
    selected_bonds=None,
    reference_bond=None,
    reference_J=None,
    xlabel=None,
    ylabel="Bond population",
    title=None,
    savepath=None,
    show=True,
    use_ms_blocks=None,
    eig_tol=1e-10,
    ms_tol=1e-12,
):
    """
    Plot the projector-based population of selected bonds versus the ratio J'/J.

    Parameters
    ----------
    system : SpinSystem
        Arbitrary spin system.
    varied_bond
        Bond whose Heisenberg coupling is scanned (this plays the role of J').
    j_values : sequence of float
        Values assigned to the varied bond during the scan.
    selected_bonds : sequence or None
        Bonds to plot. If None, all bonds are shown.
    reference_bond, reference_J
        How to define the denominator J in J'/J. If neither is given, the
        function tries to infer a unique common J from the unscanned bonds.
    """
    ratios, populations, scan, reference_value = compute_bond_population_scan(
        system=system,
        varied_bond=varied_bond,
        j_values=j_values,
        selected_bonds=selected_bonds,
        reference_bond=reference_bond,
        reference_J=reference_J,
        use_ms_blocks=use_ms_blocks,
        eig_tol=eig_tol,
        ms_tol=ms_tol,
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=120)

    _apply_publication_style(ax)

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#17becf",
        "#9467bd",
        "#f2b600",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]
    markers = ["s", "o", "^", "x", "*", "v", "D", "P", "<", ">"]

    scanned_key = tuple(scan.scanned_bond)
    for idx, (bond_key, values) in enumerate(populations.items()):
        marker = markers[idx % len(markers)]
        is_scanned = (bond_key == scanned_key)
        label = f"Bond {bond_key}" + (" [varied bond]" if is_scanned else "")
        ax.plot(
            ratios,
            values,
            "-",
            color=colors[idx % len(colors)],
            marker=marker,
            linewidth=1.6 if is_scanned else 1.3,
            markersize=4.2 if is_scanned else 3.8,
            markeredgewidth=1.0 if marker == "x" else 0.0,
            label=label,
        )

    if xlabel is None:
        xlabel = r"$J' / J$"

    if title is None:
        title = (
            f"Bond populations vs $J'/J$ for varied bond {scanned_key} "
            f"(reference $J$ = {reference_value:g})"
        )

    ax.set_xlabel(xlabel, fontsize=18, color="black")
    ax.set_ylabel(ylabel, fontsize=18, color="black")
    ax.set_title(title, fontsize=18, color="black")

    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        fontsize=16,
    )

    ax.grid(False)
    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax, {
        "ratios": ratios,
        "populations": populations,
        "scan": scan,
        "reference_J": reference_value,
    }


@dataclass
class KahnPopulationScanResult:
    scanned_bond: Tuple[int, int]
    bond_index: int
    j_values: np.ndarray
    ratio_values: np.ndarray
    ms_target: float
    temperature: float
    site_populations: Dict[int, np.ndarray]
    sector_ground_energies: np.ndarray


def _kahn_unique_ms_values(system: SpinSystem) -> np.ndarray:
    """
    Return all physical total-Ms values available in the Hilbert space.
    """
    ms_diag = system.space.total_sz_diagonal()
    return np.unique(np.rint(2.0 * ms_diag).astype(int)) / 2.0


def infer_kahn_ms_target(
    system: SpinSystem,
    *,
    eig_tol: float = 1e-10,
    ms_tol: float = 1e-12,
) -> float:
    """
    Infer the magnetic component used in Kahn-style spin populations.

    We diagonalize each invariant Ms sector, identify the sectors that contain the
    absolute ground energy, and then choose the largest Ms among them. This is the
    natural zero-field proxy for the component selected by a positive Zeeman field
    within the ground multiplet.
    """
    if not system.commutes_with_total_sz(tol=10 * ms_tol):
        raise ValueError(
            "Cannot infer a Kahn Ms sector because this Hamiltonian does not "
            "conserve total S^z. Use a model with conserved S^z or pass an "
            "explicit state-based observable instead."
        )

    H = system.hamiltonian()
    sector_mins = []
    for ms in _kahn_unique_ms_values(system):
        Hblk, _ = system.restrict_to_ms(H, ms, tol=10 * ms_tol)
        if Hblk.size == 0:
            continue
        evals = np.linalg.eigvalsh(Hblk)
        sector_mins.append((float(np.real_if_close(evals[0])), float(ms)))

    if not sector_mins:
        raise RuntimeError("No Ms sector could be diagonalized.")

    global_min = min(E for E, _ in sector_mins)
    candidate_ms = [ms for E, ms in sector_mins if abs(E - global_min) < eig_tol]
    return float(max(candidate_ms))


def kahn_spin_populations(
    system: SpinSystem,
    *,
    selected_sites: Optional[Sequence[int]] = None,
    ms_target: Optional[float] = None,
    temperature: float = 0.0,
    g_factor: float = 2.0,
    k_B: Optional[float] = None,
    k_B_unit: str = DEFAULT_K_B_UNIT,
    eig_tol: float = 1e-10,
    ms_tol: float = 1e-12,
) -> Tuple[Dict[int, float], float, float]:
    """
    Kahn-style local spin populations P_i = g <S_i^z>.

    The populations are evaluated in a fixed total-Ms sector, corresponding to the
    magnetic component selected by a positive field within the ground multiplet.
    If ``temperature`` is zero, a uniform average over the exactly degenerate
    lowest states of that sector is used. If ``temperature`` is positive, a
    canonical average is performed within that same Ms sector.

    Returns
    -------
    populations : dict
        Mapping site -> P_i = g <S_i^z>.
    ms_target : float
        The Ms sector actually used.
    sector_ground_energy : float
        Lowest energy in that sector.
    """
    if not system.commutes_with_total_sz(tol=10 * ms_tol):
        raise ValueError(
            "Kahn spin populations require a conserved total S^z in this helper."
        )

    if selected_sites is None:
        selected_sites = tuple(range(system.n_sites))
    else:
        selected_sites = tuple(int(i) for i in selected_sites)

    for site in selected_sites:
        if not (0 <= site < system.n_sites):
            raise IndexError(f"Site index {site} is out of range for this system.")

    if ms_target is None:
        ms_target = infer_kahn_ms_target(system, eig_tol=eig_tol, ms_tol=ms_tol)
    ms_target = float(ms_target)

    H = system.hamiltonian()
    Hblk, idx = system.restrict_to_ms(H, ms_target, tol=10 * ms_tol)
    evals, evecs = np.linalg.eigh(Hblk)
    evals = np.asarray(np.real_if_close(evals), dtype=float)
    evecs = np.asarray(evecs, dtype=complex)

    E0 = float(evals[0])
    if temperature <= 0:
        state_idx = np.where(np.abs(evals - E0) < eig_tol)[0]
        weights = np.full(state_idx.size, 1.0 / state_idx.size, dtype=float)
    else:
        k_B_value = resolve_k_b(k_B, energy_unit=k_B_unit)
        beta = 1.0 / (k_B_value * float(temperature))
        shifted = evals - np.min(evals)
        boltz = np.exp(-beta * shifted)
        Z = np.sum(boltz)
        if Z <= 0:
            raise RuntimeError("Invalid partition function in Kahn population average.")
        state_idx = np.arange(evals.size)
        weights = boltz / Z

    populations: Dict[int, float] = {}
    for site in selected_sites:
        Sz = system.space.site_operator(site, "z")
        value = 0.0
        for w, k in zip(weights, state_idx):
            psi = system._embed_sector_vector(evecs[:, k], idx, system.total_dim)
            norm = np.vdot(psi, psi).real
            exp_val = (np.vdot(psi, Sz @ psi) / norm).real
            value += float(w) * float(exp_val)
        populations[site] = float(g_factor) * value

    return populations, ms_target, E0


def _resolve_reference_coupling(
    system: SpinSystem,
    *,
    varied_bond: Union[int, Tuple[int, int], Bond],
    reference_bond: Optional[Union[int, Tuple[int, int], Bond]] = None,
    reference_J: Optional[float] = None,
    tol: float = 1e-12,
) -> float:
    """
    Determine the fixed J used in the ratio p = J'/J.
    """
    if reference_J is not None:
        if abs(reference_J) <= tol:
            raise ValueError("reference_J must be non-zero.")
        return float(reference_J)

    varied_index, _ = system._resolve_bond_reference(varied_bond)

    if reference_bond is not None:
        ref_index, ref = system._resolve_bond_reference(reference_bond)
        if ref_index == varied_index:
            raise ValueError("reference_bond must be different from the varied bond.")
        if abs(ref.J) <= tol:
            raise ValueError("The selected reference bond has J = 0.")
        return float(ref.J)

    other_J = [float(b.J) for k, b in enumerate(system.bonds) if k != varied_index]
    if not other_J:
        raise ValueError(
            "Could not infer a reference J automatically because the system has only "
            "one bond. Pass reference_J explicitly."
        )

    if not np.allclose(other_J, other_J[0], atol=tol, rtol=0.0):
        raise ValueError(
            "Could not infer a unique reference J from the remaining bonds. "
            "Pass reference_bond=... or reference_J=... explicitly."
        )
    if abs(other_J[0]) <= tol:
        raise ValueError("The inferred reference J is zero.")
    return float(other_J[0])


def scan_kahn_spin_populations(
    *,
    system: SpinSystem,
    bond: Union[int, Tuple[int, int], Bond],
    j_values: Sequence[float],
    selected_sites: Optional[Sequence[int]] = None,
    reference_bond: Optional[Union[int, Tuple[int, int], Bond]] = None,
    reference_J: Optional[float] = None,
    ms_target: Optional[float] = None,
    temperature: float = 0.0,
    g_factor: float = 2.0,
    k_B: Optional[float] = None,
    k_B_unit: str = DEFAULT_K_B_UNIT,
    eig_tol: float = 1e-10,
    ms_tol: float = 1e-12,
) -> KahnPopulationScanResult:
    """
    Scan one bond coupling J' and compute Kahn-style site spin populations versus p=J'/J.

    Note: Kahn's definition is site-based, not bond-based.
    """
    bond_index, target_bond = system._resolve_bond_reference(bond)
    j_values = np.asarray(j_values, dtype=float)
    if j_values.ndim != 1 or j_values.size == 0:
        raise ValueError("j_values must be a non-empty 1D sequence.")

    if selected_sites is None:
        selected_sites = tuple(range(system.n_sites))
    else:
        selected_sites = tuple(int(i) for i in selected_sites)

    J_ref = _resolve_reference_coupling(
        system,
        varied_bond=bond,
        reference_bond=reference_bond,
        reference_J=reference_J,
    )

    ratio_values = j_values / J_ref
    site_series = {site: np.empty(j_values.size, dtype=float) for site in selected_sites}
    sector_ground_energies = np.empty(j_values.size, dtype=float)
    chosen_ms: Optional[float] = None
    resolved_k_B = resolve_k_b(k_B, energy_unit=k_B_unit)

    for k, Jprime in enumerate(j_values):
        scanned = system.with_bond_coupling(bond_index, float(Jprime))
        pops, used_ms, E0_sector = kahn_spin_populations(
            scanned,
            selected_sites=selected_sites,
            ms_target=ms_target,
            temperature=temperature,
            g_factor=g_factor,
            k_B=resolved_k_B,
            k_B_unit=k_B_unit,
            eig_tol=eig_tol,
            ms_tol=ms_tol,
        )
        if chosen_ms is None:
            chosen_ms = float(used_ms)
        elif ms_target is None and abs(float(used_ms) - float(chosen_ms)) > 10 * ms_tol:
            # keep the physically selected sector point-by-point, but warn via value change
            chosen_ms = float(used_ms)
        sector_ground_energies[k] = E0_sector
        for site, value in pops.items():
            site_series[site][k] = value

    return KahnPopulationScanResult(
        scanned_bond=(target_bond.i, target_bond.j),
        bond_index=bond_index,
        j_values=j_values,
        ratio_values=ratio_values,
        ms_target=float(chosen_ms if chosen_ms is not None else 0.0),
        temperature=float(temperature),
        site_populations=site_series,
        sector_ground_energies=sector_ground_energies,
    )


def plot_kahn_spin_population_scan(
    scan: Optional[KahnPopulationScanResult] = None,
    *,
    system: Optional[SpinSystem] = None,
    bond: Optional[Union[int, Tuple[int, int], Bond]] = None,
    j_values: Optional[Sequence[float]] = None,
    selected_sites: Optional[Sequence[int]] = None,
    reference_bond: Optional[Union[int, Tuple[int, int], Bond]] = None,
    reference_J: Optional[float] = None,
    ms_target: Optional[float] = None,
    temperature: float = 0.0,
    g_factor: float = 2.0,
    k_B: Optional[float] = None,
    k_B_unit: str = DEFAULT_K_B_UNIT,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    eig_tol: float = 1e-10,
    ms_tol: float = 1e-12,
):
    """
    Plot Kahn-style site spin populations P_i = g <S_i^z> versus p = J'/J.
    """
    if scan is None:
        if system is None or bond is None or j_values is None:
            raise ValueError(
                "Either pass scan=... or pass system=..., bond=..., and j_values=..."
            )
        scan = scan_kahn_spin_populations(
            system=system,
            bond=bond,
            j_values=j_values,
            selected_sites=selected_sites,
            reference_bond=reference_bond,
            reference_J=reference_J,
            ms_target=ms_target,
            temperature=temperature,
            g_factor=g_factor,
            k_B=k_B,
            k_B_unit=k_B_unit,
            eig_tol=eig_tol,
            ms_tol=ms_tol,
        )

    x = np.asarray(scan.ratio_values, dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=120)
    _apply_publication_style(ax)

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#17becf",
        "#9467bd",
        "#f2b600",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]
    markers = ["s", "o", "^", "x", "*", "v", "D", "P", "<", ">"]

    for idx, (site, values) in enumerate(scan.site_populations.items()):
        marker = markers[idx % len(markers)]
        ax.plot(
            x,
            values,
            "-",
            color=colors[idx % len(colors)],
            marker=marker,
            linewidth=1.6,
            markersize=4.2,
            markeredgewidth=1.0 if marker == "x" else 0.0,
            label=rf"$P_{{{site}}}$",
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)

    if xlabel is None:
        i, j = scan.scanned_bond
        xlabel = rf"$J'_{{{i}{j}}}/J$"
    if ylabel is None:
        ylabel = rf"Kahn spin population $P_i = {g_factor:g}\,\langle S_i^z \rangle$"
    if title is None:
        if temperature > 0:
            title = rf"Kahn site populations vs $J'/J$ at $T={temperature:g}$"
        else:
            title = r"Kahn site populations vs $J'/J$"

    ax.set_xlabel(xlabel, fontsize=18, color="black")
    ax.set_ylabel(ylabel, fontsize=18, color="black")
    ax.set_title(title, fontsize=18, color="black")

    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        fontsize=16,
    )

    ax.grid(False)
    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def _normalize_state_vector(psi: np.ndarray) -> np.ndarray:
    """Return a normalized copy of a state vector."""
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    norm = np.sqrt(np.vdot(psi, psi).real)
    if norm <= 0:
        raise ValueError("State vector has zero norm.")
    return psi / norm


def _expectation_value(psi: np.ndarray, op: np.ndarray) -> float:
    """Return <psi|op|psi> for a normalized or unnormalized state."""
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    norm = np.vdot(psi, psi).real
    if norm <= 0:
        raise ValueError("State vector has zero norm.")
    return float(np.real_if_close(np.vdot(psi, op @ psi) / norm).real)


def _operator_variance(psi: np.ndarray, op: np.ndarray) -> float:
    """Return Var(op) = <op^2> - <op>^2 in state psi."""
    exp1 = _expectation_value(psi, op)
    exp2 = _expectation_value(psi, op @ op)
    var = float(np.real_if_close(exp2 - exp1 ** 2).real)
    return max(var, 0.0)


def _complex_to_text(z: complex, precision: int = 6, tol: float = 1e-12) -> str:
    """Compact pretty-printer for complex amplitudes."""
    z = complex(z)
    re = 0.0 if abs(z.real) < tol else z.real
    im = 0.0 if abs(z.imag) < tol else z.imag
    if im == 0.0:
        return f"{re:+.{precision}f}"
    if re == 0.0:
        return f"{im:+.{precision}f}j"
    sign = "+" if im >= 0 else "-"
    return f"{re:+.{precision}f}{sign}{abs(im):.{precision}f}j"


def build_total_spin_operators(system: SpinSystem) -> Dict[str, np.ndarray]:
    """
    Build Sx_tot, Sy_tot, Sz_tot and S^2_tot for the full Hilbert space.
    """
    sx = np.zeros((system.total_dim, system.total_dim), dtype=complex)
    sy = np.zeros_like(sx)
    sz = np.zeros_like(sx)
    for site in range(system.n_sites):
        sx += system.space.site_operator(site, "x")
        sy += system.space.site_operator(site, "y")
        sz += system.space.site_operator(site, "z")
    s2 = sx @ sx + sy @ sy + sz @ sz
    return {"Sx": sx, "Sy": sy, "Sz": sz, "S2": s2}


def allowed_total_spins(system: SpinSystem) -> np.ndarray:
    """
    Conservative list of physically allowed total-spin values from 0/1/2 step 1/2
    up to sum_i s_i, matching the parity of the total spin sum.
    """
    smax2 = int(round(2.0 * sum(system.spins)))
    parity = smax2 % 2
    vals = [k / 2.0 for k in range(parity, smax2 + 1, 2)]
    return np.asarray(vals, dtype=float)


def infer_total_spin_from_s2(
    s2_value: float,
    *,
    allowed_spins_values: Optional[Sequence[float]] = None,
    tol: float = 1e-8,
) -> Optional[float]:
    """
    Infer S from an S(S+1) expectation value when it is sharp enough.
    """
    if allowed_spins_values is None:
        s_est = 0.5 * (-1.0 + math.sqrt(max(1.0 + 4.0 * float(s2_value), 0.0)))
        if abs(s_est * (s_est + 1.0) - s2_value) < tol:
            return float(s_est)
        return None

    candidates = np.asarray(allowed_spins_values, dtype=float)
    target = candidates * (candidates + 1.0)
    idx = int(np.argmin(np.abs(target - float(s2_value))))
    if abs(target[idx] - float(s2_value)) < tol:
        return float(candidates[idx])
    return None


def state_quantum_numbers(
    system: SpinSystem,
    psi: np.ndarray,
    *,
    operator_cache: Optional[Dict[str, np.ndarray]] = None,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Inspect one state and infer its main spin quantum numbers when possible.

    Returned keys include:
    - S, Ms when they can be identified sharply
    - S2_expectation, Sz_expectation and their variances
    - local_sz expectations for each site
    - kahn_populations = 2 <S_i^z>
    """
    psi = _normalize_state_vector(psi)
    ops = operator_cache if operator_cache is not None else build_total_spin_operators(system)
    s2_op = ops["S2"]
    sz_op = ops["Sz"]

    s2_exp = _expectation_value(psi, s2_op)
    sz_exp = _expectation_value(psi, sz_op)
    s2_var = _operator_variance(psi, s2_op)
    sz_var = _operator_variance(psi, sz_op)

    allowed_s = allowed_total_spins(system)
    S = infer_total_spin_from_s2(s2_exp, allowed_spins_values=allowed_s, tol=tol)
    Ms = None
    ms2 = int(round(2.0 * sz_exp))
    if abs(sz_exp - 0.5 * ms2) < tol:
        Ms = 0.5 * ms2

    s_residual = None
    if S is not None:
        s_residual = float(np.linalg.norm((s2_op - S * (S + 1.0) * np.eye(system.total_dim)) @ psi))
    ms_residual = None
    if Ms is not None:
        ms_residual = float(np.linalg.norm((sz_op - Ms * np.eye(system.total_dim)) @ psi))

    local_sz = {
        site: _expectation_value(psi, system.space.site_operator(site, "z"))
        for site in range(system.n_sites)
    }
    kahn_populations = {site: 2.0 * val for site, val in local_sz.items()}

    return {
        "S": S,
        "Ms": Ms,
        "S2_expectation": s2_exp,
        "Sz_expectation": sz_exp,
        "S2_variance": s2_var,
        "Sz_variance": sz_var,
        "is_S_eigenstate": bool(S is not None and s_residual is not None and s_residual < math.sqrt(max(tol, 1e-16))),
        "is_Ms_eigenstate": bool(Ms is not None and ms_residual is not None and ms_residual < math.sqrt(max(tol, 1e-16))),
        "S_residual_norm": s_residual,
        "Ms_residual_norm": ms_residual,
        "local_sz": local_sz,
        "kahn_populations": kahn_populations,
    }


def state_label_S_Ms(info: Dict[str, Any], precision: int = 6) -> str:
    """
    Format a state as |S, Ms> when sharp enough, otherwise show expectations.
    """
    if info.get("S") is not None and info.get("Ms") is not None:
        return f"|{info['S']:g}, {info['Ms']:g}>"
    s_text = f"<{info['S2_expectation']:.{precision}g}>"
    ms_text = f"<{info['Sz_expectation']:.{precision}g}>"
    return f"|S?={s_text}, Ms?={ms_text}>"


def state_decomposition(
    system: SpinSystem,
    psi: np.ndarray,
    *,
    cutoff: float = 1e-6,
    max_terms: Optional[int] = 12,
    sort_by: str = "weight",
) -> List[Dict[str, Any]]:
    """
    Expand a state in the natural product basis |m1,m2,...>.
    """
    psi = _normalize_state_vector(psi)
    data: List[Dict[str, Any]] = []
    for idx, coeff in enumerate(psi):
        weight = float(abs(coeff) ** 2)
        if weight >= cutoff:
            data.append(
                {
                    "index": idx,
                    "basis": system.space.basis_label(idx),
                    "coefficient": complex(coeff),
                    "weight": weight,
                }
            )

    key = "weight" if sort_by == "weight" else "index"
    reverse = key == "weight"
    data.sort(key=lambda item: item[key], reverse=reverse)
    if max_terms is not None:
        data = data[: int(max_terms)]
    return data


def bond_observables_in_state(
    system: SpinSystem,
    psi: np.ndarray,
    *,
    eig_tol: float = 1e-10,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Per-bond observables in a single state: local energy, projector overlap,
    projector frustration, and Kahn-like bond population 1-f.
    """
    psi = _normalize_state_vector(psi)
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for bond in system.bonds:
        overlap = system.bond_projector_overlap_in_state(psi, bond, eig_tol=eig_tol)
        energy = system.bond_energy_expectation_in_state(psi, bond)
        key = (bond.i, bond.j)
        out[key] = {
            "bond_energy": energy,
            "projector_overlap": overlap,
            "projector_frustration": 1.0 - overlap,
            "bond_population": overlap,
        }
    return out


def diagonalize_system(
    system: SpinSystem,
    *,
    use_ms_blocks: Optional[bool] = None,
    ms_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the full eigensystem as (energies, row-wise eigenvectors).
    """
    H = system.hamiltonian()
    if use_ms_blocks is None:
        use_ms_blocks = system.commutes_with_total_sz()

    if use_ms_blocks:
        ms_diag = system.space.total_sz_diagonal()
        unique_ms = np.unique(np.rint(2.0 * ms_diag).astype(int)) / 2.0
        all_evals: List[float] = []
        all_vecs: List[np.ndarray] = []
        for ms in unique_ms:
            Hblk, idx = system.restrict_to_ms(H, ms, tol=10 * ms_tol)
            if Hblk.size == 0:
                continue
            evals_blk, evecs_blk = np.linalg.eigh(Hblk)
            for e, v in zip(evals_blk, evecs_blk.T):
                all_evals.append(float(np.real_if_close(e)))
                all_vecs.append(system._embed_sector_vector(v, idx, system.total_dim))
        order = np.argsort(all_evals)
        evals = np.asarray([all_evals[k] for k in order], dtype=float)
        vecs = np.asarray([all_vecs[k] for k in order], dtype=complex)
    else:
        evals, evecs = np.linalg.eigh(H)
        evals = np.asarray(np.real_if_close(evals), dtype=float)
        vecs = np.asarray(evecs.T, dtype=complex)
    return evals, vecs


def spectrum_table(
    system: SpinSystem,
    *,
    n_states: Optional[int] = None,
    use_ms_blocks: Optional[bool] = None,
    ms_tol: float = 1e-12,
    tol: float = 1e-8,
) -> List[Dict[str, Any]]:
    """
    Build a convenient spectrum table with energy and inferred spin labels.
    """
    evals, vecs = diagonalize_system(system, use_ms_blocks=use_ms_blocks, ms_tol=ms_tol)
    ops = build_total_spin_operators(system)
    if n_states is None:
        n_states = len(evals)
    rows: List[Dict[str, Any]] = []
    for idx, (E, psi) in enumerate(zip(evals[:n_states], vecs[:n_states])):
        q = state_quantum_numbers(system, psi, operator_cache=ops, tol=tol)
        rows.append(
            {
                "index": idx,
                "energy": float(E),
                "label": state_label_S_Ms(q),
                "S": q["S"],
                "Ms": q["Ms"],
                "S2_expectation": q["S2_expectation"],
                "Sz_expectation": q["Sz_expectation"],
                "vector": psi,
                "quantum_numbers": q,
            }
        )
    return rows


def print_basis(system: SpinSystem) -> None:
    """Print the natural product basis used internally."""
    print("Product basis:")
    for idx in range(system.total_dim):
        print(f"  {idx:>3d} : {system.space.basis_label(idx)}")


def print_spectrum_summary(
    system: SpinSystem,
    *,
    n_states: Optional[int] = None,
    use_ms_blocks: Optional[bool] = None,
    ms_tol: float = 1e-12,
    tol: float = 1e-8,
) -> List[Dict[str, Any]]:
    """
    Print a compact spectrum table: index, energy, inferred |S,Ms> label, and
    raw expectations.
    """
    rows = spectrum_table(
        system,
        n_states=n_states,
        use_ms_blocks=use_ms_blocks,
        ms_tol=ms_tol,
        tol=tol,
    )
    print("Spectrum summary")
    print("=" * 88)
    print(f"{'idx':>4s}  {'energy':>14s}  {'state label':>18s}  {'<S^2>':>12s}  {'<Sz>':>12s}")
    print("-" * 88)
    for row in rows:
        print(
            f"{row['index']:>4d}  {row['energy']:>14.8f}  {row['label']:>18s}  "
            f"{row['S2_expectation']:>12.6f}  {row['Sz_expectation']:>12.6f}"
        )
    print("=" * 88)
    return rows


def print_state_report(
    system: SpinSystem,
    psi: np.ndarray,
    *,
    name: str = "state",
    coeff_cutoff: float = 1e-6,
    max_terms: int = 12,
    eig_tol: float = 1e-10,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Detailed human-readable report for one state.
    """
    psi = _normalize_state_vector(psi)
    q = state_quantum_numbers(system, psi, tol=tol)
    bond_obs = bond_observables_in_state(system, psi, eig_tol=eig_tol)
    decomposition = state_decomposition(
        system,
        psi,
        cutoff=coeff_cutoff,
        max_terms=max_terms,
    )
    energy = _expectation_value(psi, system.hamiltonian())

    print(f"\n{name}")
    print("=" * 88)
    print(f"Energy               : {energy:.10f}")
    print(f"Configuration        : {state_label_S_Ms(q)}")
    print(f"<S^2>, Var(S^2)      : {q['S2_expectation']:.10f}, {q['S2_variance']:.3e}")
    print(f"<Sz>,  Var(Sz)       : {q['Sz_expectation']:.10f}, {q['Sz_variance']:.3e}")
    print("Local <Sz_i>         : " + ", ".join(f"site {i} = {v:+.6f}" for i, v in q['local_sz'].items()))
    print("Kahn populations 2<Sz_i> : " + ", ".join(f"site {i} = {v:+.6f}" for i, v in q['kahn_populations'].items()))
    print("\nBond observables")
    print("-" * 88)
    print(f"{'bond':>10s}  {'<h_ij>':>14s}  {'overlap':>14s}  {'f_proj':>14s}")
    for bond_key, obs in bond_obs.items():
        print(
            f"{str(bond_key):>10s}  {obs['bond_energy']:>14.8f}  "
            f"{obs['projector_overlap']:>14.8f}  {obs['projector_frustration']:>14.8f}"
        )

    print("\nLargest basis components")
    print("-" * 88)
    for term in decomposition:
        print(
            f"{term['basis']:>18s}   coeff = {_complex_to_text(term['coefficient'])}   "
            f"|c|^2 = {term['weight']:.8f}"
        )

    return {
        "energy": energy,
        "quantum_numbers": q,
        "bond_observables": bond_obs,
        "decomposition": decomposition,
    }


def print_ground_state_report(
    system: SpinSystem,
    *,
    use_ms_blocks: Optional[bool] = None,
    eig_tol: float = 1e-10,
    ms_tol: float = 1e-12,
    coeff_cutoff: float = 1e-6,
    max_terms: int = 12,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Print a report for the exact ground-state manifold and the two built-in
    frustration definitions.
    """
    gs = system.ground_state_manifold(
        use_ms_blocks=use_ms_blocks,
        eig_tol=eig_tol,
        ms_tol=ms_tol,
    )
    fr_proj = system.frustration(
        use_ms_blocks=use_ms_blocks,
        eig_tol=eig_tol,
        ms_tol=ms_tol,
    )
    fr_energy = system.energy_frustration(
        use_ms_blocks=use_ms_blocks,
        eig_tol=eig_tol,
        ms_tol=ms_tol,
    )

    print("\nGround-state manifold")
    print("=" * 88)
    print(f"Ground-state energy  : {gs.energy:.10f}")
    print(f"Degeneracy           : {gs.degeneracy}")
    print(f"Projector frustration: average = {fr_proj.average:.8f}, total = {fr_proj.total:.8f}")
    print(f"Energy frustration   : global  = {fr_energy.global_metric:.8f}, total = {fr_energy.total:.8f}")
    print("Projector per bond   : " + ", ".join(f"{k}={v:.6f}" for k, v in fr_proj.per_bond.items()))
    print("Energy per bond      : " + ", ".join(f"{k}={v:.6f}" for k, v in fr_energy.per_bond.items()))

    reports = []
    for idx, psi in enumerate(gs.vectors):
        reports.append(
            print_state_report(
                system,
                psi,
                name=f"ground state #{idx}",
                coeff_cutoff=coeff_cutoff,
                max_terms=max_terms,
                eig_tol=eig_tol,
                tol=tol,
            )
        )

    return {
        "ground_state": gs,
        "projector_frustration": fr_proj,
        "energy_frustration": fr_energy,
        "state_reports": reports,
    }


def inspect_scan_point(
    system: SpinSystem,
    *,
    bond: Union[int, Tuple[int, int], Bond],
    J: float,
    use_ms_blocks: Optional[bool] = None,
    eig_tol: float = 1e-10,
    ms_tol: float = 1e-12,
    coeff_cutoff: float = 1e-6,
    max_terms: int = 12,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Replace one bond coupling by J and immediately print the ground-state report
    for that modified system.
    """
    scanned = system.with_bond_coupling(bond, J)
    print("\nScanned system")
    print("=" * 88)
    print(f"Varied bond value set to J = {float(J):g}")
    return print_ground_state_report(
        scanned,
        use_ms_blocks=use_ms_blocks,
        eig_tol=eig_tol,
        ms_tol=ms_tol,
        coeff_cutoff=coeff_cutoff,
        max_terms=max_terms,
        tol=tol,
    )



__all__ = [


    'Bond',


    'CouplingScanResult',


    'SpinSystem',


    'plot_coupling_scan',


    '_resolve_selected_bond_keys',


    'compute_bond_population_scan',


    'plot_bond_population_scan',


    'KahnPopulationScanResult',


    '_kahn_unique_ms_values',


    'infer_kahn_ms_target',


    'kahn_spin_populations',


    'scan_kahn_spin_populations',


    'plot_kahn_spin_population_scan',


    '_normalize_state_vector',


    '_expectation_value',


    '_operator_variance',


    '_complex_to_text',


    'build_total_spin_operators',


    'allowed_total_spins',


    'infer_total_spin_from_s2',


    'state_quantum_numbers',


    'state_label_S_Ms',


    'state_decomposition',


    'bond_observables_in_state',


    'diagonalize_system',


    'spectrum_table',


    'print_basis',


    'print_spectrum_summary',


    'print_state_report',


    'print_ground_state_report',


    'inspect_scan_point',


    '_resolve_reference_coupling',


    '_resolve_reference_coupling_for_bond_pop',


    '_resolve_scan_reference_value',


    '_resolve_scan_bond_indices',


    '_equilateral_triangle_scan_indices',


    'plot_frustration_vs_d_over_j',


]
