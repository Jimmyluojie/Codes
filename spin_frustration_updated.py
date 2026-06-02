
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = np.ndarray


def _validate_spin(j: float) -> float:
    j = float(j)
    if j < 0 or not np.isclose(2 * j, round(2 * j)):
        raise ValueError(
            f"Invalid spin {j!r}. Spins must be non-negative integers or half-integers."
        )
    return j


@lru_cache(maxsize=None)
def spin_matrices(j: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the SU(2) spin-j matrices (I, Sx, Sy, Sz) in the basis
    |j,m> with m = j, j-1, ..., -j.
    """
    j = _validate_spin(j)

    d = int(round(2 * j + 1))
    m = np.arange(j, -j - 1, -1, dtype=float)

    Jz = np.diag(m).astype(complex)

    # coefficients for J_+
    coeff = np.sqrt(j * (j + 1) - m[1:] * (m[1:] + 1))
    Jp = np.diag(coeff, k=1).astype(complex)
    Jm = Jp.conj().T

    Jx = 0.5 * (Jp + Jm)
    Jy = (Jp - Jm) / (2.0j)
    I = np.eye(d, dtype=complex)

    return I, Jx, Jy, Jz


@dataclass(frozen=True)
class Bond:
    """
    Ordered bond. The DMI vector is interpreted for the ordered pair (i, j)
    through D · (S_i x S_j). Reversing the bond order changes the sign of the
    DMI contribution unless D is also negated.
    """
    i: int
    j: int
    J: float
    D: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.i == self.j:
            raise ValueError("A bond must connect two different sites.")
        if len(self.D) != 3:
            raise ValueError("D must be a 3-component vector.")


@dataclass
class GroundStateManifold:
    energy: float
    vectors: np.ndarray  # shape (degeneracy, dim)
    degeneracy: int


@dataclass
class FrustrationResult:
    ground_energy: float
    degeneracy: int
    per_bond: Dict[Tuple[int, int], float]
    average: float
    total: float
    global_metric: float
    definition: str


@dataclass
class CouplingScanResult:
    scanned_bond: Tuple[int, int]
    bond_index: int
    scan_param: str
    j_values: np.ndarray
    ground_energies: np.ndarray
    degeneracies: np.ndarray
    global_average: np.ndarray
    global_total: np.ndarray
    global_metric: np.ndarray
    per_bond: Dict[Tuple[int, int], np.ndarray]
    definition: str


class SpinHilbertSpace:
    """
    Hilbert-space utilities for arbitrary local spins.
    """

    def __init__(self, spins: Sequence[float]):
        if len(spins) == 0:
            raise ValueError("At least one spin is required.")
        self.spins = tuple(_validate_spin(s) for s in spins)
        self.n_sites = len(self.spins)
        self.dims = tuple(int(round(2 * s + 1)) for s in self.spins)
        self.total_dim = int(np.prod(self.dims, dtype=int))
        self._identities = tuple(np.eye(d, dtype=complex) for d in self.dims)
        self._site_ops_cache: Dict[Tuple[int, str], np.ndarray] = {}         #cache for site operators S_axis(site)
        self._pair_permutation_cache: Dict[Tuple[int, int], Tuple[Tuple[int, ...], Tuple[int, ...], int]] = {}
        self._total_sz_diag: Optional[np.ndarray] = None

    def local_spin_matrices(self, site: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return spin_matrices(self.spins[site])

    def site_operator(self, site: int, axis: str) -> np.ndarray:
        """
        Return the full-space operator S_axis(site).
        """
        key = (site, axis.lower())
        if key in self._site_ops_cache:
            return self._site_ops_cache[key]

        _, sx, sy, sz = self.local_spin_matrices(site)
        op_local = {"x": sx, "y": sy, "z": sz}[axis.lower()]

        mats = list(self._identities)
        mats[site] = op_local
        op = mats[0]
        for M in mats[1:]:
            op = np.kron(op, M)

        self._site_ops_cache[key] = op
        return op

    def total_sz_operator(self) -> np.ndarray:
        out = np.zeros((self.total_dim, self.total_dim), dtype=complex)
        for site in range(self.n_sites):
            out += self.site_operator(site, "z")
        return out

    def total_sz_diagonal(self) -> np.ndarray:
        """
        Diagonal of S^z_tot in the natural product basis, built without forming the
        full operator.
        """
        if self._total_sz_diag is not None:
            return self._total_sz_diag

        site_ms = []
        for s in self.spins:
            d = int(round(2 * s + 1))
            site_ms.append(np.array([s - k for k in range(d)], dtype=float))

        out = site_ms[0]
        for arr in site_ms[1:]:
            out = (out[:, None] + arr[None, :]).reshape(-1)

        self._total_sz_diag = out
        return out

    def _pair_layout(self, i: int, j: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        key = (i, j)
        if key in self._pair_permutation_cache:
            return self._pair_permutation_cache[key]

        if not (0 <= i < self.n_sites and 0 <= j < self.n_sites):
            raise IndexError("Site index out of range.")
        if i == j:
            raise ValueError("Two-site operator requires i != j.")

        perm = (i, j) + tuple(k for k in range(self.n_sites) if k not in (i, j))
        inv = tuple(np.argsort(perm))
        local_dim = self.dims[i] * self.dims[j]
        self._pair_permutation_cache[key] = (perm, inv, local_dim)
        return perm, inv, local_dim

    def apply_two_site_operator(
        self,
        psi: np.ndarray,
        op_ij: np.ndarray,
        i: int,
        j: int,
    ) -> np.ndarray:
        """
        Apply a two-site operator op_ij defined on H_i ⊗ H_j directly to a full
        state vector without explicitly embedding a huge full-space matrix.
        """
        psi = np.asarray(psi, dtype=complex).reshape(-1)
        if psi.size != self.total_dim:
            raise ValueError(
                f"State has size {psi.size}, expected {self.total_dim}."
            )

        perm, inv, local_dim = self._pair_layout(i, j)
        psi_tensor = psi.reshape(self.dims)
        psi_perm = np.transpose(psi_tensor, perm).reshape(local_dim, -1)
        out_perm = op_ij @ psi_perm
        out = np.transpose(out_perm.reshape(self.dims), inv).reshape(-1)
        return out

    def embed_two_site_operator(
        self,
        op_ij: np.ndarray,
        i: int,
        j: int,
    ) -> np.ndarray:
        """
        Explicitly embed a two-site operator into the full Hilbert space.
        Useful for debugging, but avoided in the main frustration workflow.
        """
        M = np.zeros((self.total_dim, self.total_dim), dtype=complex)
        for col in range(self.total_dim):
            e = np.zeros(self.total_dim, dtype=complex)
            e[col] = 1.0
            M[:, col] = self.apply_two_site_operator(e, op_ij, i, j)
        return M

    def basis_label(self, index: int) -> str:
        """
        Product-basis label |m1,m2,...>.
        """
        if not (0 <= index < self.total_dim):
            raise IndexError("Basis index out of range.")

        idx = index
        labels = []
        for dim, s in zip(reversed(self.dims), reversed(self.spins)):
            digit = idx % dim
            idx //= dim
            m = s - digit
            labels.append(f"{m:g}")
        return "|" + ",".join(reversed(labels)) + ">"


class SpinSystem:
    """
    - arbitrary graph through an explicit bond list
    - arbitrary local spins
    - no global-variable bugs
    - frustration computed without full-space bond projectors
    - cached single-site operators and tensor-permutation layouts
    """

    def __init__(self, spins: Sequence[float], bonds: Sequence[Bond]):
        self.space = SpinHilbertSpace(spins)
        self.bonds = tuple(bonds)
        self._validate_bonds()

        self._hamiltonian_cache: Optional[np.ndarray] = None
        self._local_projector_cache: Dict[Tuple[int, int, float, Tuple[float, float, float]], np.ndarray] = {}
        self._local_bond_h_cache: Dict[Tuple[int, int, float, Tuple[float, float, float]], np.ndarray] = {}

    @property
    def n_sites(self) -> int:
        return self.space.n_sites

    @property
    def spins(self) -> Tuple[float, ...]:
        return self.space.spins

    @property
    def dims(self) -> Tuple[int, ...]:
        return self.space.dims

    @property
    def total_dim(self) -> int:
        return self.space.total_dim

    def _validate_bonds(self) -> None:
        seen = set()
        for b in self.bonds:
            if not (0 <= b.i < self.n_sites and 0 <= b.j < self.n_sites):
                raise IndexError(f"Bond {(b.i, b.j)} uses an invalid site index.")
            if (b.i, b.j) in seen:
                raise ValueError(f"Duplicate ordered bond {(b.i, b.j)}.")
            seen.add((b.i, b.j))

    def commutes_with_total_sz(self, tol: float = 1e-12) -> bool:
        """
        Heuristic exact condition for this model:
        Heisenberg always conserves S^z_tot; DMI conserves it only when D_x=D_y=0
        on every bond.
        """
        for b in self.bonds:
            if abs(b.D[0]) > tol or abs(b.D[1]) > tol:
                return False
        return True

    def local_bond_hamiltonian(self, bond: Bond) -> np.ndarray:
        """
        Two-site Hamiltonian h_ij acting only on H_i ⊗ H_j:
            J S_i·S_j + D·(S_i × S_j)
        """
        key = (bond.i, bond.j, float(bond.J), tuple(map(float, bond.D)))
        if key in self._local_bond_h_cache:
            return self._local_bond_h_cache[key]

        _, sxi, syi, szi = self.space.local_spin_matrices(bond.i)
        _, sxj, syj, szj = self.space.local_spin_matrices(bond.j)

        Hloc = bond.J * (
            np.kron(sxi, sxj) +
            np.kron(syi, syj) +
            np.kron(szi, szj)
        )

        Dx, Dy, Dz = bond.D
        if Dx or Dy or Dz:
            Hloc += (
                Dx * (np.kron(syi, szj) - np.kron(szi, syj))
                + Dy * (np.kron(szi, sxj) - np.kron(sxi, szj))
                + Dz * (np.kron(sxi, syj) - np.kron(syi, sxj))
            )

        self._local_bond_h_cache[key] = Hloc
        return Hloc

    def hamiltonian(self) -> np.ndarray:
        """
        Full Hamiltonian in the product basis.
        """
        if self._hamiltonian_cache is not None:
            return self._hamiltonian_cache

        sx_list = [self.space.site_operator(i, "x") for i in range(self.n_sites)]
        sy_list = [self.space.site_operator(i, "y") for i in range(self.n_sites)]
        sz_list = [self.space.site_operator(i, "z") for i in range(self.n_sites)]

        H = np.zeros((self.total_dim, self.total_dim), dtype=complex)
        for b in self.bonds:
            i, j = b.i, b.j
            H += b.J * (
                sx_list[i] @ sx_list[j]
                + sy_list[i] @ sy_list[j]
                + sz_list[i] @ sz_list[j]
            )
            Dx, Dy, Dz = b.D
            if Dx or Dy or Dz:
                H += (
                    Dx * (sy_list[i] @ sz_list[j] - sz_list[i] @ sy_list[j])
                    + Dy * (sz_list[i] @ sx_list[j] - sx_list[i] @ sz_list[j])
                    + Dz * (sx_list[i] @ sy_list[j] - sy_list[i] @ sx_list[j])
                )

        self._hamiltonian_cache = H
        return H

    def restrict_to_ms(self, H: np.ndarray, ms_target: float, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restricts the Hamiltonian matrix H to the subspace where the total spin
        projection matches ms_target within a tolerance tol, returning the
        reduced matrix and the corresponding indices.

        This operation is only valid if H commutes with S^z_tot. If not, fixed-Ms
        sectors are not invariant and truncating to one sector is not physical.
        """
        Sz_tot = self.space.total_sz_operator()
        comm = H @ Sz_tot - Sz_tot @ H
        comm_norm = np.linalg.norm(comm, ord="fro")
        scale = max(
            np.linalg.norm(H, ord="fro") * np.linalg.norm(Sz_tot, ord="fro"),
            1.0,
        )
        if comm_norm > tol * scale:
            raise ValueError(
                "Cannot restrict to a fixed Ms sector because H does not commute "
                "with S^z_tot within tolerance. Use full-space diagonalization "
                "(use_ms_blocks=False) or remove non-conserving terms."
            )

        ms_diag = self.space.total_sz_diagonal()
        idx = np.where(np.abs(ms_diag - ms_target) < tol)[0]
        return H[np.ix_(idx, idx)], idx

    @staticmethod
    def _embed_sector_vector(v_sector: np.ndarray, idx: np.ndarray, full_dim: int) -> np.ndarray:
        """
        Creates a complex vector of length full_dim, embedding the values from
        v_sector at positions specified by idx, with zeros elsewhere.
        """
        v = np.zeros(full_dim, dtype=complex)
        v[idx] = np.asarray(v_sector).reshape(-1)
        return v

    def ground_state_manifold(
        self,
        use_ms_blocks: Optional[bool] = None,
        eig_tol: float = 1e-10,
        ms_tol: float = 1e-12,
    ) -> GroundStateManifold:
        """
        Compute the exact ground-state manifold.

        If use_ms_blocks is None, the code automatically uses M_s blocks only when
        the Hamiltonian conserves S^z_tot.
        """
        H = self.hamiltonian()
        if use_ms_blocks is None:
            use_ms_blocks = self.commutes_with_total_sz()

        if use_ms_blocks:
            ms_diag = self.space.total_sz_diagonal()
            # Total M_s is always an integer or half-integer for physical spins.
            unique_ms = np.unique(np.rint(2.0 * ms_diag).astype(int)) / 2.0

            all_evals: List[float] = []
            all_vecs: List[np.ndarray] = []

            for ms in unique_ms:
                Hblk, idx = self.restrict_to_ms(H, ms, tol=10 * ms_tol)
                if Hblk.size == 0:
                    continue
                evals_blk, evecs_blk = np.linalg.eigh(Hblk)
                for e, v in zip(evals_blk, evecs_blk.T):
                    all_evals.append(float(np.real_if_close(e)))
                    all_vecs.append(self._embed_sector_vector(v, idx, self.total_dim))

            order = np.argsort(all_evals)
            evals = np.asarray([all_evals[k] for k in order], dtype=float)
            vecs = np.asarray([all_vecs[k] for k in order], dtype=complex)
        else:
            evals, evecs = np.linalg.eigh(H)
            evals = np.asarray(np.real_if_close(evals), dtype=float)
            vecs = evecs.T.copy()

        E0 = float(evals[0])
        gs_idx = np.where(np.abs(evals - E0) < eig_tol)[0]
        gs_vecs = vecs[gs_idx]
        return GroundStateManifold(
            energy=E0,
            vectors=gs_vecs,
            degeneracy=len(gs_idx),
        )

    @staticmethod
    def local_ground_projector(Hloc: np.ndarray, eig_tol: float = 1e-10) -> Tuple[float, np.ndarray, int]:
        """
        Returns the ground state energy, projector onto the ground state
        subspace, and its degeneracy for a local Hamiltonian Hloc by
        diagonalizing it and selecting eigenvectors within eig_tol of the lowest
        eigenvalue.
        """
        evals, evecs = np.linalg.eigh(Hloc)
        E0 = float(np.real_if_close(evals[0]))
        idx = np.where(np.abs(evals - E0) < eig_tol)[0]
        P = np.zeros_like(Hloc, dtype=complex)
        for k in idx:
            v = evecs[:, k]
            P += np.outer(v, v.conj())
        return E0, P, len(idx)

    def local_bond_projector(self, bond: Bond, eig_tol: float = 1e-10) -> np.ndarray:
        """
        Returns the local ground state projector for a given bond in the spin
        system, using a cached value if available. Computes the projector via
        the local bond Hamiltonian and stores it for future use. The eig_tol
        parameter sets the eigenvalue tolerance.
        """
        key = (bond.i, bond.j, float(bond.J), tuple(map(float, bond.D)))
        if key in self._local_projector_cache:
            return self._local_projector_cache[key]

        _, P, _ = self.local_ground_projector(self.local_bond_hamiltonian(bond), eig_tol=eig_tol)
        self._local_projector_cache[key] = P
        return P

    def bond_projector_overlap_in_state(
        self,
        psi: np.ndarray,
        bond: Bond,
        eig_tol: float = 1e-10,
    ) -> float:
        """
        Return <psi|P_ij|psi> using the local projector P_ij, applied directly in the
        full Hilbert space without explicitly building a full projector matrix.
        """
        psi = np.asarray(psi, dtype=complex).reshape(-1)
        norm = np.vdot(psi, psi).real
        if norm <= 0:
            raise ValueError("State vector has zero norm.")

        Pij = self.local_bond_projector(bond, eig_tol=eig_tol)
        Ppsi = self.space.apply_two_site_operator(psi, Pij, bond.i, bond.j)
        return float((np.vdot(psi, Ppsi) / norm).real)

    def bond_energy_expectation_in_state(
        self,
        psi: np.ndarray,
        bond: Bond,
    ) -> float:
        """
        Return <psi|h_ij|psi> for one bond Hamiltonian h_ij.
        """
        psi = np.asarray(psi, dtype=complex).reshape(-1)
        norm = np.vdot(psi, psi).real
        if norm <= 0:
            raise ValueError("State vector has zero norm.")

        hij = self.local_bond_hamiltonian(bond)
        hpsi = self.space.apply_two_site_operator(psi, hij, bond.i, bond.j)
        return float((np.vdot(psi, hpsi) / norm).real)


    def energy_frustration(
        self,
        use_ms_blocks: Optional[bool] = None,
        eig_tol: float = 1e-10,
        ms_tol: float = 1e-12,
    ) -> FrustrationResult:
        """
        Energy-normalized frustration:
            F = (E_GS - E_min) / (E_max - E_min)

        Here E_min and E_max are obtained by summing the minimum and maximum
        eigenvalues of each local bond Hamiltonian h_ij. The per-bond values are
        defined analogously from the bond-resolved expectation values.
        """
        gs = self.ground_state_manifold(
            use_ms_blocks=use_ms_blocks,
            eig_tol=eig_tol,
            ms_tol=ms_tol,
        )

        per_bond: Dict[Tuple[int, int], float] = {}
        emin_total = 0.0
        emax_total = 0.0

        for bond in self.bonds:
            hloc = self.local_bond_hamiltonian(bond)
            evals = np.linalg.eigvalsh(hloc)
            emin = float(np.real_if_close(evals[0]))
            emax = float(np.real_if_close(evals[-1]))
            emin_total += emin
            emax_total += emax

            e_avg = 0.0
            for psi in gs.vectors:
                e_avg += self.bond_energy_expectation_in_state(psi, bond)
            e_avg /= gs.degeneracy

            denom = emax - emin
            if abs(denom) < eig_tol:
                per_bond[(bond.i, bond.j)] = 0.0
            else:
                per_bond[(bond.i, bond.j)] = (e_avg - emin) / denom

        total = float(sum(per_bond.values()))
        average = total / len(self.bonds) if self.bonds else 0.0

        denom_global = emax_total - emin_total
        if abs(denom_global) < eig_tol:
            global_metric = 0.0
        else:
            global_metric = (gs.energy - emin_total) / denom_global

        return FrustrationResult(
            ground_energy=gs.energy,
            degeneracy=gs.degeneracy,
            per_bond=per_bond,
            average=average,
            total=total,
            global_metric=float(global_metric),
            definition="energy",
        )

    def energy_lower_bound_gap(
        self,
        use_ms_blocks: Optional[bool] = None,
        eig_tol: float = 1e-10,
        ms_tol: float = 1e-12,
    ) -> float:
        """
        Return the absolute frustration gap

            Delta = E0 - sum_{<ij>} e_ij^min

        where E0 is the exact ground-state energy of the full Hamiltonian and
        e_ij^min is the minimum eigenvalue of the local bond Hamiltonian h_ij.

        A strictly positive gap indicates frustration in the energetic lower-bound
        sense: no global state can minimize every bond simultaneously.
        """
        gs = self.ground_state_manifold(
            use_ms_blocks=use_ms_blocks,
            eig_tol=eig_tol,
            ms_tol=ms_tol,
        )

        emin_total = 0.0
        for bond in self.bonds:
            evals = np.linalg.eigvalsh(self.local_bond_hamiltonian(bond))
            emin_total += float(np.real_if_close(evals[0]))

        gap = float(gs.energy - emin_total)
        if abs(gap) < eig_tol:
            return 0.0
        return gap

    def is_frustrated_lower_bound(
        self,
        *,
        criterion_tol: float = 1e-10,
        use_ms_blocks: Optional[bool] = None,
        eig_tol: float = 1e-10,
        ms_tol: float = 1e-12,
    ) -> bool:
        """
        Boolean form of the criterion E0 > sum e_ij^min.

        Returns True when the lower-bound gap is greater than criterion_tol.
        """
        gap = self.energy_lower_bound_gap(
            use_ms_blocks=use_ms_blocks,
            eig_tol=eig_tol,
            ms_tol=ms_tol,
        )
        return bool(gap > criterion_tol)

    def frustration(
        self,
        use_ms_blocks: Optional[bool] = None,
        eig_tol: float = 1e-10,
        ms_tol: float = 1e-12,
    ) -> FrustrationResult:
        """
        Compute the projector-based frustration
            f_ij = 1 - Tr(P_ij rho_GS),
        with rho_GS the equal-weight mixture over the exact ground-state manifold.
        """
        gs = self.ground_state_manifold(
            use_ms_blocks=use_ms_blocks,
            eig_tol=eig_tol,
            ms_tol=ms_tol,
        )

        f_bonds: Dict[Tuple[int, int], float] = {}
        for bond in self.bonds:
            overlap_sum = 0.0
            for psi in gs.vectors:
                overlap_sum += self.bond_projector_overlap_in_state(psi, bond, eig_tol=eig_tol)
            overlap = overlap_sum / gs.degeneracy
            f_bonds[(bond.i, bond.j)] = 1.0 - overlap

        total = float(sum(f_bonds.values()))
        average = total / len(self.bonds) if self.bonds else 0.0
        return FrustrationResult(
            ground_energy=gs.energy,
            degeneracy=gs.degeneracy,
            per_bond=f_bonds,
            average=average,
            total=total,
            global_metric=average,
            definition="projector",
        )

    def frustration_by_definition(
        self,
        definition: str = "projector",
        use_ms_blocks: Optional[bool] = None,
        eig_tol: float = 1e-10,
        ms_tol: float = 1e-12,
    ) -> FrustrationResult:
        """
        Dispatch frustration calculation by definition name.

        Supported definitions:
        - "projector": local-projector incompatibility
        - "energy": energy-normalized frustration
        """
        definition = str(definition).strip().lower()
        if definition == "projector":
            return self.frustration(
                use_ms_blocks=use_ms_blocks,
                eig_tol=eig_tol,
                ms_tol=ms_tol,
            )
        if definition == "energy":
            return self.energy_frustration(
                use_ms_blocks=use_ms_blocks,
                eig_tol=eig_tol,
                ms_tol=ms_tol,
            )
        raise ValueError("definition must be 'projector' or 'energy'.")

    def _resolve_bond_reference(self, bond: Union[int, Tuple[int, int], Bond]) -> Tuple[int, Bond]:
        """
        Resolves a bond reference given as an index, ordered pair, or Bond
        object, returning a tuple of the bond index and the corresponding Bond
        instance. Raises an error if the reference is invalid or ambiguous.
        """
        if isinstance(bond, int):
            if not (0 <= bond < len(self.bonds)):
                raise IndexError(f"Bond index {bond} out of range.")
            return bond, self.bonds[bond]

        if isinstance(bond, Bond):
            for idx, existing in enumerate(self.bonds):
                if existing == bond:
                    return idx, existing
            raise ValueError("The provided Bond object is not present in this SpinSystem.")

        if isinstance(bond, tuple) and len(bond) == 2:
            key = tuple(int(x) for x in bond)
            matches = [idx for idx, existing in enumerate(self.bonds) if (existing.i, existing.j) == key]
            if not matches:
                raise ValueError(f"No bond with ordered pair {key} exists in this SpinSystem.")
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple bonds match {key}; use an explicit bond index instead of the pair."
                )
            idx = matches[0]
            return idx, self.bonds[idx]

        raise TypeError("bond must be a bond index, an ordered pair (i, j), or a Bond object.")

    def with_bond_coupling(self, bond: Union[int, Tuple[int, int], Bond], J: float) -> "SpinSystem":
        """
        HELPER METHOD FOR SCANNING:
        Updates the coupling constant J 
        """
        bond_index, _ = self._resolve_bond_reference(bond)
        new_bonds = list(self.bonds)
        target = new_bonds[bond_index]
        new_bonds[bond_index] = Bond(i=target.i, j=target.j, J=float(J), D=target.D)
        return SpinSystem(spins=self.spins, bonds=new_bonds)

    def scan_bond_coupling(
        self,
        bond: Union[int, Tuple[int, int], Bond],
        j_values: Optional[Sequence[float]] = None,
        *,
        scan_values: Optional[Sequence[float]] = None,
        scan_param: str = "J",
        frustration_definition: str = "projector",
        use_ms_blocks: Optional[bool] = None,
        eig_tol: float = 1e-10,
        ms_tol: float = 1e-12,
    ) -> CouplingScanResult:
        """
        Vary the Heisenberg coupling J on one bond and track the global and partial
        frustration across the scan.

        Parameters
        ----------
        bond
            Bond to scan. Can be the bond index in ``self.bonds``, an ordered pair
            ``(i, j)``, or the exact ``Bond`` instance.
        j_values / scan_values
            Sequence of values to assign to the scanned bond parameter.
            ``j_values`` is kept for backward compatibility.
        scan_param
            One of {"J", "Dx", "Dy", "Dz"}.
        """
        bond_index, target_bond = self._resolve_bond_reference(bond)

        if j_values is not None and scan_values is not None:
            raise ValueError("Pass only one of j_values or scan_values, not both.")
        values_in = scan_values if scan_values is not None else j_values
        if values_in is None:
            raise ValueError("You must pass scan_values (or j_values for backward compatibility).")

        scan_param = str(scan_param).strip()
        if scan_param not in {"J", "Dx", "Dy", "Dz"}:
            raise ValueError("scan_param must be one of: 'J', 'Dx', 'Dy', 'Dz'.")

        j_values = np.asarray(values_in, dtype=float)
        if j_values.ndim != 1 or j_values.size == 0:
            raise ValueError("scan values must be a non-empty 1D sequence.")

        energies = np.empty(j_values.size, dtype=float)
        degeneracies = np.empty(j_values.size, dtype=int)
        global_average = np.empty(j_values.size, dtype=float)
        global_total = np.empty(j_values.size, dtype=float)
        global_metric = np.empty(j_values.size, dtype=float)
        per_bond_series = {
            (b.i, b.j): np.empty(j_values.size, dtype=float)
            for b in self.bonds
        }

        for k, value in enumerate(j_values):
            if scan_param == "J":
                scanned = self.with_bond_coupling(bond_index, float(value))
            else:
                new_bonds = list(self.bonds)
                old = new_bonds[bond_index]
                dvec = list(map(float, old.D))
                dvec[{"Dx": 0, "Dy": 1, "Dz": 2}[scan_param]] = float(value)
                new_bonds[bond_index] = Bond(i=old.i, j=old.j, J=float(old.J), D=tuple(dvec))
                scanned = SpinSystem(spins=self.spins, bonds=new_bonds)
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

        return CouplingScanResult(
            scanned_bond=(target_bond.i, target_bond.j),
            bond_index=bond_index,
            scan_param=scan_param,
            j_values=j_values,
            ground_energies=energies,
            degeneracies=degeneracies,
            global_average=global_average,
            global_total=global_total,
            global_metric=global_metric,
            per_bond=per_bond_series,
            definition=str(frustration_definition).strip().lower(),
        )

    def plot_bond_coupling_scan(
        self,
        bond: Union[int, Tuple[int, int], Bond],
        j_values: Sequence[float],
        include_global_total: bool = False,
        include_per_bond: bool = True,
        show_scanned_bond_only: bool = False,
        frustration_definition: str = "projector",
        use_ms_blocks: Optional[bool] = None,
        eig_tol: float = 1e-10,
        ms_tol: float = 1e-12,
        ax=None,
    ):
        """
        Plot the frustration as a function of the coupling J on a selected bond.

        By default the plot contains:
        - the global average frustration
        - one curve for each bond's partial frustration

        Set ``include_global_total=True`` to also plot the summed frustration.
        Set ``show_scanned_bond_only=True`` to show only the varying bond among the
        partial-frustration curves.
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator

        scan = self.scan_bond_coupling(
            bond=bond,
            j_values=j_values,
            frustration_definition=frustration_definition,
            use_ms_blocks=use_ms_blocks,
            eig_tol=eig_tol,
            ms_tol=ms_tol,
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        x = scan.j_values
        ax.plot(x, scan.global_metric, linewidth=2.5, label=f'{"energy-based frustration" if scan.definition == "energy" else scan.definition.capitalize() + " global metric"}')
        if include_global_total:
            ax.plot(x, scan.global_total, linewidth=2.0, linestyle='--', label='Global total frustration')

        if include_per_bond:
            for bond_key, y in scan.per_bond.items():
                if show_scanned_bond_only and bond_key != scan.scanned_bond:
                    continue
                label = f'Partial frustration {bond_key}'
                if bond_key == scan.scanned_bond:
                    label += ' [scanned bond]'
                ax.plot(x, y, linestyle=':', label=label)

        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.2)
        ax.tick_params(
            axis='both',
            which='major',
            direction='in',
            top=True,
            right=True,
            length=6,
            width=1.2,
            colors='black',
            labelsize=10,
        )
        ax.tick_params(
            axis='both',
            which='minor',
            direction='in',
            top=True,
            right=True,
            length=3,
            width=1.0,
            colors='black',
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.set_xlabel(f'J on bond {scan.scanned_bond}')
        ax.set_ylabel('Frustration')
        ax.set_title(f'Frustration scan for bond {scan.scanned_bond}')
        ax.grid(False)
        ax.legend(frameon=True, framealpha=1.0, edgecolor='black', facecolor='white', fancybox=False)
        fig.tight_layout()
        return fig, ax, scan

    @classmethod
    def from_ring(
        cls,
        spins: Sequence[float],
        J: Sequence[float],
        D: Optional[Sequence[Sequence[float]]] = None,
        periodic: bool = True,
    ) -> "SpinSystem":
        """
        Convenience constructor matching the old notebook's nearest-neighbor ring
        convention.
        """
        n = len(spins)
        if len(J) != n and periodic:
            raise ValueError("For a periodic ring, J must have length N.")
        if not periodic and len(J) != n - 1:
            raise ValueError("For an open chain, J must have length N-1.")

        if D is None:
            if periodic:
                D = [(0.0, 0.0, 0.0)] * n
            else:
                D = [(0.0, 0.0, 0.0)] * (n - 1)

        bonds = []
        last = n if periodic else n - 1
        for i in range(last):
            j = (i + 1) % n
            bonds.append(Bond(i=i, j=j, J=float(J[i]), D=tuple(float(x) for x in D[i])))
        return cls(spins=spins, bonds=bonds)
    
    


# def _demo() -> None:
#     # AFM triangle of three spin-1/2 sites
#     system = SpinSystem.from_ring(
#         spins=[0.5, 0.5, 0.5],
#         J=[10.0, 10.0, 10.0],
#         D=[(0.0, 0.0, 0.0)] * 3,
#         periodic=True,
#     )
#     result = system.frustration()
#     print("Ground-state energy:", result.ground_energy)
#     print("Ground-state degeneracy:", result.degeneracy)
#     print("Per-bond frustration:", result.per_bond)

#     scan_values = np.linspace(-2.0, 2.0, 21)
#     _, _, scan = system.plot_bond_coupling_scan((0, 1), scan_values, show_scanned_bond_only=True)
#     print("Scanned J values:", scan.j_values)
#     print("Global average frustration:", scan.global_average)
#     print("Average frustration:", result.average)
#     print("Total frustration:", result.total)


# if __name__ == "__main__":
#     _demo()
