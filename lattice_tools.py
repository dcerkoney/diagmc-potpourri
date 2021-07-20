#!/usr/bin/env python3

# Package imports
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from types import LambdaType
from matplotlib import cm
from itertools import product
from scipy.integrate import quad
from scipy import fftpack, interpolate, optimize


##############################################
# Generic lattice (and HEG) helper functions #
##############################################


def safe_filename(dir, savename, file_extension='pdf', overwrite=False):
    # Optionally avoid overwriting duplicate filenames
    filename = pathlib.PosixPath(dir) / f'{savename}.{file_extension}'
    if not overwrite:
        dup_count = 1
        while filename.is_file():
            filename = pathlib.PosixPath(
                dir) / f'{savename}({dup_count}).{file_extension}'
            dup_count += 1
    return filename


def batch_dot(a, b):
    '''Vector dot product for batch (integration) numpy
       arrays (vectors with an extra batch index).'''
    return np.sum(a*b, axis=1).reshape((a.shape[0], 1))


def difference_n_torus(n1, n2, N, a=None):
    '''Get the difference between two vectors on a square n-torus
       of side length L = N a, where a is the lattice constant,
       and N is the number of sites per direction.'''
    del_n1_n2 = np.minimum(np.abs(n1 - n2), N - np.abs(n1 - n2))
    if a:
        return a * del_n1_n2.astype(n1.dtype)
    else:
        return del_n1_n2.astype(n1.dtype)


def distance_n_torus(N, a, n1, n2):
    '''Get the distance between two vectors on a square n-torus
       of side length L = N a, where a is the lattice constant,
       and N is the number of sites per direction.'''
    del_12 = np.minimum(np.abs(n1 - n2), N - np.abs(n1 - n2))
    return a * np.sqrt(np.dot(del_12, del_12))


def distance_n_torus_batch(N, a, n1, n2):
    '''Get the distance between two batches of vectors on a square
       n-torus of side length L = N a, where a is the lattice
       constant, and N is the number of sites per direction.'''
    del_12 = np.minimum(np.abs(n1 - n2), N - np.abs(n1 - n2))
    return a * np.sqrt(batch_dot(del_12, del_12))


def antiperiodic(my_fn, x, period):
    '''Get the antiperiodic extension of f(x) 
       limited to the domain [0, T) for scalar x.'''
    sign = 1
    while x >= period:
        sign *= -1
        x -= period
    while x < 0:
        sign *= -1
        x += period
    return sign * my_fn(x)


def antiperiodic_batch(my_vec_fn, x, period):
    '''Get the antiperiodic extension of f(x)
       limited to the domain [0, T) for vector x.'''
    signs = np.ones(x.shape, dtype=float)
    x_shifted = np.copy(x)
    while np.any(x_shifted >= period):
        signs[x_shifted >= period] *= -1
        x_shifted[x_shifted >= period] -= period
    while np.any(x_shifted < 0):
        signs[x_shifted < 0] *= -1
        x_shifted[x_shifted < 0] += period
    return signs * my_vec_fn(x_shifted)


def ferm(x, is1D=False):
    '''Hyperbolic function representation of the Fermi
       function. Takes a number or numpy array as argument
       and places a finite cutoff at abs(x) = cutoff.'''
    # Defines the largest np.float64 which can be exponentiated (roughly 700)
    cutoff = np.finfo('d').max
    # Fermi function for a scalar arument x
    if is1D:
        if x > cutoff:
            return 0.0
        elif x < -cutoff:
            return 1.0
        else:
            return (1.0 - np.tanh(x / 2.0)) / 2.0
    # Fermi function for n-d array x
    else:
        y = x.copy()
        y[x > cutoff] = 0.0
        y[x < -cutoff] = 1.0
        y[np.abs(x) <= cutoff] = (
            1.0 - np.tanh(y[np.abs(x) <= cutoff] / 2.0)) / 2.0
        return y


def ferm_direct(x, is1D=False):
    '''Canonical (numerically naive) representation of the Fermi
       function; takes a number or numpy array as argument, and
       places a finite cutoff at abs(x) = cutoff.'''
    # Defines the largest np.float64 which can be exponentiated (roughly 700)
    cutoff = np.log(np.finfo('d').max)
    # Fermi function for a scalar arument x
    if is1D:
        if x > cutoff:
            return 0.0
        elif x < -cutoff:
            return 1.0
        else:
            return 1.0 / (1.0 + np.exp(x))
    # Fermi function for n-d array x
    else:
        y = x.copy()
        y[x > cutoff] = 0.0
        y[x < -cutoff] = 1.0
        y[np.abs(x) <= cutoff] = 1.0 / (1.0 + np.exp(y[np.abs(x) <= cutoff]))
        return y


def lat_epsilon_k(k, lat_const, t_hop, qp_rescale=1.0, meshgrid=False):
    '''Dispersion relation of the hypercubic lattice; assumes k is a numpy array
       whose last axis runs over the Cartesian k-point coefficients (k_1, ..., k_d),
       unless otherwise specified as a numpy meshgrid.'''
    if meshgrid:
        return -2 * t_hop * np.sum(np.cos(k) * lat_const, axis=0) * qp_rescale
    else:
        return -2 * t_hop * np.sum(np.cos(k) * lat_const, axis=len(k.shape)-1) * qp_rescale


def lat_epsilon_k_meshgrid(dim, n_site_pd, lat_const, t_hop, qp_rescale=1.0):
    '''Dispersion relation of the hypercubic lattice, evaluated on the entire k-mesh.'''
    kscale = 2.0*np.pi / float(n_site_pd * lat_const)
    ki_bz = kscale * np.arange(np.floor(-n_site_pd / 2.0) + 1,
                               np.floor(n_site_pd / 2.0) + 1, dtype=int)
    ki_meshes = [[ki_bz]] * dim
    k_meshgrid = np.asarray(np.meshgrid(*ki_meshes))
    epsilon_k_meshgrid = lat_epsilon_k(
        k=k_meshgrid,
        lat_const=lat_const,
        t_hop=t_hop,
        qp_rescale=qp_rescale,
        meshgrid=True,
    )
    return k_meshgrid, epsilon_k_meshgrid


def fill_band(dim, num_elec, n_site_pd, lat_const, t_hop, plot_filling=False):
    '''Get the Fermi momentum and energy by filling the band with N_e electrons.'''
    # Reciprocal lattice scale
    k_scale = 2.0*np.pi / float(n_site_pd*lat_const)
    # Get all the momentum magnitudes and coordinates in the lattice
    # (includes momentum and spin degeneracies)
    ek_levels, k_coords = get_all_k_sigma_states(
        dim=dim,
        n_site_pd=n_site_pd,
        lat_const=lat_const,
        k_scale=k_scale,
        t_hop=t_hop,
    )
    # Get the N_e filled levels and respective k-point coordinates using
    # the argpartition algorithm (sorts only the first num_elec array entries)
    filled_indices = np.argpartition(ek_levels, kth=num_elec)[:num_elec]
    filled_levels = ek_levels[filled_indices]
    filled_coords = k_coords[filled_indices]
    assert filled_levels.size == num_elec
    assert filled_coords.shape == (num_elec, dim)
    # The Fermi momentum is the maximum of these filled levels
    ef = np.max(filled_levels)
    # The Fermi wavevectors are defined as those states where k * k = 2 ef; due to
    # the inversion symmetry, we can transform all vectors to the first quadrant.
    # Note that we list ALL degenerate Fermi wavevectors here, even though not
    # all such momentum states are filled (at half-filling, e.g., half are!).
    kf_vecs = []
    for i, k_coord in enumerate(k_coords):
        if np.allclose(ek_levels[i], ef, rtol=1e-13):
            # Transform the coordinate to the first quadrant
            kf_vecs.append(np.abs(k_coord))
            # The vector generated by reflection about
            # x = y is also in the first quadrant
            kf_vecs.append(np.abs(k_coord[::-1]))
    kf_vecs = k_scale * np.unique(np.asarray(kf_vecs), axis=0)
    # Plot the momentum state occupancy in the Brillouin zone
    if plot_filling:
        P_BZ = n_site_pd // 2
        range_BZ = np.arange(-P_BZ, P_BZ)
        xx_BZ, yy_BZ = np.meshgrid(range_BZ, range_BZ)
        filling_BZ = np.zeros(xx_BZ.shape)
        for i in range_BZ:
            for j in range_BZ:
                for coord in filled_coords:
                    if np.all([i, j] == coord):
                        filling_BZ[i + P_BZ, j + P_BZ] += 1
        print(filling_BZ)
        print(num_elec)
        print(np.sum(filling_BZ))
        # Draw the Brillouin zone boundaries, and the reduced Brillouin zone boundaries at half-filling
        fig, ax = plt.subplots()
        x = k_scale * np.linspace(-P_BZ, P_BZ, num=200)
        y = np.arccos((-ef / (2.0 * t_hop)) - np.cos(x))
        ax.plot(x/k_scale, y/k_scale, color='C0')
        ax.plot(x/k_scale, -y/k_scale, color='C0')
        ax.plot([-P_BZ, -P_BZ], [P_BZ, -P_BZ], 'r-')
        ax.plot([-P_BZ, P_BZ], [-P_BZ, -P_BZ], 'r-')
        ax.plot([P_BZ, P_BZ], [P_BZ, -P_BZ], 'r--')
        ax.plot([-P_BZ, P_BZ], [P_BZ, P_BZ], 'r--')
        ax.plot([-P_BZ, 0], [0, -P_BZ], 'c-')
        ax.plot([-P_BZ, 0], [0, P_BZ], 'c-')
        ax.plot([P_BZ, 0], [0, -P_BZ], 'c--')
        ax.plot([P_BZ, 0], [0, P_BZ], 'c--')
        ax.scatter(xx_BZ, yy_BZ, c=filling_BZ, zorder=10,
                   cmap=cm.get_cmap('Greys'), edgecolors='k')
        ax.set_xlim(-P_BZ - 0.5, P_BZ + 0.5)
        ax.set_ylim(-P_BZ - 0.5, P_BZ + 0.5)
        ax.set_xlabel(r'$\mathbf{k}_x$')
        ax.set_ylabel(r'$\mathbf{k}_y$')
        ax.set_xticks((-P_BZ, 0, P_BZ))
        ax.set_yticks((-P_BZ, 0, P_BZ))
        ax.set_xticklabels((r'$-\pi/a$', r'$0$', r'$\pi/a$'))
        ax.set_yticklabels((r'$-\pi/a$', r'$0$', r'$\pi/a$'))
        ax.set_aspect('equal')
        fig.savefig('filled_momentum_states_n_site_pd='+str(n_site_pd)+'.pdf')
        plt.close('all')
    return ef, kf_vecs


def get_all_k_sigma_states(dim, n_site_pd, lat_const, k_scale, t_hop, shift=False):
    '''Returns a numpy array of all momentum states on the lattice,
       including all momentum / spin degeneracies (for band filling).'''
    # Calculate k-magnitudes relative to the Gamma point
    unshifted_indices = np.arange(
        1 - n_site_pd // 2, 1 + n_site_pd // 2, dtype=int)
    indices = unshifted_indices
    if shift:
        shift = n_site_pd // 2
        indices = np.concatenate(
            (unshifted_indices[shift:], unshifted_indices[:shift]))
    # print(indices)
    # Get the list of distances and coordinates
    if dim == 2:
        def epsilon_k(i, j): return lat_epsilon_k(
            k=k_scale * np.array([i, j]),
            lat_const=lat_const,
            t_hop=t_hop,
            meshgrid=False,
        )
        ek_levels_spinless = np.asarray(
            [epsilon_k(i, j) for i in indices for j in indices])
        k_coords_spinless = np.asarray(
            [[i, j] for i in indices for j in indices])
    elif dim == 3:
        def epsilon_k(i, j, k): return lat_epsilon_k(
            k=k_scale * np.array([i, j, k]),
            lat_const=lat_const,
            t_hop=t_hop,
            meshgrid=False,
        )
        ek_levels_spinless = np.asarray(
            [epsilon_k(i, j, k) for i in indices for j in indices for k in indices])
        k_coords_spinless = np.asarray(
            [[i, j, k] for i in indices for j in indices for k in indices])
    else:
        raise ValueError('Invalid spatial dimension d = '+str(dim) +
                         '; one of the following is required: d = {2, 3}!')
    # Put in the spin degeneracy by explicitly doubling each level (necessary if num_elec is odd)
    ek_levels = np.repeat(ek_levels_spinless, 2)
    k_coords = np.repeat(k_coords_spinless, 2, axis=0)
    # Remove double-counting at the Brillouin zone edge
    return ek_levels, k_coords


def test_distance_roundoff(n_site_pd, lat_const):
    '''Test for no roundoff error in the distance function.'''
    n1_batch_test = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 2, 3],
                              [1, 2, 3],
                              [1, 7, 3],
                              [1, 7, 3]], dtype=int)
    n2_batch_test = np.array([[0, 1, 1],
                              [2, 1, 1],
                              [1, 1, 3],
                              [1, 3, 3],
                              [1, 7, 2],
                              [1, 7, 4]], dtype=int)
    del_r_batch_test = distance_n_torus_batch(
        N=n_site_pd,
        a=lat_const,
        n1=n1_batch_test,
        n2=n2_batch_test,
    )
    # Test for roundoff error in distance function for nearest neighbors
    assert np.allclose(del_r_batch_test, lat_const)
    if np.allclose(del_r_batch_test, lat_const) and np.any(del_r_batch_test != lat_const):
        raise ValueError(
            'Rounding error in distance calculation gives del_r_nn != a!')
    return


def ext_hub_scHF_num_elec_from_mu0(dim, mu0, n_site_pd, lat_const, z_coord,
                                   t_hop, beta, U_loc, V_nn=0, qp_rescale=1.0):
    '''Find the bare chemical potential mu0 for a given number of electrons
       in the lattice, using the lattice dispersion relation and (Extended)
       Hubbard Hartree self-energy, Sigma_H = ((U / 2) + z V) n.'''
    n_site = int(n_site_pd ** dim)
    vol_lat = (lat_const * n_site_pd) ** dim
    kscale = 2.0*np.pi / float(n_site_pd*lat_const)
    ki_bz = kscale * np.arange(np.floor(-n_site_pd/2.0) + 1,
                               np.floor(n_site_pd/2.0) + 1, dtype=int)
    ki_meshes = [[ki_bz]] * dim
    k_meshgrid = np.asarray(np.meshgrid(*ki_meshes))
    assert k_meshgrid[0].shape == dim * (n_site_pd,)
    epsilon_k_meshgrid = lat_epsilon_k(
        k=k_meshgrid,
        lat_const=lat_const,
        t_hop=t_hop,
        qp_rescale=qp_rescale,
        meshgrid=True,
    )

    def mu0_implicit(num_elec):
        # Solve the implicit equation in terms of the Extended
        # Hubbard Hartree-reduced chemical potential, which is
        # itself a function of N_e
        n0 = num_elec / vol_lat
        mu_tilde = mu0 - ((U_loc / 2.0) + (z_coord * V_nn)) * n0
        ferm_meshgrid = ferm(beta*(epsilon_k_meshgrid - mu_tilde))
        num_elec_trial = int(round(2 * np.sum(ferm_meshgrid)))
        print('N_e(mu0 = '+str(mu0)+') = '+str(num_elec_trial))
        return np.floor(num_elec - num_elec_trial)
    # Numerically invert the implicit equation for N_e(mu0) to get mu0(N_e)
    return int(optimize.brentq(mu0_implicit, a=0, b=2*n_site, maxiter=1000))


def lat_mu_tilde_from_num_elec(dim, num_elec, n_site_pd, lat_const, z_coord,
                               U_loc, V_nn, t_hop, beta, qp_rescale=1.0):
    '''Find the Extended Hubbard Hartree-reduced chemical potential mu_tilde for a 
       given number of electrons in the lattice, using the lattice dispersion relation.'''
    vol_lat = (lat_const * n_site_pd) ** dim
    n0 = num_elec / vol_lat
    # Solve the implicit equation in N_e in terms of the Hartree-reduced
    # chemical potential, mu_tilde := mu - ((U / 2) + z V) n
    mu_tilde = lat_mu0_from_num_elec(
        dim, num_elec, n_site_pd, lat_const, t_hop, beta, qp_rescale)
    # Recover mu = mu_tilde + ((U / 2) + z V) n
    mu = mu_tilde + ((U_loc / 2.0) + (z_coord * V_nn)) * n0
    return mu, mu_tilde


def lat_mu0_from_num_elec(dim, num_elec, n_site_pd, lat_const, t_hop, beta, qp_rescale=1.0):
    '''Find the noninteracting chemical potential mu0 for a given number
       of electrons in the lattice, using the lattice dispersion relation.'''
    kscale = 2.0*np.pi / float(n_site_pd*lat_const)
    ki_bz = kscale * np.arange(np.floor(-n_site_pd/2.0) + 1,
                               np.floor(n_site_pd/2.0) + 1, dtype=int)
    ki_meshes = [[ki_bz]] * dim
    k_meshgrid = np.asarray(np.meshgrid(*ki_meshes))
    epsilon_k_meshgrid = lat_epsilon_k(
        k=k_meshgrid,
        lat_const=lat_const,
        t_hop=t_hop,
        qp_rescale=qp_rescale,
        meshgrid=True,
    )

    def num_elec_implicit(mu0):
        ferm_meshgrid = ferm(beta*(epsilon_k_meshgrid - mu0))
        num_elec_trial = int(round(2 * np.sum(ferm_meshgrid)))
        print('N_e(mu0 = '+str(mu0)+' ) = '+str(num_elec_trial))
        return num_elec - num_elec_trial
    # Numerically invert the implicit equation for N_e(mu0) to get mu0(N_e)
    # NOTE: to get exactly mu = 0 at half-filling (N_e = N ^ d), it is important
    #       for symmetry reasons that for the endpoints we take a = -b.
    return optimize.brentq(num_elec_implicit, a=-10, b=10, maxiter=1000)


def lat_num_elec_from_mu0(dim, mu0, n_site_pd, lat_const, t_hop, beta, qp_rescale=1.0):
    '''Find the number of electrons in the lattice, for a given noninteracting
       chemical potential mu0 using the lattice dispersion relation.'''
    kscale = 2.0*np.pi / float(n_site_pd*lat_const)
    ki_bz = kscale * np.arange(np.floor(-n_site_pd/2.0) + 1,
                               np.floor(n_site_pd/2.0) + 1, dtype=int)
    ki_meshes = [[ki_bz]] * dim
    k_meshgrid = np.asarray(np.meshgrid(*ki_meshes))
    assert k_meshgrid[0].shape == dim * (n_site_pd,)
    epsilon_k_meshgrid = lat_epsilon_k(
        k=k_meshgrid,
        lat_const=lat_const,
        t_hop=t_hop,
        qp_rescale=qp_rescale,
        meshgrid=True,
    )
    ferm_meshgrid = ferm(beta*(epsilon_k_meshgrid - mu0))
    num_elec = int(round(2 * np.sum(ferm_meshgrid)))
    print('N_e(mu_tilde = '+str(mu0)+' ) = '+str(num_elec))
    return num_elec


def lat_mu_F_from_n_exthub(dim, num_elec, n_site_pd, lat_const, V_nn, t_hop,
                           beta, delta_tau, gn_r_tau, qp_rescale=1.0):
    '''Find the chemical potential for a given number of electrons
       in the lattice using the lattice dispersion relation and HFI rules,
       so that the (momentum-dependent) Fock self-energy is included. We
       have absorbed the Hartree self-energy into the chemical potential,
       so that half-filling in the HI series corresponds to mu = 0.'''
    kscale = 2.0*np.pi / float(n_site_pd*lat_const)
    ki_bz = kscale * np.arange(np.floor(-n_site_pd/2.0) + 1,
                               np.floor(n_site_pd/2.0) + 1, dtype=int)
    ki_meshes = [[ki_bz]] * dim
    k_meshgrid = np.asarray(np.meshgrid(*ki_meshes))
    epsilon_k_meshgrid = lat_epsilon_k(
        k=k_meshgrid,
        lat_const=lat_const,
        t_hop=t_hop,
        qp_rescale=qp_rescale,
        meshgrid=True,
    )
    sigma_F_k_meshgrid = sigma_F_k_exthub(
        k=k_meshgrid,
        gn_r_tau=gn_r_tau,
        delta_tau=delta_tau,
        lat_const=lat_const,
        dim=dim,
        V_nn=V_nn,
        t_hop=t_hop,
        beta=beta,
        qp_rescale=qp_rescale,
        meshgrid=True,
    )

    def num_elec_implicit(mu):
        ferm_meshgrid = ferm(
            beta * (epsilon_k_meshgrid - mu + sigma_F_k_meshgrid))
        num_elec_trial = int(round(2 * np.sum(ferm_meshgrid)))
        print('N_e(mu = '+str(mu)+' ) = '+str(num_elec_trial))
        return num_elec - num_elec_trial
    # Numerically invert the implicit equation for N_e(mu) to get mu(N_e)
    return optimize.brentq(num_elec_implicit, a=-10, b=10, maxiter=1000)


def sigma_F_k_exthub(k, gn_r_tau, delta_tau, lat_const, dim, V_nn,
                     t_hop, beta, qp_rescale=1.0, meshgrid=False):
    '''Fock self-energy for the extended Hubbard model.'''
    epsilon_k = lat_epsilon_k(
        k=k,
        lat_const=lat_const,
        t_hop=t_hop,
        qp_rescale=qp_rescale,
        meshgrid=meshgrid,
    )
    return -(V_nn / float(t_hop)) * (lat_const ** dim) * gn_r_tau((lat_const, beta - delta_tau)) * epsilon_k


def lheg_mu_from_n0(dim, num_elec, n_site_pd, lat_const, beta):
    '''Find the noninteracting mu0 for a given number of electrons
       in the lattice, using the HEG dispersion relation (for LHEG model).'''
    kscale = 2.0*np.pi / float(n_site_pd*lat_const)
    ki_bz = kscale * np.arange(np.floor(-n_site_pd/2.0) + 1,
                               np.floor(n_site_pd/2.0) + 1, dtype=int)
    ki_meshes = [[ki_bz]] * dim
    k_meshgrid = np.asarray(np.meshgrid(*ki_meshes))

    def num_elec_implicit(mu):
        ferm_meshgrid = ferm(beta*(np.sum(k_meshgrid**2, axis=0)/2.0 - mu))
        num_elec_trial = int(round(2 * np.sum(ferm_meshgrid)))
        # print('N_e(mu = '+str(mu)+' ) = '+str(num_elec_trial))
        return num_elec - num_elec_trial
    # Numerically invert the implicit equation for N_e(mu) to get mu(N_e)
    return optimize.brentq(num_elec_implicit, a=-1e2, b=1e2, maxiter=1000)


class LatticeDensity:
    def __init__(self, dim, beta, t_hop, n_site_pd, lat_const,
                 target_mu=None, target_rho=None, sigma=None,
                 verbose=False, qp_rescale=1.0):
        self.verbose = verbose
        self.dim = dim
        self.beta = beta
        self.t_hop = t_hop
        self.n_site_pd = n_site_pd
        self.lat_const = lat_const
        self.qp_rescale = qp_rescale
        self.vol_lat = (n_site_pd * lat_const) ** dim
        self.k_mesh, self.epsilon_k_mesh = lat_epsilon_k_meshgrid(
            dim, n_site_pd, lat_const, t_hop, qp_rescale)
        if sigma is not None:
            if not isinstance(sigma, LambdaType):
                raise ValueError(
                    "Self-energy must be parametrized in terms of momentum k and electron number N_e!")
            self.sigma = sigma
        else:
            self.sigma = lambda k, rho: 0.0
        if target_mu is None and target_rho is None:
            raise ValueError(
                'The user must supply either the target density or chemical potential!')
        if target_mu is None:
            # num_elec able to most closely produce the target density
            num_elec = int(round((n_site_pd ** dim) * target_rho))
            # If the rounding process gives zero electrons in the system, choose N_e = 1 instead
            if num_elec == 0:
                num_elec = 1
                print('\nWarning: using minimal non-zero electron density for N = ' +
                      str(n_site_pd)+', L = '+str(self.lat_length) +
                      ': target_rho = '+str(target_rho)+'.')
            self.mu = self.mu_from_num_elec(num_elec)
            self.num_elec = num_elec
        else:
            self.num_elec = self.num_elec_from_mu(target_mu)
            # The exact mu value corresponding to num_elec electrons does
            # not necessarily equal target_mu (due to the coarse-graining)
            self.mu = self.mu_from_num_elec(self.num_elec)
        # The coarse-grained density (may not exactly equal target density)
        self.rho = self.num_elec / self.vol_lat

    def num_elec_from_mu(self, target_mu):
        '''Find the chemical potential mu for a given number of electrons 
           in the lattice, using the lattice dispersion relation and
           self-energy (if applicable).'''
        def mu_implicit(num_elec):
            ferm_meshgrid = ferm(self.beta * (self.epsilon_k_mesh - target_mu +
                                              self.sigma(self.k_mesh, num_elec / self.vol_lat)))
            num_elec_trial = int(round(2 * np.sum(ferm_meshgrid)))
            if self.verbose:
                print('    N_e(mu = '+str(target_mu)+') = '+str(num_elec_trial))
            return np.floor(num_elec - num_elec_trial)
        # Numerically invert the implicit equation for N_e(mu) to get mu(N_e)
        n_site = int(self.n_site_pd ** self.dim)
        if self.verbose:
            print(f'\nInverting: mu(N_e) = {target_mu} (target mu)...')
        num_elec_opt = int(optimize.brentq(mu_implicit, a=0, b=2*n_site, maxiter=1000))
        if self.verbose:
            print(f'Done! Optimized value: N_e = {num_elec_opt}')
        return num_elec_opt

    def mu_from_num_elec(self, target_num_elec):
        '''Find the chemical potential mu for a given number of electrons
           in the lattice, using the lattice dispersion relation.'''
        def num_elec_implicit(mu):
            ferm_meshgrid = ferm(self.beta * (self.epsilon_k_mesh - mu +
                                              self.sigma(self.k_mesh, target_num_elec / self.vol_lat)))
            num_elec_trial = int(round(2 * np.sum(ferm_meshgrid)))
            if self.verbose:
                print('    N_e(mu = '+str(mu)+' ) = '+str(num_elec_trial))
            return target_num_elec - num_elec_trial
        # Numerically invert the implicit equation for N_e(mu) to get mu(N_e)
        if self.verbose:
            print(f'\nInverting: N_e(mu) = {target_num_elec} (target num. elec)...')
        # NOTE: to get exactly mu = 0 at half-filling (N_e = N ^ d), it is important
        #       for symmetry reasons that for the endpoints we take a = -b.
        mu_opt = optimize.brentq(num_elec_implicit, a=-10, b=10, maxiter=1000)
        if self.verbose:
            print(f'Done! Optimized value: mu = {mu_opt}')
        return mu_opt


#################################################################
# Lattice propagator (finite-mesh G0 and W0) getters/converters #
#################################################################


def g0_k4(k4, mu, lat_const, t_hop, qp_rescale=1.0, meshgrid=False):
    '''Defines the Matsubara Green's function G_0(k, ik_n) on the lattice (scalar 
       valued at fixed k and ik_n). k can be a single vector or a numpy array of 
       vectors if (meshgrid == False), or a numpy meshgrid (i.e., all k-points in 
       the 1BZ) if (meshgrid == True). ikn can be a single frequency or an array.'''
    # k4 is a Matsubara momentum 4-vector, k4 := (\mathbf{k}, ik_n)
    k, ikn = k4
    epsilon_k = lat_epsilon_k(
        k=k, lat_const=lat_const, t_hop=t_hop, qp_rescale=qp_rescale, meshgrid=meshgrid)
    # Use broadcasting to build a whole-array result of shape (k.shape + ikn.shape); numpy will
    # auto-broadcast \epsilon_k and mu to this shape, as they differ along one axis only
    id_elemwise_prod = np.ones(epsilon_k.shape + ikn.shape)
    return 1.0 / ((ikn * id_elemwise_prod) - epsilon_k[..., np.newaxis] + mu)


def g0_k_tau(k, tau, beta, dim, mu, lat_const, t_hop, qp_rescale=1.0):
    # NOTE: Zero-tau limits other than the choice ztl = 1, which enforces a continuous
    #       principle interval (\tau \in [0 = 0^{+}, \beta = \beta^{-}]) were deprecated
    '''Get G_0(k, tau) on the lattice for various deduced batch shapes of k and tau.'''
    def g0_kvec_tau(kvec, tau, beta, mu, lat_const, t_hop, qp_rescale=1.0, batch=False, zero_tau_limit=1):
        '''Serial/vectorized imaginary time electron Green's function; the zero_tau_limit parameter  
           controls the discontinuity at \tau = 0; the value -1 corresponds to normal-ordering 
           (G_0(k, \tau = 0) = f_k). We can also choose to throw away the value at \tau = 0
           for safety -- this is zero_tau_limit = 0. Lastly, we can try to get the correct
            behavior by grabbing the sign of tau = +/- 0.0; this is None option.'''
        # Catch exceptional cases with a cutoff
        cutoff = np.log(np.finfo('d').max) / 2.0
        # 1D case; use regular functions
        if (batch == False):
            # Manually enforce fermionic anti-periodicity
            eta = 1.0
            tau_shifted = tau
            while tau_shifted >= beta:
                eta *= -1.0
                tau_shifted -= beta
            while tau_shifted < 0:
                eta *= -1.0
                tau_shifted += beta
            # Get the dispersion for the current k-point(s)
            epsilon_k = lat_epsilon_k(
                k=kvec, lat_const=lat_const, t_hop=t_hop, qp_rescale=qp_rescale)
            # Define auxilliary variables for convenience
            x = beta * (epsilon_k - mu) / 2.0
            y = (2 * tau_shifted / beta) - 1.0
            result = 0.0
            # Use the midpoint at the discontinuous point tau_shifted = 0: (G_0(k, 0^{+}) - G_0(k, 0^{-}))/2 = f_k - 1/2
            if tau_shifted == 0 and zero_tau_limit == None:
                return 0.0
            elif tau_shifted == 0 and zero_tau_limit == 0:
                result = -eta * (ferm(2*x, is1D=True) - 0.5)
            elif x > cutoff:
                result = -eta * np.exp(-x*(y + 1.0))
            elif x < -cutoff:
                result = -eta * np.exp(-x*(y - 1.0))
            else:
                result = -eta * np.exp(-x*y) / (2.0*np.cosh(x))
            # Adjust the value at zero time if we use normal-ordering
            if tau_shifted == 0 and zero_tau_limit == -1:
                result *= -np.exp(-2.0*x)
            return result
        # Whole-array (batch) functions
        else:
            # Manually enforce fermionic anti-periodicity for each time in the batch vector tau
            eta = np.ones(tau.shape)
            tau_shifted = np.copy(tau)
            while np.any(tau_shifted >= beta):
                eta[tau_shifted >= beta] *= -1.0
                tau_shifted[tau_shifted >= beta] -= beta
            while np.any(tau_shifted < 0):
                eta[tau_shifted < 0] *= -1.0
                tau_shifted[tau_shifted < 0] += beta
            # Get the dispersion for the current k-point(s)
            epsilon_kvec = lat_epsilon_k(
                k=kvec, lat_const=lat_const, t_hop=t_hop, qp_rescale=qp_rescale)
            # Define auxilliary variables for convenience
            x = beta * (epsilon_kvec - mu) / 2.0
            y = (2 * tau_shifted / beta) - 1.0
            result = np.zeros(x.shape)
            # Exponential underflow
            idx_uf = (x > cutoff)
            result[idx_uf] = -eta[idx_uf] * \
                np.exp(-x[idx_uf] * (y[idx_uf] + 1.0))
            # Exponential overflow
            idx_of = (x < -cutoff)
            result[idx_of] = -eta[idx_of] * \
                np.exp(-x[idx_of] * (y[idx_of] - 1.0))
            # Print any remaining overflow cases
            if np.any(result == np.inf):
                print('Overflow in G_0 exponential with argument x=',
                      x[result == np.inf])
            # Unmasked portion of the dispersion relation
            idx_ok = (np.abs(x) <= cutoff)
            # Get the well-behaved values of G_0(k, \tau_shifted)
            result[idx_ok] = -eta[idx_ok] * \
                np.exp(-x[idx_ok] * y[idx_ok]) / (2.0 * np.cosh(x[idx_ok]))
            # Use the midpoint at the discontinuous point tau_shifted = 0: (G_0(k, 0^{+}) - G_0(k, 0^{-}))/2 = f_k - 1/2
            idx_zt = (tau_shifted == 0)
            if zero_tau_limit == None:
                result[idx_zt] = 0.0
            if zero_tau_limit == 0:
                result[idx_zt] = -eta[idx_zt] * (ferm(2*x[idx_zt]) - 0.5)
            if zero_tau_limit == -1:
                result[idx_zt] *= -np.exp(-2*x[idx_zt])
            return result

    def g0_kvec_meshgrid_tau(k_meshgrid, tau, beta, mu, lat_const, t_hop, qp_rescale=1.0, zero_tau_limit=1):
        '''Meshgrid imaginary time electron Green's function; the zero_tau_limit parameter 
           controls the discontinuity at \tau = 0; the value -1 corresponds to normal-ordering
           (G_0(k, \tau = 0) = f_k). We can also choose to throw away the value at \tau = 0
           for safety -- this is zero_tau_limit = 0. Lastly, we can try to get the correct
           behavior by grabbing the sign of tau = +/- 0.0; this is None option.'''
        # Catch exceptional cases with a cutoff
        cutoff = np.log(np.finfo('d').max) / 2.0
        # Manually enforce fermionic anti-periodicity
        eta = 1.0
        tau_shifted = tau
        # Leaves the \tau = \beta point unshifted to numerically
        # enforce [0, \beta] := [0^{+}, \beta^{-}]
        # while tau_shifted > beta:
        while tau_shifted >= beta:
            eta *= -1.0
            tau_shifted -= beta
            # zero_tau_limit *= -1
        while tau_shifted < 0:
            eta *= -1.0
            tau_shifted += beta
            # zero_tau_limit *= -1
        # Get the dispersion for the current k-point(s)
        epsilon_k_meshgrid = lat_epsilon_k(
            k=k_meshgrid, lat_const=lat_const, t_hop=t_hop, qp_rescale=qp_rescale, meshgrid=True)
        # Define auxilliary variables for convenience
        x = beta * (epsilon_k_meshgrid - mu) / 2.0
        y = (2 * tau_shifted / beta) - 1.0
        result = np.zeros(x.shape)
        # Exponential underflow
        idx_uf = (x > cutoff)
        result[idx_uf] = -eta * np.exp(-x[idx_uf] * (y + 1.0))
        # Exponential overflow
        idx_of = (x < -cutoff)
        result[idx_of] = -eta * np.exp(-x[idx_of] * (y - 1.0))
        # Print any remaining overflow cases
        if np.any(result == np.inf):
            print('Overflow in G_0 exponential with argument x=',
                  x[result == np.inf])
        # Unmasked portion of the dispersion relation
        idx_ok = (np.abs(x) <= cutoff)
        # Get the well-behaved values of G_0(k, \tau_shifted)
        result[idx_ok] = -eta * \
            np.exp(-x[idx_ok] * y) / (2.0 * np.cosh(x[idx_ok]))
        # Use the midpoint at the discontinuous point tau_shifted = 0: (G_0(k, 0^{+}) - G_0(k, 0^{-}))/2 = f_k - 1/2
        if tau_shifted == 0:
            if zero_tau_limit == None:
                result = 0.0
            if zero_tau_limit == 0:
                result = -eta * (ferm(2*x) - 0.5)
            if zero_tau_limit == -1:
                result *= -np.exp(-2*x)
        return result
    # k magnitude and no batch index
    if np.isscalar(k) and np.isscalar(tau):
        raise ValueError('G_0(k) is not isotropic for a lattice dispersion'
                         + 'relation; must evaluate as a function of vector k!')
    # k meshgrid and scalar tau
    elif (not np.isscalar(k)) and (k.ndim == (1 + dim)) and (k.shape[0] == dim) and np.isscalar(tau):
        return g0_kvec_meshgrid_tau(k, tau, beta, mu, lat_const, t_hop, qp_rescale)
    # k vector and no batch index
    elif (not np.isscalar(k)) and np.isscalar(tau):
        return g0_kvec_tau(k, tau, beta, mu, lat_const, t_hop, qp_rescale, batch=False)
    # k magnitude and batch index
    elif k.shape == tau.shape:
        raise ValueError('G_0(k) is not isotropic for a lattice dispersion'
                         + 'relation; must evaluate as a function of vector k!')
    # k vector and batch index
    else:
        return g0_kvec_tau(k, tau, beta, mu, lat_const, t_hop, qp_rescale, batch=True)


def get_lat_g0_r_tau(lat_const, n_site_pd, t_hop, taulist, dim, beta,
                     mu, n0, delta_tau, qp_rescale=1.0, save_dir=".",
                     plots=False, overwrite=True):
    '''Obtain the real space, imaginary time lattice Green's function exactly via IFFT.'''
    kscale = 2.0 * np.pi / float(n_site_pd*lat_const)
    rscale = lat_const
    # Original (unshifted) k list in the Brillouin zone; we shift negative
    # points to the end to be consistent with the numpy FFT definitions
    # ki_bz = kscale * np.arange(1 - n_site_pd // 2, 1 + n_site_pd // 2, dtype=int)
    # idx_ki_Gamma = (n_site_pd - 1) // 2
    ki_bz = kscale * np.arange(- n_site_pd // 2, n_site_pd // 2, dtype=int)
    idx_ki_Gamma = (n_site_pd + 1) // 2
    # 1D momentum and position FFT lists
    ki_GX_list = np.concatenate((ki_bz[idx_ki_Gamma:], ki_bz[:idx_ki_Gamma]))
    assert (ki_GX_list[0] == 0) and (ki_bz[idx_ki_Gamma] == 0)
    # Distance vectors (modulo the lattice metric) have
    # components ranging from 0 to n_site_pd // 2, so we
    # only need to store G(R, R') = G(R - R') over the
    # first octant of the hypercubic lattice
    ri_list = rscale * np.arange(0, 1 + n_site_pd // 2)
    assert len(ri_list) == 1 + n_site_pd // 2
    # ri_list = rscale * np.arange(0, n_site_pd)
    # Momentum and position grids for whole-array function evaluation
    ki_meshes = [[ki_GX_list]] * dim
    k_meshgrid = np.asarray(np.meshgrid(*ki_meshes))
    # Shift endpoints (tau = 0, beta) to delta and beta - delta, if present
    taulist_cont = np.copy(taulist)
    taulist_cont[taulist_cont == 0] = min(delta_tau, taulist_cont[1])
    taulist_cont[taulist_cont == beta] = max(beta - delta_tau, taulist_cont[-2])
    # First, get G_0(k, \tau) for each 3D k vector and tau point
    mesh_shape = dim * [len(ki_GX_list)] + [len(taulist)]
    g0_k_tau_mesh = np.zeros(mesh_shape)
    for itau in range(len(taulist)):
        g0_k_tau_mesh[..., itau] = g0_k_tau(
            k=k_meshgrid,
            # The small shift when \tau = \beta ensures that we
            # numerically define [0, \beta] := [0^{+}, \beta^{-}]
            # tau=taulist[itau] - beta * delta_tau * (taulist[itau] == beta),
            tau=taulist_cont[itau],
            # tau=taulist[itau],
            beta=beta,
            dim=dim,
            mu=mu,
            lat_const=lat_const,
            t_hop=t_hop,
            qp_rescale=qp_rescale,
        )

    # Then, do the IFFT (the factor of 1 / N^3 is already present in the ifftn normalization)
    g0_ifft_mesh = np.real(np.fft.ifftn(
        g0_k_tau_mesh, axes=np.arange(dim))) / lat_const**dim
    assert np.allclose(np.imag(np.fft.ifftn(
        g0_ifft_mesh, axes=np.arange(dim))) / lat_const**dim, 0.0)
    # Interpolate in r_i's and tau
    r_tau_mesh = dim * (ri_list,) + (taulist,)
    if dim == 2:
        g0_ifft_red_mesh = g0_ifft_mesh[:len(ri_list), :len(ri_list), :]
    elif dim == 3:
        g0_ifft_red_mesh = g0_ifft_mesh[:len(
            ri_list), :len(ri_list), :len(ri_list), :]
    else:
        raise ValueError('Spatial dimensionality d = '+str(dim) +
                         ' not yet implemented; choose from the following: [2, 3]!')
    g0_r_tau_interp = interpolate.RegularGridInterpolator(
        r_tau_mesh, g0_ifft_red_mesh, bounds_error=False, fill_value=0.0)

    # Build an (N x ... x N) (d times) matrix of imaginary-time
    # interpolants (interpolation in r is unnecessary on the lattice)
    g0_r_tau_interp_mtx = np.empty(dim * (n_site_pd,), dtype=object)
    for idx_ri in np.ndindex(dim * (n_site_pd,)):
        g0_r_tau_interp_mtx[idx_ri] = interpolate.interp1d(
            taulist, g0_ifft_mesh[idx_ri], kind='cubic')

    # Plot the results for the lattice Green's function
    if plots:
        # lat_length = n_site_pd * lat_const
        # Lists for plots
        small_rx_list = ri_list[:5]
        # big_rxlist = ri_list
        print(small_rx_list)
        little_taulist = np.linspace(taulist.min(), taulist.max(), num=5)
        # Set the minimal and maximal times to plot as delta_tau and beta - delta_tau, respectively,
        # since the values of G_0 at tau = 0, beta are not recorded (set to zero)
        # little_taulist[0] = taulist[1]
        # little_taulist[-1] = taulist[-2]
        big_taulist = np.sort(np.unique(np.concatenate(
            ([delta_tau], np.linspace(taulist.min(), taulist.max(), num=1001)))))
        mp_taulist = np.unique(np.concatenate(
            (big_taulist - beta, big_taulist, big_taulist + beta)))
        highlight_indices = []
        for i in range(len(mp_taulist)):
            for n in [-1., 0., 1., 2.]:
                if np.abs(mp_taulist[i] - n*beta) < big_taulist[2]:
                    highlight_indices.append(i)
        # # Plot vs tau inlaid for the integration interval
        # fig1, ax1 = plt.subplots()
        # for this_rx in small_rx_list:
        #     r_tau_eval = (this_rx * np.ones(big_taulist.shape),) + \
        #         (dim - 1) * (np.zeros(big_taulist.shape),) + (big_taulist,)
        #     ax1.plot(big_taulist/beta, g0_r_tau_interp(r_tau_eval), '-',
        #              label=r'$r_x/a = $'+str('%g' % (this_rx / rscale)))
        # ax1.set_xlabel(r'$\tau/\beta$')
        # ax1.set_ylabel(r'$G_0(r_x,\tau)$')
        # ax1.legend(loc='best')
        # fig1.tight_layout()
        # savename1 = safe_filename(
        #     dir=save_dir,
        #     savename=f'lat_g0_rx_tau_vs_tau_d={dim}',
        #     file_extension='pdf',
        #     overwrite=overwrite,
        # )
        # fig1.savefig(savename1)
        # # Zoom in on the short-time behavior
        # fig1, ax1 = plt.subplots()
        # for this_rx in small_rx_list:
        #     r_tau_eval = (this_rx * np.ones(big_taulist[big_taulist < 0.05].shape),) + (dim - 1) * (np.zeros(big_taulist[big_taulist < 0.05].shape),) + (big_taulist[big_taulist < 0.05],)
        #     ax1.plot(big_taulist[big_taulist < 0.05]/beta, g0_r_tau_interp(), 'o-', label=r'$r/a = $'+str('%g' % this_rx))
        # ax1.set_xlabel(r'$\tau/\beta$')
        # ax1.set_ylabel(r'$G_0(r_x,\tau)$')
        # ax1.legend(loc='best')
        # fig1.tight_layout()
        # fig1.savefig('lat_g0_r_tau_vs_tau_d='+str(dim)+'_short_time.pdf')
        # Plot vs tau over a few periods
        fig1, ax1 = plt.subplots(len(small_rx_list[:-1]), sharex=True)
        fig2, ax2 = plt.subplots(len(small_rx_list[:-1]), sharex=True)
        for irx, this_rx in enumerate(small_rx_list[:-1]):
            adjust_tau = np.copy(mp_taulist)
            adjust_tau[(adjust_tau % beta) == 0] = delta_tau
            # adjust_tau[(adjust_tau % beta) == 0] = -delta_tau
            g0_r_tau_list = antiperiodic_batch(
                my_vec_fn=lambda tau_ap: g0_r_tau_interp(
                    (this_rx * np.ones(tau_ap.shape),) + (dim - 1) * (np.zeros(tau_ap.shape),) + (tau_ap,)),
                x=mp_taulist,
                period=beta,
            )
            g0_r_ptau_list = antiperiodic_batch(
                my_vec_fn=lambda tau_ap: g0_r_tau_interp(
                    (this_rx * np.ones(tau_ap.shape),) + (dim - 1) * (np.zeros(tau_ap.shape),) + (tau_ap,)),
                x=adjust_tau,
                period=beta,
            )
            g0_r_mtau_list = antiperiodic_batch(
                my_vec_fn=lambda tau_ap: g0_r_tau_interp(
                    (this_rx * np.ones(tau_ap.shape),) + (dim - 1) * (np.zeros(tau_ap.shape),) + (tau_ap,)),
                # x=-adjust_tau,
                x=beta-adjust_tau,
                period=beta,
            )
            # Plot G_0
            ax1[irx].plot(mp_taulist/beta, g0_r_tau_list, '-',
                          markersize=1, label=r'$r/a = $'+str('%g' % this_rx))
            ax1[irx].plot(mp_taulist[highlight_indices]/beta,
                          g0_r_tau_list[highlight_indices], 'o', color='C1', markersize=2)
            ax1[irx].plot(mp_taulist[mp_taulist % beta == 0]/beta,
                          g0_r_tau_list[mp_taulist % beta == 0], 'o', color='r', markersize=2)
            ax1[irx].set_ylabel(r'$G_0(r_x,\tau)$')
            ax1[irx].legend(loc='best')
            # Plot P_0
            ax2[irx].plot(mp_taulist/beta, 2*g0_r_ptau_list*g0_r_mtau_list,
                          '-', markersize=1, label=r'$r/a = $'+str('%g' % this_rx))
            ax2[irx].plot(mp_taulist[highlight_indices]/beta, 2*g0_r_mtau_list[highlight_indices]
                          * g0_r_ptau_list[highlight_indices], 'o', color='C1', markersize=2)
            ax2[irx].plot(mp_taulist[mp_taulist % beta == 0]/beta, 2*g0_r_mtau_list[mp_taulist %
                                                                                    beta == 0]*g0_r_ptau_list[mp_taulist % beta == 0], 'o', color='r', markersize=2)
            ax2[irx].set_ylabel(r'$P_0(r_x,\tau)$')
            ax2[irx].legend(loc='best')
        ax1[-1].set_xlabel(r'$\tau/\beta$')
        fig1.tight_layout()
        savename1 = safe_filename(
            dir=save_dir,
            savename=f'lat_g0_rx_tau_vs_tau_d={dim}_mp',
            file_extension='pdf',
            overwrite=overwrite,
        )
        fig1.savefig(savename1)
        ax2[-1].set_xlabel(r'$\tau/\beta$')
        fig2.tight_layout()
        savename2 = safe_filename(
            dir=save_dir,
            savename=f'lat_p0_rx_tau_vs_tau_d={dim}_mp',
            file_extension='pdf',
            overwrite=overwrite,
        )
        fig2.savefig(savename2)
        # Plot vs r for a small window of r magnitudes (to check correspondence to continuum Green's function)
        fig3, ax3 = plt.subplots()
        ax3.axvline(x=lat_const, color='0.5',
                    linestyle='-', label=r'$r_x = a$')
        for this_tau in little_taulist:
            r_tau_eval = ((ri_list / rscale),) + (dim - 1) * \
                (np.zeros(ri_list.shape),) + (this_tau*np.ones(ri_list.shape),)
            ax3.plot((ri_list / rscale), g0_r_tau_interp(r_tau_eval), 'o-',
                     markersize=3, label=r'$\tau/\beta = $'+str('%g' % (this_tau/beta)))
        if n0 == 1:
            ax3.plot((ri_list / rscale), np.ones((ri_list / rscale).shape)
                     * n0 / 2.0, 'k--', label=r'$G(\mathbf{0},0^-) = n_0 / 2$')
            ax3.plot((ri_list / rscale), -np.ones((ri_list / rscale).shape)
                     * n0 / 2.0, 'k--', label=r'$G(\mathbf{0},\beta^-) = -n_0 / 2$')
            if dim == 3:
                ax3.plot((ri_list / rscale), -np.ones((ri_list / rscale).shape)
                         * n0, 'k--', label=r'$G(\mathbf{0},\beta/2) = -n_0$')
                # print(-n0 / g0_r_tau_interp((0.0, beta/2.0)), -g0_r_tau_interp((0.0, beta/2.0)) / n0)
        ax3.set_xlabel(r'$r_x/a$')
        ax3.set_ylabel(r'$G_0(r_x,\tau)$')
        ax3.legend(loc='lower right')
        fig3.tight_layout()
        ax3.set_xlim(0, min(5, (ri_list / rscale)[-1]))
        if n0 == 1:
            ax3.set_ylim(-1.1*n0/2.0, 1.1*n0/2.0)
        ax3_tick_spacing = 0.1
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(ax3_tick_spacing))
        savename3 = safe_filename(
            dir=save_dir,
            savename=f'lat_g0_rx_tau_vs_r_corresp_d={dim}',
            file_extension='pdf',
            overwrite=overwrite,
        )
        fig3.savefig(savename3)
        # Plot vs r for the whole window of r magnitudes
        fig4, ax4 = plt.subplots()
        ax4.axvline(x=lat_const, color='0.5',
                    linestyle='-', label=r'$r_x = a$')
        for this_tau in little_taulist:
            r_tau_eval = ((ri_list / rscale),) + (dim - 1) * \
                (np.zeros(ri_list.shape),) + (this_tau*np.ones(ri_list.shape),)
            ax4.plot((ri_list / rscale), g0_r_tau_interp(r_tau_eval), '-',
                     markersize=3, label=r'$\tau/\beta = $'+str('%g' % (this_tau/beta)))
        if n0 == 1:
            ax4.plot((ri_list / rscale), np.ones((ri_list / rscale).shape)
                     * n0 / 2.0, 'k--', label=r'$G(\mathbf{0},0^-) = n_0 / 2$')
            ax4.plot((ri_list / rscale), -np.ones((ri_list / rscale).shape)
                     * n0 / 2.0, 'k--', label=r'$G(\mathbf{0},\beta^-) = -n_0 / 2$')
            if dim == 3:
                ax4.plot((ri_list / rscale), -np.ones((ri_list / rscale).shape)
                         * n0, 'k--', label=r'$G(\mathbf{0},\beta/2) = -n_0$')
        ax4.set_xlabel(r'$r_x/a$')
        ax4.set_ylabel(r'$G_0(r_x,\tau)$')
        ax4.legend(loc='lower right')
        if n0 == 1:
            ax4.set_ylim(-1.1*n0/2.0, 1.1*n0/2.0)
        ax4_tick_spacing = 0.1
        ax4.yaxis.set_major_locator(ticker.MultipleLocator(ax4_tick_spacing))
        fig4.tight_layout()
        savename4 = safe_filename(
            dir=save_dir,
            savename=f'lat_g0_rx_tau_vs_r_d={dim}',
            file_extension='pdf',
            overwrite=overwrite,
        )
        fig4.savefig(savename4)
        # Done plotting
        plt.close('all')
    return g0_r_tau_interp_mtx, g0_ifft_mesh


def get_lat_g0_k_tau_from_g0_r_tau(g0_r_tau_ifft_mesh, lat_const, num_elec, n_site_pd, taulist,
                                   dim, beta, t_hop, mu, delta_tau, qp_rescale=1.0,
                                   plots=False, plot_filling=False):
    '''Obtain the momentum-space, imaginary time lattice Green's function exactly via FFT.'''
    # FFT lists; k_list contains n_samples momentum points evenly spaced on [0,n_samples-1],
    # and r_list is a list of n_samples real-space points evenly spaced on [0,n_samples-1]/(2*\pi)
    kscale = 2.0*np.pi / float(n_site_pd*lat_const)
    # Original (unshifted) k list in the Brillouin zone
    ki_bz = kscale * np.arange(- n_site_pd // 2, n_site_pd // 2, dtype=int)
    idx_ki_Gamma = (n_site_pd + 1) // 2
    # Shifted k-list for FFT
    ki_GX_list = np.concatenate((ki_bz[idx_ki_Gamma:], ki_bz[:idx_ki_Gamma]))
    assert (ki_GX_list[0] == 0) and (ki_bz[idx_ki_Gamma] == 0)
    # print(ki_bz)
    # Momentum and position grids for whole-array function evaluation
    ki_meshes = [[ki_GX_list]] * dim
    k_meshgrid = np.asarray(np.meshgrid(*ki_meshes))
    # Do the FFT (the factor of a^3 assures that this is a proper inverse of the IFFT)
    g0_k_tau_fft_mesh = np.real(np.fft.fftn(g0_r_tau_ifft_mesh,
                                            axes=np.arange(dim))) * lat_const**dim
    # Interpolate in r_i's and tau
    ki_bz_for_fftshift = kscale * \
        np.arange(- n_site_pd // 2, n_site_pd // 2, dtype=int)
    k_tau_mesh_sign_ordered = dim * (ki_bz_for_fftshift,) + (taulist,)
    g0_k_tau_interp = interpolate.RegularGridInterpolator(
        k_tau_mesh_sign_ordered, np.fft.fftshift(g0_k_tau_fft_mesh, axes=np.arange(dim)), bounds_error=False, fill_value=0.0)

    # Shift endpoints to delta and beta - delta, if present
    taulist_cont = np.copy(taulist)
    taulist_cont[taulist_cont == 0] = min(beta * delta_tau, taulist_cont[1])
    taulist_cont[taulist_cont == beta] = max(
        beta * (1 - delta_tau), taulist_cont[-2])

    # Compare to exact G_0(k, \tau) (should be equal up to a small floating-point error)
    for itau in range(len(taulist)):
        g0_k_tau_exact = g0_k_tau(
            k=k_meshgrid,
            # tau=taulist[itau] - beta * delta_tau * (taulist[itau] == beta),
            # tau=taulist[itau],
            tau=taulist_cont[itau],
            beta=beta,
            dim=dim,
            mu=mu,
            lat_const=lat_const,
            t_hop=t_hop,
            qp_rescale=qp_rescale,
        )
        assert np.allclose(g0_k_tau_fft_mesh[..., itau], g0_k_tau_exact)

    # Plot the results vs both k and tau
    if plots:
        # Tests the band filling scheme
        _, kf_vecs = fill_band(
            dim=dim,
            num_elec=num_elec,
            n_site_pd=n_site_pd,
            lat_const=lat_const,
            t_hop=t_hop,
            plot_filling=plot_filling,
        )
        # print(ef, '\n', kf_vecs)
        # Build an ordered path of k-points in the Brillouin zone; we choose the high-symmetry
        # path for the simple square lattice (\Gamma - X - M - \Gamma), discarding duplicate
        # coordinates at the path vertices (accounted for in plotting step).
        N_edge = n_site_pd // 2
        nk_coords_Gamma_X = [[x, 0] for x in range(0, N_edge + 1)]
        nk_coords_X_M = [[N_edge, y] for y in range(1, N_edge + 1)]
        nk_coords_M_Gamma = [[xy, xy] for xy in range(0, N_edge)[::-1]]
        # Indices for the high-symmetry points
        idx_Gamma1 = 0
        idx_X = len(nk_coords_Gamma_X) - 1
        idx_M = len(nk_coords_Gamma_X) + len(nk_coords_X_M) - 1
        idx_Gamma2 = len(nk_coords_Gamma_X) + \
            len(nk_coords_X_M) + len(nk_coords_M_Gamma) - 1
        # Build the full ordered high-symmetry path
        path_nk_coords = np.concatenate(
            (nk_coords_Gamma_X, nk_coords_X_M, nk_coords_M_Gamma))
        path_nk_coords_shifted = path_nk_coords - \
            n_site_pd * (path_nk_coords == N_edge)
        path_k_coords = kscale * path_nk_coords
        path_k_coords_shifted = kscale * path_nk_coords_shifted

        # Find the corresponding indices in the full k_list
        i_path = np.arange(len(path_k_coords))
        i_path_kf_locs = []
        for i, this_k_coord in enumerate(path_k_coords):
            for kf_vec in kf_vecs:
                if np.allclose(this_k_coord, kf_vec):
                    print(this_k_coord, i)
                    i_path_kf_locs.append(i)
        i_path_kf_locs = np.asarray(i_path_kf_locs)

        # Lists for plots
        small_kx_list = ki_bz[len(ki_bz) / 2:: len(ki_bz) / 10]
        little_taulist = np.linspace(taulist.min(), taulist.max(), num=5)
        big_taulist = np.linspace(taulist.min(), taulist.max(), num=1001)
        big_klist = path_k_coords
        # Plot vs tau
        fig1, ax1 = plt.subplots()
        for this_kx in small_kx_list:
            k_tau_eval = (this_kx * np.ones(big_taulist.shape),) + \
                (dim - 1) * (np.zeros(big_taulist.shape),) + (big_taulist,)
            ax1.plot(big_taulist/beta, g0_k_tau_interp(k_tau_eval),
                     '-', label=r'$k_x = $'+str('%g' % this_kx))
        ax1.set_xlabel(r'$\tau/\beta$')
        ax1.set_ylabel(r'$G_0(k_x,\tau)$')
        ax1.legend(loc='best')
        fig1.tight_layout()
        fig1.savefig('lat_g0_kx_tau_vs_tau_d='+str(dim)+'.pdf')
        # Plot vs r
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        for _, this_tau in enumerate(little_taulist):
            k_tau_eval = (ki_bz[ki_bz >= 0],) + (dim - 1) * (np.zeros(
                ki_bz[ki_bz >= 0].shape),) + (this_tau * np.ones(ki_bz[ki_bz >= 0].shape),)
            ax3.plot(ki_bz[ki_bz >= 0], g0_k_tau_interp(
                k_tau_eval), '-', markersize=3, label=r'$\tau/\beta = $'+str('%g' % (this_tau / beta)))
            # Plot on the k path in the Brillouin zone
            g0_k_tau_path = [g0_k_tau_interp(
                tuple(coord) + (this_tau,)) for coord in path_k_coords_shifted]
            ax4.plot(i_path, g0_k_tau_path, linestyle='-',
                     label=r'$\tau/\beta = $'+str('%g' % (this_tau / beta)))
        ax3.axhline(-0.5, color='k', linestyle='--', zorder=0)
        ax3.set_xlabel(r'$k_x$')
        ax3.set_ylabel(r'$G_0(k_x,\tau)$')
        ax3.legend(loc='best')
        fig3.tight_layout()
        fig3.savefig('lat_g0_kx_tau_vs_kx_d='+str(dim)+'.pdf')
        if np.max(big_klist) > 5:
            ax3.set_xlim(0, 5)
            fig3.savefig('lat_g0_kx_tau_vs_kx_corresp_d='+str(dim)+'.pdf')
        # Label the Fermi wavevectors along the path, if applicable
        if len(i_path_kf_locs) > 0:
            ax4.axvline(x=i_path_kf_locs[0], linestyle='-', color='0.0',
                        zorder=-1, linewidth=1, label=r'$\mathbf{k}_F$')
            for i_path_kf_loc in i_path_kf_locs[1:]:
                ax4.axvline(x=i_path_kf_loc, linestyle='-',
                            color='0.0', zorder=-1, linewidth=1)
        ax4.legend(loc='best')
        # Add some evenly-spaced minor ticks to the axis
        n_minor_ticks = 9
        assert (len(i_path) - 1) % 9 == 0
        minor_ticks = np.arange(0, len(i_path), len(i_path) / n_minor_ticks)
        print(minor_ticks)
        ax4.set_xticks(minor_ticks, minor=True)
        # Label the high-symmetry points
        ax4.set_xticks((idx_Gamma1, idx_X, idx_M, idx_Gamma2))
        ax4.set_xticklabels((r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$'))
        ax4.set_ylabel(r'$G_0(\mathbf{{k}}, \tau)$')
        ax4.set_xlim(left=i_path[0], right=i_path[-1])
        ax4.grid(True, color='k', linestyle=':', which='minor')
        ax4.grid(True, color='k', linestyle=':', which='major')
        fig4.tight_layout()
        fig4.savefig('lat_g0_k_tau_vs_k_d='+str(dim)+'_high_symm_path.pdf')
        # Done plotting
        plt.close('all')
    return g0_k_tau_interp, g0_k_tau_fft_mesh


def get_pi0_q4_path_from_g0_r_tau_quad(g0_r_tau_ifft_red_mesh, path_q_coords, inu_list,
                                       tau_list, beta, delta_tau, dim, n_site_pd, lat_const,
                                       verbose=False):
    '''Calculates the electron-hole ('polarization') bubble $\Pi_0(q, i\nu)$ numerically
       exactly via Gaussian quadrature, given $G_0(r, \tau)$ on the lattice. Also returns a
       d-dimensional array of 1d interpolants in Matsubara frequency, $\Pi_0(q_\mathbf{n})(i\nu)$,
       which will only be relevant when inu_list is sparse (does not contain all freqs up to m_max).'''
    tau_list_cont = np.copy(tau_list)
    tau_list_cont[tau_list_cont < delta_tau] = delta_tau
    tau_list_cont[tau_list_cont > beta - delta_tau] = beta - delta_tau

    n_q_path = len(path_q_coords)
    n_nu = len(inu_list)
    # Calculate the first-order static charge susceptibility at fixed spin
    # sigma numerically exactly from the bare Green's function (via quadrature)
    full_ri_list = lat_const * np.arange(0, n_site_pd)
    r_coords = np.asarray(list(product(full_ri_list, full_ri_list)))
    print(r_coords)
    # pi0_sigma_ex_means_inu_interp = np.zeros(n_q_path, dtype=object)
    pi0_sigma_ex_means_path = np.zeros((n_q_path, n_nu))
    pi0_sigma_ex_errs_path = np.zeros((n_q_path, n_nu))
    for iq, this_q_coord in enumerate(path_q_coords):
        for inu, this_nu in enumerate(inu_list.imag):
            f_tau = np.zeros(len(tau_list))
            for this_r_coord in r_coords:
                this_nr_coord = (this_r_coord / lat_const).astype(int)
                this_del_nr = difference_n_torus(
                    n1=this_nr_coord, n2=np.zeros(dim, dtype=int), N=n_site_pd)
                g0_tau_p_this_r_coord = g0_r_tau_ifft_red_mesh[this_del_nr[0],
                                                               this_del_nr[1]]
                g0_tau_m_this_r_coord = g0_r_tau_ifft_red_mesh[this_del_nr[0],
                                                               this_del_nr[1]][::-1]
                # Now multiply by the FT factor; if \tau = \beta is present in the list,
                # we shift it to \beta^{-} = \beta - \delta_{\tau} so that the function
                # is evaluated on the continuous interval [0^{+}, \beta^{-}]
                f_tau += -g0_tau_p_this_r_coord * g0_tau_m_this_r_coord * \
                    np.cos(np.dot(this_q_coord, this_r_coord)) * \
                    np.cos(this_nu * tau_list)
                # np.cos(this_nu * tau_list_cont)
            f_tau_interp = interpolate.interp1d(
                tau_list, f_tau, kind='cubic', bounds_error=None, fill_value=0.0)
            # mean, err = quad(lambda tau: f_tau_interp(tau), 0.0, beta - delta_tau,
            # mean, err = quad(lambda tau: f_tau_interp(tau), delta_tau, beta - delta_tau,
            mean, err = quad(lambda tau: f_tau_interp(tau), 0.0, beta,
                             epsabs=1e-8, limit=int(max(100, 5*beta)))
            if verbose:
                print('chi_ex for q4(', iq, ',', inu, '): ', mean, '+/-', err)
            pi0_sigma_ex_means_path[iq, inu] = mean
            pi0_sigma_ex_errs_path[iq, inu] = err
        # # Build a 1d linear interpolant in frequency at each
        # # k-point and add it to the interpolant matrix for \Pi_0
        # pi0_sigma_ex_means_inu_interp[iq] = interpolate.interp1d(
        #     inu_list.imag,
        #     pi0_sigma_ex_means_path[iq, :],
        #     # We use linear interpolation instead of, e.g., cubic, to avoid the
        #     # need to manually enforce causality (sign-definiteness of \Pi_0)
        #     kind='linear',
        #     bounds_error=None,
        #     fill_value=0.0,
        # )
        # # Make sure we did not break causality via oscillations in the cubic interpolation,
        # # that is, check that \Pi_0(\mathbf{q}_i)(i\nu) is sign-definite on the interval [i\nu_0, i\nu_{n_{nu} - 1}]
        # m_max = int(np.max(inu_list.imag * (beta / (2 * np.pi))))
        # print(m_max)
        # nu_list_full = (2 * np.pi / beta) * np.arange(m_max + 1)
        # pi0_dense = pi0_sigma_ex_means_inu_interp[iq](nu_list_full)
        # assert np.unique(np.sign(pi0_dense[pi0_dense != 0])).size < 2
    # return pi0_sigma_ex_means_inu_interp, pi0_sigma_ex_means_path, pi0_sigma_ex_errs_path
    return pi0_sigma_ex_means_path, pi0_sigma_ex_errs_path


def get_pi0_q4_from_g0_r_tau_fft(g0_r_tau, n_nu, dim, beta, delta_tau, n_site_pd, lat_const):
    '''Defines the electron-hole ('polarization') bubble \Pi_0(q, i\nu) on the lattice. 
       Performs FFT from the result in the (r, \tau) representation, where \Pi_0 is a
       product rather than a convolution of G_0's.

       NOTE: It is assumed that G_0 is spin-independent, such that we trace over
             the spin factor in \Pi_0 analytically, \Pi_0 = 2 \Pi^\sigma_0.'''

    assert isinstance(g0_r_tau, np.ndarray)
    # If G_0(r) is an interpolant in \tau, upsample it on a uniform mesh
    if isinstance(g0_r_tau.flat[0], interpolate.interp1d):
        print('''Warning: using lossy upsampling from nonuniform interpolant data to a dense uniform mesh!''')
        # The FFT tau mesh should be consistent with the ifft data for G_0(r, \tau), that is,
        # uniform and corresponding to: inu_mesh = (2 \pi i / \beta) range(n_nu)
        tau_unif_mesh = beta * np.arange(n_nu + 1) / float(n_nu)
        tau_unif_cont = np.copy(tau_unif_mesh)
        tau_unif_cont[tau_unif_cont == 0] = min(delta_tau, tau_unif_cont[1])
        tau_unif_cont[tau_unif_cont == beta] = max((beta - delta_tau), tau_unif_cont[-2])
        # Get G_0 on the uniform mesh via linear interpolation (lossy upsampling)
        g0_r_tau_fft = np.empty(dim * (n_site_pd,) + (n_nu + 1,))
        for idx_ri_tj in np.ndindex(g0_r_tau_fft.shape):
            g0_r_tau_fft[idx_ri_tj] = g0_r_tau[idx_ri_tj[:dim]](
                tau_unif_cont[idx_ri_tj[dim]])
            # Make sure we did not break causality via oscillations in the cubic interpolation,
            # that is, check that G_0(\mathbf{r}_i)(\tau) is sign-definite on the interval [0^{+}, beta^{-}]
            assert np.unique(
                np.sign(g0_r_tau_fft[idx_ri_tj][g0_r_tau_fft[idx_ri_tj] != 0])).size < 2
    # If G_0(r, \tau) is a (dim + 1) array of mesh data, we can work with it
    # directly; it is assumed that the spacing in \tau is uniform in this case
    # TODO: Find a nice way to check it without adding tau_list input if possible!
    else:
        g0_r_tau_fft = g0_r_tau

    # Make sure that the employed tau mesh is odd,
    # i.e., symmetric about \tau = \beta / 2
    assert g0_r_tau_fft.shape[-1] % 2 == 1

    # \Pi_0(r, \tau) = -2 G_0(r, \tau) G_0(r, \beta - \tau)
    pi0_r_tau = -2 * g0_r_tau_fft * g0_r_tau_fft[..., ::-1]
    # print(pi0_r_tau[dim*(0,) + (0,)], pi0_r_tau[dim*(0,) + (-1,)])
    # print(pi0_r_tau[dim*(0,) + (1,)], pi0_r_tau[dim*(0,) + (-2,)])

    # Check that the values at \tau = 0, \beta have been defined via the correct limit, i.e.,
    # \Pi_0(r, 0) := \Pi_0(r, 0^{+}) = \Pi_0(r, \beta^{-}) := \Pi_0(r, \beta)
    print('Verifying bosonic symmetry in Pi_0(r, tau)...', end='')
    assert np.all(pi0_r_tau[..., :] == pi0_r_tau[..., ::-1])
    print('OK')

    # Do the spatial FFT (the factor of a^3 assures that this is a proper inverse of the IFFT)
    pi0_q_tau = np.fft.fftn(
        pi0_r_tau, axes=np.arange(dim)).real * lat_const**dim

    print('Verifying bosonic symmetry in Pi_0(q, tau)...', end='')
    assert np.all(pi0_q_tau[..., :] == pi0_q_tau[..., ::-1])
    print('OK')

    # Alternatively, we could employ the fftshift method to get the data on a Gamma-centered
    # k-point mesh, but this would require shifting all k-path indices by n_site_pd // 2
    # pi0_q_tau = np.fft.fftshift(np.fft.fftn(pi0_r_tau, axes=np.arange(
    #     dim)).real, axes=np.arange(dim)) * lat_const**dim

    # Finally, do the temporal IFFT (the factor of \beta assures that this is a proper inverse of the FFT);
    # we should not include the periodic point, \tau = 0, twice, and hence remove the data at \tau = \beta
    # from the list for purposes of the IFFT to Matsubara frequency
    pi0_q4 = beta * fftpack.ifft(pi0_q_tau[..., :-1], axis=-1).real
    assert pi0_q4.shape == dim * (n_site_pd,) + (n_nu,)
    return pi0_q4, pi0_q_tau


def get_lat_wstar_q(pi0_q_inu, lat_const, n_site_pd, n_nu, dim, beta):
    '''Gets the lattice RPA screened interaction in the momentum space representations,
       W_{*}(q, ~). The imaginary-time result is obtained via FFT from the numerically
       exact interaction in the (q, i\nu) representation to (q, \tau) space. It is
       assumed that \Pi(q, i\nu) is a d-dimensional array of frequency interpolants.

       NOTE: The singularity at the \Gamma point is treated by simply omitting that contribution, 
       W_{*} = W_{0} \delta_{q \ne 0} (i.e., Andrey's simple scheme. This worked sufficiently
       well, he says, for one-shot GW calculations, although we use this scheme here purely in the
       interest of simplicity of our initial implementation of a W-like interaction in real-space).

       NOTE: We assume the \Pi_0 data is on a uniform frequency mesh, so that:
           m_list = np.arange(n_nu),
           tau_fftlist = beta * np.arange(n_nu + 1) / float(n_nu).'''

    kscale = 2.0 * np.pi / float(n_site_pd * lat_const)
    q_tau_mesh_shape = dim * (n_site_pd,) + (n_nu + 1,)
    assert pi0_q_inu.shape == dim * (n_site_pd,) + (n_nu,)

    # Coulomb and dynamic part of the RPA screened interaction with q = 0
    # divergences removed (we call them V_{*} and \widetilde{W}_{*})
    vstar_q = np.zeros(dim * (n_site_pd,))
    wstar_tilde_q_inu = np.zeros(pi0_q_inu.shape)
    # Stores the 1D FFT from nu to tau for the dynamic part of the interaction
    wstar_tilde_q_tau = np.zeros(q_tau_mesh_shape)

    iter_idx_q = np.ndindex(vstar_q.shape)
    # Use a zero value for the singular q = 0 point
    next(iter_idx_q)
    # Build the other matrix elements
    for idx_qi in iter_idx_q:
        # V(q) = 4\pi / |q|^2
        vstar_q[idx_qi] = 4 * np.pi / kscale**2 / np.dot(idx_qi, idx_qi)
        # Reduced interaction for FFT: \widetilde{W}_0(q, i\nu) / V(q)
        # = V(q) \Pi_0(q, i\nu) / (1 - V(q) \Pi_0(q, i\nu))
        wtilde_over_v_star_q_inu = vstar_q[idx_qi] * \
            pi0_q_inu[idx_qi] / (1.0 - vstar_q[idx_qi] * pi0_q_inu[idx_qi])
        # \widetilde{W}_0(q, i\nu) = V(q)^2 \Pi_0(q, i\nu) / (1 - V(q) \Pi_0(q, i\nu))
        wstar_tilde_q_inu[idx_qi] = wtilde_over_v_star_q_inu * vstar_q[idx_qi]
        # Get the values via FFT for \tau < \beta
        wstar_tilde_q_tau[idx_qi][:-1] = np.real(2 * fftpack.fft(
            wtilde_over_v_star_q_inu) - wtilde_over_v_star_q_inu[0]) / beta

    # wstar_tilde_q_tau[..., :-1] = np.real(
    #     2 * fftpack.fft(wstar_tilde_q_inu, axis=-1) - wstar_tilde_q_inu[..., 0, np.newaxis]) / beta

    # The interaction is bosonic, so the values at \tau = 0, \beta
    # should be equal, and determined from the zero-frequency term;
    # note that in practice, we have defined \tau = 0 := 0^{+}
    wstar_tilde_q_tau[..., -1] = wstar_tilde_q_tau[..., 0]

    # Add back the missing factor of V_{*}(q) to the interactions, now that the FFT has been performed
    iter_idx_q = np.ndindex(vstar_q.shape)
    # Use a zero value for the singular q = 0 point
    next(iter_idx_q)
    for idx_qi in iter_idx_q:
        # V(q) = 4\pi / |q|^2
        vstar_q[idx_qi] = 4 * np.pi / kscale**2 / np.dot(idx_qi, idx_qi)
        wstar_tilde_q_tau[idx_qi] *= vstar_q[idx_qi]

    print('Verifying bosonic symmetry in Wstar(q, tau)...', end='')
    assert np.all(wstar_tilde_q_tau[..., :] == wstar_tilde_q_tau[..., ::-1])
    print('OK')

    return vstar_q, wstar_tilde_q_tau, wstar_tilde_q_inu


def get_lat_wstar_r(vstar_q, wstar_tilde_q4, wstar_tilde_q_tau, lat_const, dim):
    '''Gets the lattice RPA screened interaction in the position space representations,
       W_{*}(r, ~). Obtains the lattice RPA screened interaction W_{*}(r, \tau) via 
       (exact) FFT of the approximate momentum-space result W_{*}(q, \tau), given as
       input. Since V_{*}(r) is now the FFT of V(q) \delta_{q \ne 0}, it must be
       stored in matrix format as for the dynamic part, \widetilde{W}_{*}(r, \tau).'''
    # Do the 3D IFFTs from k to r
    vstar_r = np.real(np.fft.ifftn(vstar_q)) / lat_const**dim

    wstar_tilde_r_inu = np.real(np.fft.ifftn(
        wstar_tilde_q4, axes=np.arange(dim))) / lat_const**dim

    wstar_tilde_r_tau = np.real(np.fft.ifftn(
        wstar_tilde_q_tau, axes=np.arange(dim))) / lat_const**dim

    print('Verifying bosonic symmetry in Wstar(r, tau)...', end='')
    assert np.all(wstar_tilde_r_tau[..., :] == wstar_tilde_r_tau[..., ::-1])
    print('OK')

    return vstar_r, wstar_tilde_r_tau, wstar_tilde_r_inu
