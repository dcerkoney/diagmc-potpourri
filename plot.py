#!/usr/bin/env python3

# Package imports
import os
import re
import sys
import glob
import h5py
import pathlib
import contextlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import product
from packaging import version

if version.parse(matplotlib.__version__) < version.parse("3.4"):
    # Ignore findfont warnings (buggy for matplotlib<3.4)
    import logging
    logging.getLogger('matplotlib.font_manager').disabled = True

# Local script imports
from lattice_tools import fill_band


def complex_h5view(dataset):
    '''Get a view of HDF5 composite re/im data as complex type.'''
    return dataset[:].view(complex)


def load_params_h5(run_subdir, job_name):
    ''' Load the run subdirectory results from HDF5. '''
    # If there are multiple matches, load params from
    # the first match (they will be consistent anyway)
    rundata_path = glob.glob(f"./{run_subdir}/*{job_name}*.h5")[0]
    run_data = h5py.File(rundata_path, 'r')
    params = dict(run_data.attrs)
    if 'diag_typestring' in params:
        params['diag_typestring'] = params['diag_typestring'].decode()
    params.update({'lat_length': params['lat_const'] * params['n_site_pd']})
    run_data.close()
    return params


def get_triqs_stat_chi_ch_RPA(params, path_k_coords):
    '''Use TRIQS TPRF to compute the RPA static charge susceptibility.'''
    from triqs_tprf.tight_binding import TBLattice
    from triqs_tprf.lattice_utils import k_space_path
    from triqs.gf import MeshImFreq, Idx
    from triqs_tprf.lattice import lattice_dyson_g0_wk
    from triqs_tprf.lattice_utils import imtime_bubble_chi0_wk
    from triqs.operators import n, c
    from triqs_tprf.OperatorUtils import quartic_tensor_from_operator
    from triqs_tprf.OperatorUtils import quartic_permutation_symmetrize
    from triqs_tprf.rpa_tensor import lose_spin_degree_of_freedom
    from triqs_tprf.rpa_tensor import split_quartic_tensor_in_charge_and_spin
    from triqs_tprf.lattice import solve_rpa_PH

    H = TBLattice(
        units=[(1, 0, 0), (0, 1, 0)],
        hopping={
            # nearest neighbour hopping -t
            (0, +1): -params['t_hop'] * np.eye(2),
            (0, -1): -params['t_hop'] * np.eye(2),
            (+1, 0): -params['t_hop'] * np.eye(2),
            (-1, 0): -params['t_hop'] * np.eye(2),
        },
        orbital_positions=[(0, 0, 0)]*2,
        orbital_names=['up', 'down'],
    )

    e_k = H.on_mesh_brillouin_zone(
        n_k=(params['n_site_pd'], params['n_site_pd'], 1))

    G = np.array([0.0, 0.0, 0.0]) * 2.*np.pi
    X = np.array([0.5, 0.0, 0.0]) * 2.*np.pi
    M = np.array([0.5, 0.5, 0.0]) * 2.*np.pi

    paths = [(G, X), (X, M), (M, G)]

    k_vecs, _, _ = k_space_path(paths)
    kx, ky, kz = k_vecs.T

    e_k_interp = np.vectorize(lambda kx, ky, kz: e_k([kx, ky, kz])[0, 0].real)
    e_k_interp = e_k_interp(kx, ky, kz)

    wmesh = MeshImFreq(beta=params['beta'], S='Fermion', n_max=100)
    g0_wk = lattice_dyson_g0_wk(mu=params['mu_tilde'], e_k=e_k, mesh=wmesh)
    chi00_wk = imtime_bubble_chi0_wk(g0_wk, nw=1)

    # Analytic, but slow!
    # from triqs_tprf.lattice import lindhard_chi00_wk
    # chi00_wk_analytic = lindhard_chi00_wk(e_k=e_k, beta=params['beta'], mu=params['mu_tilde'], nw=1)

    def chi_ch_contraction(chi):
        """ Computes Tr[ I \chi I ] / 2"""
        I = np.eye(2)
        chi_ch = chi[0, 0, 0, 0].copy()
        chi_ch.data[:] = np.einsum('wqabcd,ab,cd->wq', chi.data, I, I)[:, :]
        chi_ch = chi_ch[Idx(0), :]
        return chi_ch / 2.0

    def interpolate_chi(chi, k_vecs):
        assert(k_vecs.shape[1] == 3)
        chi_interp = np.zeros(
            [k_vecs.shape[0]] + list(chi.target_shape), dtype=np.complex)
        for kidx, (kx, ky, kz) in enumerate(k_vecs):
            chi_interp[kidx] = chi((kx, ky, kz))
        return chi_interp

    H_int = n(0, 0) * n(0, 1)
    fundamental_operators = [c(0, 0), c(0, 1)]

    V_int_abcd = quartic_tensor_from_operator(H_int, fundamental_operators)
    V_int_abcd = quartic_permutation_symmetrize(V_int_abcd)

    V_abcd = np.zeros_like(V_int_abcd)
    for a, b, c, d in product(range(V_abcd.shape[0]), repeat=4):
        V_abcd[a, b, c, d] = V_int_abcd[b, d, a, c]

    # Get a general charge interaction tensor, and tune the strength by U
    V_c_abcd, _ = split_quartic_tensor_in_charge_and_spin(V_abcd)
    U_c = params['U_loc'] * V_c_abcd

    chi00_wk_spinless = lose_spin_degree_of_freedom(chi00_wk, spin_fast=False)
    chi_ch = chi_ch_contraction(solve_rpa_PH(chi00_wk_spinless, U_c))

    k_vecs = np.hstack((path_k_coords, np.zeros((len(path_k_coords), 1))))
    triqs_stat_chi_ch_RPA = interpolate_chi(chi_ch, k_vecs).real
    return triqs_stat_chi_ch_RPA


def plot_chi_n_ch(params, run_subdir, job_name, logfile):
    '''Plots the nth order correction to the static charge susceptibility.'''
    # Read relevant input params from log file
    n_threads = 1
    with open(logfile) as input_logfile:
        for line in input_logfile:
            # Find the line containing the number of MPI threads
            match_n_threads = re.search(
                r'MPI run with (\S+) thread\(s\)', line)
            if match_n_threads:
                n_threads = int(float(match_n_threads.group(1)))
                # print(f'\nn_threads: {n_threads}')
                break

    # Load results from h5
    chi_path = pathlib.PosixPath(
        f'./{run_subdir}') / f"{params['save_name']}_{job_name}.h5"
    chi_ch_run_data = h5py.File(chi_path, 'r')
    # NOTE: We define the bubble with an extra minus sign, following the usual
    #       Dyson equation convention, in contrast with Kristjan's code (and Mahan)
    chi_n_ch_means = -1 * \
        complex_h5view(chi_ch_run_data[f"V{params['max_order']}_meas_mean"])
    chi_n_ch_errs = None
    if n_threads > 1:
        chi_n_ch_errs = chi_ch_run_data[f"V{params['max_order']}_meas_stderr"][:]
    chi_ch_run_data.close()

    # Charge and longitudinal spin susceptibility measurements are real, although
    # we may find a small noisy imaginary part in practice; throw it out
    assert np.allclose(chi_n_ch_means.imag, 0)
    chi_n_ch_means = chi_n_ch_means.real

    # Deduce number of measured frequency points
    n_nu_meas = len(chi_n_ch_means) / float(params['n_k_meas'])
    assert n_nu_meas == int(n_nu_meas)
    n_nu_meas = int(n_nu_meas)

    # List of Matsubara frequencies at which the susceptibility was measured
    # m_list = np.arange(n_nu_meas)
    # om_list = (2 * np.pi / params['beta']) * m_list
    # print('\nList of measurement Matsubara frequencies:\n', om_list * (p.beta / np.pi))

    # Tests the band filling scheme
    ef, kf_vecs = fill_band(
        dim=params['dim'],
        num_elec=params['num_elec'],
        n_site_pd=params['n_site_pd'],
        lat_const=params['lat_const'],
        t_hop=params['t_hop'],
    )
    # print('List of k-points on the Fermi surface:\n', kf_vecs)

    # Numerical roundoff issues occur at half-filling, so manually round the Fermi energy to zero
    if params['n0'] == 1:
        # Double-check that we actually obtained a near-zero answer for ef
        # (i.e., that the k_F vectors are legitimate)
        assert np.allclose(ef, params['ef'])

    # Build an ordered path of k-points in the Brillouin zone; we choose the high-symmetry
    # path for the simple square lattice (\Gamma - X - M - \Gamma), discarding duplicate
    # coordinates at the path vertices (accounted for in plotting step).
    n_path_edges = 3
    n_edge = int(np.floor(params['n_site_pd'] / 2.0))
    nk_coords_Gamma_X = [[x, 0] for x in range(0, n_edge + 1)]
    nk_coords_X_M = [[n_edge, y] for y in range(1, n_edge + 1)]
    # NOTE: We have introduced the duplicate \Gamma point at
    #       the end of the k-point list for plotting purposes
    nk_coords_M_Gamma = [[xy, xy] for xy in range(0, n_edge)[::-1]]
    # Indices for the high-symmetry points
    idx_Gamma1 = 0
    idx_X = len(nk_coords_Gamma_X) - 1
    idx_M = len(nk_coords_Gamma_X) + len(nk_coords_X_M) - 1
    idx_Gamma2 = len(nk_coords_Gamma_X) + \
        len(nk_coords_X_M) + len(nk_coords_M_Gamma) - 1
    # Build the full ordered high-symmetry path
    path_nk_coords = np.concatenate(
        (nk_coords_Gamma_X, nk_coords_X_M, nk_coords_M_Gamma))
    # path_nk_coords = path_nk_coords - n_site_pd * (path_nk_coords == 30)
    k_scale = 2.0 * np.pi / float(params['lat_length'])
    path_k_coords = k_scale * path_nk_coords
    n_k_plot = len(path_k_coords)

    # Find the corresponding indices in the full k_list
    i_path = np.arange(len(path_k_coords))
    i_path_kf_locs = []
    for i, this_k_coord in enumerate(path_k_coords):
        for kf_vec in kf_vecs:
            if np.all(this_k_coord == kf_vec):
                i_path_kf_locs.append(i)
    i_path_kf_locs = np.asarray(i_path_kf_locs)
    # If we missed the Fermi surface along the M-\Gamma path
    # due to coarse-graining, set the locations manually
    if (len(i_path_kf_locs) == 1):
        i_path_kf_locs = [params['n_k_meas'] /
                          3.0, params['n_k_meas'] * 5 / 6.0]

    # The missing k-point is the duplicate \Gamma point
    assert n_k_plot == params['n_k_meas'] + 1
    assert len(chi_n_ch_means) == params['n_k_meas'] * n_nu_meas

    # Reshape the calculated susceptibility data into
    chi_n_ch_calc_means = chi_n_ch_means.reshape(
        (n_nu_meas, params['n_k_meas']))
    chi_n_ch_calc_errs = None
    if n_threads > 1:
        chi_n_ch_calc_errs = chi_n_ch_errs.reshape(
            (n_nu_meas, params['n_k_meas']))

    ##########################################################################
    # Plot the charge susceptibility for the first few Matsubara frequencies #
    ##########################################################################

    print('\nPlotting charge susceptibility corrections...', end='', flush=True)

    colorlist = ['orchid', 'cornflowerblue',
                 'turquoise', 'chartreuse', 'greenyellow']

    # Plot the static susceptibility if n_nu_meas = 1
    if n_nu_meas == 1:
        # Make singleton axis list for compatibility with the static case (n_nu_meas = 1)
        fig, axes = plt.subplots()
        axes.axvline(x=i_path_kf_locs[0], linestyle='-', color='0.0',
                     zorder=-1, linewidth=1, label=r'$\mathbf{k}_F$')
        for i_path_kf_loc in i_path_kf_locs[1:]:
            axes.axvline(x=i_path_kf_loc, linestyle='-',
                         color='0.0', zorder=-1, linewidth=1)
        axes = [axes]
    # Plot the susceptibility for at most the first 5 frequency points as a function of momentum
    else:
        fig, axes = plt.subplots(1, min(n_nu_meas, 5),
                                 figsize=(4 * min(n_nu_meas, 5), 3.5))

    for iom in range(min(n_nu_meas, 5)):
        axes[iom].plot(i_path, chi_n_ch_calc_means[iom, :][i_path % params['n_k_meas']], 'o-',
                       markersize=2.5, color=colorlist[iom],
                       label=rf"$n = {params['n0']:g},\; U = {params['U_loc']:g},\; \beta = {params['beta']:g}$")
        if n_threads > 1:
            axes[iom].fill_between(
                i_path, chi_n_ch_calc_means[iom, :][i_path % params['n_k_meas']] +
                chi_n_ch_calc_errs[iom, :][i_path % params['n_k_meas']],
                chi_n_ch_calc_means[iom, :][i_path % params['n_k_meas']] -
                chi_n_ch_calc_errs[iom, :][i_path % params['n_k_meas']],
                color=colorlist[iom], alpha=0.3)
        axes[iom].legend(loc='upper left')
        # Add some evenly-spaced minor ticks to the axis
        n_minor_ticks = 9
        path_nk_coords = np.concatenate(
            (nk_coords_Gamma_X, nk_coords_X_M, nk_coords_M_Gamma))
        minor_ticks = []
        for i in range(n_path_edges):
            offset = i * n_edge
            minor_ticks += (offset + np.arange(0, n_edge, n_edge /
                                               (n_minor_ticks // 3), dtype=float)).tolist()
        mask_major = [i for i in np.arange(
            n_minor_ticks) if i not in [0, 3, 6, 9]]
        minor_ticks = np.asarray(minor_ticks)[mask_major]
        axes[iom].set_xticks(minor_ticks, minor=True)
        # Label the high-symmetry points
        axes[iom].set_xticks((idx_Gamma1, idx_X, idx_M, idx_Gamma2))
        axes[iom].set_xticklabels((r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$'))
        iqmstring = r''
        # if iom == 0:
        #     iqmstring = r'=0'
        axes[iom].set_title(rf"$\chi^{{({params['n_intn']})}}_{{\mathrm{{ch}}}}$" +
                            rf'$[G_{{H}}, U](\mathbf{{q}}, iq_{iom}{iqmstring})$', pad=13)
        axes[iom].text(
            x=0.0365,
            y=0.815,
            s=fr"$N_{{\mathrm{{thread}}}}={n_threads},\, N_{{\mathrm{{meas}}}} = {params['n_meas']:g}$",
            horizontalalignment='left',
            verticalalignment='center',
            transform=axes[iom].transAxes,
            bbox=dict(boxstyle="round, pad=0.4", ec=(
                0.8, 0.8, 0.8), fc=(1, 1, 1), alpha=0.8),
        )
        # Set the same plot range and ticks as the Kozik paper for easy visual correspondence
        axes[iom].set_xlim(left=i_path[0], right=i_path[-1])
        # axes[iom].set_ylim(bottom=-.65, top=-.35)
        axes[iom].grid(True, color='k', linestyle=':', which='minor')
        axes[iom].grid(True, color='k', linestyle=':', which='major')
        fig.tight_layout()
    savepath = pathlib.PosixPath(
        f'./{run_subdir}') / f"chi_{params['n_intn']}_ch_2dsqhub_{job_name}.pdf"
    fig.savefig(savepath)
    plt.close('all')

    print('done!')
    return


def plot_static_chi_ch_together(params, run_subdir, job_name, logfile, plot_rpa=True):
    '''Plots the bare static charge susceptibility, as well
       as the bare result plus nth order correction.'''
    # Read relevant input params from log file
    n_threads = 1
    with open(logfile) as input_logfile:
        for line in input_logfile:
            # Find the line containing the number of MPI threads
            match_n_threads = re.search(
                r'MPI run with (\S+) thread\(s\)', line)
            if match_n_threads:
                n_threads = int(float(match_n_threads.group(1)))
                # print(f'\nn_threads: {n_threads}')
                break

    # Load results from h5
    chi_path = pathlib.PosixPath(
        f'./{run_subdir}') / f"{params['save_name']}_{job_name}.h5"
    chi_ch_run_data = h5py.File(chi_path, 'r')
    # NOTE: We define the bubble with an extra minus sign, following the usual
    #       Dyson equation convention, in contrast with Kristjan's code (and Mahan)
    chi_n_ch_means = -1 * \
        complex_h5view(chi_ch_run_data[f"V{params['max_order']}_meas_mean"])
    chi_n_ch_errs = None
    if n_threads > 1:
        chi_n_ch_errs = chi_ch_run_data[f"V{params['max_order']}_meas_stderr"][:]
    chi_ch_run_data.close()

    # Charge and (longitudinal) spin susceptibility measurements should be real!
    assert np.allclose(chi_n_ch_means.imag, 0)
    chi_n_ch_means = chi_n_ch_means.real

    # Import the bare polarization bubble to get the zeroth-order static charge susceptibility
    found_consistent_poln = True
    consistent_poln_path = None
    propr_paths = glob.glob(f'./propagators/*/lat_pi0_q4.h5')
    for propr_path in propr_paths:
        pi0_path = pathlib.PosixPath(propr_path)
        pi0_data = h5py.File(pi0_path, 'r')
        # Check for consistency of relevant attributes
        params_consistent = (params['beta'] == pi0_data.attrs['beta']
                             and params['n_site_pd'] == pi0_data.attrs['n_site_pd'])
        if params_consistent:
            consistent_poln_path = pi0_path
            found_consistent_poln = True
            break
    if found_consistent_poln:
        print(
            f'Found consistent polarization bubble data in propagator subdirectory {consistent_poln_path}, loading it...', end='')
    else:
        raise ValueError(
            'No MCMC parameters found which are consistent with the supplied polarization bubble data!')
    # As above, chi_n_ch(q) = -P_n(q)
    pi0_shape_3d = pi0_data['lat_pi0_q4_data'].attrs['shape']
    chi_1_ch_ex_data = -1 * \
        pi0_data['lat_pi0_q4_data'][:].reshape(pi0_shape_3d)
    pi0_data.close()
    print('done!')

    # Deduce number of measured frequency points
    n_nu_meas = len(chi_n_ch_means) / float(params['n_k_meas'])
    assert n_nu_meas == int(n_nu_meas)
    n_nu_meas = int(n_nu_meas)

    # List of Matsubara frequencies at which the susceptibility was measured
    # m_list = np.arange(n_nu_meas)
    # om_list = (2 * np.pi / params['beta']) * m_list
    # print('\nList of measurement Matsubara frequencies:\n', om_list)

    # Tests the band filling scheme
    ef, kf_vecs = fill_band(
        dim=params['dim'],
        num_elec=params['num_elec'],
        n_site_pd=params['n_site_pd'],
        lat_const=params['lat_const'],
        t_hop=params['t_hop'],
    )
    # print('List of k-points on the Fermi surface:\n', kf_vecs)

    # Numerical roundoff issues occur at half-filling, so manually round the Fermi energy to zero
    if params['n0'] == 1:
        # Double-check that we actually obtained a near-zero answer for ef
        # (i.e., that the k_F vectors are legitimate)
        assert np.allclose(ef, params['ef'])

    # Build an ordered path of k-points in the Brillouin zone; we choose the high-symmetry
    # path for the simple square lattice (\Gamma - X - M - \Gamma), discarding duplicate
    # coordinates at the path vertices (accounted for in plotting step).
    n_path_edges = 3
    n_edge = int(np.floor(params['n_site_pd'] / 2.0))
    nk_coords_Gamma_X = [[x, 0] for x in range(0, n_edge + 1)]
    nk_coords_X_M = [[n_edge, y] for y in range(1, n_edge + 1)]
    # NOTE: We have introduced the duplicate \Gamma point at
    #       the end of the k-point list for plotting purposes
    nk_coords_M_Gamma = [[xy, xy] for xy in range(0, n_edge)[::-1]]
    # Indices for the high-symmetry points
    idx_Gamma1 = 0
    idx_X = len(nk_coords_Gamma_X) - 1
    idx_M = len(nk_coords_Gamma_X) + len(nk_coords_X_M) - 1
    idx_Gamma2 = len(nk_coords_Gamma_X) + \
        len(nk_coords_X_M) + len(nk_coords_M_Gamma) - 1
    # Build the full ordered high-symmetry path
    path_nk_coords = np.concatenate(
        (nk_coords_Gamma_X, nk_coords_X_M, nk_coords_M_Gamma))
    # path_nk_coords = path_nk_coords - n_site_pd * (path_nk_coords == 30)
    k_scale = 2.0 * np.pi / float(params['lat_length'])
    path_k_coords = k_scale * path_nk_coords
    n_k_plot = len(path_k_coords)

    # Find the corresponding indices in the full k_list
    i_path = np.arange(len(path_k_coords))
    i_path_kf_locs = []
    for i, this_k_coord in enumerate(path_k_coords):
        for kf_vec in kf_vecs:
            if np.all(this_k_coord == kf_vec):
                i_path_kf_locs.append(i)
    i_path_kf_locs = np.asarray(i_path_kf_locs)
    # If we missed the Fermi surface along the M-\Gamma path
    # due to coarse-graining, set the locations manually
    if len(i_path_kf_locs) == 1:
        # print(i_path_kf_locs)
        i_path_kf_locs = [params['n_k_meas'] /
                          3.0, params['n_k_meas'] * 5 / 6.0]

    # The missing k-point is the duplicate \Gamma point
    assert n_k_plot == params['n_k_meas'] + 1
    assert len(chi_n_ch_means) == params['n_k_meas'] * n_nu_meas

    # Reshape the calculated susceptibility data into a 2D array
    chi_n_ch_calc_means = chi_n_ch_means.reshape(
        (n_nu_meas, params['n_k_meas']))
    chi_n_ch_calc_errs = None
    if n_threads > 1:
        chi_n_ch_calc_errs = chi_n_ch_errs.reshape(
            (n_nu_meas, params['n_k_meas']))

    # Get the exact noninteracting static susceptibility along the k-path
    chi_0_ch_exact = np.zeros((n_nu_meas, params['n_k_meas']))
    # Skip the duplicate \Gamma point when filling the array
    for ik, nk in enumerate(path_nk_coords[:-1]):
        slice_obj = tuple(nk) + (slice(None, n_nu_meas),)
        chi_0_ch_exact[:, ik] = chi_1_ch_ex_data[slice_obj]

    chi_n_ch_tot_means = chi_0_ch_exact + chi_n_ch_calc_means
    chi_n_ch_tot_errs = None
    if n_threads > 1:
        chi_n_ch_tot_errs = chi_n_ch_calc_errs

    ##########################################################
    # Plot static charge susceptibility vs expansion order n #
    ##########################################################

    if plot_rpa:
        # Surpress TRIQS stdout/stderr
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            triqs_chi_0_ch_RPA = get_triqs_stat_chi_ch_RPA(
                params, path_k_coords)

    print('Plotting static charge susceptibility...', end='', flush=True)

    colorlist = ['darkorchid', 'orchid', 'cornflowerblue']

    # Plot the susceptibility for the first 5 frequency points as a function of momentum
    fig, ax = plt.subplots()
    ax.axvline(x=i_path_kf_locs[0], linestyle='-', color='0.0',
               zorder=-1, linewidth=1, label=r'$\mathbf{k}_{\mathrm{F}}$')
    for i_path_kf_loc in i_path_kf_locs[1:]:
        ax.axvline(x=i_path_kf_loc, linestyle='-',
                   color='0.0', zorder=-1, linewidth=1)

    # The expansion order n denotes order in U, i.e., N = (order - 1) = n_intn
    ax.plot(i_path, chi_0_ch_exact[0, :][i_path % params['n_k_meas']], 'o-',
            markersize=2.5, color=colorlist[0], label=rf'$n=0$')

    ax.plot(i_path, chi_n_ch_tot_means[0, :][i_path % params['n_k_meas']], 'o-',
            markersize=2.5, color=colorlist[1], label=rf"$n={params['n_intn']}$")

    if plot_rpa:
        ax.plot(i_path, triqs_chi_0_ch_RPA[i_path % params['n_k_meas']], 'o-',
                markersize=2.5, color=colorlist[2], label='RPA')

    if n_threads > 1:
        ax.fill_between(i_path, chi_n_ch_tot_means[0, :][i_path % params['n_k_meas']]
                        + chi_n_ch_tot_errs[0, :][i_path % params['n_k_meas']],
                        chi_n_ch_tot_means[0, :][i_path % params['n_k_meas']] -
                        chi_n_ch_tot_errs[0, :][i_path % params['n_k_meas']],
                        color=colorlist[1], alpha=0.3)
    ax.legend(loc='upper left')
    # Add some evenly-spaced minor ticks to the axis
    n_minor_ticks = 9
    path_nk_coords = np.concatenate(
        (nk_coords_Gamma_X, nk_coords_X_M, nk_coords_M_Gamma))
    minor_ticks = []
    for i in range(n_path_edges):
        offset = i * n_edge
        minor_ticks += (offset + np.arange(0, n_edge, n_edge /
                                           (n_minor_ticks // 3), dtype=float)).tolist()
    mask_major = [i for i in np.arange(
        n_minor_ticks) if i not in [0, 3, 6, 9]]
    minor_ticks = np.asarray(minor_ticks)[mask_major]
    ax.set_xticks(minor_ticks, minor=True)
    # Label the high-symmetry points
    ax.set_xticks((idx_Gamma1, idx_X, idx_M, idx_Gamma2))
    ax.set_xticklabels((r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$'))
    ax.set_title(
        rf'Static charge susceptibility $\chi_{{\mathrm{{ch}}}}[G_{{H}}, U](\mathbf{{q}})$ to $\mathcal{{O}}\hspace{{-0.5ex}}\left(U^n\right)$', pad=13)
    ax.text(
        x=0.021,
        y=0.76 + 0.022 - 0.05 * (plot_rpa),
        s=fr"$n = {params['n0']:g},\; U = {params['U_loc']:g},\; \beta = {params['beta']:g}$",
        # s=fr"$U = {params['U_loc']:g},\; \beta = {params['beta']:g},\; \widetilde{{\mu}} = {params['mu_tilde']:g}$",
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round, pad=0.4", ec=(
            0.8, 0.8, 0.8), fc=(1, 1, 1), alpha=0.8),
    )
    ax.text(
        x=0.021,
        y=0.685 + 0.022 - 0.05 * (plot_rpa),
        s=fr"$N_{{\mathrm{{thread}}}}={n_threads},\, N_{{\mathrm{{meas}}}} = {params['n_meas']:g}$",
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round, pad=0.4", ec=(
            0.8, 0.8, 0.8), fc=(1, 1, 1), alpha=0.8),
    )
    # Set the same plot range and ticks as the Kozik paper for easy visual correspondence
    ax.set_xlim(left=i_path[0], right=i_path[-1])
    # ax.set_ylim(bottom=-.65, top=-.35)
    ax.grid(True, color='k', linestyle=':', which='minor')
    ax.grid(True, color='k', linestyle=':', which='major')
    fig.tight_layout()
    savepath = pathlib.PosixPath(
        f'./{run_subdir}') / f'static_chi_ch_2dsqhub_{job_name}.pdf'
    fig.savefig(savepath)
    plt.close('all')
    print('done!\n')
    return


def plot_self_en(params, run_subdir, job_name, logfile):
    '''Plots the nth order correction to the charge susceptibility
       at the first frequency point, omega_0 = pi T.'''
    # Read relevant input params from log file
    n_threads = 1
    with open(logfile) as input_logfile:
        for line in input_logfile:
            # Find the line containing the number of MPI threads
            match_n_threads = re.search(
                r'MPI run with (\S+) thread\(s\)', line)
            if match_n_threads:
                n_threads = int(float(match_n_threads.group(1)))
                print(f'\nn_threads: {n_threads}')
                break

    for i in params.items():
        print(i)

    # Load results from h5
    self_en_path = pathlib.PosixPath(
        f'./{run_subdir}') / f"{params['save_name']}_{job_name}.h5"
    self_en_run_data = h5py.File(self_en_path, 'r')
    self_en_means = complex_h5view(self_en_run_data[f"V{params['max_order']}_meas_mean"])
    self_en_errs = None
    if n_threads > 1:
        self_en_errs = self_en_run_data[f"V{params['max_order']}_meas_stderr"][:]
    self_en_run_data.close()

    # Deduce number of measured frequency points
    n_om_meas = len(self_en_means) / float(params['n_k_meas'])
    assert n_om_meas == int(n_om_meas)
    n_om_meas = int(n_om_meas)

    # List of Matsubara frequencies at which the susceptibility was measured
    n_list = np.arange(n_om_meas)
    om_list = (2 * n_list + 1) * (np.pi / params['beta'])
    # print('\nList of measurement Matsubara frequencies:\n', om_list)

    # Tests the band filling scheme
    ef, kf_vecs = fill_band(
        dim=params['dim'],
        num_elec=params['num_elec'],
        n_site_pd=params['n_site_pd'],
        lat_const=params['lat_const'],
        t_hop=params['t_hop'],
    )
    # print('List of k-points on the Fermi surface:\n', kf_vecs)

    # Numerical roundoff issues occur at half-filling, so manually round the Fermi energy to zero
    if params['n0'] == 1:
        # Double-check that we actually obtained a near-zero answer for ef
        # (i.e., that the k_F vectors are legitimate)
        assert np.allclose(ef, params['ef'])

    # Build an ordered path of k-points in the Brillouin zone; we choose the high-symmetry
    # path for the simple square lattice (\Gamma - X - M - \Gamma), discarding duplicate
    # coordinates at the path vertices (accounted for in plotting step).
    n_path_edges = 3
    n_edge = int(np.floor(params['n_site_pd'] / 2.0))
    nk_coords_Gamma_X = [[x, 0] for x in range(0, n_edge + 1)]
    nk_coords_X_M = [[n_edge, y] for y in range(1, n_edge + 1)]
    # NOTE: We have introduced the duplicate \Gamma point at
    #       the end of the k-point list for plotting purposes
    nk_coords_M_Gamma = [[xy, xy] for xy in range(0, n_edge)[::-1]]
    # Indices for the high-symmetry points
    idx_Gamma1 = 0
    idx_X = len(nk_coords_Gamma_X) - 1
    idx_M = len(nk_coords_Gamma_X) + len(nk_coords_X_M) - 1
    idx_Gamma2 = len(nk_coords_Gamma_X) + \
        len(nk_coords_X_M) + len(nk_coords_M_Gamma) - 1
    # Build the full ordered high-symmetry path
    path_nk_coords = np.concatenate(
        (nk_coords_Gamma_X, nk_coords_X_M, nk_coords_M_Gamma))
    # path_nk_coords = path_nk_coords - n_site_pd * (path_nk_coords == 30)
    k_scale = 2.0 * np.pi / float(params['lat_length'])
    path_k_coords = k_scale * path_nk_coords
    n_k_plot = len(path_k_coords)

    # Find the corresponding indices in the full k_list
    i_path = np.arange(len(path_k_coords))
    i_path_kf_locs = []
    for i, this_k_coord in enumerate(path_k_coords):
        for kf_vec in kf_vecs:
            if np.all(this_k_coord == kf_vec):
                i_path_kf_locs.append(i)
    i_path_kf_locs = np.asarray(i_path_kf_locs)
    # If we missed the Fermi surface along the M-\Gamma path
    # due to coarse-graining, set the locations manually
    if (len(i_path_kf_locs) == 1):
        i_path_kf_locs = [params['n_k_meas'] /
                          3.0, params['n_k_meas'] * 5 / 6.0]

    # The missing k-point is the duplicate \Gamma point
    assert n_k_plot == params['n_k_meas'] + 1
    assert len(self_en_means) == params['n_k_meas'] * n_om_meas

    # Reshape the calculated susceptibility data into
    self_en_means = self_en_means.reshape((n_om_meas, params['n_k_meas']))
    if n_threads > 1:
        self_en_errs = self_en_errs.reshape((n_om_meas, params['n_k_meas']))

    # Estimate the local self-energy by averaging over k-points
    local_self_en_means = np.mean(self_en_means, axis=-1)
    local_self_en_errs = np.mean(self_en_errs, axis=-1)

    print('\nPlotting self energy corrections...', end='', flush=True)

    colorlist = ['orchid', 'cornflowerblue', 'turquoise', 'chartreuse',
                 'greenyellow', 'gold', 'orange', 'orangered', 'red', 'firebrick']

    n_om_plot = min(n_om_meas, len(colorlist))

    component_names = ['Re', 'Im']
    component_colors = ['r', 'b']

    def cplx_cmpts(sigma):
        return [sigma.real, sigma.imag]

    ###########################################################
    # Plot the self energy for the lowest Matsubara frequency #
    ###########################################################

    fig, ax = plt.subplots()
    ax.axvline(x=i_path_kf_locs[0], linestyle='-', color='0.0',
                 zorder=-1, linewidth=1, label=r'$\mathbf{k}_F$')
    for i_path_kf_loc in i_path_kf_locs[1:]:
        ax.axvline(x=i_path_kf_loc, linestyle='-',
                   color='0.0', zorder=-1, linewidth=1)
    for i, self_en_means_cmpt in enumerate(cplx_cmpts(self_en_means)):
        ax.plot(i_path, self_en_means_cmpt[0, :][i_path % params['n_k_meas']], 'o-',
                markersize=2.5, color=component_colors[i],
                label=component_names[i])
        if n_threads > 1:
            ax.fill_between(
                i_path, self_en_means_cmpt[0, :][i_path % params['n_k_meas']] +
                self_en_errs[0, :][i_path % params['n_k_meas']],
                self_en_means_cmpt[0, :][i_path % params['n_k_meas']] -
                self_en_errs[0, :][i_path % params['n_k_meas']],
                color=component_colors[i], alpha=0.25)
    ax.legend(loc='upper right')
    # Add some evenly-spaced minor ticks to the axis
    n_minor_ticks = 9
    path_nk_coords = np.concatenate(
        (nk_coords_Gamma_X, nk_coords_X_M, nk_coords_M_Gamma))
    minor_ticks = []
    for i in range(n_path_edges):
        offset = i * n_edge
        minor_ticks += (offset + np.arange(0, n_edge, n_edge /
                                           (n_minor_ticks // 3), dtype=float)).tolist()
    mask_major = [i for i in np.arange(
        n_minor_ticks) if i not in [0, 3, 6, 9]]
    minor_ticks = np.asarray(minor_ticks)[mask_major]
    ax.set_xticks(minor_ticks, minor=True)
    # Label the high-symmetry points
    ax.set_xticks((idx_Gamma1, idx_X, idx_M, idx_Gamma2))
    ax.set_xticklabels((r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$'))
    ax.set_title(rf"$\Sigma^{{({params['n_intn']})}}$" +
                 rf'$[G_{{H}}, U](\mathbf{{k}}, i\omega_0 = \pi T)$', pad=13)
    ax.text(
        x=0.021,
        y=0.953,
        s=fr"$n = {params['n0']:g},\; U = {params['U_loc']:g},\; \beta = {params['beta']:g}$",
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round, pad=0.4", ec=(
            0.8, 0.8, 0.8), fc=(1, 1, 1), alpha=1.0),
    )
    ax.text(
        x=0.021,
        y=0.878,
        s=fr"$N_{{\mathrm{{thread}}}}={n_threads},\, N_{{\mathrm{{meas}}}} = {params['n_meas']:g}$",
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round, pad=0.4", ec=(
            0.8, 0.8, 0.8), fc=(1, 1, 1), alpha=1.0),
    )
    # Set the same plot range and ticks as the Kozik paper for easy visual correspondence
    ax.set_xlim(left=i_path[0], right=i_path[-1])
    # ax.set_ylim(bottom=-.65, top=-.35)
    ax.grid(True, color='k', linestyle=':', which='minor')
    ax.grid(True, color='k', linestyle=':', which='major')
    fig.tight_layout()
    savepath = pathlib.PosixPath(
        f'./{run_subdir}') / f"sigma_{params['n_intn']}_iom0_2dsqhub_{job_name}.pdf"
    fig.savefig(savepath)
    plt.close('all')

    ####################################################################
    # Plot the self energy for the first several Matsubara frequencies #
    ####################################################################

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    axes = [ax1, ax2]
    figs = [fig1, fig2]

    for i, ax in enumerate(axes):
        self_en_means_compt = cplx_cmpts(self_en_means)[i]
        for iom in range(n_om_plot):
            ax.plot(i_path, self_en_means_compt[iom, :][i_path % params['n_k_meas']], 'o-',
                    markersize=2.5, color=colorlist[iom],
                    label=rf"$\omega_{iom}$")
                    # label=rf"$\omega_{iom} = {2 * n_list[iom] + 1} \pi T$")
            if n_threads > 1:
                ax.fill_between(
                    i_path, self_en_means_compt[iom, :][i_path % params['n_k_meas']] +
                    self_en_errs[iom, :][i_path % params['n_k_meas']],
                    self_en_means_compt[iom, :][i_path % params['n_k_meas']] -
                    self_en_errs[iom, :][i_path % params['n_k_meas']],
                    color=colorlist[iom], alpha=0.3)
        ax.axvline(x=i_path_kf_locs[0], linestyle='-', color='0.0',
                   zorder=-1, linewidth=1, label=r'$\mathbf{k}_F$')
        for i_path_kf_loc in i_path_kf_locs[1:]:
            ax.axvline(x=i_path_kf_loc, linestyle='-',
                       color='0.0', zorder=-1, linewidth=1)
        legend = ax.legend(loc='upper left', ncol=3, framealpha=1.0)
        # Add some evenly-spaced minor ticks to the axis
        n_minor_ticks = 9
        path_nk_coords = np.concatenate(
            (nk_coords_Gamma_X, nk_coords_X_M, nk_coords_M_Gamma))
        minor_ticks = []
        for i_tick in range(n_path_edges):
            offset = i_tick * n_edge
            minor_ticks += (offset + np.arange(0, n_edge, n_edge /
                                               (n_minor_ticks // 3), dtype=float)).tolist()
        mask_major = [i for i in np.arange(
            n_minor_ticks) if i not in [0, 3, 6, 9]]
        minor_ticks = np.asarray(minor_ticks)[mask_major]
        ax.set_xticks(minor_ticks, minor=True)
        # Label the high-symmetry points
        ax.set_xticks((idx_Gamma1, idx_X, idx_M, idx_Gamma2))
        ax.set_xticklabels((r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$'))
        ax.set_title(rf"{component_names[i]}$\Sigma^{{({params['n_intn']})}}$" +
                     rf'$[G_{{H}}, U](\mathbf{{k}}, i\omega_n)$', pad=13)
        ax.text(
            x=0.021,
            y=0.76 - 0.03,
            s=fr"$n = {params['n0']:g},\; U = {params['U_loc']:g},\; \beta = {params['beta']:g}$",
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes,
            bbox=dict(boxstyle="round, pad=0.4", ec=(
                0.8, 0.8, 0.8), fc=(1, 1, 1), alpha=1.0),
        )
        ax.text(
            x=0.021,
            y=0.685 - 0.03,
            s=fr"$N_{{\mathrm{{thread}}}}={n_threads},\, N_{{\mathrm{{meas}}}} = {params['n_meas']:g}$",
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes,
            bbox=dict(boxstyle="round, pad=0.4", ec=(
                0.8, 0.8, 0.8), fc=(1, 1, 1), alpha=1.0),
        )
        # Set the same plot range and ticks as the Kozik paper for easy visual correspondence
        ax.set_xlim(left=i_path[0], right=i_path[-1])
        # ax.set_ylim(bottom=-.65, top=-.35)
        ax.grid(True, color='k', linestyle=':', which='minor')
        ax.grid(True, color='k', linestyle=':', which='major')
        figs[i].tight_layout()
        savepath = pathlib.PosixPath(
            f'./{run_subdir}') / f"{component_names[i].lower()}_sigma_{params['n_intn']}_2dsqhub_{job_name}.pdf"
        figs[i].savefig(savepath)
    plt.close('all')

    ######################################################
    # Plot the local self energy vs. Matsubara frequency #
    ######################################################

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    axes = [ax1, ax2]
    figs = [fig1, fig2]

    for i, ax in enumerate(axes):
        local_self_en_means_compt = cplx_cmpts(local_self_en_means)[i]
        ax.axhline(y=0, linestyle='--', color='0.0', zorder=-1, linewidth=1)
        ax.plot(om_list, local_self_en_means_compt, 'o-',
                markersize=2.5, color=component_colors[i])
        if n_threads > 1:
            ax.fill_between(om_list, local_self_en_means_compt + local_self_en_errs,
                            local_self_en_means_compt - local_self_en_errs,
                            color=component_colors[i], alpha=0.3)
        ax.set_title(rf"{component_names[i]}$\Sigma^{{({params['n_intn']})}}_{{\mathrm{{loc}}}}$" +
                     rf'$[G_{{H}}, U](i\omega_n)$', pad=13)
        ax.text(
            x=0.73 - 0.02,
            y=0.88,
            s=fr"$n = {params['n0']:g},\; U = {params['U_loc']:g},\; \beta = {params['beta']:g}$",
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes,
            bbox=dict(boxstyle="round, pad=0.4", ec=(
                0.8, 0.8, 0.8), fc=(1, 1, 1), alpha=1.0),
        )
        ax.text(
            x=0.625 - 0.02,
            y=0.88 - 0.08,
            s=fr"$N_{{\mathrm{{thread}}}}={n_threads},\, N_{{\mathrm{{meas}}}} = {params['n_meas']:g}$",
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes,
            bbox=dict(boxstyle="round, pad=0.4", ec=(
                0.8, 0.8, 0.8), fc=(1, 1, 1), alpha=1.0),
        )
        # Set the same plot range and ticks as the Kozik paper for easy visual correspondence
        ax.set_xlim(left=0)
        ax.set_xlabel(rf'$\omega_n = (2 n + 1) \pi T$')
        # ax.set_ylim(top=0.005)
        ax.grid(True, color='k', linestyle=':')
        figs[i].tight_layout()
        savepath = pathlib.PosixPath(
            f'./{run_subdir}') / f"{component_names[i].lower()}_sigma_{params['n_intn']}_loc_2dsqhub_{job_name}.pdf"
        figs[i].savefig(savepath)
    plt.close('all')

    print('done!')
    return


def main():
    """
    Generate charge susceptibility plots for the Hubbard model on a finite square lattice.
    """
    # Should the RPA result from TRIQS TPRF be added to the static plots for comparison?
    plot_rpa = True

    # Use TeX / TeX font in plots
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    # Parse command line argument(s)
    if len(sys.argv[1:]) == 0:
        logfiles = glob.glob('*/*run*.log')
        if not logfiles:
            raise ValueError('No logfile found in the working directory!')
        # Get a list of job IDs corresponding to
        # all logfiles in the working directory
        filenames = [os.path.splitext(lf)[0] for lf in logfiles]
        job_ids = [int(fname.rsplit('run_')[-1]) for fname in filenames]
        job_names = [f'run_{str(jid)}' for jid in job_ids]
        run_subdirs = [os.path.dirname(lf) for lf in logfiles]
        for i in range(len(logfiles)):
            # Get parameters
            print(
                f"\nGenerating plots for run subdirectory '{run_subdirs[i]}':\n")
            # Ignore this subdir if there is no complete run data in it
            if not glob.glob(f'{run_subdirs[i]}/*run*.h5'):
                print('\nNo run data found in this working directory, skipping it!\n')
                continue
            params = load_params_h5(run_subdirs[i], job_names[i])
            # Generate plots
            if glob.glob(f'{run_subdirs[i]}/*chi_ch*.h5'):
                # Add polarization-specific extra parameters
                n_intn = params['max_order'] - 1
                save_name = 'chi_ch_hub_2dsqlat'
                # Prefix for saved files
                params.update({'save_name': save_name, 'n_intn': n_intn})
                # Plot susceptibilities
                plot_chi_n_ch(
                    params, run_subdirs[i], job_names[i], logfiles[i])
                plot_static_chi_ch_together(
                    params, run_subdirs[i], job_names[i], logfiles[i], plot_rpa)
            if glob.glob(f'{run_subdirs[i]}/*self_en*.h5'):
                # Add self-energy-specific extra parameters
                n_intn = params['max_order']
                save_name = 'self_en_dyn_hub_2dsqlat'
                # Prefix for saved files
                params.update({'save_name': save_name, 'n_intn': n_intn})
                # Plot self energies
                plot_self_en(
                    params, run_subdirs[i], job_names[i], logfiles[i])
    elif len(sys.argv[1:]) == 1:
        if sys.argv[1] == 'latest':
            all_logfiles = glob.glob('*/*run*.log')
            if not all_logfiles:
                raise ValueError(
                    'No logfiles found in the working directory!')
            # Get a list of job IDs corresponding to
            # all logfiles in the working directory
            filenames = [os.path.splitext(lf)[0] for lf in all_logfiles]
            job_ids = [int(fname.rsplit('run_')[-1]) for fname in filenames]
            # Choose the most recent logfile, i.e., the one with maximum job ID
            # latest_job_id = np.max(job_ids)
            logfile = all_logfiles[np.argmax(job_ids)]
            job_name = f'run_{str(np.max(job_ids))}'
            run_subdir = os.path.dirname(logfile)
            print('\nSelected the following (most recent)' +
                  f" run subdirectory:\n'{run_subdir}'")
            print(f"\nUsing logfile:\n'{logfile}'")
            # Ignore this subdir if there is no complete run data in it
            if not glob.glob(f'{run_subdir}/*run*.h5'):
                raise ValueError(
                    'No run data found in the working directory!')
            # Get parameters
            params = load_params_h5(run_subdir, job_name)
            # Generate plots
            if glob.glob(f'{run_subdir}/*chi_ch*.h5'):
                # Add polarization-specific extra parameters
                n_intn = params['max_order'] - 1
                save_name = 'chi_ch_hub_2dsqlat'
                # Prefix for saved files
                params.update({'save_name': save_name, 'n_intn': n_intn})
                # Plot susceptibilities
                plot_chi_n_ch(params, run_subdir, job_name, logfile)
                plot_static_chi_ch_together(
                    params, run_subdir, job_name, logfile, plot_rpa)
            if glob.glob(f'{run_subdir}/*self_en*.h5'):
                # Add self-energy-specific extra parameters
                n_intn = params['max_order']
                save_name = 'self_en_dyn_hub_2dsqlat'
                # Prefix for saved files
                params.update({'save_name': save_name, 'n_intn': n_intn})
                # Plot self energies
                plot_self_en(
                    params, run_subdir, job_name, logfile)

        else:
            # Unescape special chars in dirname and interpret as string literal
            run_subdir = (sys.argv[1]).replace('\\', '')
            logfile = glob.glob(f'{run_subdir}/*run*.log')
            if not logfile:
                raise ValueError('No logfile found in the run directory!')
            logfile = logfile[0]
            # Ignore this subdir if there is no complete run data in it
            if len(glob.glob(f'{run_subdir}/*run*.h5')) == 0:
                raise ValueError('No run data found in the working directory!')
            filename = os.path.splitext(logfile)[0]
            job_id = int(filename.rsplit('run_')[-1])
            job_name = f'run_{job_id}'
            print(
                f"Manually selected the following run subdirectory:\n'{run_subdir}'")
            print(f"\nUsing logfile:\n'{logfile}'")
            # Get parameters
            params = load_params_h5(run_subdir, job_name)
            # Generate plots
            if len(glob.glob(f'{run_subdir}/*chi_ch*.h5')) > 0:
                # Add polarization-specific extra parameters
                n_intn = params['max_order'] - 1
                save_name = 'chi_ch_hub_2dsqlat'
                # Prefix for saved files
                params.update({'save_name': save_name, 'n_intn': n_intn})
                # Plot susceptibilities
                plot_chi_n_ch(params, run_subdir, job_name, logfile)
                plot_static_chi_ch_together(
                    params, run_subdir, job_name, logfile, plot_rpa)
            if len(glob.glob(f'{run_subdir}/*self_en*.h5')) > 0:
                # Add self-energy-specific extra parameters
                n_intn = params['max_order']
                save_name = 'self_en_dyn_hub_2dsqlat'
                # Prefix for saved files
                params.update({'save_name': save_name, 'n_intn': n_intn})
                # Plot self energies
                plot_self_en(
                    params, run_subdir, job_name, logfile)
        # Finally, print the run parameters
        print("\nRun parameters:")
        for item in params.items():
            print(item)
    else:
        raise ValueError("\nPlease supply a single run directory!\n" +
                         "Usage: 'python aggregate_and_plot_chi_ch.py run_dir'" +
                         "\nIf no directory is specified, the most recent run " +
                         "in the working directory will be used, if it exists." +
                         "\nIf keyword 'all' is specified, plots for all run " +
                         "subdirectories will be generated.")
    return


# End of main()
if __name__ == '__main__':
    main()
