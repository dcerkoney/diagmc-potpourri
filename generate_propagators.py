#!/usr/bin/env python3

# Package imports
import sys
import h5py
import pathlib
import optparse
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.special import gamma

# Local script imports
from lattice_tools import (
    LatticeDensity,
    test_distance_roundoff,
    safe_filename,
    lat_epsilon_k,
    fill_band,
    get_lat_g0_r_tau,
    get_pi0_q4_from_g0_r_tau_fft,
    get_pi0_q4_path_from_g0_r_tau_quad,
)


def approx_eq_float(a, b, abs_tol=2*sys.float_info.epsilon):
    """Define approximate float equality using an absolute tolerance;
       for direct fp subtraction, the error upper bound is 2 epsilon,
       but this can increase with additional operations, e.g. a =? b*c."""
    return abs(a - b) <= abs_tol


def to_timestamp(dt, rtype=int, epoch=datetime(1970, 1, 1)):
    '''Converts a datetime object into a POSIX timestamp (either of (1) integer
       type to millisecond precision, or (2) float type, to microsecond precision).'''
    if rtype not in [int, float]:
        raise ValueError(
            'Return type must be either integer or (double-precision) float!')
    td = dt - epoch
    ts = td.total_seconds()
    if rtype == int:
        return int(round(ts))
    return ts


def main():
    """Get the lattice Green's function and polarization bubble to Hartree level for Hubbard-type theories."""
    usage = """usage: %prog [ options ]"""
    parser = optparse.OptionParser(usage)
    parser.add_option("--dim", type="int",    default=2,
                      help="Spatial dimension of the electron gas (default is 2); allowed values: {2, 3}.")
    parser.add_option("--target_n0", type="float",  default=None,
                      help="Target density in units of the lattice constant; since the number of electrons "
                      + "is coarse-grained, the actual density may differ slightly. Default (n0 = 1) "
                      + "corresponds to half-filling (mu = 0).")
    parser.add_option("--target_mu0", type="float",  default=None,
                      help="Target (noninteracting) chemical potential. If supplied, we work at "
                      + "fixed chemical potential and variable density; otherwise, we use a "
                      + "fixed density and variable chemical potential.")
    parser.add_option("--t_hop", type="float", default=1.0,
                      help="The tight-binding hopping parameter t.")
    parser.add_option("--U_loc", type="float",  default=1.0,
                      help="Onsite Hubbard interaction in units of t.")
    parser.add_option("--beta", type="float",  default=1.0,
                      help="Inverse temperature in units of 1/t.")
    parser.add_option("--n_site_pd", type="int",    default=30,
                      help="Number of sites per direction.")
    parser.add_option("--lat_const", type="float",    default=1.0,
                      help="Lattice constant, in Bohr radii (for working at fixed "
                      + "'N' and 'a'; we will calculate 'V' on-the-fly).")
    parser.add_option("--lat_length", type="float",    default=None,
                      help="Lattice length, in Bohr radii (for working at "
                      + "fixed V; we will calculate 'a' on-the-fly).")
    parser.add_option("--n_tau",  type="int",   default=2**9,
                      help="Number of tau points in the nonuniform mesh "
                      + "used for downsampling (an even number).")
    parser.add_option("--n_nu",  type="int",   default=2**9,
                      help="Number of bosonic frequency points (an even number).")
    parser.add_option("--save_dir",   type="string", default="propagators",
                      help="Subdirectory to save results to, if applicable")
    parser.add_option("--save",   default=False,  action="store_true",
                      help="Save propagator data to h5?")
    parser.add_option("--overwrite",   default=False,  action="store_true",
                      help="Overwrite existing propagator data?")
    parser.add_option("--plot_g0",   default=False,  action="store_true",
                      help="Option for plotting the lattice Green's functions.")
    parser.add_option("--plot_pi0",   default=False,  action="store_true",
                      help="Option for plotting the polarization bubble P_0.")

    # Next, parse  the arguments and collect all options into a dictionary
    (options, _) = parser.parse_args()
    optdict = vars(options)

    # Get job start time for purposes of generating a UNIX timestamp ID
    starttime = datetime.utcnow()
    timestamp = to_timestamp(starttime, rtype=int)
    propr_dir = pathlib.PosixPath(f'proprs_{timestamp}')
    print(f'Timestamp: {timestamp} (UTC)\nJob started at: {starttime}\n')

    # Make directory in which to save propagator figs/data, if applicable and necessary
    if optdict['save']:
        save_dir = pathlib.PosixPath(optdict['save_dir']) / propr_dir
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = "."

    # Currently, we only implement the 2D and 3D cases
    dim = optdict['dim']
    if dim not in [2, 3]:
        raise ValueError(
            'd = '+str(optdict['dim'])+' is not a currently supported spatial dimension; d must be 2 or 3!')
    # The coordination number for hypercubic lattices is 2 * d (two nearest neighbors per Cartesian axis)
    z_coord = 2 * dim

    # Determine the lattice constant and length
    if (optdict['lat_const'] is None) and (optdict['lat_length'] is None):
        raise ValueError(
            'The user must supply either the lattice constant or lattice length.')
    # Get the lattice length if it was not supplied
    elif optdict['lat_length'] is None:
        lat_const = optdict['lat_const']
        lat_length = optdict['lat_const'] * optdict['n_site_pd']
    # Get the lattice constant if it was not supplied
    else:
        lat_length = optdict['lat_length']
        lat_const = optdict['lat_length'] / float(optdict['n_site_pd'])
        # Ensure that the derived and user-supplied lattice lengths are equal within floating-point precision
        assert approx_eq_float(
            lat_length, lat_const * optdict['n_site_pd'], abs_tol=4*sys.float_info.epsilon)
    # Make sure there is no roundoff error in the nearest-neighbor distance calculations
    test_distance_roundoff(n_site_pd=optdict['n_site_pd'], lat_const=lat_const)

    # Convert to Hartrees from energy units of t
    U_loc = optdict['U_loc'] / optdict['t_hop']
    beta = optdict['beta'] * optdict['t_hop']
    n_site_pd = optdict['n_site_pd']

    # Sigma[k, rho] = Sigma_H[rho] |_{U_loc}
    # (must be parametrized in terms of k, rho for general density getter)
    def sigma_hartree(k, rho):
        return (U_loc * rho / 2.0)

    # Determine the density and reduced chemical potential
    lat_dens_getter = LatticeDensity(dim=dim,
                                     beta=beta,
                                     t_hop=optdict['t_hop'],
                                     n_site_pd=n_site_pd,
                                     lat_const=lat_const,
                                     target_mu=optdict['target_mu0'],
                                     target_rho=optdict['target_n0'],
                                     sigma=sigma_hartree)
    mu = lat_dens_getter.mu
    n0 = lat_dens_getter.rho
    num_elec = lat_dens_getter.num_elec

    # (Hartree) reduced chemical potential with which we parametrize G_H
    mu_tilde = mu - sigma_hartree(_, n0)

    print('N_e: ', num_elec, '\nn_H: ', n0,
          '\nmu_H: ', mu, '\nmu_tilde_H: ', mu_tilde)

    # Volume of a 'dim'-dimensional ball
    def rad_d_ball(vol): return (vol * gamma(1.0 + dim / 2.0)
                                 )**(1.0 / float(dim)) / np.sqrt(np.pi)
    # Get the Wigner-Seitz radius
    rs = rad_d_ball(1.0 / n0)
    print('r_s(n_H): '+str(rs))
    # Get the Fermi momentum and one corresponding k-point by explicitly filling the band
    k_scale = 2.0 * np.pi / float(lat_length)
    # r_scale = lat_const
    ef, kf_vecs = fill_band(
        dim=dim,
        num_elec=num_elec,
        n_site_pd=n_site_pd,
        lat_const=lat_const,
        t_hop=optdict['t_hop'],
    )
    # print(ef, '\n', kf_vecs)
    # Numerical roundoff issues occur at half-filling, so manually round the Fermi energy to zero
    if optdict['target_n0'] == 1:
        # Double-check that we actually obtained a near-zero answer for ef
        assert np.allclose(ef, 0)
        # Then, shift it to the exact value at half-filling
        ef = 0.0

    # NOTE: the quasiparticle dispersion relation only shows up in G_0,
    #       so there is no qp rescale factor implemented here!
    kf_docstring = "{k_F} in the first quadrant: \n["
    for i, kf_vec in enumerate(kf_vecs):
        this_ek = lat_epsilon_k(
            k=kf_vec,
            lat_const=lat_const,
            t_hop=optdict['t_hop'],
            meshgrid=False,
        )
        assert np.allclose(ef, this_ek, rtol=1e-13)
        kf_docstring += "k"+str(np.round(kf_vec / k_scale).astype(int))
        if i < len(kf_vecs) - 1:
            kf_docstring += ",\t"
            if (i+1) % 4 == 0:
                kf_docstring += "\n "
        else:
            kf_docstring += "]"
    # kf_docstring += str(kf_vecs)
    print('{k_F} in the first quadrant: \n', kf_vecs,
          '\n', 'e_F =', ef, '\n', 'mu_n_{HF} = ', mu)

    # UV cutoff defined by the lattice (longest scattering length in the BZ)
    llambda_lat = dim * (np.pi / lat_const)**2 / 2.0

    # Build an ordered path of k-points in the Brillouin zone; we choose the high-symmetry
    # path for the simple square lattice (\Gamma - X - M - \Gamma), discarding duplicate
    # coordinates at the path vertices (accounted for in plotting step).
    N_edge = int(np.floor(n_site_pd / 2.0))
    nk_coords_Gamma_X = [[x, 0] for x in range(0, N_edge + 1)]
    nk_coords_X_M = [[N_edge, y] for y in range(1, N_edge + 1)]
    nk_coords_M_Gamma = [[xy, xy] for xy in range(1, N_edge)[::-1]]
    # Indices for the high-symmetry points
    idx_Gamma1 = 0
    idx_X = len(nk_coords_Gamma_X) - 1
    idx_M = len(nk_coords_Gamma_X) + len(nk_coords_X_M) - 1
    idx_Gamma2 = len(nk_coords_Gamma_X) + \
        len(nk_coords_X_M) + len(nk_coords_M_Gamma) - 1
    # Build the full ordered high-symmetry path
    path_nk_coords = np.concatenate(
        (nk_coords_Gamma_X, nk_coords_X_M, nk_coords_M_Gamma))
    path_k_coords = k_scale * path_nk_coords
    n_k_pts = len(path_k_coords)

    # Find the corresponding indices in the full k_list
    i_path = np.arange(len(path_k_coords))
    i_path_kf_locs = []
    for i, this_k_coord in enumerate(path_k_coords):
        for kf_vec in kf_vecs:
            if np.all(this_k_coord == kf_vec):
                i_path_kf_locs.append(i)
    i_path_kf_locs = np.asarray(i_path_kf_locs)

    print(r'High-symmetry path: \Gamma - X - M - \Gamma')
    print('Number of k-points in the path: ', n_k_pts)
    print('Fermi momentum indices along path:\n', i_path_kf_locs)
    print('Fermi momentum coordinates along path:\n',
          path_nk_coords[i_path_kf_locs])
    print('k'+str(path_nk_coords[len(i_path) // 3]) +
          ': '+str(path_k_coords[len(i_path) // 3]))
    assert len(np.unique(path_k_coords, axis=0)) == len(path_k_coords)

    # Cubic taulist; smallest time mesh point in G_0 calculation is \beta * \delta_\tau = 1e-10
    delta_tau = 1e-10 / beta
    tau_powlist_left = (np.linspace(delta_tau ** (1.0 / 3.0), 0.5 ** (1.0 / 3.0),
                                    num=optdict['n_tau'] // 2) ** 3)
    tau_powlist_right = (1.0 - tau_powlist_left[::-1])
    # tau_powlist = beta * \
    tau_list = beta * np.concatenate(
        ([0.0, delta_tau], tau_powlist_left[1:-1], [0.5], tau_powlist_right[1:-1], [1.0 - delta_tau, 1.0]))

    if optdict['save']:
        log_savename = safe_filename(
            dir=save_dir,
            savename=propr_dir,
            file_extension='log',
            overwrite=optdict['overwrite'],
        )
        param_file = open(log_savename, 'w+')
        param_printout = (
            '-----------------------------'+'\n'
            'Lattice G_0(r, tau) from FFT'+'\n'
            '-----------------------------'+'\n'
            f'Job ID# (POSIX integer timestamp): {timestamp}'
            '\n'
            f'Spatial dimension: d = {dim}\n'
            f"t (hopping parameter): {optdict['t_hop']}\n"
            f"beta / t (inverse temperature): {optdict['beta']}\n"
            f"U / t (onsite (Hubbard) interaction): {optdict['U_loc']}\n"
            f"Magnetization ((n_up - n_down) / n_tot): 0.0\n" +
            f'Lattice coordination number: z = {z_coord}\n'
            f"Bandwidth: w = 2zt = {2 * z_coord * optdict['t_hop']}\n"
            '\n'
            f'Number of sites per direction: N = {n_site_pd}\n'
            f'Total number of electrons: N_e = {num_elec}\n'
            f'Lattice constant: a = {lat_const}\n'
            f'Lattice length: L = {lat_length}\n'
            f'Lattice UV cutoff: Lambda_{{lat}} = {llambda_lat}\n'
            f'Imaginary-time infinitesimal (for G_0 hard cutoff and limits): {delta_tau * beta}\n'
            '\n'
            f'mu_tilde_H: {mu_tilde}\n'
            f'mu_H: {mu}\n'
            f'n_0 = n = {n0}\n'
            f'r_s: {rs}\n'
            f'e_F: {ef}\n'
            f'{kf_docstring}\n'
            '\n'
            f"Number of nonuniform tau mesh points: {optdict['n_tau']}\n" +
            f"Number of frequency mesh points: {optdict['n_nu']}\n" +
            '\n'
        )
        param_file.write(param_printout)
        param_file.close()

    #########################################################
    # First, get the (Hartree) lattice Green's function G_0 #
    #########################################################

    # Get G_0(r, tau) from 3D IFFT of G_0(k, tau)
    g0_r_tau_interp_mtx, g0_r_tau_ifft_mesh = get_lat_g0_r_tau(
        lat_const=lat_const,
        n_site_pd=n_site_pd,
        t_hop=optdict['t_hop'],
        taulist=tau_list,
        dim=dim,
        beta=beta,
        mu=mu_tilde,
        n0=n0,
        delta_tau=delta_tau,
        plots=optdict['plot_g0'],
        overwrite=optdict['overwrite'],
        save_dir=save_dir,
    )

    tau_dense_unif = beta * \
        np.arange(optdict['n_nu'] + 1) / float(optdict['n_nu'])
    n_tau_unif = len(tau_dense_unif[:-1])

    # Get G_0 on the uniform dense tau mesh for FFT without
    # upsampling, which we will use to obtain \Pi_0 and W_0
    _, g0_r_tau_ifft_dense_mesh = get_lat_g0_r_tau(
        lat_const=lat_const,
        n_site_pd=n_site_pd,
        t_hop=optdict['t_hop'],
        taulist=tau_dense_unif,
        dim=dim,
        beta=beta,
        mu=mu_tilde,
        n0=n0,
        delta_tau=delta_tau,
        # plots=optdict['plot_g0'],
        plots=False,
        overwrite=optdict['overwrite'],
        save_dir=save_dir,
    )

    # Defines the maximum possible index value in the reduced
    # difference vector mesh, max(nr_1 - nr_2) = (N // 2) + 1
    r_red_cut = (n_site_pd // 2) + 1
    # Get G_0 over the irreducible set of lattice
    # distance vectors, i.e., the first orthant of L;
    # also, remove the tau = beta point used for plotting
    r_red_slice = dim * (slice(r_red_cut),) + (slice(-1),)

    if optdict['save']:
        # Save the lattice G_0(r, tau) data to h5
        g0_h5file = save_dir / 'lat_g0_rt.h5'
        # Optionally avoid overwriting duplicate filenames
        if not optdict['overwrite']:
            dup_count = 1
            while g0_h5file.is_file():
                g0_h5file = save_dir / f'lat_g0_rt({dup_count}).h5'
                dup_count += 1
        # Open H5 file and write attributes/data to it
        h5file = h5py.File(g0_h5file, 'w')
        h5file.attrs['n0'] = n0
        h5file.attrs['mu'] = mu
        h5file.attrs['mu_tilde'] = mu_tilde
        h5file.attrs['lat_const'] = lat_const
        h5file.attrs['n_site_pd'] = n_site_pd
        h5file.attrs['dim'] = optdict['dim']
        h5file.attrs['beta'] = optdict['beta']
        h5file.attrs['t_hop'] = optdict['t_hop']
        h5file.attrs['U_loc'] = optdict['U_loc']
        h5file.attrs['n_tau'] = optdict['n_tau']

        g0_r_tau_irred_mesh = g0_r_tau_ifft_mesh[r_red_slice]
        g0_irred_1d = g0_r_tau_irred_mesh.flatten(order='C')

        dataset_g0 = h5file.create_dataset('lat_g0_rt_data', data=g0_irred_1d)
        dataset_g0.attrs['shape'] = g0_r_tau_ifft_mesh[r_red_slice].shape

        # Save tau on [0, beta)
        assert(len(tau_list[:-1]) == optdict['n_tau'])
        dataset_tau = h5file.create_dataset('tau_mesh', data=tau_list[:-1])
        # dataset_tau.attrs['n_tau'] = optdict['n_tau']

        # Write to disk
        h5file.close()

    ####################################################################################
    # Then, get the lattice polarization bubble \Pi_0 and RPA screened interaction W_0 #
    ####################################################################################

    # Now, get \Pi_0 on the full dense tau mesh
    pi0_q4_dense, _ = get_pi0_q4_from_g0_r_tau_fft(
        g0_r_tau=g0_r_tau_ifft_dense_mesh,
        n_nu=optdict['n_nu'],
        dim=dim,
        beta=beta,
        delta_tau=delta_tau,
        n_site_pd=n_site_pd,
        lat_const=lat_const,
    )

    if optdict['plot_pi0']:
        # Get \Pi_0 via upsampling from the non-uniform tau mesh
        pi0_q4_upsampled, _ = get_pi0_q4_from_g0_r_tau_fft(
            g0_r_tau=g0_r_tau_interp_mtx,
            n_nu=optdict['n_nu'],
            dim=dim,
            beta=beta,
            delta_tau=delta_tau,
            n_site_pd=n_site_pd,
            lat_const=lat_const,
        )
        mlist_plot = np.arange(2)
        # Inclusive endpoints for integration (and to ensure odd number of time points)
        r_red_slice_incl = dim * (slice(r_red_cut),) + (slice(None),)
        # Get the polarization bubble along the k-path via quadrature integration in \tau; while
        # this approach would be very inefficient to calculate \Pi_0 on the entire Brillouin
        # zone, we use it as a benchmark for the FFT methods along the high-symmetry path
        pi0_sigma_quad_path, _ = get_pi0_q4_path_from_g0_r_tau_quad(
            g0_r_tau_ifft_red_mesh=g0_r_tau_ifft_mesh[r_red_slice_incl],
            path_q_coords=path_k_coords,
            inu_list=(2j * np.pi / beta) * mlist_plot.astype(float),
            tau_list=tau_list,
            beta=beta,
            delta_tau=delta_tau,
            dim=dim,
            n_site_pd=n_site_pd,
            lat_const=lat_const,
            # verbose=True,
        )
        for this_m in mlist_plot:
            # Get the polarization bubbles along the k-path
            pi0_om_dense_path = np.zeros(n_k_pts)
            pi0_om_upsampled_path = np.zeros(n_k_pts)
            for iq, this_nq_coord in enumerate(path_nk_coords):
                pi0_om_dense_path[iq] = pi0_q4_dense[tuple(
                    this_nq_coord) + (this_m,)]
                pi0_om_upsampled_path[iq] = pi0_q4_upsampled[tuple(
                    this_nq_coord) + (this_m,)]
            # Check that they are all essentially equal
            if not np.allclose(pi0_om_dense_path, pi0_om_upsampled_path):
                print('Maximal absolute error between dense and upsampled FFT results along the k-path:\n',
                      np.max(np.abs(pi0_om_dense_path - pi0_om_upsampled_path)))
            if not np.allclose(pi0_om_dense_path, 2 * pi0_sigma_quad_path[..., this_m]):
                print('Maximal absolute error between (dense) FFT and quad results along the k-path:\n',
                      np.max(np.abs(pi0_om_dense_path - 2 * pi0_sigma_quad_path[..., this_m])))
            if not np.allclose(pi0_om_dense_path, 2 * pi0_sigma_quad_path[..., this_m]):
                print('Maximal absolute error between upsampled FFT and quad results along the k-path:\n',
                      np.max(np.abs(pi0_om_upsampled_path - 2 * pi0_sigma_quad_path[..., this_m])))
                # fig, ax = plt.subplots()
                # ax.plot(np.abs(pi0_om_upsampled_path - 2 *
                #                pi0_sigma_quad_path[..., this_m]))
                # savename = safe_filename(
                #     dir=save_dir,
                #     savename=f'abs_err_quad_upsampled_m={this_m}',
                #     file_extension='pdf',
                #     overwrite=optdict['overwrite'],
                # )
                # ax.set_ylabel('Absolute error')
                # ax.grid(True)
                # fig.savefig(savename)

            # Plot the results
            fig, ax = plt.subplots()
            i_path = range(len(path_k_coords))
            # print(len(i_path))
            ax.axvline(x=i_path_kf_locs[0], linestyle='-', color='0.0',
                       zorder=-1, linewidth=1, label=r'$\mathbf{k}_F$')
            for i_path_kf_loc in i_path_kf_locs[1:]:
                ax.axvline(x=i_path_kf_loc, linestyle='-',
                           color='0.0', zorder=-1, linewidth=1)

            ax.plot(i_path, 2 * pi0_sigma_quad_path[..., this_m], color='k', alpha=1,
                    label=r'$m = {}$ (FT via quadrature on dense cubic interpolant)'.format(this_m))
            # ax.plot(i_path, pi0_om_dense_path, '.-', color='b', alpha=1,
            ax.plot(i_path, pi0_om_dense_path, '-', color='b', alpha=1,
                    label=r'$m = {}$ (FFT on full uniform dense mesh)'.format(this_m))
            # ax.plot(i_path, pi0_om_upsampled_path, '.-', color='r', alpha=1,
            ax.plot(i_path, pi0_om_upsampled_path, '-', color='r', alpha=1,
                    label=r'$m = {}$ (FFT using sparse nonuniform upsampling)'.format(this_m))
            ax.legend(loc='best')
            # Add some evenly-spaced minor ticks to the axis
            n_minor_ticks = 9
            # assert (len(i_path)) % 9 == 0
            minor_ticks = np.arange(
                0, len(i_path), len(i_path) / n_minor_ticks)
            # print(minor_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            # Label the high-symmetry points
            ax.set_xticks((idx_Gamma1, idx_X, idx_M, idx_Gamma2))
            ax.set_xticklabels((r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$'))
            ax.set_ylabel(r'$\Pi_0(\mathbf{{q}}, i\nu_m)$')
            ax.set_title(r'$\Pi_0(\mathbf{{q}}, i\nu_m), n_{\tau} = $' + str(
                optdict['n_tau']) + r', $n_{\nu} = $'+str(optdict['n_nu']))
            # Set the same plot range and ticks as the Kozik paper for easy visual correspondence
            ax.set_xlim(left=i_path[0], right=i_path[-1])
            # ax.set_ylim(bottom=-.65, top=-.35)
            ax.grid(True, color='k', linestyle=':', which='minor')
            ax.grid(True, color='k', linestyle=':', which='major')
            fig.tight_layout()
            savename = safe_filename(
                dir=save_dir,
                savename=(f"lat_pi0_q_inu{this_m}_N={n_site_pd}_beta={beta:g}" +
                          f"_n_tau={optdict['n_tau']}_n_nu={optdict['n_nu']}" +
                          f"_fft_upsampled"),
                file_extension='pdf',
                overwrite=optdict['overwrite'],
            )
            fig.savefig(savename)

    if optdict['save']:
        # Save the lattice Pi_0(q, i nu) data to h5
        pi0_h5file = save_dir / 'lat_pi0_q4.h5'
        # Optionally avoid overwriting duplicate filenames
        if not optdict['overwrite']:
            dup_count = 1
            while pi0_h5file.is_file():
                pi0_h5file = save_dir / f'lat_pi0_q4({dup_count}).h5'
                dup_count += 1
        # Open H5 file and write attributes/data to it
        h5file = h5py.File(pi0_h5file, 'w')
        h5file.attrs['n0'] = n0
        h5file.attrs['mu'] = mu
        h5file.attrs['mu_tilde'] = mu_tilde
        h5file.attrs['lat_const'] = lat_const
        h5file.attrs['n_site_pd'] = n_site_pd
        h5file.attrs['n_tau_unif'] = n_tau_unif
        h5file.attrs['n_nu'] = optdict['n_nu']
        h5file.attrs['dim'] = optdict['dim']
        h5file.attrs['beta'] = optdict['beta']
        h5file.attrs['t_hop'] = optdict['t_hop']
        h5file.attrs['U_loc'] = optdict['U_loc']

        k_red_cut = (n_site_pd // 2) + 1
        k_red_slice = dim * (slice(k_red_cut),) + (slice(optdict['n_nu']),)

        pi0_q4_irred_mesh = pi0_q4_dense[k_red_slice]
        pi0_irred_1d = pi0_q4_irred_mesh.flatten(order='C')

        dataset_pi0 = h5file.create_dataset(
            'lat_pi0_q4_data', data=pi0_irred_1d)
        dataset_pi0.attrs['shape'] = pi0_q4_dense[k_red_slice].shape

        dataset_tau = h5file.create_dataset(
            'tau_unif_mesh', data=tau_dense_unif[:-1])
        # dataset_tau.attrs['n_tau_unif'] = n_tau_unif

        dataset_nu = h5file.create_dataset(
            'nu_unif_mesh', data=list(range(optdict['n_nu'])))
        # dataset_nu.attrs['n_nu'] = optdict['n_nu']

        # Write to disk
        h5file.close()

    return


# End of main()
if __name__ == '__main__':
    main()
