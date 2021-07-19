#!/usr/bin/env python3

# Package imports
import sys
import glob
import json
import h5py
import optparse
import numpy as np
import matplotlib.pyplot as plt

from ruamel.yaml import YAML
from pathlib import Path, PosixPath
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

    # Save / plot options
    parser.add_option("--save_dir",   type="string", default="propagators",
                      help="Subdirectory to save results to, if applicable")
    parser.add_option("--plot_g0",   default=False,  action="store_true",
                      help="Option for plotting the lattice Green's function.")
    parser.add_option("--plot_pi0",   default=False,  action="store_true",
                      help="Option for plotting the polarization bubble P_0.")
    parser.add_option("--dry_run",   default=False,  action="store_true",
                      help="Perform a dry run, i.e., don't save any propagator data/plots.")
    
    # Optional flags for overriding config file params
    parser.add_option("--n_tau",  type="int",   default=None,
                      help="Number of tau points in the nonuniform mesh "
                      + "used for downsampling (an even number).")
    parser.add_option("--n_nu",  type="int",   default=None,
                      help="Number of bosonic frequency points in "
                      + "the uniform FFT mesh (an even number).")
    parser.add_option("--target_mu", type="float",  default=None,
                      help="Target (noninteracting) chemical potential. If supplied, we work at "
                      + "fixed chemical potential and variable density; otherwise, we use a "
                      + "fixed density and variable chemical potential.")
    parser.add_option("--target_n0", type="float",  default=None,
                      help="Target density in units of the lattice constant; since the number of electrons "
                      + "is coarse-grained, the actual density may differ slightly. Default (n0 = 1) "
                      + "corresponds to half-filling (mu0 ~= 0).")
    parser.add_option("--dim", type="int",    default=2,
                      help="Spatial dimension of the lattice (default is 2); allowed values: {2, 3}.")
    parser.add_option("--t_hop", type="float", default=None,
                      help="The tight-binding hopping parameter t.")
    parser.add_option("--U_loc", type="float",  default=None,
                      help="Onsite Hubbard interaction in Hartrees.")
    parser.add_option("--beta", type="float",  default=None,
                      help="Inverse temperature in inverse Hartrees.")
    parser.add_option("--n_site_pd", type="int",    default=None,
                      help="Number of sites per direction.")
    parser.add_option("--lat_const", type="float",    default=None,
                      help="Lattice constant in Bohr radii.")
    parser.add_option("--lat_length", type="float",    default=None,
                      help="Lattice length in Bohr radii (for working at "
                      + "fixed V: calculate 'a' on-the-fly).")

    # Next, parse  the arguments and collect all options into a dictionary
    (options, _) = parser.parse_args()
    optdict = vars(options)

    # Parse YAML config file
    yaml = YAML(typ='rt')
    yaml.default_flow_style = False
    cfg = yaml.load(PosixPath("config.yml"))

    # Update config with user-supplied cmdline args where applicable
    cfg_override_string = ''
    for k, v in optdict.items():
        for group in cfg:
            try:
                if (k in cfg[group].keys()) and (v is not None) and (v != cfg[group][k]):
                    cfg_override_string += "\nNOTE: Overriding config file setting" + \
                        f" '{k} = {cfg[group][k]}' with cmdline value: {v}"
                    cfg[group][k] = v
            except AttributeError:
                pass
    if cfg_override_string:
        cfg_override_string += '\n'

    # Reference param groups directly for brevity (cfg is still modified)
    p_phys = cfg['phys_params']
    p_propr = cfg['propr_params']

    # Get job start time for purposes of generating a UNIX timestamp ID
    starttime = datetime.utcnow()
    job_id = to_timestamp(starttime, rtype=int)
    p_propr['propr_job_id'] = job_id
    print(f'\nTimestamp: {job_id} (UTC)\nJob started at: {starttime}')

    # Currently, we only implement the 2D and 3D cases
    if p_phys['dim'] not in [2, 3]:
        raise ValueError(
            'd = '+str(p_phys['dim'])+' is not a currently supported spatial dimension; d must be 2 or 3!')
    # The coordination number for hypercubic lattices is 2 * d (two nearest neighbors per Cartesian axis)
    # z_coord = 2 * dim

    # Determine the lattice constant and length
    if (p_phys['lat_const'] is None) and (p_phys['lat_length'] is None):
        raise ValueError(
            'The user must supply either the lattice constant or lattice length.')
    # Derive the lattice length if a lattice constant was supplied
    elif p_phys['lat_const'] is not None:
        p_phys['lat_length'] = p_phys['lat_const'] * p_phys['n_site_pd']
    # Get the lattice constant if it was not supplied
    else:
        p_phys['lat_const'] = p_phys['lat_length'] / float(p_phys['n_site_pd'])
        # Ensure that the derived and user-supplied lattice lengths are equal within floating-point precision
        assert approx_eq_float(
            p_phys['lat_length'], p_phys['lat_const'] * p_phys['n_site_pd'], abs_tol=4*sys.float_info.epsilon)
    # Make sure there is no roundoff error in the nearest-neighbor distance calculations
    test_distance_roundoff(
        n_site_pd=p_phys['n_site_pd'], lat_const=p_phys['lat_const'])

    print(p_phys['lat_length'], p_phys['n_site_pd'], p_phys['lat_const'])

    # Sigma[k, rho] = Sigma_H[rho] |_{U_loc}
    # (must be parametrized in terms of k, rho for general density getter)
    def sigma_hartree(k, rho):
        return (p_phys['U_loc'] * rho / 2.0)

    # Determine the density and reduced chemical potential
    lat_dens_getter = LatticeDensity(dim=p_phys['dim'],
                                     beta=p_phys['beta'],
                                     t_hop=p_phys['t_hop'],
                                     n_site_pd=p_phys['n_site_pd'],
                                     lat_const=p_phys['lat_const'],
                                     target_mu=p_phys['target_mu'],
                                     target_rho=p_phys['target_n0'],
                                     sigma=sigma_hartree, verbose=True)

    # Update physical params   
    p_phys['vol_lat'] = p_phys['lat_length'] ** p_phys['dim']
    p_phys['n_site'] = p_phys['n_site_pd'] ** p_phys['dim']
    p_phys['num_elec'] = lat_dens_getter.num_elec
    p_phys['n0'] = lat_dens_getter.rho
    p_phys['mu'] = lat_dens_getter.mu

    # (Hartree) reduced chemical potential with which we parametrize G_H
    p_phys['mu_tilde'] = p_phys['mu'] - sigma_hartree(_, p_phys['n0'])

    print('\nN_e: ', p_phys['num_elec'], '\nn_H: ', p_phys['n0'], '\nV: ', p_phys['vol_lat'],
          '\nmu_H: ', p_phys['mu'], '\nmu_tilde_H: ', p_phys['mu_tilde'])

    # Volume of a 'dim'-dimensional ball
    def rad_d_ball(vol): return (vol * gamma(1.0 + p_phys['dim'] / 2.0)
                                 )**(1.0 / float(p_phys['dim'])) / np.sqrt(np.pi)
    # Get the Wigner-Seitz radius
    p_phys['rs'] = float(rad_d_ball(1.0 / p_phys['n0']))
    print('r_s(n_H): '+str(p_phys['rs']))

    # Get the Fermi momentum and one corresponding k-point by explicitly filling the band
    k_scale = 2.0 * np.pi / float(p_phys['lat_length'])
    # r_scale = lat_const
    ef, kf_vecs = fill_band(
        dim=p_phys['dim'],
        num_elec=p_phys['num_elec'],
        n_site_pd=p_phys['n_site_pd'],
        lat_const=p_phys['lat_const'],
        t_hop=p_phys['t_hop'],
    )
    p_phys['ef'] = float(ef)
    print(p_phys['ef'], '\n', kf_vecs)

    # Numerical roundoff issues occur at half-filling, so manually round the Fermi energy to zero
    if p_phys['target_n0'] == 1:
        # Double-check that we actually obtained a near-zero answer for ef
        assert np.allclose(p_phys['ef'], 0)
        print(p_phys['ef'])
        # Then, shift it to the exact value at half-filling
        p_phys['ef'] = 0.0

    # NOTE: the quasiparticle dispersion relation only shows up in G_0,
    #       so there is no qp rescale factor implemented here!
    kf_docstring = "\n{k_F} in the first quadrant: \n["
    for i, kf_vec in enumerate(kf_vecs):
        this_ek = lat_epsilon_k(
            k=kf_vec,
            lat_const=p_phys['lat_const'],
            t_hop=p_phys['t_hop'],
            meshgrid=False,
        )
        assert np.allclose(p_phys['ef'], this_ek, rtol=1e-13)
        kf_docstring += "k"+str(np.round(kf_vec / k_scale).astype(int))
        if i < len(kf_vecs) - 1:
            kf_docstring += ",  "
            if (i+1) % 4 == 0:
                kf_docstring += "\n "
        else:
            kf_docstring += "]"
    print(kf_docstring)
    # print(f"\n{{k_F}} in the first quadrant: \n{kf_vecs}\ne_F = {p_phys['ef']}\nmu_n_{{HF}} = {p_phys['mu']}")

    # UV cutoff defined by the lattice (longest scattering length in the BZ)
    # llambda_lat = p_phys['dim'] * (np.pi / p_phys['lat_const'])**2 / 2.0

    # Build an ordered path of k-points in the Brillouin zone; we choose the high-symmetry
    # path for the simple square lattice (\Gamma - X - M - \Gamma), discarding duplicate
    # coordinates at the path vertices (accounted for in plotting step).
    N_edge = int(np.floor(p_phys['n_site_pd'] / 2.0))
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

    print('\nHigh-symmetry path: G-X-M-G')
    print('Number of k-points in the path: ', n_k_pts)
    if len(i_path_kf_locs) > 0:
        print('Fermi momentum indices along path:\n', i_path_kf_locs)
        print('Fermi momentum coordinates along path:\n',
              path_nk_coords[i_path_kf_locs])
    print('k'+str(path_nk_coords[len(i_path) // 3]) +
          ': '+str(path_k_coords[len(i_path) // 3]))
    assert len(np.unique(path_k_coords, axis=0)) == len(path_k_coords)

    # Defines the maximum possible index value in the reduced
    # difference vector mesh, max(nr_1 - nr_2) = (N // 2) + 1
    p_phys['n_site_irred'] = (p_phys['n_site_pd'] // 2) + 1

    # Use a cubic tau mesh; the smallest time mesh point is delta_tau (in units of beta)
    tau_powlist_left = p_phys['beta'] * (np.linspace(p_propr['delta_tau'] / p_phys['beta'] ** (1.0 / 3.0),
                                                     0.5 ** (1.0 / 3.0), num=p_propr['n_tau'] // 2) ** 3)
    tau_powlist_right = p_phys['beta'] - tau_powlist_left[::-1]
    tau_list = np.concatenate(([0.0, p_propr['delta_tau']], tau_powlist_left[1:-1], [0.5 * p_phys['beta']],
                               tau_powlist_right[1:-1], [p_phys['beta'] - p_propr['delta_tau'], p_phys['beta']]))
    assert(len(tau_list[:-1]) == p_propr['n_tau'])

    # Check for any existing propagator data in save_dir matching
    # the current parameters; if none exists, generate them
    consistent_proprs_exist = False
    consistent_propr_path = None
    propr_paths = glob.glob(f"./{optdict['save_dir']}/*/lat_g0_rt.h5")
    for propr_path in propr_paths:
        g0_path = PosixPath(propr_path)
        g0_data = h5py.File(g0_path, 'r')
        # Check for consistency of relevant attributes
        try:
            params_consistent = (p_phys['n0'] == g0_data.attrs['n0']
                                 and p_phys['mu'] == g0_data.attrs['mu']
                                 and p_phys['mu_tilde'] == g0_data.attrs['mu_tilde']
                                 and p_phys['lat_const'] == g0_data.attrs['lat_const']
                                 and p_phys['n_site_pd'] == g0_data.attrs['n_site_pd']
                                 and p_phys['dim'] == g0_data.attrs['dim']
                                 and p_phys['beta'] == g0_data.attrs['beta']
                                 and p_phys['t_hop'] == g0_data.attrs['t_hop']
                                 and p_phys['U_loc'] == g0_data.attrs['U_loc']
                                 and p_propr['n_tau'] == g0_data.attrs['n_tau'])
        except AttributeError:
            continue
        if params_consistent:
            consistent_propr_path = g0_path
            consistent_proprs_exist = True
            break
    if consistent_proprs_exist:
        print(f'\nFound consistent propagator data in subdirectory {consistent_propr_path},'
              + ' updating paths in cfg...', end='')
        consistent_save_dir = consistent_propr_path.parent
        consistent_job_id = consistent_save_dir.name.split('proprs_')[-1]
        p_propr['propr_save_dir'] = str(consistent_save_dir)
        p_propr['propr_job_id'] = str(consistent_job_id)
        print('done!')
    else:
        print('\nNo consistent propagator data found for the supplied config, generating it now...')

    if not consistent_proprs_exist:
        # Make directory in which to save new propagator figs/data (if applicable and necessary)
        if not optdict['dry_run']:
            propr_dir = PosixPath(f'proprs_{job_id}')
            save_dir = PosixPath(optdict['save_dir']) / propr_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            p_propr['propr_save_dir'] = save_dir
        else:
            save_dir = "."
            p_propr['propr_save_dir'] = "N/A"

    # Summarize any config settings overriden by cmdline flags, and print the updated config
    cfg_update_string = ('Proposed config settings' if optdict['dry_run']
                         else 'Updated config settings')
    print(f'{cfg_override_string}\n{cfg_update_string}:')
    json.dump(cfg, sys.stdout, indent=4, sort_keys=True)
    print('\n')

    if not optdict['dry_run']:
        # Save updated config back to file
        yaml.dump(cfg, PosixPath("config.yml"))

        # Dump YAML config to JSON for use with the C++ MCMC driver
        with open("config.json", 'w') as f:
            json.dump(cfg, f, indent=4, sort_keys=True)


    # Nothing more to do after updating configs in this case
    if consistent_proprs_exist: 
        return

    #########################################################
    # First, get the (Hartree) lattice Green's function G_0 #
    #########################################################

    # Get G_0(r, tau) from 3D IFFT of G_0(k, tau)
    g0_r_tau_interp_mtx, g0_r_tau_ifft_mesh = get_lat_g0_r_tau(
        lat_const=p_phys['lat_const'],
        n_site_pd=p_phys['n_site_pd'],
        t_hop=p_phys['t_hop'],
        taulist=tau_list,
        dim=p_phys['dim'],
        beta=p_phys['beta'],
        mu=p_phys['mu_tilde'],
        n0=p_phys['n0'],
        delta_tau=p_propr['delta_tau'],
        save_dir=save_dir,
        plots=optdict['plot_g0'],
    )

    tau_dense_unif = p_phys['beta'] * np.arange(p_propr['n_nu'] + 1) / float(p_propr['n_nu'])
    assert(len(tau_dense_unif[:-1]) == p_propr['n_nu'])

    # Get G_0 on the uniform dense tau mesh for FFT without
    # upsampling, which we will use to obtain \Pi_0 and W_0
    _, g0_r_tau_ifft_dense_mesh = get_lat_g0_r_tau(
        lat_const=p_phys['lat_const'],
        n_site_pd=p_phys['n_site_pd'],
        t_hop=p_phys['t_hop'],
        taulist=tau_dense_unif,
        dim=p_phys['dim'],
        beta=p_phys['beta'],
        mu=p_phys['mu_tilde'],
        n0=p_phys['n0'],
        delta_tau=p_propr['delta_tau'],
        save_dir=save_dir,
        plots=False,
    )

    if not optdict['dry_run']:
        # Save the lattice G_0(r, tau) data to h5
        g0_h5file = save_dir / f'lat_g0_rt_{job_id}.h5'
        # Open H5 file and write attributes/data to it
        h5file = h5py.File(g0_h5file, 'w')
        h5file.attrs['n0'] = p_phys['n0']
        h5file.attrs['mu'] = p_phys['mu']
        h5file.attrs['mu_tilde'] = p_phys['mu_tilde']
        h5file.attrs['lat_const'] = p_phys['lat_const']
        h5file.attrs['n_site_pd'] = p_phys['n_site_pd']
        h5file.attrs['dim'] = p_phys['dim']
        h5file.attrs['beta'] = p_phys['beta']
        h5file.attrs['t_hop'] = p_phys['t_hop']
        h5file.attrs['U_loc'] = p_phys['U_loc']
        h5file.attrs['n_tau'] = p_propr['n_tau']

        # Get G_0 over the irreducible set of lattice
        # distance vectors, i.e., the first orthant of L;
        # also, remove the tau = beta point used for plotting
        r_red_slice = p_phys['dim'] * \
            (slice(p_phys['n_site_irred']),) + (slice(-1),)
        g0_r_tau_irred_mesh = g0_r_tau_ifft_mesh[r_red_slice]
        g0_irred_1d = g0_r_tau_irred_mesh.flatten(order='C')
        dataset_g0 = h5file.create_dataset('lat_g0_rt_data', data=g0_irred_1d)
        dataset_g0.attrs['shape'] = g0_r_tau_ifft_mesh[r_red_slice].shape
        
        # Save tau on [0, beta)
        h5file.create_dataset('tau_mesh', data=tau_list[:-1])
        # Write to disk
        h5file.close()

    ####################################################################################
    # Then, get the lattice polarization bubble \Pi_0 and RPA screened interaction W_0 #
    ####################################################################################

    # Now, get \Pi_0 on the full dense tau mesh
    pi0_q4_dense, _ = get_pi0_q4_from_g0_r_tau_fft(
        g0_r_tau=g0_r_tau_ifft_dense_mesh,
        n_nu=p_propr['n_nu'],
        dim=p_phys['dim'],
        beta=p_phys['beta'],
        delta_tau=p_propr['delta_tau'],
        n_site_pd=p_phys['n_site_pd'],
        lat_const=p_phys['lat_const'],
    )

    if optdict['plot_pi0']:
        # Get \Pi_0 via upsampling from the non-uniform tau mesh
        pi0_q4_upsampled, _ = get_pi0_q4_from_g0_r_tau_fft(
            g0_r_tau=g0_r_tau_interp_mtx,
            n_nu=p_propr['n_nu'],
            dim=p_phys['dim'],
            beta=p_phys['beta'],
            delta_tau=p_propr['delta_tau'],
            n_site_pd=p_phys['n_site_pd'],
            lat_const=p_phys['lat_const'],
        )
        mlist_plot = np.arange(2)
        # Inclusive endpoints for integration (and to ensure odd number of time points)
        r_red_slice_incl = p_phys['dim'] * (slice(p_phys['n_site_irred']),) + (slice(None),)
        # Get the polarization bubble along the k-path via quadrature integration in \tau; while
        # this approach would be very inefficient to calculate \Pi_0 on the entire Brillouin
        # zone, we use it as a benchmark for the FFT methods along the high-symmetry path
        pi0_sigma_quad_path, _ = get_pi0_q4_path_from_g0_r_tau_quad(
            g0_r_tau_ifft_red_mesh=g0_r_tau_ifft_mesh[r_red_slice_incl],
            path_q_coords=path_k_coords,
            inu_list=(2j * np.pi / p_phys['beta']) * mlist_plot.astype(float),
            tau_list=tau_list,
            beta=p_phys['beta'],
            delta_tau=p_propr['delta_tau'],
            dim=p_phys['dim'],
            n_site_pd=p_phys['n_site_pd'],
            lat_const=p_phys['lat_const'],
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
                # )
                # ax.set_ylabel('Absolute error')
                # ax.grid(True)
                # fig.savefig(savename)
            # Plot the results
            fig, ax = plt.subplots()
            i_path = range(len(path_k_coords))
            # print(len(i_path))
            if len(i_path_kf_locs) > 0:
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
                p_propr['n_tau']) + r', $n_{\nu} = $'+str(p_propr['n_nu']))
            # Set the same plot range and ticks as the Kozik paper for easy visual correspondence
            ax.set_xlim(left=i_path[0], right=i_path[-1])
            # ax.set_ylim(bottom=-.65, top=-.35)
            ax.grid(True, color='k', linestyle=':', which='minor')
            ax.grid(True, color='k', linestyle=':', which='major')
            fig.tight_layout()
            savename = safe_filename(
                dir=save_dir,
                savename=(f"lat_pi0_q_inu{this_m}_N={p_phys['n_site_pd']}_beta={p_phys['beta']:g}" +
                          f"_n_tau={p_propr['n_tau']}_n_nu={p_propr['n_nu']}" +
                          f"_fft_upsampled"),
                file_extension='pdf',
                overwrite=True,
            )
            fig.savefig(savename)

    if not optdict['dry_run']:
        # Save the lattice Pi_0(q, i nu) data to h5
        pi0_h5file = save_dir / f'lat_pi0_q4_{job_id}.h5'
        # Open H5 file and write attributes/data to it
        h5file = h5py.File(pi0_h5file, 'w')
        h5file.attrs['n0'] = p_phys['n0']
        h5file.attrs['mu'] = p_phys['mu']
        h5file.attrs['mu_tilde'] = p_phys['mu_tilde']
        h5file.attrs['lat_const'] = p_phys['lat_const']
        h5file.attrs['n_site_pd'] = p_phys['n_site_pd']
        h5file.attrs['n_nu'] = p_propr['n_nu']
        h5file.attrs['dim'] = p_phys['dim']
        h5file.attrs['beta'] = p_phys['beta']
        h5file.attrs['t_hop'] = p_phys['t_hop']
        h5file.attrs['U_loc'] = p_phys['U_loc']
        h5file.attrs['n_tau_unif'] = p_propr['n_nu']

        k_red_slice = p_phys['dim'] * \
            (slice(p_phys['n_site_irred']),) + (slice(p_propr['n_nu']),)
        pi0_q4_irred_mesh = pi0_q4_dense[k_red_slice]
        pi0_irred_1d = pi0_q4_irred_mesh.flatten(order='C')
        dataset_pi0 = h5file.create_dataset(
            'lat_pi0_q4_data', data=pi0_irred_1d)
        dataset_pi0.attrs['shape'] = pi0_q4_dense[k_red_slice].shape
        nu_mesh_unif = list(range(p_propr['n_nu']))

        h5file.create_dataset('nu_unif_mesh', data=nu_mesh_unif)
        h5file.create_dataset('tau_unif_mesh', data=tau_dense_unif[:-1])
        # Write to disk
        h5file.close()

    return


# End of main()
if __name__ == '__main__':
    main()
