#!/usr/bin/env python3

# Standard imports
import json
import glob
import argparse
from pathlib import PosixPath
from datetime import datetime
from ruamel.yaml import YAML

# Package imports
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Local script imports
from misc_tools import (
    json_pretty_dumps,
    safe_filename,
    to_timestamp,
)
from lattice_tools import (
    fill_band,
    get_lat_g0_r_tau,
    get_pi0_q4_from_g0_r_tau_fft,
    get_pi0_q4_path_from_g0_r_tau_quad,
)


def simple_cubic_high_symm_path(lat_const, n_site_pd, save=True):
    '''
    Build an ordered path of k-points in the Brillouin zone; we choose the high-symmetry
    path for the simple square lattice (Gamma -- X -- M -- Gamma), discarding duplicate
    coordinates at the path vertices (accounted for in plotting step).
    '''
    N_edge = int(np.floor(n_site_pd / 2.0))
    nk_coords_Gamma_X = [[x, 0] for x in range(0, N_edge + 1)]
    nk_coords_X_M = [[N_edge, y] for y in range(1, N_edge + 1)]
    nk_coords_M_Gamma = [[xy, xy] for xy in range(1, N_edge)[::-1]]
    # Build the full ordered high-symmetry path
    path_nk_coords = np.concatenate(
        (nk_coords_Gamma_X, nk_coords_X_M, nk_coords_M_Gamma))
    # Labels / indices of the high-symmetry points of the k-path (G -- X -- M -- G)
    high_symm_path = ['$\\Gamma$', '$X$', '$M$', '$\\Gamma$']
    high_symm_indices = [0, N_edge, 2 * N_edge, 3 * N_edge]
    # Scale factor for k-points
    k_scale = 2.0 * np.pi / float(lat_const * n_site_pd)
    # Duplicate points are not stored explicitly in the path
    assert len(np.unique(path_nk_coords, axis=0)) == len(path_nk_coords)
    if save:
        k_path_info = {"lat_const": lat_const,
                       "n_site_pd": n_site_pd,
                       "k_scale": k_scale,
                       "high_symm_path": high_symm_path,
                       "high_symm_indices": high_symm_indices,
                       "k_path": path_nk_coords.tolist()}
        # Using js-beautify for indentation with more human-readable JSON arrays
        # k_path_info_serialized = jsbeautifier.beautify(json.dumps(k_path_info))
        with open('k_path_info.json', 'w') as f:
            json_pretty_dumps(k_path_info, f, keep_array_indentation=False)
    return path_nk_coords


def main():
    '''Get the lattice Green's function and polarization bubble to Hartree level for Hubbard-type theories.'''
    usage = '''usage: %prog [ options ]'''
    parser = argparse.ArgumentParser(usage)

    # Propagator mesh sizes (optional flags for overriding config file params)
    parser.add_argument("--n_tau", type="int",   default=None,
                      help="number of tau points in the nonuniform mesh "
                      + "used for downsampling (an even number)")
    parser.add_argument("--n_nu", type="int",   default=None,
                      help="number of bosonic frequency points in "
                      + "the uniform FFT mesh (an even number)")

    # Save / plot options
    parser.add_argument("--config", type="string", default="config.yml",
                      help="relative path of the config file to be used (default: 'config.yml')")
    parser.add_argument("--propr_save_dir", type="string", default="propagators",
                      help="subdirectory to save results to, if applicable")
    parser.add_argument("--plot_g0", default=False,  action="store_true",
                      help="generate plots for the lattice Green's function")
    parser.add_argument("--plot_pi0", default=False,  action="store_true",
                      help="generate plots for the polarization bubble")
    parser.add_argument("--dry_run", default=False,  action="store_true",
                      help="perform a dry run (don't update config file or save propagator data)")

    # Next, parse  the arguments and collect all arguments into a dictionary
    (args, _) = parser.parse_args()
    optdict = vars(args)
    
    # Parse YAML config file
    yaml_interface = YAML(typ='rt')
    yaml_interface.default_flow_style = False
    config_file = PosixPath(optdict['config'])
    try:
        cfg = yaml_interface.load(config_file)
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(
            f"The specified config file '{config_file}' was not found!") from fnfe
    except Exception as exn:
        raise OSError(f"The specified config file '{config_file}' is invalid YAML!") from exn

    # Override the config with user-supplied cmdline args where applicable
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
    p_phys = cfg['phys']
    p_propr = cfg['propr']

    # Check that required config options are defined as expected
    # (in case this script is called directly by the user)
    required_phys_keys = ['dim', 'n_site_pd', 'lat_const',
                          'lat_length', 'beta', 't_hop', 'mu_tilde', 'n0']
    required_propr_keys = ['delta_tau', 'n_nu', 'n_tau']
    required_group_keys = {'phys': required_phys_keys, 'propr': required_propr_keys}
    for group, keys in required_group_keys.items():
        for key in keys:
            if key not in cfg[group]:
                raise KeyError(
                    f"Required YAML key '{key}' in group '{group}' is missing from {config_file}!")

    # Get job start time for purposes of generating a UNIX timestamp ID
    start_time = datetime.utcnow()
    job_id = to_timestamp(start_time, rtype=int)
    p_propr['job_id'] = job_id
    print(f'\nTimestamp: {job_id} (UTC)\nJob started at: {start_time}')

    # Check for any existing propagator data in the propagator save_dir
    # matching the current parameters; if none exists, generate them
    consistent_proprs_exist = False
    consistent_propr_path = None
    propr_paths = glob.glob(f"./{optdict['propr_save_dir']}/*/lat_g0_rt*.h5")
    print(propr_paths)
    for propr_path in propr_paths:
        g0_path = PosixPath(propr_path)
        g0_data = h5py.File(g0_path, 'r')
        try:
            j_cfg = json.loads(json.dumps(cfg))
            g0_cfg = json.loads(g0_data.attrs['config'])
            # Ignore propagator job_id / save_dir for purposes of this consistency check
            for d in (j_cfg['propr'], g0_cfg['propr']):
                for k in ('job_id', 'save_dir'):
                    d.pop(k, None)
            # Check for consistency of physical/propagator parameters between the
            # currently loaded propagator data and the current run configuration
            params_consistent = (j_cfg['phys'] == g0_cfg['phys']
                                 and j_cfg['propr'] == g0_cfg['propr'])
        except (AttributeError, KeyError):
            # Handle any HDF5 attribute / JSON key errors gently; they
            # merely imply this Green's function data is incompatible
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
        p_propr['save_dir'] = str(consistent_save_dir)
        p_propr['job_id'] = int(consistent_job_id)
        print('done!')
    else:
        print('\nNo consistent propagator data found for the supplied config, generating it now...')

    if not consistent_proprs_exist:
        # Make directory in which to save new propagator figs/data (if applicable and necessary)
        propr_dir = PosixPath(f'proprs_{job_id}')
        save_dir = PosixPath(optdict['propr_save_dir']) / propr_dir
        p_propr['save_dir'] = str(save_dir)
        if not optdict['dry_run']:
            save_dir.mkdir(parents=True, exist_ok=True)

    # Build the high-symmetry (\Gamma -- X -- M -- \Gamma) path
    # and save it to 'kpath.dat' unless this is a dry-run
    path_nk_coords = simple_cubic_high_symm_path(
        lat_const=p_phys['lat_const'],
        n_site_pd=p_phys['n_site_pd'],
        save=(not optdict['dry_run']))

    # Update number of measurement k-points in the
    # YAML config file to the above k-path length
    n_k_pts = len(path_nk_coords)
    cfg['mcmc']['n_k_meas'] = n_k_pts

    # Include scale factor for k-points
    k_scale = 2.0 * np.pi / float(p_phys['lat_length'])
    path_k_coords = k_scale * path_nk_coords

    # Use a cubic tau mesh; the smallest time mesh point is delta_tau (in units of beta)
    tau_powlist_left = p_phys['beta'] * (
        np.linspace(
            p_propr['delta_tau'] / p_phys['beta'] ** (1.0 / 3.0),
            0.5 ** (1.0 / 3.0),
            num=p_propr['n_tau'] // 2) ** 3)
    tau_powlist_right = p_phys['beta'] - tau_powlist_left[::-1]
    tau_list = np.concatenate(
        ([0.0, p_propr['delta_tau']],
         tau_powlist_left[1: -1],
         [0.5 * p_phys['beta']],
         tau_powlist_right[1: -1],
         [p_phys['beta'] - p_propr['delta_tau'],
          p_phys['beta']]))
    assert(len(tau_list[:-1]) == p_propr['n_tau'])

    # UV cutoff defined by the lattice (longest scattering length in the BZ)
    # llambda_lat = p_phys['dim'] * (np.pi / p_phys['lat_const'])**2 / 2.0

    if not optdict['dry_run']:
        # Save updated config back to file
        yaml_interface.dump(cfg, config_file)
        # Dump YAML config to JSON to be used by the C++ MCMC driver
        with open(config_file.with_suffix('.json'), 'w') as f:
            json_pretty_dumps(cfg, f, wrap_line_length=79)

    # Nothing more to do after updating the config in either of these cases
    if consistent_proprs_exist or optdict['dry_run']:
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

    tau_dense_unif = p_phys['beta'] * \
        np.arange(p_propr['n_nu'] + 1) / float(p_propr['n_nu'])
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

    # Save the lattice G_0(r, tau) data to h5
    g0_h5file = save_dir / f'lat_g0_rt_{job_id}.h5'
    # Open H5 file and write config/data to it
    h5file = h5py.File(g0_h5file, 'w')
    # Serialize config to JSON, and save as an H5 string attribute
    h5file.attrs['config'] = json.dumps(cfg)

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
        r_red_slice_incl = p_phys['dim'] * \
            (slice(p_phys['n_site_irred']),) + (slice(None),)
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
                print(
                    'Maximal absolute error between dense and upsampled FFT results along the k-path:\n',
                    np.max(np.abs(pi0_om_dense_path - pi0_om_upsampled_path)))
            if not np.allclose(pi0_om_dense_path, 2 * pi0_sigma_quad_path[..., this_m]):
                print('Maximal absolute error between (dense) FFT and quad results along the k-path:\n',
                      np.max(np.abs(pi0_om_dense_path - 2 * pi0_sigma_quad_path[..., this_m])))
            if not np.allclose(pi0_om_dense_path, 2 * pi0_sigma_quad_path[..., this_m]):
                print(
                    'Maximal absolute error between upsampled FFT and quad results along the k-path:\n',
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
            # Indices of high-symmetry points
            N_edge = int(np.floor(p_phys['n_site_pd'] / 2.0))
            high_symm_indices = [0, N_edge, 2 * N_edge, 3 * N_edge]
            # Get the Fermi wavevectors by explicitly filling the band
            _, kf_vecs = fill_band(
                dim=p_phys['dim'],
                num_elec=p_phys['num_elec'],
                n_site_pd=p_phys['n_site_pd'],
                lat_const=p_phys['lat_const'],
                t_hop=p_phys['t_hop'],
            )
            # Get indices of any path k-points on the Fermi surface for plotting
            i_path = np.arange(len(path_k_coords))
            i_path_kf_locs = []
            for i, this_k_coord in enumerate(path_k_coords):
                for kf_vec in kf_vecs:
                    if np.all(this_k_coord == kf_vec):
                        i_path_kf_locs.append(i)
            i_path_kf_locs = np.asarray(i_path_kf_locs)
            # If we missed the Fermi surface along the M-\Gamma path
            # due to coarse-graining, set the locations manually
            # when at half-filling for illustrative purposes
            if (len(i_path_kf_locs) < 2) and (cfg['phys']['n0'] == 1):
                i_path_kf_locs = [n_k_pts / 3.0, n_k_pts * 5 / 6.0]
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
            ax.set_xticks(high_symm_indices)
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
                          "_fft_upsampled"),
                file_extension='pdf',
                overwrite=True,
            )
            fig.savefig(savename)

    # Save the lattice Pi_0(q, i nu) data to h5
    pi0_h5file = save_dir / f'lat_pi0_q4_{job_id}.h5'
    # Open H5 file and write attributes/data to it
    h5file = h5py.File(pi0_h5file, 'w')
    # Serialize config to JSON, and save as an H5 string attribute
    h5file.attrs['config'] = json.dumps(cfg)
    # Number of tau points in the uniform mesh is the same as n_nu
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
