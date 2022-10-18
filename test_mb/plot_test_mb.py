#!/usr/bin/env python3

'''
Plots the mth-order correction to the charge susceptibility in a multiband
Hubbard model with n degenerate bands, and compares to the analytical result
(n ^ F(m) times the one-band result, where F(m) = # fermion loops at order m in U)
'''

# Standard imports
import os
import glob
import pathlib

# Package imports
import json
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from packaging import version

# Local imports
from lattice_tools import fill_band

if version.parse(matplotlib.__version__) < version.parse("3.4"):
    # Ignore findfont warnings (buggy for matplotlib<3.4)
    import logging
    logging.getLogger('matplotlib.font_manager').disabled = True


def complex_h5view(dataset):
    '''Get a view of HDF5 composite re/im data as complex type.'''
    return dataset[:].view(complex)


def convert(array):
    '''Converts x from a numpy array to a list, if applicable'''
    if hasattr(array, "tolist"):
        return array.tolist()
    raise TypeError(array)


def load_json_cfg_from_h5(run_subdir, verbose=False):
    ''' Load the run subdirectory results from HDF5. '''
    # If there are multiple matches, load cfg from
    # the first match (they will be consistent anyway)
    rundata_path = glob.glob(f"./{run_subdir}/*run_*.h5")[0]
    run_data = h5py.File(rundata_path, 'r')
    extras = dict(run_data.attrs)
    # Get the original run config from the corresponding H5 attribute
    cfg = json.loads(extras.pop('config'))
    # Add extra H5 attributes to the JSON config's MCMC option group
    cfg['mcmc'].update(extras)
    run_data.close()
    if verbose:
        print(f"\nJSON config for run_{cfg['mcmc']['job_id']} at directory " +
              f"'{run_subdir}':\n{json.dumps(cfg, indent=4, default=convert)}")
    return cfg


def plot_chi_n_ch(cfg):
    '''Plots the nth order correction to the static charge susceptibility.'''
    # Load reference results from h5, if present in this subdirectory
    refdata_path = pathlib.PosixPath(f"./{cfg['mcmc']['save_dir']}") / "reference.h5"
    ref_exists = os.path.isfile(refdata_path)
    if ref_exists:
        ref_chi_ch_data = h5py.File(refdata_path, 'r')
        # NOTE: We define the bubble with an extra minus sign, following the usual
        #       Dyson equation convention, in contrast with Kristjan's code (and Mahan)
        chi_n_ch_ref_means = -1 * \
            complex_h5view(ref_chi_ch_data[f"V{cfg['mcmc']['max_order']}_meas_mean"])
        chi_n_ch_ref_errs = None
        if cfg['mcmc']['n_threads'] > 1:
            chi_n_ch_ref_errs = ref_chi_ch_data[f"V{cfg['mcmc']['max_order']}_meas_stderr"][:]
        ref_chi_ch_data.close()

    # Load run results from h5
    chi_path = (
        pathlib.PosixPath(f"./{cfg['mcmc']['save_dir']}") /
        f"{cfg['mcmc']['save_name']}_run_{cfg['mcmc']['job_id']}.h5"
    )
    chi_ch_run_data = h5py.File(chi_path, 'r')
    # NOTE: We define the bubble with an extra minus sign, following the usual
    #       Dyson equation convention, in contrast with Kristjan's code (and Mahan)
    chi_n_ch_means = -1 * \
        complex_h5view(chi_ch_run_data[f"V{cfg['mcmc']['max_order']}_meas_mean"])
    chi_n_ch_errs = None
    if cfg['mcmc']['n_threads'] > 1:
        chi_n_ch_errs = chi_ch_run_data[f"V{cfg['mcmc']['max_order']}_meas_stderr"][:]
    chi_ch_run_data.close()

    # Charge and longitudinal spin susceptibility measurements are real, although
    # we may find a small noisy imaginary part in practice; throw it out
    assert np.allclose(chi_n_ch_means.imag, 0)
    chi_n_ch_means = chi_n_ch_means.real
    if ref_exists:
        assert np.allclose(chi_n_ch_ref_means.imag, 0)
        chi_n_ch_ref_means = chi_n_ch_ref_means.real

    # Deduce number of measured frequency points
    n_nu_meas = len(chi_n_ch_means) / float(cfg['mcmc']['n_k_meas'])
    assert n_nu_meas == int(n_nu_meas)
    n_nu_meas = int(n_nu_meas)

    # List of Matsubara frequencies at which the susceptibility was measured
    # m_list = np.arange(n_nu_meas)
    # om_list = (2 * np.pi / cfg['beta']) * m_list
    # print('\nList of measurement Matsubara frequencies:\n', om_list * (p.beta / np.pi))

    # Tests the band filling scheme
    ef, kf_vecs = fill_band(
        dim=cfg['phys']['dim'],
        num_elec=cfg['phys']['num_elec'],
        n_site_pd=cfg['phys']['n_site_pd'],
        lat_const=cfg['phys']['lat_const'],
        t_hop=cfg['phys']['t_hop'],
    )
    # print('List of k-points on the Fermi surface:\n', kf_vecs)

    # Numerical roundoff issues occur at half-filling, so manually round the Fermi energy to zero
    if cfg['phys']['n0'] == 1:
        # Double-check that we actually obtained a near-zero answer for ef
        # (i.e., that the k_F vectors are legitimate)
        assert np.allclose(ef, 0.0)

    # Build the high-symmetry (\Gamma -- X -- M -- \Gamma) path
    # and save it to 'kpath.dat' unless this is a dry-run
    with open(pathlib.PosixPath(f"./{cfg['mcmc']['save_dir']}") / 'k_path_info.json') as f:
        k_path_info = json.load(f)
        path_nk_coords = np.asarray(k_path_info['k_path'])
        k_scale = k_path_info['k_scale']
        high_symm_indices = k_path_info['high_symm_indices']
        high_symm_path = k_path_info['high_symm_path']

    # Build the high-symmetry (\Gamma -- X -- M -- \Gamma) path
    # and save it to 'kpath.dat' unless this is a dry-run
    with open(pathlib.PosixPath(f"./{cfg['mcmc']['save_dir']}") / 'graph_info.json') as f:
        graph_info = json.load(f)
        n_loops = np.asarray(graph_info['n_loops'])
    print(n_loops)

    # This test requires that the number of fermionic loops is the
    # same for each diagram at the calculated order
    assert np.all(n_loops == n_loops[0])
    n_loops = n_loops[0]

    # Include scale factor for k-points
    path_k_coords = k_scale * path_nk_coords
    n_path_edges = 3
    n_edge = len(path_k_coords) / n_path_edges

    # Add in the (redundant) Gamma point to the end of the path for plotting purposes
    i_path = np.arange(len(path_k_coords) + 1)

    # Find the corresponding indices in the full k_list
    i_path_kf_locs = []
    for i, this_k_coord in enumerate(path_k_coords):
        for kf_vec in kf_vecs:
            if np.all(this_k_coord == kf_vec):
                i_path_kf_locs.append(i)
    i_path_kf_locs = np.asarray(i_path_kf_locs)
    # If we missed the Fermi surface along the M-\Gamma path
    # due to coarse-graining, set the locations manually
    # when at half-filling
    if (len(i_path_kf_locs) < 2) and (cfg['phys']['n0'] == 1):
        i_path_kf_locs = [cfg['mcmc']['n_k_meas'] / 3.0, cfg['mcmc']['n_k_meas'] * 5 / 6.0]

    # Reshape the calculated susceptibility data for plotting
    chi_n_ch_calc_means = chi_n_ch_means.reshape(
        (n_nu_meas, cfg['mcmc']['n_k_meas']))
    chi_n_ch_calc_errs = None
    if cfg['mcmc']['n_threads'] > 1:
        chi_n_ch_calc_errs = chi_n_ch_errs.reshape(
            (n_nu_meas, cfg['mcmc']['n_k_meas']))
    if ref_exists:
        chi_n_ch_calc_ref_means = chi_n_ch_ref_means.reshape(
            (n_nu_meas, cfg['mcmc']['n_k_meas']))
        chi_n_ch_calc_ref_errs = None
        if cfg['mcmc']['n_threads'] > 1:
            chi_n_ch_calc_ref_errs = chi_n_ch_ref_errs.reshape(
                (n_nu_meas, cfg['mcmc']['n_k_meas']))

    chi_means_plot = [chi_n_ch_calc_means]
    chi_errs_plot = [chi_n_ch_calc_errs]
    if ref_exists:
        chi_means_plot.insert(0, (cfg['phys']['n_band'] ** n_loops) * chi_n_ch_calc_ref_means)
        chi_errs_plot.insert(0, (cfg['phys']['n_band'] ** n_loops) * chi_n_ch_calc_ref_errs)

    ##########################################################################
    # Plot the charge susceptibility for the first few Matsubara frequencies #
    ##########################################################################

    print('\nPlotting charge susceptibility corrections...', end='', flush=True)

    colorlists_plot = [['orchid', 'cornflowerblue',
                        'turquoise', 'chartreuse', 'greenyellow']]
    if ref_exists:
        colorlists_plot.insert(0, len(colorlists_plot[0]) * ['dimgray'])
        # colorlists_plot.append(['darkorchid', 'royalblue',
        #                         'mediumturquoise', 'limegreen', 'yellowgreen'])
    linestyles_plot = ['--', 'o-']
    n_bands_plot = [1, cfg['phys']['n_band']]

    # Plot the static susceptibility if n_nu_meas = 1
    if n_nu_meas == 1:
        # Make singleton axis list for compatibility with the static case (n_nu_meas = 1)
        fig, axes = plt.subplots()
        if len(i_path_kf_locs) > 0:
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
        for i, chi_mean in enumerate(chi_means_plot):
            ref_prefactor = ''
            if ref_exists:
                color = colorlists_plot[i][iom]
                linestyle = linestyles_plot[i]
                if i == 0:
                    ref_prefactor = rf"{cfg['phys']['n_band'] ** n_loops}"
            else:
                color = 'dimgray'
                linestyle = '--'
            axes[iom].plot(
                i_path, chi_mean[iom, :][i_path % cfg['mcmc']['n_k_meas']],
                linestyle, markersize=2.5, color=color,
                label=(rf"${ref_prefactor}\chi^{{(2)}}_{{\mathrm{{ch}}, "
                       rf"{n_bands_plot[i]}\,\mathrm{{band}}}} \, "
                       rf"(n = {cfg['phys']['n0']:g},\; U = {cfg['phys']['U_loc']:g},\; "
                       rf"\beta = {cfg['phys']['beta']:g})$"))
            if cfg['mcmc']['n_threads'] > 1:
                axes[iom].fill_between(
                    i_path, chi_mean[iom, :][i_path % cfg['mcmc']['n_k_meas']] +
                    chi_errs_plot[i][iom, :][i_path % cfg['mcmc']['n_k_meas']],
                    chi_mean[iom, :][i_path % cfg['mcmc']['n_k_meas']] -
                    chi_errs_plot[i][iom, :][i_path % cfg['mcmc']['n_k_meas']],
                    color=color, alpha=0.3)
        axes[iom].legend(loc='upper left')
        # Add some evenly-spaced minor ticks to the axis
        n_minor_ticks = 9
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
        axes[iom].set_xticks(high_symm_indices)
        axes[iom].set_xticklabels(high_symm_path)
        iqmstring = r''
        # if iom == 0:
        #     iqmstring = r'=0'
        axes[iom].set_title(rf"$\chi^{{({cfg['diag']['n_intn']})}}_{{\mathrm{{ch}}}}"
                            rf'[G_{{H}}, U](\mathbf{{q}}, iq_{iom}{iqmstring})$', pad=13)
        axes[iom].text(
            # x=0.0225, y=0.825,
            # x=0.0365, y=0.815,
            x=0.0225, y=0.735 + 0.07 * (not ref_exists),
            s=(
                fr"$N_{{\mathrm{{thread}}}}={cfg['mcmc']['n_threads']},"
                fr"\, N_{{\mathrm{{meas}}}} = {cfg['mcmc']['n_meas']:g}$"
            ),
            horizontalalignment='left', verticalalignment='center', transform=axes[iom].transAxes,
            bbox=dict(
                boxstyle="round, pad=0.4", ec=(0.8, 0.8, 0.8),
                fc=(1, 1, 1),
                alpha=0.8),)
        # Set the same plot range and ticks as the Kozik paper for easy visual correspondence
        axes[iom].set_xlim(left=i_path[0], right=i_path[-1])
        # axes[iom].set_ylim(bottom=-.65, top=-.35)
        axes[iom].grid(True, color='k', linestyle=':', which='minor')
        axes[iom].grid(True, color='k', linestyle=':', which='major')
        fig.tight_layout()
    savepath = (
        pathlib.PosixPath(f"./{cfg['mcmc']['save_dir']}") /
        f"chi_{cfg['diag']['n_intn']}_ch_2dsqhub_run_{cfg['mcmc']['job_id']}.pdf"
    )
    fig.savefig(savepath)
    plt.close('all')

    print('done!')


def main():
    """
    Generate charge susceptibility plots for the Hubbard model on a finite square lattice.
    """
    # Use TeX / TeX font in plots
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    # Parse command line argument(s)
    logfiles = glob.glob('*/*run*.log')
    if not logfiles:
        raise ValueError('No logfiles found in the working directory!')
    # Get a list of job IDs corresponding to
    # all logfiles in the working directory
    run_subdirs = [os.path.dirname(lf) for lf in logfiles]
    for i in range(len(logfiles)):
        # Get JSON config
        cfg = load_json_cfg_from_h5(run_subdirs[i])
        plot_chi_n_ch(cfg)


# End of main()
if __name__ == '__main__':
    main()
