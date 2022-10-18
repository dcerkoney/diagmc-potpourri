#!/usr/bin/env python3

# Package imports
import sys
import json
import optparse
import warnings
import subprocess
import multiprocessing
import numpy as np

from ruamel.yaml import YAML
from pathlib import PosixPath
from datetime import datetime
from scipy.special import gamma

# Local script imports
from misc_tools import (
    json_pretty_dumps,
    approx_eq_float,
)
from lattice_tools import (
    LatticeDensity,
    fill_band,
)


def apply_runtime_config_updates(yaml_interface, cfg, config_file, dry_run=False):
    """Update the YAML config with parameters which are derived / defined at runtime."""
    # Reference the physical param group directly for brevity (cfg modified in-place)
    p_phys = cfg['phys']

    # Currently, we only implement the 2D and 3D cases
    if p_phys['dim'] not in [2, 3]:
        raise ValueError(
            'd = ' + str(p_phys['dim']) +
            ' is not a currently supported spatial dimension; d must be 2 or 3!')

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
            p_phys['lat_length'],
            p_phys['lat_const'] * p_phys['n_site_pd'],
            abs_tol=4 * sys.float_info.epsilon)

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

    # Defines the maximum possible index value in the reduced
    # difference vector mesh, max(nr_1 - nr_2) = (N // 2) + 1
    p_phys['n_site_irred'] = (p_phys['n_site_pd'] // 2) + 1

    # (Hartree) reduced chemical potential with which we parametrize G_H
    p_phys['mu_tilde'] = p_phys['mu'] - sigma_hartree(k=None, rho=p_phys['n0'])

    print('\nN_e: ', p_phys['num_elec'], '\nn_H: ', p_phys['n0'], '\nV: ', p_phys['vol_lat'],
          '\nmu_H: ', p_phys['mu'], '\nmu_tilde_H: ', p_phys['mu_tilde'])

    # Volume of a 'dim'-dimensional ball
    def rad_d_ball(vol): return (vol * gamma(1.0 + p_phys['dim'] / 2.0)
                                 )**(1.0 / float(p_phys['dim'])) / np.sqrt(np.pi)

    # Get the Wigner-Seitz radius
    p_phys['rs'] = float(rad_d_ball(1.0 / p_phys['n0']))
    print('r_s(n_H): '+str(p_phys['rs']))

    # Get the Fermi energy by explicitly filling the band
    print('Filling band...', end='')
    ef, _ = fill_band(
        dim=p_phys['dim'],
        num_elec=p_phys['num_elec'],
        n_site_pd=p_phys['n_site_pd'],
        lat_const=p_phys['lat_const'],
        t_hop=p_phys['t_hop'],
    )
    print(f"done!\nFound Fermi energy E_F = {p_phys['ef']} (Hartrees)")
    p_phys['ef'] = ef

    # Numerical roundoff issues occur at half-filling, so manually round the Fermi energy to zero
    if p_phys['target_n0'] == 1:
        # Double-check that we actually obtained a near-zero answer for ef
        assert np.allclose(p_phys['ef'], 0)
        # Then, shift it to the exact value at half-filling
        p_phys['ef'] = 0.0

    if not dry_run:
        # Dump updated YAML config back to file
        yaml_interface.dump(cfg, config_file)
        # Dump YAML config to JSON, to be used by the C++ MCMC driver
        with open(config_file.with_suffix('.json'), 'w') as f:
            json_pretty_dumps(cfg, f, wrap_line_length=79)
    return cfg


def main():
    """Perform the MCMC run."""
    usage = """usage: %prog [ options ]"""
    parser = optparse.OptionParser(usage)

    # Optional flags for overriding config file params
    parser.add_option("--target_mu", type="float",  default=None,
                      help="Target (noninteracting) chemical potential. If supplied, we work at "
                      + "fixed chemical potential and variable density; otherwise, we use a "
                      + "fixed density and variable chemical potential.")
    parser.add_option("--target_n0", type="float", default=None,
                      help="Target density in units of the lattice constant; since the number " +
                      "of electrons is coarse-grained, the actual density may differ slightly. " +
                      "Default (n0 = 1) corresponds to half-filling (mu0 ~= 0).")
    parser.add_option("--lat_length", type="float",    default=None,
                      help="Lattice length in Bohr radii for working at fixed V. "
                      + "If supplied, the lattice constant is deduced from the "
                      + "lattice volume and number of sites per direction.")
    parser.add_option("--lat_const", type="float",    default=None,
                      help="lattice constant in Bohr radii")
    parser.add_option("--n_site_pd", type="int",    default=None,
                      help="number of lattice sites per direction")
    parser.add_option("--n_tau", type="int",   default=None,
                      help="number of tau points in the nonuniform mesh "
                      + "used for downsampling (an even number)")
    parser.add_option("--n_nu", type="int",   default=None,
                      help="number of bosonic frequency points in "
                      + "the uniform FFT mesh (an even number)")
    parser.add_option("--dim", type="int", default=2,
                      help="spatial dimension of the lattice (allowed values: {2, 3})")
    parser.add_option("--beta", type="float",  default=None,
                      help="inverse temperature in inverse Hartrees")
    parser.add_option("--t_hop", type="float", default=None,
                      help="tight-binding hopping parameter t")
    parser.add_option("--U_loc", type="float",  default=None,
                      help="onsite Hubbard interaction in Hartrees")
    parser.add_option("--n_band", type="int",  default=None,
                      help="number of bands for multi-band Hubbard runs")
    parser.add_option("--n_threads", type="int",  default=None,
                      help="number of MPI threads for the MCMC run")

    # Save / plot options
    parser.add_option("--config", type="string", default="config.yml",
                      help="relative path of the config file to be used (default: 'config.yml')")
    parser.add_option("--propr_save_dir", type="string", default="propagators",
                      help="subdirectory to save results to, if applicable")
    parser.add_option("--plot_mcmc", default=False,  action="store_true",
                      help="generate plots for this MCMC run")
    parser.add_option("--plot_g0", default=False,  action="store_true",
                      help="generate plots for the lattice Green's function")
    parser.add_option("--plot_pi0", default=False,  action="store_true",
                      help="generate plots for the polarization bubble")
    parser.add_option("--dry_run", default=False, action="store_true",
                      help="perform a dry run to display proposed run configuration " +
                           "and propagator data (if applicable)")

    # Next, parse the arguments and collect all options into a dictionary
    (options, _) = parser.parse_args()
    optdict = vars(options)

    # Log job start time
    start_time = datetime.utcnow()
    print(f'\nJob started at: {start_time}\n')

    # Parse YAML config file
    yaml_interface = YAML(typ='rt')
    yaml_interface.default_flow_style = False
    config_file = PosixPath(optdict['config'])
    try:
        cfg = yaml_interface.load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The specified config file '{config_file}' was not found!")
    except Exception:
        raise OSError(
            f"The specified config file '{config_file}' is invalid YAML!")

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

    # Update the config with physical parameters derived at runtime
    cfg = apply_runtime_config_updates(yaml_interface, cfg, config_file, optdict['dry_run'])

    # Collect relevant arguments to be forwarded to generate_propagators.py
    forwarded_flags = ['--n_tau', str(cfg['propr']['n_tau']), '--n_nu', str(cfg['propr']['n_nu'])]
    gen_propr_options = ['config', 'propr_save_dir', 'plot_g0', 'plot_pi0', 'dry_run']
    forwarded_flags += [elem for k, v in optdict.items() for elem in (f'--{k}', f'{v}')
                        if (k in gen_propr_options and v is not (None or False))]
    print(forwarded_flags)

    # Link the run config to appropriate propagator data, or generate it if not available
    print("\nCalling script 'generate_propagators.py'...")
    subprocess.run(['python3', 'generate_propagators.py'] + forwarded_flags,
                   stdout=sys.stdout, stderr=sys.stderr)

    if not optdict['dry_run']:
        # Reload the YAML config (which may be updated by generate_propagators.py)
        cfg = yaml_interface.load(config_file)

    # Summarize any config settings overriden by cmdline flags, and print the updated config
    cfg_update_string = ('Proposed config settings' if optdict['dry_run']
                         else 'Updated config settings')
    # Using js-beautify for more human-readable JSON array indentation
    print(f'{cfg_override_string}\n{cfg_update_string}:')
    print(json_pretty_dumps(cfg, wrap_line_length=79))

    if not optdict['dry_run']:
        # Deduce the appropriate Open MPI prefix (if any) for this run
        openmpi_prefix = []
        n_cores = multiprocessing.cpu_count()
        n_threads = cfg['mcmc']['n_threads']
        if n_threads > 1:
            # Basic MPI run
            openmpi_prefix = ['mpiexec', '-n', f'{n_threads}']
            # MPI run using hyperthreading
            if n_threads == 2 * n_cores:
                openmpi_prefix.append('--use-hwthread-cpus')
            # Oversubscribed MPI run
            elif n_threads > n_cores:
                warnings.warn(
                    "Oversubscribing the number of MPI slots (n_threads > n_cores)", RuntimeWarning)
                openmpi_prefix.append('--oversubscribe')

        # Call the C++ executable to perform the MCMC run
        print("\nCalling MCMC C++ executable mb_hub_2dsqlat_rt_mcmc_cf_meas...")
        subprocess.run(openmpi_prefix + ['./mb_hub_2dsqlat_rt_mcmc_cf_meas'],
                       stdout=sys.stdout, stderr=sys.stderr)

        # Reformats the updated config.json file using jsbeautify
        with open(config_file.with_suffix('.json'), 'r+') as f:
            cfg = json.load(f)
            f.seek(0)
            f.truncate(0)
            json_pretty_dumps(cfg, f, wrap_line_length=79, keep_array_indentation=False)

        # Optionally, plot the results of the MCMC run
        if optdict['plot_mcmc']:
            print("\nCalling script 'plot.py'...")
            subprocess.run(['python3', 'plot.py', 'latest'],
                           stdout=sys.stdout, stderr=sys.stderr)

    # Log job end time
    end_time = datetime.utcnow()
    print(f'\nJob ended at: {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    return


# End of main()
if __name__ == '__main__':
    main()
