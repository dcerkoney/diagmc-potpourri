#!/usr/bin/env python3
import io
import math
import pstats
import cProfile
import numpy as np
from collections import defaultdict

# User-defined module(s)
from diaggen_tools import *  # pylint: disable=unused-wildcard-import


# Draw a batch of Feynman diagrams to a bulk .tex file; this will generate a .pdf
# with all of them in a grid (optionally: labeled vertices), as well as standalone
# postscript .1 files.
def draw_bulk_feynmp(graphs, n_legs=0, fside='left', savename='diagrams.tex'):
    if fside not in ['left', 'right']:
        raise ValueError(
            "Fermion line side preference for drawing must be either 'left' or 'right'!")
    # Draw each possible nth order diagram in a separate file
    n_graphs = 0
    n_vertices = None
    diagram_str = ''
    for i, g in enumerate(graphs):
        n_graphs += 1
        if not n_vertices:
            n_vertices = len(g)
        # Get back the psi and phi lists corresponding to this graph
        psi_list, phi_list = graph_al_to_pg(g)
        # Draw vertices; there are 2n
        diagram_str += (
            r'\begin{fmffile}{diagram_'+str(i)+'}'+'\n'
            r'\begin{fmfgraph*}(100,100)'+'\n'
            r'\fmfsurroundn{v}{'+str(n_vertices)+'}'+'\n'
            r'\fmfdotn{v}{'+str(n_vertices)+'}'+'\n'
        )
        # Draw Fermion lines; there are 2n
        for j in range(n_vertices):
            # One will be missing for self-energies
            if psi_list[j] != -1:
                diagram_str += (r'\fmf{fermion,'+fside+'=0.3}{v['
                                + str(j + 1)+'],v['
                                + str(psi_list[j] + 1)+']}\n')
        # Freeze the diagram shape after drawing Fermion lines
        # diagram_str += ('\n'+r'\fmffreeze'+'\n')
        # Draw interaction lines; there are n
        for j in range(n_vertices):
            # If the vertex is an external one, don't
            # label it, and add a small bosonic leg
            if phi_list[j] == -1:
                # Separately label external vertices
                diagram_str += (
                    r'\fmflabel{$'+str(j + 1)+r'_{\mathrm{ext}}$}{v['+str(j + 1)+']}\n'
                    r'\fmfv{d.sh=circle, d.fi=full, d.si=2thick, f=(1,,0,,0)}{v['+str(j + 1)+']}\n'
                )
                continue
            else:
                diagram_str += (r'\fmflabel{$'+str(j + 1)+r'$}{v['+str(j + 1)+']}'+'\n')
            # Ignoring mirrored connections, add the bosonic lines
            if phi_list[j] != -2:
                diagram_str += (r'\fmf{photon}{v['
                                + str(j + 1)+'],v['
                                + str(phi_list[j] + 1)+']}\n')
        # Find and label the incoming/outgoing self-energy external vertices
        for j in range(n_vertices):
            if psi_list[j] == -1:
                # Found the outgoing vertex; label it!
                diagram_str += (
                    r'\fmflabel{$'+str(j + 1)+r'_{\mathrm{ext}}$}{v['+str(j + 1)+']}\n'
                    r'\fmfv{d.sh=circle, d.fi=full, d.si=2thick, f=(1,,0,,0)}{v['+str(j + 1)+']}\n'
                )
                continue
            # Found the incoming vertex; label it!
            if j not in psi_list:
                diagram_str += (
                    r'\fmflabel{$'+str(j + 1)+r'_{\mathrm{ext}}$}{v['+str(j + 1)+']}\n'
                    r'\fmfv{d.sh=circle, d.fi=full, d.si=2thick, f=(1,,0,,0)}{v['+str(j + 1)+']}\n'
                )
                continue
        # End of this feynmf file
        diagram_str += (
            r'\end{fmfgraph*}'+'\n'
            r'\end{fmffile}'+'\n'
            r'\hspace{8ex}'+'\n'
        )
        # Spaces the drawings as nearly as possible to a square grid
        if ((i + 1) % int(math.ceil(math.sqrt(len(graphs))))) == 0:
            # diagram_str += '\n'+r'\vspace{4ex}'+'\n'
            diagram_str += '\n'+r'\vspace{8ex}'+'\n'
    if n_graphs == 0:
        print('No graphs to draw!')
        return
    # Make the new .tex file
    myfile = open(str(savename), 'w+')
    # Preamble
    preamble = (
        r'\documentclass[varwidth=\maxdimen,border=25pt]{standalone}'+'\n'
        r'\usepackage[force]{feynmp-auto}'+'\n'
        '\n'
        r'\begin{document}'+'\n'
        '\n'
    )
    myfile.write(preamble)
    # Write feynmf diagram contents to file
    myfile.write(diagram_str)
    # Write end of doc statements to file
    eod = '\n'+r'\end{document}'+'\n'
    myfile.write(eod)
    # Done writing; close the file and return
    myfile.close()
    # End of diagram creation loop
    return


# Draw a batch of Hugenholtz diagrams to a bulk .tex file; this will generate a .pdf
# with all of them in a grid (optionally: labeled vertices), as well as standalone
# postscript .1 files.
def draw_bulk_hhz_feynmp(hhz_graphs, fside='left', savename='hhz_diagrams.tex'):
    if fside not in ['left', 'right']:
        raise ValueError(
            "Fermion line side preference for drawing must be either 'left' or 'right'!")
    # Draw each possible nth order diagram in a separate file
    n_graphs = 0
    n_vertices = None
    diagram_str = ''
    for i, g in enumerate(hhz_graphs):
        n_graphs += 1
        if not n_vertices:
            n_vertices = len(g)
        # Draw vertices; there are n
        diagram_str += (
            r'\begin{fmffile}{hhz_diagram_'+str(i)+'}'+'\n'
            r'\begin{fmfgraph*}(100,100)'+'\n'
            r'\fmfsurroundn{v}{'+str(n_vertices)+'}'+'\n'
            r'\fmfdotn{v}{'+str(n_vertices)+'}'+'\n'
        )
        # Draw the fermion lines; there are 2n for vacuum graphs, plus
        # 2 for polarization graphs after the external leg addition
        phantom_string = r'\fmf{phantom}{v['+str(n_vertices)+'],'
        for v_start in g:
            phantom_string += 'v['+str(v_start + 1)+'],'
            diagram_str += r'\fmflabel{$' + \
                str(v_start + 1)+r'$}{v['+str(v_start + 1)+']}'+'\n'
            for v_end in g[v_start]:
                # Mark any external vertices (where the connection
                # is either (i, -1) or (-1, i)) by color and label
                if v_end == -1:
                    diagram_str += (
                        r'\fmflabel{$'+str(v_start + 1) +
                        r'_{\mathrm{ext}}$}{v['+str(v_start + 1)+']}'+'\n'
                        r'\fmfv{d.sh=circle, d.fi=full, d.si=2thick, f=(1,,0,,0)}{v['+str(
                            v_start + 1)+']}'+'\n'
                    )
                else:
                    if v_start == v_end:
                        tension = 2
                        diagram_str += (
                            r'\fmf{fermion, tension='+str(tension)+'}{v['+str(
                                v_start + 1)+'],v['+str(v_end + 1)+']}\n'
                        )
                    else:
                        for i_line, _ in enumerate(g[v_start][v_end]):
                            arc_angle = 0.3 + (0.2 * i_line)
                            diagram_str += (
                                r'\fmf{fermion,'+fside+'='+str(arc_angle)+'}{v['+str(
                                    v_start + 1)+'],v['+str(v_end + 1)+']}\n'
                            )
        phantom_string = phantom_string[:-1]
        phantom_string += '}'
        # End of this feynmf file
        diagram_str += (
            phantom_string+'\n'
            r'\end{fmfgraph*}'+'\n'
            r'\end{fmffile}'+'\n'
            r'\hspace{15ex}'+'\n'
        )
        # Spaces the drawings as nearly as possible to a square grid
        if ((i + 1) % int(math.ceil(math.sqrt(len(hhz_graphs))))) == 0:
            # diagram_str += '\n'+r'\vspace{4ex}'+'\n'
            diagram_str += '\n'+r'\vspace{15ex}'+'\n'
    if n_graphs == 0:
        print('No graphs to draw!')
        return   
    # Make the new .tex file
    myfile = open(str(savename), 'w+')
    # Preamble
    preamble = (
        r'\documentclass[varwidth=\maxdimen,border=35pt]{standalone}'+'\n'
        r'\usepackage[force]{feynmp-auto}'+'\n'
        '\n'
        r'\begin{document}'+'\n'
        '\n'
    )
    myfile.write(preamble)
    # Write feynmf diagram contents to file
    myfile.write(diagram_str)
    # Write end of doc statements to file
    eod = '\n'+r'\end{document}'+'\n'
    myfile.write(eod)
    # Done writing; close the file and return
    myfile.close()
    # End of diagram creation loop
    return


# Generates and draws all naive (including topologically indistinct)
# connected Hugenholtz vacuum diagrams of the specified order
def generate_naive_hhz_vacuum_diags(order=1, draw=True):
    naive_hhz_graphs = get_connected_subset(get_naive_vacuum_hhz_diags(order))
    if draw:
        draw_bulk_hhz_feynmp(naive_hhz_graphs, fside='left',
                             savename='naive_hhz_diagrams_n='+str(order)+'.tex')
    return


# Generates and draws all naive (including topologically indistinct)
# connected Hugenholtz vacuum diagrams of the specified order
def generate_hhz_vacuum_diags(order=1, draw=True):
    naive_disconnected_hhz_graphs = get_naive_vacuum_hhz_diags(order)
    naive_hhz_graphs = get_connected_subset(naive_disconnected_hhz_graphs)
    distinct_hhz_graphs = rem_top_equiv_al(naive_hhz_graphs)
    # distinct_hhz_graphs = naive_hhz_graphs
    if draw:
        draw_bulk_hhz_feynmp(distinct_hhz_graphs, fside='left',
                             savename='hhz_vacuum_diagrams_n='+str(order)+'.tex')
    print('{}! = {}'.format(2*order, math.factorial(2*order)))
    print('Number of naive Hugenholtz vacuum diagrams of order {}:'.format(
        order), len(naive_disconnected_hhz_graphs))
    print('Number of naive connected Hugenholtz vacuum diagrams of order {}:'.format(
        order), len(naive_hhz_graphs))
    print('Number of topologically distinct connected Hugenholtz vacuum diagrams of order {}:'.format(
        order), len(distinct_hhz_graphs))
    return


# Generates and draws naive (including topologically indistinct)
# connected vacuum Feynman diagrams of the specified order by
# expanding the distinct connected vacuum Hugenholtz diagrams
def generate_bare_vacuum_diags_from_hhz(order=1, draw=True):
    naive_disconnected_hhz_graphs = get_naive_vacuum_hhz_diags(order)
    naive_hhz_graphs = get_connected_subset(naive_disconnected_hhz_graphs)
    distinct_hhz_graphs = rem_top_equiv_al(naive_hhz_graphs)
    print('Unwrapping Hugenholtz diagrams...')
    n_verts_feyn = 2 * order
    feyn_graphs = []
    for graph in distinct_hhz_graphs:
        # First, we double all vertex labels in the graph; this makes room for the new vertex labels
        # in such a manner as to preserve our pairwise convention for the bosonic connections
        g_shifted = map_vertices_defaultdict(
            graph, vmap=map(lambda x: 2*x, graph.keys()))
        feyn_graphs.extend(rem_top_equiv_al(unwrap_hhz_to_feyn(g_shifted, n_verts_feyn)))
    print('Done!')

    if draw:
        draw_bulk_feynmp(feyn_graphs, n_legs=0, fside='left',
                         savename='feyn_vacuum_diagrams_bare_n='+str(order)+'.tex')

    print('{}! = {}'.format(2*order, math.factorial(2*order)))
    print('Number of naive Hugenholtz vacuum diagrams of order {}:'.format(
        order), len(naive_disconnected_hhz_graphs))
    print('Number of naive connected Hugenholtz vacuum diagrams of order {}:'.format(
        order), len(naive_hhz_graphs))
    print('Number of topologically distinct connected Hugenholtz vacuum diagrams of order {}:'.format(
        order), len(distinct_hhz_graphs))
    print('Number of bare connected Feynman vacuum diagrams of order {}:'.format(
        order), len(feyn_graphs))
    print('n_{feyn} <= n_{hhz} * 2^n =', (2**order * len(distinct_hhz_graphs)))
    assert len(feyn_graphs) <= 2**order * len(distinct_hhz_graphs)
    return


# Generates and draws naive (including topologically indistinct)
# connected vacuum Feynman diagrams of the specified order by
# expanding the distinct connected vacuum Hugenholtz diagrams
def generate_bHI_vacuum_diags_from_hhz(order=1, draw=True):
    naive_disconnected_hhz_graphs = get_naive_vacuum_hhz_diags(order)
    naive_hhz_graphs = get_connected_subset(naive_disconnected_hhz_graphs)
    distinct_hhz_graphs = rem_top_equiv_al(naive_hhz_graphs)
    print('Unwrapping Hugenholtz diagrams with 1BI rules...')
    n_verts_feyn = 2 * order
    feyn_graphs = []
    for graph in distinct_hhz_graphs:
        # First, we double all vertex labels in the graph; this makes room for the new vertex labels
        # in such a manner as to preserve our pairwise convention for the bosonic connections
        g_shifted = map_vertices_defaultdict(graph, vmap=map(lambda x: 2*x, graph.keys()))
        expanded_graphs = unwrap_hhz_to_feyn_with_irred_rule(g_shifted, n_verts_feyn, is_irred=is_1BI)
        feyn_graphs.extend(rem_top_equiv_al(expanded_graphs, n_verts_feyn))
    print('Done!')

    if draw:
        draw_bulk_feynmp(feyn_graphs, n_legs=0, fside='left',
                         savename='feyn_vacuum_diagrams_bHI_n='+str(order)+'.tex')

    print('{}! = {}'.format(2*order, math.factorial(2*order)))
    print('Number of naive Hugenholtz vacuum diagrams of order {}:'.format(
        order), len(naive_disconnected_hhz_graphs))
    print('Number of naive connected Hugenholtz vacuum diagrams of order {}:'.format(
        order), len(naive_hhz_graphs))
    print('Number of topologically distinct connected Hugenholtz vacuum diagrams of order {}:'.format(
        order), len(distinct_hhz_graphs))
    print('Number of bHI (HEG) Feynman vacuum diagrams of order {}:'.format(
        order), len(feyn_graphs))
    print('n_{feyn} <= n_{hhz} * 2^n =', (2**order * len(distinct_hhz_graphs)))
    assert len(feyn_graphs) <= 2**order * len(distinct_hhz_graphs)
    return


def generate_diff_GW_BSE2_charge_poln_graphs(order, use_hhz=True, draw=False, save=True, g_fmt='pg', save_name=None, verbose=False):
    # NOTE: We use the defaultdict structure for the adjacency list format,
    #       for compatibility with other functions in this library
    if g_fmt not in ['al', 'pg', 'sel']:
        raise ValueError("Graph save format must be either the permutation group ('pg'), " +
                         "adjacency list ('al'), or split edge list ('sel') representation!")
    # Define a default save name
    if save_name is None:
        save_name = 'charge_poln_diags_diff_gw_bse2.npz'
    # There are 2(n - 1) vertices to the generating vacuum diagrams if n is the polarization diagram order
    n_verts_vac_feyn = 2 * (order - 1)
    n_verts_poln = 2 * order
    # The BSE2 approximation, by construction, is exact up to 2nd order,
    # so the set of missing diagrams, (GW / BSE2)_{n<=2} is empty
    if order <= 2:
        print(f'By construction, the BSE2 approximation is exact '
              'up to 2nd order (no diagrams for n = {order})!')
        return {}
    # Otherwise, there are 2(n - 1) vertices to the base
    # vacuum diagrams if n is the polarization diagram order;
    # derive the polarization graphs from these by gluing two legs
    else:
        if use_hhz:
            n_verts_vac_hhz = int(n_verts_vac_feyn / 2)
            print('Generating Hugenholtz diagrams...')
            distinct_hhz_graphs = rem_top_equiv_al(get_connected_subset(get_naive_vacuum_hhz_diags(n_verts_vac_hhz)))
            print('Done!\n')
            print('Unwrapping Hugenholtz diagrams with 1BI rules...')
            distinct_vacuum_graphs_1BI = []
            for graph in distinct_hhz_graphs:
                # First, we double all vertex labels in the graph; this makes room for the new vertex labels
                # in such a manner as to preserve our pairwise convention for the bosonic connections
                g_shifted = map_vertices_defaultdict(graph, vmap=map(lambda x: 2*x, graph.keys()))
                # Then, we recursively unwrap the Hugenholtz diagrams into the set of all contained Feynman diagrams
                expanded_graphs = unwrap_hhz_to_feyn_with_irred_rule(g_shifted, n_verts_vac_feyn, is_irred=is_1BI)
                distinct_vacuum_graphs_1BI.extend(rem_top_equiv_al(expanded_graphs, n_verts_vac_feyn))
            print('Done!\n')
        else:
            # Generate all fermionic connections
            psi_all = get_feyn_vacuum_perms(n_verts_vac_feyn)
            # Build all the naive vacuum graphs
            all_vacuum_graphs = []
            for i in range(psi_all.shape[0]):
                all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
            all_vacuum_graphs = np.asarray(all_vacuum_graphs)
            # Get all distinct (1BI) vacuum graphs
            distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
            distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)

        # Now, get all naive (PBI) polarization graphs using the distinct 1BI vacuum graphs
        poln_graphs_1BI = get_poln_graphs(distinct_vacuum_graphs_1BI)
        
        # Get all naive 1BI + 2BI + 2FI (bold) Feynman diagrams
        poln_graphs_bold = get_bold_subset(poln_graphs_1BI, diag_type='poln', legs=[0, 1])

        # Finally, get the distinct subset of these graphs
        gw_poln_graphs = rem_top_equiv_al(poln_graphs_bold, fixed_pts=[0, 1])

        # Compute the set difference of GW and BSE2 diagrams at this order, modulo topological equivalence
        bse2_poln_graphs = get_BSE2_graphs(order=order, diag_type='poln')
        diff_gw_bse2_poln_graphs = get_diag_set_diff(gw_poln_graphs, bse2_poln_graphs)

        # The polarization is a two-leg correlation function,
        # and we always fix the COM coordinate by convention
        n_legs = 2
        n_fixed = 1
        # Number of integrated variables of each type; wlog, we work in COM coordinates, and
        # hence place one vertex (the incoming leg for correlation functions) at the origin
        has_legs = (n_legs > 0)
        # The number of internal boson lines is (order - 1) for polarization
        # diagrams, and we then include the two external lines in the count
        # n_intn = order + 1
        n_intn = order - 1
        # A dynamic interaction means that the times of each vertex
        # are integrated independently, less one (for the COM coord)
        n_times = n_verts_poln - n_fixed
        # We have one integrated position per vertex for a non-local
        # interaciton, less fixed vertices (one for COM coord)
        n_posns = n_verts_poln - n_fixed
        # Get the (number of) fermion loops in each polarization graph
        n_diags = len(diff_gw_bse2_poln_graphs)
        n_loops = np.zeros(n_diags, dtype=int)
        loops = np.zeros(n_diags, dtype=object)
        neighbors = np.zeros(n_diags, dtype=object)
        for i in range(n_diags):
            loops[i] = get_cycles(diff_gw_bse2_poln_graphs[i])
            n_loops[i] = len(loops[i])
            neighbors[i] = get_nearest_neighbor_list(
                graph=diff_gw_bse2_poln_graphs[i],
                sort=True,
                signed=True
            )
        # Calculate the maximum number of spins at this order
        n_spins_max = int(np.max(n_loops))
        # The symmetry factor P = 1 for all correlation functions!
        symm_factors = np.ones(n_diags, dtype=int).tolist()
        # Store all the graph information in a dictionary
        graph_info = {
            'has_legs': has_legs,
            'n_legs': n_legs,
            'n_fixed': n_fixed,
            'n_intn': n_intn,
            'n_verts': n_verts_poln,
            'n_times': n_times,
            'n_posns': n_posns,
            'n_spins_max': n_spins_max,
            'n_diags': n_diags,
            'n_loops': n_loops,
            'loops': loops,
            'neighbors': neighbors,
            'symm_factors': symm_factors,
        }
        # Build an .npz file containing the graph information / relevant params
        graph_info[f'graphs_{g_fmt}'] = []
        for i in range(n_diags):
            # Keep the graphs in the adjacency list representation
            if g_fmt == 'al':
                this_graph = diff_gw_bse2_poln_graphs[i]
            # Converts to the split edge list representation
            elif g_fmt == 'sel':
                this_graph = graph_al_to_split_el(
                    diff_gw_bse2_poln_graphs[i])
            # Converts to the permutation group representation
            else:
                this_graph = graph_al_to_pg(diff_gw_bse2_poln_graphs[i], flag_mirrored=False)
            # Add this diagram to the .npz file
            graph_info[f'graphs_{g_fmt}'].append(this_graph)
            # graph_info['graph_{}_{}'.format(i, g_fmt)] = this_graph
        # Optionally save the graph and relevant loop/combinatorial information
        if save:
            # Save the graph information to a .npz file
            np.savez(save_name, **graph_info)
            # Log info about the diagram set cardinalities
            with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
                info_block = (
                    'Number of vacuum vertices: ' + str(n_verts_vac_feyn) + '\n'
                    'Number of polarization vertices: ' +
                    str(n_verts_poln) + '\n'
                    'Diagram order (number of interaction lines plus one): ' +
                    str(order) + '\n'
                    '\nTotal number of disconnected EG diagrams: ' +
                    str(math.factorial(n_verts_vac_feyn)) + '\n'
                    'Number of topologically distinct disconnected EG vacuum diagrams: ' +
                    str(len(distinct_vacuum_graphs)) + '\n'
                    'Number of topologically distinct 1BI (G0, V) vacuum diagrams: ' + str(
                        len(distinct_vacuum_graphs_1BI)) + '\n'
                    'Number of topologically distinct 1BI and bold (H + GW) polarization diagrams: ' + str(
                        len(gw_poln_graphs)) + '\n'
                    'Number of topologically distinct BSE2 polarization diagrams: ' +
                    str(len(bse2_poln_graphs)) + '\n'
                    'Number of topologically distinct (H + GW) mod ' +
                    'BSE2 charge polarization diagrams: ' +
                    str(n_diags) + '\n'
                )
                diagram_file.write(info_block)
        # Draws the diagrams in a latex file using feynmp
        if draw:
            draw_bulk_feynmp(diff_gw_bse2_poln_graphs, n_legs=2,
                             savename=f'charge_poln_n={order}_diff_gw_bse2.tex')
            draw_bulk_feynmp(gw_poln_graphs, n_legs=2,
                             savename=f'charge_poln_n={order}_gw.tex')
            draw_bulk_feynmp(bse2_poln_graphs, n_legs=2,
                             savename=f'charge_poln_n={order}_bse2.tex')
        return graph_info


def generate_diff_GW_BSE2_self_energy_graphs(order, use_hhz=True, draw=False, save=True,
                                            g_fmt='pg', save_name=None, verbose=False):
    # NOTE: We use the defaultdict structure for the adjacency list format,
    #       for compatibility with other functions in this library
    if g_fmt not in ['al', 'pg', 'sel']:
        raise ValueError(
            "Graph save format must be either the permutation group ('pg'), " +
            "adjacency list ('al'), or split edge list ('sel') representation!")
    # Define a default save name
    if save_name is None:
        save_name = 'charge_poln_diags_diff_gw_bse2.npz'
    # There are 2n vertices to the generating vacuum diagrams if n is the self-energy diagram order
    n_verts = 2 * order
    # The BSE2 approximation, by construction, is exact up to 2nd order,
    # so the set of missing diagrams, (GW / BSE2)_{n<=2} is empty
    if order <= 2:
        print('By construction, the BSE2 approximation is exact up to 2nd order (no diagrams for n = {})!'.format(order))
        return {}
    # Otherwise, there are 2(n - 1) vertices to the base
    # vacuum diagrams if n is the polarization diagram order;
    # derive the polarization graphs from these by gluing two legs
    else:
        if use_hhz:
            n_verts_hhz = order
            print('Generating Hugenholtz diagrams...')
            distinct_hhz_graphs = rem_top_equiv_al(get_connected_subset(get_naive_vacuum_hhz_diags(n_verts_hhz)))
            print('Done!\n')
            print('Unwrapping Hugenholtz diagrams with 1BI rules...')
            distinct_vacuum_graphs_1BI = []
            for graph in distinct_hhz_graphs:
                # First, we double all vertex labels in the graph; this makes room for the new vertex labels
                # in such a manner as to preserve our pairwise convention for the bosonic connections
                g_shifted = map_vertices_defaultdict(
                    graph, vmap=map(lambda x: 2*x, graph.keys()))
                # Then, we recursively unwrap the Hugenholtz diagrams into the set of all contained Feynman diagrams
                expanded_graphs = unwrap_hhz_to_feyn_with_irred_rule(g_shifted, n_verts, is_irred=is_1BI)
                distinct_vacuum_graphs_1BI.extend(rem_top_equiv_al(expanded_graphs))
            print('Done!\n')
        else:
            # Generate all fermionic connections
            psi_all = get_feyn_vacuum_perms(n_verts)
            # Build all the naive vacuum graphs
            all_vacuum_graphs = []
            for i in range(psi_all.shape[0]):
                all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
            all_vacuum_graphs = np.asarray(all_vacuum_graphs)
            # Get all distinct (1BI) vacuum graphs
            distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
            distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)

        # Now, get all naive (PBI) self-energy graphs using the distinct 1BI vacuum graphs
        self_en_graphs_1BI = get_self_energy_graphs(distinct_vacuum_graphs_1BI)

        # Get all naive 1BI + 2BI + 2FI (bold) Feynman diagrams
        self_en_graphs_bold = get_bold_subset(self_en_graphs_1BI, diag_type='self_en', legs=[0, 1])

        # Finally, get the distinct subset of these graphs
        gw_self_en_graphs = rem_top_equiv_al(self_en_graphs_bold, fixed_pts=[0, 1])

        # Compute the set difference of GW mod BSE2 diagrams at this order, modulo topological equivalence
        bse2_self_en_graphs = get_BSE2_graphs(order=order, diag_type='self_en')
        diff_gw_bse2_self_en_graphs = get_diag_set_diff(gw_self_en_graphs, bse2_self_en_graphs)

        # Now, relabel all graphs to fit the external leg conventions
        bse2_self_en_graphs = enforce_self_en_ext_pair_convn(bse2_self_en_graphs)
        diff_gw_bse2_self_en_graphs = enforce_self_en_ext_pair_convn(diff_gw_bse2_self_en_graphs)

        # NOTE: To be implemented later: swap internal vertices until we fit the
        #       internal leg (alternating) convention (not necessary up to third order)

        # Check that the graphs conform to the alternating convention
        for g in diff_gw_bse2_self_en_graphs:
            for v in g.keys()[4::2]:
                assert (v + 1) in g[v] and 'b' in g[v][v + 1]

        # The self-energy is a two-leg correlation function,
        # and we always fix the COM coordinate by convention
        n_legs = 2
        n_fixed = 1
        # Number of integrated variables of each type; wlog, we work in COM coordinates, and
        # hence place one vertex (the incoming leg for correlation functions) at the origin
        has_legs = (n_legs > 0)
        # The number of internal boson lines is (order - 1) for self-energy
        # diagrams, and we then include the two external lines in the count
        n_intn = order
        # A dynamic interaction means that the times of each vertex
        # are integrated independently, less one (for the COM coord)
        n_times = n_verts - n_fixed
        # We have one integrated position per vertex for a non-local
        # interaciton, less fixed vertices (one for COM coord)
        n_posns = n_verts - n_fixed
        # Get the (number of) fermion loops in each self-energy graph
        n_diags = len(diff_gw_bse2_self_en_graphs)
        n_loops = np.zeros(n_diags, dtype=int)
        loops = np.zeros(n_diags, dtype=object)
        neighbors = np.zeros(n_diags, dtype=object)
        for i in range(n_diags):
            loops[i] = get_cycles(diff_gw_bse2_self_en_graphs[i])
            n_loops[i] = len(loops[i])
            neighbors[i] = get_nearest_neighbor_list(
                graph=diff_gw_bse2_self_en_graphs[i],
                sort=True,
                signed=True
            )
        # Calculate the maximum number of spins at this order
        n_spins_max = int(np.max(n_loops))
        # The symmetry factor P = 1 for all correlation functions!
        symm_factors = np.ones(n_diags, dtype=int).tolist()
        # Store all the graph information in a dictionary
        graph_info = {
            'has_legs': has_legs,
            'n_legs': n_legs,
            'n_fixed': n_fixed,
            'n_intn': n_intn,
            'n_verts': n_verts,
            'n_times': n_times,
            'n_posns': n_posns,
            'n_spins_max': n_spins_max,
            'n_diags': n_diags,
            'n_loops': n_loops,
            'loops': loops,
            'neighbors': neighbors,
            'symm_factors': symm_factors,
        }
        # Build an .npz file containing the graph information / relevant params
        graph_info[f'graphs_{g_fmt}'] = []
        for i in range(n_diags):
            # Keep the graphs in the adjacency list representation
            if g_fmt == 'al':
                this_graph = diff_gw_bse2_self_en_graphs[i]
            # Converts to the split edge list representation
            elif g_fmt == 'sel':
                this_graph = graph_al_to_split_el(
                    diff_gw_bse2_self_en_graphs[i])
            # Converts to the permutation group representation
            else:
                this_graph = graph_al_to_pg(diff_gw_bse2_self_en_graphs[i])
            # Add this diagram to the .npz file
            graph_info[f'graphs_{g_fmt}'].append(this_graph)
            # graph_info['graph_{}_{}'.format(i, g_fmt)] = this_graph
        # Optionally save the graph and relevant loop/combinatorial information
        if save:
            # Save the graph information to a .npz file
            np.savez(save_name, **graph_info)
            # Log info about the diagram set cardinalities
            with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
                info_block = (
                    'Number of vacuum vertices: ' + str(n_verts) + '\n'
                    'Number of self-energy vertices: ' +
                    str(n_verts) + '\n'
                    'Diagram order (number of interaction lines plus one): ' +
                    str(order) + '\n'
                    '\nTotal number of disconnected EG diagrams: ' +
                    str(math.factorial(n_verts)) + '\n'
                    'Number of topologically distinct 1BI (G0, V) vacuum diagrams: ' + str(
                        len(distinct_vacuum_graphs_1BI)) + '\n'
                    'Number of topologically distinct 1BI and bold (H + GW) self-energy diagrams: ' + str(
                        len(gw_self_en_graphs)) + '\n'
                    'Number of topologically distinct BSE2 self-energy diagrams: ' +
                    str(len(bse2_self_en_graphs)) + '\n'
                    'Number of topologically distinct (H + GW) mod ' +
                    'BSE2 charge self-energy diagrams: ' +
                    str(n_diags) + '\n'
                )
                diagram_file.write(info_block)
        # Draws the diagrams in a latex file using feynmp
        if draw:
            draw_bulk_feynmp(diff_gw_bse2_self_en_graphs, n_legs=2,
                             savename='self_en_n='+str(order)+'_diff_gw_bse2.tex')
            draw_bulk_feynmp(bse2_self_en_graphs, n_legs=2,
                             savename='self_en_n='+str(order)+'_bse2.tex')
        return graph_info


# Generate all topologically distinct nth-order vacuum diagrams in the bare series (no irreducibility rules).
def generate_disconnected_bare_vacuum_diagrams(order=1, save_name='vacuum_diagrams.npz', draw=True):
    if order == 0:
        print('No graphs to draw!')
        return (0, 0)
    # There are 2n vertices to a diagram, if n is the diagram order
    n_verts = 2 * order
    # Generate all fermionic connections
    psi_all = get_feyn_vacuum_perms(n_verts)
    # Build all the naive vacuum graphs
    all_vacuum_graphs = []
    for i in range(psi_all.shape[0]):
        all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
    all_vacuum_graphs = np.asarray(all_vacuum_graphs)

    # Get all distinct (1BI) vacuum graphs
    distinct_vacuum_graphs = list(rem_top_equiv_al(all_vacuum_graphs))
    # Get the number of fermion loops in each vacuum graph
    n_loops = np.zeros(len(distinct_vacuum_graphs), dtype=int)
    loops = np.zeros(len(distinct_vacuum_graphs), dtype=object)
    for i in range(len(distinct_vacuum_graphs)):
        loops[i] = get_cycles(distinct_vacuum_graphs[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_vacuum_graphs)):
        psi, phi = graph_al_to_pg(distinct_vacuum_graphs[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)
    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Diagram order (number of interaction lines): ' +
            str(order) + '\n'
            '\nTotal number of disconnected vacuum diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct disconnected vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_vacuum_graphs, n_legs=0, fside='right',
                         savename='vacuum_n='+str(order)+'_disconnected_bare.tex')
    # Return the number of diagrams generated, and the number of vertices at this order
    return (len(distinct_vacuum_graphs), n_verts)


# Generate all topologically distinct nth-order vacuum diagrams in the bare series (no irreducibility rules).
def generate_bare_vacuum_diagrams(order=1, save_name='vacuum_diagrams.npz', draw=True):
    if order == 0:
        print('No graphs to draw!')
        return (0, 0)
    # There are 2n vertices to a diagram, if n is the diagram order
    n_verts = 2 * order
    # Generate all fermionic connections
    psi_all = get_feyn_vacuum_perms(n_verts)
    # Build all the naive vacuum graphs
    all_vacuum_graphs = []
    for i in range(psi_all.shape[0]):
        all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
    all_vacuum_graphs = np.asarray(all_vacuum_graphs)
    # Get all distinct (1BI) vacuum graphs
    distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
    distinct_vacuum_graphs_connected = get_connected_subset(distinct_vacuum_graphs)
    # Get the number of fermion loops in each vacuum graph
    n_loops = np.zeros(len(distinct_vacuum_graphs_connected), dtype=int)
    loops = np.zeros(len(distinct_vacuum_graphs_connected), dtype=object)
    for i in range(len(distinct_vacuum_graphs_connected)):
        loops[i] = get_cycles(distinct_vacuum_graphs_connected[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_vacuum_graphs_connected)):
        psi, phi = graph_al_to_pg(distinct_vacuum_graphs_connected[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)
    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Diagram order (number of interaction lines): ' +
            str(order) + '\n'
            '\nTotal number of disconnected vacuum diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct disconnected vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs)) + '\n'
            'Number of topologically distinct connected (bare) vacuum diagrams: ' + str(
                len(distinct_vacuum_graphs_connected)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_vacuum_graphs_connected, n_legs=0,
                         fside='right', savename='vacuum_n='+str(order)+'_bare.tex')
    # Return the number of diagrams generated, and the number of vertices at this order
    return (len(distinct_vacuum_graphs_connected), n_verts)


# Generate all topologically distinct nth-order vacuum diagrams with bold Hartree (1BI) irreducibility rules.
def generate_bHI_vacuum_diagrams(order=1, save_name='vacuum_diagrams.npz', draw=True):
    if order == 0:
        print('No graphs to draw!')
        return (0, 0)
    # There are 2n vertices to a diagram, if n is the diagram order
    n_verts = 2 * order
    # Generate all fermionic connections
    psi_all = get_feyn_vacuum_perms(n_verts)
    # Build all the naive vacuum graphs
    all_vacuum_graphs = []
    for i in range(psi_all.shape[0]):
        all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
    all_vacuum_graphs = np.asarray(all_vacuum_graphs)
    # Get all distinct (1BI) vacuum graphs
    distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
    distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)
    # Get the number of fermion loops in each vacuum graph
    n_loops = np.zeros(len(distinct_vacuum_graphs_1BI), dtype=int)
    loops = np.zeros(len(distinct_vacuum_graphs_1BI), dtype=object)
    for i in range(len(distinct_vacuum_graphs_1BI)):
        loops[i] = get_cycles(distinct_vacuum_graphs_1BI[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_vacuum_graphs_1BI)):
        psi, phi = graph_al_to_pg(distinct_vacuum_graphs_1BI[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)
    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Diagram order (number of interaction lines): ' +
            str(order) + '\n'
            '\nTotal number of disconnected vacuum diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct disconnected vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs)) + '\n'
            'Number of topologically distinct bHI (1BI) vacuum diagrams: ' + str(
                len(distinct_vacuum_graphs_1BI)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_vacuum_graphs_1BI, n_legs=0,
                         fside='right', savename='vacuum_n='+str(order)+'_bHI.tex')
    # Return the number of diagrams generated, and the number of vertices at this order
    return (len(distinct_vacuum_graphs_1BI), n_verts)


# Generate all topologically distinct nth-order vacuum diagrams with HF irreducibility
# rules; since the bold Hatree term is trivially obtained, by HFI we mean bHI + FI,
# i.e., no general Hartree terms at all, and no bare Fock insertions (implies no scHF insertions).
def generate_HFI_vacuum_diagrams(order=1, save_name='vacuum_diagrams.npz', draw=True):
    if order == 0:
        print('No graphs to draw!')
        return (0, 0)
    # There are 2n vertices to a diagram, if n is the diagram order
    n_verts = 2 * order
    # Generate all fermionic connections
    psi_all = get_feyn_vacuum_perms(n_verts)
    # Build all the naive vacuum graphs
    all_vacuum_graphs = []
    for i in range(psi_all.shape[0]):
        all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
    all_vacuum_graphs = np.asarray(all_vacuum_graphs)
    # Get all distinct (1BI) vacuum graphs
    distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
    distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)
    distinct_vacuum_graphs_HFI = get_FI_subset(distinct_vacuum_graphs_1BI)
    # Get the number of fermion loops in each vacuum graph
    n_loops = np.zeros(len(distinct_vacuum_graphs_HFI), dtype=int)
    loops = np.zeros(len(distinct_vacuum_graphs_HFI), dtype=object)
    for i in range(len(distinct_vacuum_graphs_HFI)):
        loops[i] = get_cycles(distinct_vacuum_graphs_HFI[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_vacuum_graphs_HFI)):
        psi, phi = graph_al_to_pg(distinct_vacuum_graphs_HFI[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)
    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Diagram order (number of interaction lines): ' +
            str(order) + '\n'
            '\nTotal number of disconnected vacuum diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct disconnected vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs)) + '\n'
            'Number of topologically distinct 1BI vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs_1BI)) + '\n'
            'Number of topologically distinct HFI vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs_HFI)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_vacuum_graphs_HFI, n_legs=0,
                         fside='right', savename='vacuum_n='+str(order)+'_HFI.tex')
    # Return the number of diagrams generated, and the number of vertices at this order
    return (len(distinct_vacuum_graphs_HFI), n_verts)


# Generate all topologically distinct nth-order vacuum diagrams with bold HF irreducibility rules.
def generate_bHFI_vacuum_diagrams(order=1, save_name='vacuum_diagrams.npz', draw=True):
    if order == 0:
        print('No graphs to draw!')
        return (0, 0)
    # There are 2n vertices to a diagram, if n is the diagram order
    n_verts = 2 * order
    # Generate all fermionic connections
    psi_all = get_feyn_vacuum_perms(n_verts)
    # Build all the naive vacuum graphs
    all_vacuum_graphs = []
    for i in range(psi_all.shape[0]):
        all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
    all_vacuum_graphs = np.asarray(all_vacuum_graphs)
    # Get all distinct (1BI) vacuum graphs
    distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
    distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)
    distinct_vacuum_graphs_bHFI = get_bFI_subset(distinct_vacuum_graphs_1BI, diag_type='vacuum', legs=[0, 1])
    # Get the number of fermion loops in each vacuum graph
    n_loops = np.zeros(len(distinct_vacuum_graphs_bHFI), dtype=int)
    loops = np.zeros(len(distinct_vacuum_graphs_bHFI), dtype=object)
    for i in range(len(distinct_vacuum_graphs_bHFI)):
        loops[i] = get_cycles(distinct_vacuum_graphs_bHFI[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_vacuum_graphs_bHFI)):
        psi, phi = graph_al_to_pg(distinct_vacuum_graphs_bHFI[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)
    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Diagram order (number of interaction lines): ' +
            str(order) + '\n'
            '\nTotal number of disconnected vacuum diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct disconnected vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs)) + '\n'
            'Number of topologically distinct 1BI vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs_1BI)) + '\n'
            'Number of topologically distinct bold HFI vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs_bHFI)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_vacuum_graphs_bHFI, n_legs=0,
                         fside='right', savename='vacuum_n='+str(order)+'_bHFI.tex')
    # Return the number of diagrams generated, and the number of vertices at this order
    return (len(distinct_vacuum_graphs_bHFI), n_verts)


# Generate all topologically distinct nth-order vacuum diagrams with H + RPA (G0W0) irreducibility rules.
def generate_HPBI_vacuum_diagrams(order=1, use_hhz=True, draw=True, save_name='vacuum_diagrams.npz'):
    if order == 0:
        print('No graphs to draw!')
        return (0, 0)
    # There are 2n vertices to a diagram, if n is the diagram order
    n_verts_feyn = 2 * order
    if use_hhz:
        print('Generating Hugenholtz diagrams...')
        distinct_hhz_graphs = rem_top_equiv_al(get_connected_subset(
            get_naive_vacuum_hhz_diags(order)))
        print('Done!\n')
        print('Unwrapping Hugenholtz diagrams with 1BI + PBI (G0W0) rules...')
        distinct_vacuum_graphs_HPBI = []
        for graph in distinct_hhz_graphs:
            # First, we double all vertex labels in the graph; this makes room for the new vertex labels
            # in such a manner as to preserve our pairwise convention for the bosonic connections
            g_shifted = map_vertices_defaultdict(
                graph, vmap=map(lambda x: 2*x, graph.keys()))
            expanded_graphs = unwrap_hhz_to_feyn_with_irred_rule(
                g_shifted, n_verts_feyn, is_irred=lambda g: is_1BI(g) and is_PBI(g))
            if len(expanded_graphs) > 0:
                distinct_vacuum_graphs_HPBI.extend(
                    rem_top_equiv_al(expanded_graphs, n_verts_feyn))
        print('Done!')
    else:
        # Generate all fermionic connections
        psi_all = get_feyn_vacuum_perms(n_verts_feyn)
        # Build all the naive vacuum graphs
        all_vacuum_graphs = []
        for i in range(psi_all.shape[0]):
            all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
        all_vacuum_graphs = np.asarray(all_vacuum_graphs)
        # Get all distinct (1BI) vacuum graphs
        distinct_vacuum_graphs_1BI = get_1BI_subset(rem_top_equiv_al(all_vacuum_graphs))
        distinct_vacuum_graphs_HPBI = get_PBI_subset(distinct_vacuum_graphs_1BI)

    # Get the number of fermion loops in each vacuum graph
    n_loops = np.zeros(len(distinct_vacuum_graphs_HPBI), dtype=int)
    loops = np.zeros(len(distinct_vacuum_graphs_HPBI), dtype=object)
    for i in range(len(distinct_vacuum_graphs_HPBI)):
        loops[i] = get_cycles(distinct_vacuum_graphs_HPBI[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_vacuum_graphs_HPBI)):
        psi, phi = graph_al_to_pg(distinct_vacuum_graphs_HPBI[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)
    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts_feyn) + '\n'
            'Diagram order (number of interaction lines): ' +
            str(order) + '\n'
            '\nTotal number of disconnected EG diagrams: ' +
            str(math.factorial(n_verts_feyn)) + '\n'
            # 'Number of topologically distinct 1BI (G0, V) vacuum diagrams: ' + str(len(distinct_vacuum_graphs_1BI)) + '\n'
            'Number of topologically distinct 1BI and PBI (G0, W0) vacuum diagrams: ' + str(
                len(distinct_vacuum_graphs_HPBI)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_vacuum_graphs_HPBI, n_legs=0,
                         fside='right', savename='vacuum_n='+str(order)+'_HPBI.tex')
    # Return the number of diagrams generated, and the number of vertices at this order
    return (len(distinct_vacuum_graphs_HPBI), n_verts_feyn)


# Generate all topologically distinct nth-order vacuum diagrams with bold (G W) irreducibility rules,
# i.e., 1BI + 2BI + 2FI diagrams.
def generate_bold_vacuum_diagrams(order=1, use_hhz=True, draw=True, save_name='bold_vacuum_diagrams.npz'):
    if order == 0:
        print('No graphs to draw!')
        return (0, 0)
    # There are 2n vertices to a diagram, if n is the diagram order
    n_verts_feyn = 2 * order
    if use_hhz:
        print('Generating Hugenholtz vacuum diagrams...')
        distinct_hhz_graphs = rem_top_equiv_al(get_connected_subset(
            get_naive_vacuum_hhz_diags(order)))
        print('Done!\n')
        print('Unwrapping Hugenholtz vacuum diagrams with 1BI + 2BI + 2FI (GW) rules...')
        # n_verts_feyn = 2 * order
        bold_vacuum_graphs = []
        for graph in distinct_hhz_graphs:
            # First, we double all vertex labels in the graph; this makes room for the new vertex labels
            # in such a manner as to preserve our pairwise convention for the bosonic connections
            g_shifted = map_vertices_defaultdict(
                graph, vmap=map(lambda x: 2*x, graph.keys()))
            # Then, unwrap these vshifted Hugenholtz graphs to the corresponding Feynman diagrams,
            # discarding any graphs we encounter which conform to the bold graph reducibility rules
            expanded_graphs = unwrap_hhz_to_feyn_with_irred_rule(
                g_shifted, n_verts_feyn, is_irred=is_bold)
            if len(expanded_graphs) > 0:
                bold_vacuum_graphs.extend(
                    rem_top_equiv_al(expanded_graphs, n_verts_feyn))
        print('Done!\n')
    else:
        # Generate all fermionic connections
        psi_all = get_feyn_vacuum_perms(n_verts_feyn)
        # Build all the naive vacuum graphs
        all_vacuum_graphs = []
        for i in range(psi_all.shape[0]):
            all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
        all_vacuum_graphs = np.asarray(all_vacuum_graphs)
        # Get all 1BI + 2BI + 2FI (bold) Feynman diagrams
        bold_vacuum_graphs = get_bold_subset(get_1BI_subset(
            rem_top_equiv_al(all_vacuum_graphs, n_verts_feyn)))

    # Get the number of fermion loops in each vacuum graph
    n_loops = np.zeros(len(bold_vacuum_graphs), dtype=int)
    loops = np.zeros(len(bold_vacuum_graphs), dtype=object)
    for i in range(len(bold_vacuum_graphs)):
        loops[i] = get_cycles(bold_vacuum_graphs[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(bold_vacuum_graphs)):
        psi, phi = graph_al_to_pg(bold_vacuum_graphs[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)
    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts_feyn) + '\n'
            'Diagram order (number of interaction lines): ' +
            str(order) + '\n'
            '\nTotal number of disconnected EG diagrams: ' +
            str(math.factorial(n_verts_feyn)) + '\n'
            'Number of topologically distinct 1BI, 2BI and 2FI (bold) vacuum diagrams: ' + str(
                len(bold_vacuum_graphs)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(bold_vacuum_graphs, n_legs=0, fside='right',
                         savename='vacuum_n='+str(order)+'_bold.tex')
    # Return the number of diagrams generated, and the number of vertices at this order
    return (len(bold_vacuum_graphs), n_verts_feyn)


# Recursively generate all bare diagrams in the series generated by a
# first-order truncation of Hedin's equations. This is the usual GW
# approximation, i.e., \Sigma = -GW, and P = -GG, reexpanded in terms
# of the bare propagators.
def compare_HEG_and_C1_vacuum_diagrams(order=1, save_name='C1_vacuum_diagrams.npz', draw=True, use_hhz=True):
    if order == 0:
        print('No graphs to draw!')
        return (0, 0)
    # There are 2n vertices to a diagram, if n is the diagram order
    n_verts_feyn = 2 * order
    # Generates all Hugenholtz diagrams and then unwraps them
    if use_hhz:
        print('Generating Hugenholtz diagrams...')
        distinct_hhz_graphs = rem_top_equiv_al(get_connected_subset(
            get_naive_vacuum_hhz_diags(order)))
        print('Done!\n')
        print('Unwrapping Hugenholtz diagrams with 1BI rules...')
        vacuum_graphs_HEG = []
        for graph in distinct_hhz_graphs:
            # First, we double all vertex labels in the graph; this makes room for the new vertex labels
            # in such a manner as to preserve our pairwise convention for the bosonic connections
            g_shifted = map_vertices_defaultdict(
                graph, vmap=map(lambda x: 2*x, graph.keys()))
            # Then, we recursively unwrap the Hugenholtz diagrams into the set of all contained Feynman diagrams
            expanded_graphs = unwrap_hhz_to_feyn_with_irred_rule(
                g_shifted, n_verts_feyn, is_irred=is_1BI)
            vacuum_graphs_HEG.extend(
                rem_top_equiv_al(expanded_graphs, n_verts_feyn))
        print('Done!\n')
    # Generates all Feynman diagrams directly
    else:
        # Generate all fermionic connections
        psi_all = get_feyn_vacuum_perms(n_verts_feyn)
        # Build all the naive vacuum graphs
        all_vacuum_graphs = []
        for i in range(psi_all.shape[0]):
            all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
        all_vacuum_graphs = np.asarray(all_vacuum_graphs)
        # Get all distinct (1BI) vacuum graphs, i.e., the free energy graphs in the HEG
        distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
        vacuum_graphs_HEG = get_1BI_subset(distinct_vacuum_graphs)

    # Self-consistently generate all bare diagrams in the GW approximation
    vacuum_graphs_C1 = get_C1_graphs(order=order, diag_type='vacuum')
    # Now, identify the set of diagrams missing in the C1 series at this order
    vacuum_graphs_diff_HEG_C1 = get_diag_set_diff(vacuum_graphs_HEG, vacuum_graphs_C1)

    # Get the number of fermion loops in each vacuum graph
    n_loops = np.zeros(len(vacuum_graphs_C1), dtype=int)
    loops = np.zeros(len(vacuum_graphs_C1), dtype=object)
    for i in range(len(vacuum_graphs_C1)):
        loops[i] = get_cycles(vacuum_graphs_C1[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(vacuum_graphs_C1)):
        psi, phi = graph_al_to_pg(vacuum_graphs_C1[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)
    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts_feyn) + '\n'
            'Diagram order (number of interaction lines): ' +
            str(order) + '\n'
            '\nTotal number ofdisconnected EG diagrams: ' +
            str(math.factorial(n_verts_feyn)) + '\n'
            'Number of topologically distinct bHI (1BI) vacuum diagrams: ' + str(
                len(vacuum_graphs_HEG)) + '\n'
            'Number of topologically distinct C1 (GW approximation) vacuum diagrams: ' + str(
                len(vacuum_graphs_C1)) + '\n'
            'Number of vacuum diagrams missing in the C1 series (GW approximation): ' + str(
                len(vacuum_graphs_diff_HEG_C1)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(vacuum_graphs_HEG, n_legs=0, fside='right',
                         savename='vacuum_n='+str(order)+'_HEG.tex')
        draw_bulk_feynmp(vacuum_graphs_C1, n_legs=0, fside='right',
                         savename='vacuum_n='+str(order)+'_C1.tex')
        draw_bulk_feynmp(vacuum_graphs_diff_HEG_C1, n_legs=0, fside='right',
                         savename='vacuum_n='+str(order)+'_diff_HEG_C1.tex')
    # Return the number of diagrams generated, and the number of vertices at this order
    return (len(vacuum_graphs_C1), n_verts_feyn)


# Recursively generate all bare diagrams in the series generated by a
# first-order truncation of Hedin's equations. This is the usual GW
# approximation, i.e., \Sigma = -GW, and P = -GG.
def compare_HEG_and_C2_vacuum_diagrams(order=1, save_name='C2_vacuum_diagrams.npz', draw=True, use_hhz=True):
    if order == 0:
        print('No graphs to draw!')
        return (0, 0)
    # There are 2n vertices to a diagram, if n is the diagram order
    n_verts_feyn = 2 * order
    # Generates all Hugenholtz diagrams and then unwraps them
    if use_hhz:
        print('Generating Hugenholtz diagrams...')
        distinct_hhz_graphs = rem_top_equiv_al(get_connected_subset(
            get_naive_vacuum_hhz_diags(order)))
        print('Done!\n')
        print('Unwrapping Hugenholtz diagrams with 1BI rules...')
        vacuum_graphs_HEG = []
        for graph in distinct_hhz_graphs:
            # First, we double all vertex labels in the graph; this makes room for the new vertex labels
            # in such a manner as to preserve our pairwise convention for the bosonic connections
            g_shifted = map_vertices_defaultdict(
                graph, vmap=map(lambda x: 2*x, graph.keys()))
            # Then, we recursively unwrap the Hugenholtz diagrams into the set of all contained Feynman diagrams
            expanded_graphs = unwrap_hhz_to_feyn_with_irred_rule(
                g_shifted, n_verts_feyn, is_irred=is_1BI)
            vacuum_graphs_HEG.extend(
                rem_top_equiv_al(expanded_graphs, n_verts_feyn))
        print('Done!')
    # Generates all Feynman diagrams directly
    else:
        # Generate all fermionic connections
        psi_all = get_feyn_vacuum_perms(n_verts_feyn)
        # Build all the naive vacuum graphs
        all_vacuum_graphs = []
        for i in range(psi_all.shape[0]):
            all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
        all_vacuum_graphs = np.asarray(all_vacuum_graphs)
        # Get all distinct (1BI) vacuum graphs, i.e., the free energy graphs in the HEG
        distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
        vacuum_graphs_HEG = get_1BI_subset(distinct_vacuum_graphs)

    # Self-consistently generate all bare diagrams in the GW approximation
    vacuum_graphs_C2 = get_C2_graphs(order=order, diag_type='vacuum')
    # Now, identify the set of diagrams missing in the C2 series at this order
    vacuum_graphs_diff_HEG_C2 = get_diag_set_diff(vacuum_graphs_HEG, vacuum_graphs_C2)

    # Get the number of fermion loops in each vacuum graph
    n_loops = np.zeros(len(vacuum_graphs_C2), dtype=int)
    loops = np.zeros(len(vacuum_graphs_C2), dtype=object)
    for i in range(len(vacuum_graphs_C2)):
        loops[i] = get_cycles(vacuum_graphs_C2[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(vacuum_graphs_C2)):
        psi, phi = graph_al_to_pg(vacuum_graphs_C2[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)
    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts_feyn) + '\n'
            'Diagram order (number of interaction lines): ' +
            str(order) + '\n'
            '\nTotal number of disconnected EG diagrams: ' +
            str(math.factorial(n_verts_feyn)) + '\n'
            'Number of topologically distinct bHI (1BI) vacuum diagrams: ' + str(
                len(vacuum_graphs_HEG)) + '\n'
            'Number of topologically distinct C2 (2nd-order conserving approximation) vacuum diagrams: ' + str(
                len(vacuum_graphs_C2)) + '\n'
            'Number of vacuum diagrams missing in the C2 series: ' +
            str(len(vacuum_graphs_diff_HEG_C2)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(vacuum_graphs_HEG, n_legs=0, fside='right',
                         savename='vacuum_n='+str(order)+'_HEG.tex')
        draw_bulk_feynmp(vacuum_graphs_C2, n_legs=0, fside='right',
                         savename='vacuum_n='+str(order)+'_C2.tex')
        draw_bulk_feynmp(vacuum_graphs_diff_HEG_C2, n_legs=0, fside='right',
                         savename='vacuum_n='+str(order)+'_diff_HEG_C2.tex')
    # Return the number of diagrams generated, and the number of vertices at this order
    return (len(vacuum_graphs_C2), n_verts_feyn)


# Recursively generate all bare diagrams in the series generated by a
# first-order truncation of Hedin's equations. This is the usual GW
# approximation, i.e., \Sigma = -GW, and P = -GG.
def generate_C1_vacuum_diagrams(order=1, save_name='C1_vacuum_diagrams.npz', draw=True):
    if order == 0:
        print('No graphs to draw!')
        return (0, 0)
    # There are 2n vertices to a diagram, if n is the diagram order
    n_verts = 2 * order
    # Self-consistently generate all bare diagrams in the GW approximation
    vacuum_graphs_C1 = get_C1_graphs(order=order, diag_type='vacuum')
    # Get the number of fermion loops in each vacuum graph
    n_loops = np.zeros(len(vacuum_graphs_C1), dtype=int)
    loops = np.zeros(len(vacuum_graphs_C1), dtype=object)
    for i in range(len(vacuum_graphs_C1)):
        loops[i] = get_cycles(vacuum_graphs_C1[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(vacuum_graphs_C1)):
        psi, phi = graph_al_to_pg(vacuum_graphs_C1[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)
    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Diagram order (number of interaction lines): ' +
            str(order) + '\n'
            '\nTotal number of disconnected EG diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct GW approximation (C1, 1st-order conserving approximation) vacuum diagrams: ' + str(
                len(vacuum_graphs_C1)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(vacuum_graphs_C1, n_legs=0, fside='right',
                         savename='vacuum_n='+str(order)+'_C1.tex')
    # Return the number of diagrams generated, and the number of vertices at this order
    return (len(vacuum_graphs_C1), n_verts)


# Recursively generate all bare diagrams in the series generated by a
# second-order truncation of Hedin's equations, that is, GW with
# first-order (conserving!) vertex corrections included.
def generate_C2_vacuum_diagrams(order=1, save_name='C2_vacuum_diagrams.npz', draw=True):
    if order == 0:
        print('No graphs to draw!')
        return (0, 0)
    # There are 2n vertices to a diagram, if n is the diagram order
    n_verts = 2 * order

    # Self-consistently generate all bare diagrams in the GW approximation
    distinct_vacuum_graphs_C2 = get_C2_graphs(order=order, diag_type='vacuum')

    # Get the number of fermion loops in each vacuum graph
    n_loops = np.zeros(len(distinct_vacuum_graphs_C2), dtype=int)
    loops = np.zeros(len(distinct_vacuum_graphs_C2), dtype=object)
    for i in range(len(distinct_vacuum_graphs_C2)):
        loops[i] = get_cycles(distinct_vacuum_graphs_C2[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_vacuum_graphs_C2)):
        psi, phi = graph_al_to_pg(distinct_vacuum_graphs_C2[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)
    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Diagram order (number of interaction lines): ' +
            str(order) + '\n'
            '\nTotal number of disconnected EG diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct C2 (2nd-order conserving approximation) vacuum diagrams: ' + str(
                len(distinct_vacuum_graphs_C2)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_vacuum_graphs_C2, n_legs=0,
                         fside='right', savename='vacuum_n='+str(order)+'_C2.tex')
    # Return the number of diagrams generated, and the number of vertices at this order
    return (len(distinct_vacuum_graphs_C2), n_verts)


# Generate all topologically distinct nth-order vacuum diagrams with bare HFPBI
# irreducibility rules, i.e., no bare Fock self-energy or polarization bubble insertions.
def generate_HFPBI_vacuum_diagrams(order=1, save_name='vacuum_diagrams.npz', draw=True, profile=False):
    if order == 0:
        print('No graphs to draw!')
        return (0, 0)

    pr = None
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    # There are 2n vertices to a diagram, if n is the diagram order
    n_verts = 2 * order
    # Generate all fermionic connections
    psi_all = get_feyn_vacuum_perms(n_verts)

    # Build all the naive vacuum graphs
    all_vacuum_graphs = []
    for i in range(psi_all.shape[0]):
        all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
    all_vacuum_graphs = np.asarray(all_vacuum_graphs)
    # Get all distinct 1BI vacuum graphs
    distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)

    # Get the distinct HFPBI graphs
    distinct_vacuum_graphs_HFPBI = get_HFPBI_subset(distinct_vacuum_graphs)
    # Get the number of fermion loops in each vacuum graph
    n_loops = np.zeros(len(distinct_vacuum_graphs_HFPBI), dtype=int)
    loops = np.zeros(len(distinct_vacuum_graphs_HFPBI), dtype=object)
    for i in range(len(distinct_vacuum_graphs_HFPBI)):
        loops[i] = get_cycles(distinct_vacuum_graphs_HFPBI[i])
        n_loops[i] = len(loops[i])

    if profile:
        pr.disable()
        print_stream = io.StringIO()
        file_stream = open(
            'generate_HFPBI_vacuum_diagrams_n='+str(order)+'.profile', 'w+')
        for ostream in [file_stream, print_stream]:
            ps = pstats.Stats(pr, stream=ostream).sort_stats('time')
            ps.print_stats()
        print(print_stream.getvalue())
        print('n_naive_graphs: ', len(all_vacuum_graphs))
        print('n_distinct_graphs: ', len(distinct_vacuum_graphs))

    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_vacuum_graphs_HFPBI)):
        psi, phi = graph_al_to_pg(distinct_vacuum_graphs_HFPBI[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)
    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Diagram order (number of interaction lines): ' +
            str(order) + '\n'
            '\nTotal number of disconnected EG diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct disconnected EG vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs)) + '\n'
            'Number of topologically distinct HFPBI vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs_HFPBI)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_vacuum_graphs_HFPBI, n_legs=0,
                         fside='right', savename='vacuum_n='+str(order)+'_HFPBI.tex')
    # Return the number of diagrams generated, and the number of vertices at this order
    return (len(distinct_vacuum_graphs_HFPBI), n_verts)


# Generate all nth-order charge polarization diagrams with bold
# Hartree irreducibility rules. Attaches legs to the (n-1)th-order
# vacuum diagrams in all possible ways, then removes duplicates.
def generate_bHI_charge_poln_diagrams(order=1, save_name='charge_poln_diagrams.npz', draw=True):
    # Number of vertices in the vacuum diagrams at the next-lowest
    # order, from which we derive the polarization diagrams
    n_verts_vacuum = 2 * (order - 1)
    n_verts_poln = 2 * order
    # Hard-coded diagram generation for the zeroth order bubble diagram
    # (since there are no vacuum diagrams with zero interaction lines)
    if n_verts_vacuum == 0:
        # There is one diagram, and one fermion loop in the diagram
        n_diags_1BI = 1
        n_loops = np.array([1.0])
        loops = np.array([[0, 1]])
        # Simple polarization bubble is G * G
        psi = np.array([[1, 0]], dtype=int)
        # No internal boson lines (the external vertex flag is -1)
        phi = np.array([[-1, -1]], dtype=int)
        # Make the graph in the adjacency list representations
        distinct_poln_graphs_1BI = defaultdict(lambda: defaultdict(list))
        distinct_poln_graphs_1BI[0][1].append('f')
        distinct_poln_graphs_1BI[1][0].append('f')
        distinct_poln_graphs_1BI = [distinct_poln_graphs_1BI]
        # Save the graph and relevant loop/combinatorial information
        save_contents = {'n_loops': n_loops, 'loops': loops}
        # Add this set of diagram connections to the .npz file
        save_contents['psi_0'] = psi
        save_contents['phi_0'] = phi
        # Save the graph information to a .npz file
        np.savez(save_name, **save_contents)
        with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
            info_block = (
                'Number of vacuum vertices: ' + str(n_verts_vacuum) + '\n'
                'Number of polarization vertices: ' +
                str(n_verts_poln) + '\n'
                'Diagram order (number of interaction lines plus one): ' +
                str(order) + '\n'
                '\nTotal number of disconnected EG diagrams: 0' + '\n'
                'Number of topologically distinct 1BI (bHI) polarization diagrams: 1' + '\n'
            )
            diagram_file.write(info_block)
    # Otherwise, there are 2(n - 1) vertices to the base
    # vacuum diagrams if n is the polarization diagram order;
    # derive the polarization graphs from these by gluing two legs
    else:
        # Generate all fermionic connections
        psi_all = get_feyn_vacuum_perms(n_verts_vacuum)
        # Build all the naive vacuum graphs
        all_vacuum_graphs = []
        for i in range(psi_all.shape[0]):
            all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
        all_vacuum_graphs = np.asarray(all_vacuum_graphs)
        # Get all distinct (1BI) vacuum graphs
        distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
        distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)
        # Now, get all distinct (1BI) polarization graphs using the distinct 1BI vacuum graphs
        distinct_poln_graphs_1BI = list(rem_top_equiv_al(get_poln_graphs(distinct_vacuum_graphs_1BI), fixed_pts=[0, 1]))
        n_diags_1BI = len(distinct_poln_graphs_1BI)
        # Get the number of fermion loops in each polarization graph
        n_loops = np.zeros(len(distinct_poln_graphs_1BI), dtype=int)
        loops = np.zeros(len(distinct_poln_graphs_1BI), dtype=object)
        for i in range(len(distinct_poln_graphs_1BI)):
            loops[i] = get_cycles(distinct_poln_graphs_1BI[i])
            n_loops[i] = len(loops[i])
        # Unmake the graphs to get numpy arrays for \psi and \phi
        # (i.e., to get the permutation group representations of all graphs)
        save_contents = {'n_loops': n_loops, 'loops': loops}
        for i in range(n_diags_1BI):
            psi, phi = graph_al_to_pg(distinct_poln_graphs_1BI[i])
            # Add this set of diagram connections to the .npz file
            save_contents['psi_' + str(i)] = psi
            save_contents['phi_' + str(i)] = phi
        # Save the graph information to a .npz file
        np.savez(save_name, **save_contents)
        with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
            info_block = (
                'Number of vacuum vertices: ' + str(n_verts_vacuum) + '\n'
                'Number of polarization vertices: ' +
                str(n_verts_poln) + '\n'
                'Diagram order (number of interaction lines plus one): ' +
                str(order) + '\n'
                '\nTotal number of disconnected EG diagrams: ' +
                str(math.factorial(n_verts_vacuum)) + '\n'
                'Number of topologically distinct disconnected EG vacuum diagrams: ' +
                str(len(distinct_vacuum_graphs)) + '\n'
                'Number of topologically distinct 1BI vacuum diagrams: ' +
                str(len(distinct_vacuum_graphs_1BI)) + '\n'
                '\nNumber of topologically distinct 1BI (bHI) polarization diagrams: ' + str(
                    len(distinct_poln_graphs_1BI)) + '\n'
            )
            diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_poln_graphs_1BI, n_legs=2,
                         savename='charge_poln_n='+str(order)+'_bHI.tex')
    # Return the number of diagrams generated, and the
    # number of vertices in each (including external ones)
    return (n_diags_1BI, n_verts_poln)


# Generate all nth-order charge polarization diagrams with HF irreducibility rules;
# since the bold Hatree term is trivially obtained, by HFI we mean bHI + FI, i.e.,
# no Hartree terms at all, and nested Hartree-Fock subdiagrams only. Attaches legs to
# the (n-1)th-order vacuum diagrams in all possible ways, then removes duplicates.
def generate_HFI_charge_poln_diagrams(order=1, save_name='charge_poln_diagrams.npz', draw=True):
    # Number of vertices in the vacuum diagrams at the next-lowest
    # order, from which we derive the polarization diagrams
    n_verts_vacuum = 2 * (order - 1)
    n_verts_poln = 2 * order
    # Hard-coded diagram generation for the zeroth order bubble diagram
    # (since there are no vacuum diagrams with zero interaction lines)
    if n_verts_vacuum == 0:
        # There is one diagram, and one fermion loop in the diagram
        n_diags_HFI = 1
        n_loops = np.array([1.0])
        loops = np.array([[0, 1]])
        # Simple polarization bubble is G * G
        psi = np.array([[1, 0]], dtype=int)
        # No internal boson lines (the external vertex flag is -1)
        phi = np.array([[-1, -1]], dtype=int)
        # Make the graph in the adjacency list representations
        distinct_poln_graphs_HFI = defaultdict(lambda: defaultdict(list))
        distinct_poln_graphs_HFI[0][1].append('f')
        distinct_poln_graphs_HFI[1][0].append('f')
        distinct_poln_graphs_HFI = [distinct_poln_graphs_HFI]
        # Save the graph and relevant loop/combinatorial information
        save_contents = {'n_loops': n_loops, 'loops': loops}
        # Add this set of diagram connections to the .npz file
        save_contents['psi_0'] = psi
        save_contents['phi_0'] = phi
        # Save the graph information to a .npz file
        np.savez(save_name, **save_contents)
        with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
            info_block = (
                'Number of vacuum vertices: ' + str(n_verts_vacuum) + '\n'
                'Number of polarization vertices: ' +
                str(n_verts_poln) + '\n'
                'Diagram order (number of interaction lines plus one): ' +
                str(order) + '\n'
                '\nTotal number of disconnected EG diagrams: 0' + '\n'
                'Number of topologically distinct HFI polarization diagrams: 1' + '\n'
            )
            diagram_file.write(info_block)
    # Otherwise, there are 2(n - 1) vertices to the base
    # vacuum diagrams if n is the polarization diagram order;
    # derive the polarization graphs from these by gluing two legs
    else:
        # Generate all fermionic connections
        psi_all = get_feyn_vacuum_perms(n_verts_vacuum)
        # Build all the naive vacuum graphs
        all_vacuum_graphs = []
        for i in range(psi_all.shape[0]):
            all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
        all_vacuum_graphs = np.asarray(all_vacuum_graphs)
        # Get all distinct (1BI) vacuum graphs
        distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
        distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)
        # Now, get all distinct (PBI) polarization graphs using the distinct 1BI vacuum graphs
        distinct_poln_graphs_1BI = rem_top_equiv_al(get_poln_graphs(distinct_vacuum_graphs_1BI), fixed_pts=[0, 1])
        distinct_poln_graphs_HFI = get_FI_subset(distinct_poln_graphs_1BI, legs=[0, 1])
        n_diags_HFI = len(distinct_poln_graphs_HFI)
        # Get the number of fermion loops in each polarization graph
        n_loops = np.zeros(len(distinct_poln_graphs_HFI), dtype=int)
        loops = np.zeros(len(distinct_poln_graphs_HFI), dtype=object)
        for i in range(len(distinct_poln_graphs_HFI)):
            loops[i] = get_cycles(distinct_poln_graphs_HFI[i])
            n_loops[i] = len(loops[i])
        # Unmake the graphs to get numpy arrays for \psi and \phi
        # (i.e., to get the permutation group representations of all graphs)
        save_contents = {'n_loops': n_loops, 'loops': loops}
        for i in range(n_diags_HFI):
            psi, phi = graph_al_to_pg(distinct_poln_graphs_HFI[i])
            # Add this set of diagram connections to the .npz file
            save_contents['psi_' + str(i)] = psi
            save_contents['phi_' + str(i)] = phi
        # Save the graph information to a .npz file
        np.savez(save_name, **save_contents)
        with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
            info_block = (
                'Number of vacuum vertices: ' + str(n_verts_vacuum) + '\n'
                'Number of polarization vertices: ' +
                str(n_verts_poln) + '\n'
                'Diagram order (number of interaction lines plus one): ' +
                str(order) + '\n'
                '\nTotal number of disconnected EG diagrams: ' +
                str(math.factorial(n_verts_vacuum)) + '\n'
                'Number of topologically distinct disconnected EG vacuum diagrams: ' +
                str(len(distinct_vacuum_graphs)) + '\n'
                'Number of topologically distinct 1BI vacuum diagrams: ' +
                str(len(distinct_vacuum_graphs_1BI)) + '\n'
                '\nNumber of topologically distinct 1BI polarization diagrams: ' +
                str(len(distinct_poln_graphs_1BI)) + '\n'
                'Number of topologically distinct HFI polarization diagrams: ' +
                str(len(distinct_poln_graphs_HFI)) + '\n'
            )
            diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_poln_graphs_HFI, n_legs=2,
                         savename='charge_poln_n='+str(order)+'_HFI.tex')
    # Return the number of diagrams generated, and the
    # number of vertices in each (including external ones)
    return (n_diags_HFI, n_verts_poln)


# Generate all nth-order charge polarization diagrams with
# bold HF irreducibility rules. Attaches legs to the (n-1)th-order
# vacuum diagrams in all possible ways, then removes duplicates.
def generate_bHFI_charge_poln_diagrams(order=1, save_name='charge_poln_diagrams.npz', 
                                       spinless=False, draw=True):
    # Number of vertices in the vacuum diagrams at the next-lowest
    # order, from which we derive the polarization diagrams
    n_verts_vacuum = 2 * (order - 1)
    n_verts_poln = 2 * order
    # Hard-coded diagram generation for the zeroth order bubble diagram
    # (since there are no vacuum diagrams with zero interaction lines)
    if n_verts_vacuum == 0:
        # There is one diagram, and one fermion loop in the diagram
        n_diags_bHFI = 1
        n_loops = np.array([1.0])
        loops = np.array([[0, 1]])
        # Simple polarization bubble is G * G
        psi = np.array([[1, 0]], dtype=int)
        # No internal boson lines (the external vertex flag is -1)
        phi = np.array([[-1, -1]], dtype=int)
        # Make the graph in the adjacency list representations
        distinct_poln_graphs_bHFI = defaultdict(lambda: defaultdict(list))
        distinct_poln_graphs_bHFI[0][1].append('f')
        distinct_poln_graphs_bHFI[1][0].append('f')
        distinct_poln_graphs_bHFI = [distinct_poln_graphs_bHFI]
        # Save the graph and relevant loop/combinatorial information
        save_contents = {'n_loops': n_loops, 'loops': loops}
        # Add this set of diagram connections to the .npz file
        save_contents['psi_0'] = psi
        save_contents['phi_0'] = phi
        # Save the graph information to a .npz file
        np.savez(save_name, **save_contents)
        with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
            info_block = (
                'Number of vacuum vertices: ' + str(n_verts_vacuum) + '\n'
                'Number of polarization vertices: ' +
                str(n_verts_poln) + '\n'
                'Diagram order (number of interaction lines plus one): ' +
                str(order) + '\n'
                '\nTotal number of disconnected EG diagrams: 0' + '\n'
                'Number of topologically distinct bold HFI (Ext. Hub.) polarization diagrams: 1' + '\n'
            )
            diagram_file.write(info_block)
    # Otherwise, there are 2(n - 1) vertices to the base
    # vacuum diagrams if n is the polarization diagram order;
    # derive the polarization graphs from these by gluing two legs
    else:
        # Generate all fermionic connections
        psi_all = get_feyn_vacuum_perms(n_verts_vacuum)

        # Build all the naive vacuum graphs
        all_vacuum_graphs = []
        for i in range(psi_all.shape[0]):
            all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
        all_vacuum_graphs = np.asarray(all_vacuum_graphs)

        # Get all distinct vacuum graphs
        distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)

        # Now, get all distinct (PBI) polarization graphs using the distinct 1BI vacuum graphs
        distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)

        distinct_poln_graphs_1BI = rem_top_equiv_al(
            get_poln_graphs(
                distinct_vacuum_graphs_1BI,
                ext_convn='new'),
            fixed_pts=[0, 1])

        distinct_poln_graphs_bHFI = get_bFI_subset(
            distinct_poln_graphs_1BI, diag_type='charge_poln', legs=[0, 1])

        if spinless:
            distinct_poln_graphs_bHFI = get_loop_bipartite_subset(distinct_poln_graphs_bHFI)

        # Get the number of fermion loops in each polarization graph
        n_diags_bHFI = len(distinct_poln_graphs_bHFI)
        n_loops = np.zeros(len(distinct_poln_graphs_bHFI), dtype=int)
        loops = np.zeros(len(distinct_poln_graphs_bHFI), dtype=object)
        for i in range(len(distinct_poln_graphs_bHFI)):
            loops[i] = get_cycles(distinct_poln_graphs_bHFI[i])
            n_loops[i] = len(loops[i])
        # Unmake the graphs to get numpy arrays for \psi and \phi
        # (i.e., to get the permutation group representations of all graphs)
        save_contents = {'n_loops': n_loops, 'loops': loops}
        for i in range(n_diags_bHFI):
            psi, phi = graph_al_to_pg(distinct_poln_graphs_bHFI[i])
            # Add this set of diagram connections to the .npz file
            save_contents['psi_' + str(i)] = psi
            save_contents['phi_' + str(i)] = phi
        # Save the graph information to a .npz file
        np.savez(save_name, **save_contents)
        with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
            info_block = (
                'Number of vacuum vertices: ' + str(n_verts_vacuum) + '\n'
                'Number of polarization vertices: ' +
                str(n_verts_poln) + '\n'
                'Diagram order (number of interaction lines plus one): ' +
                str(order) + '\n'
                '\nTotal number of disconnected EG diagrams: ' +
                str(math.factorial(n_verts_vacuum)) + '\n'
                'Number of topologically distinct disconnected EG vacuum diagrams: ' +
                str(len(distinct_vacuum_graphs)) + '\n'
                'Number of topologically distinct 1BI (G0, V) vacuum diagrams: ' + str(
                    len(distinct_vacuum_graphs_1BI)) + '\n'
                '\nNumber of topologically distinct 1BI (G0, V) polarization diagrams: ' + str(
                    len(distinct_poln_graphs_1BI)) + '\n'
                'Number of topologically distinct bold HFI (Ext. Hub.) polarization diagrams: ' + str(
                    len(distinct_poln_graphs_bHFI)) + '\n'
            )
            diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_poln_graphs_bHFI, n_legs=2,
                         savename='charge_poln_n='+str(order)+'_bHFI.tex')
    # Return the number of diagrams generated, and the
    # number of vertices in each (including external ones)
    return (n_diags_bHFI, n_verts_poln)


# Generate all nth-order charge polarization diagrams with
# H + RPA (G0W0) irreducibility rules. Attaches legs to the (n-1)th-order
# vacuum diagrams in all possible ways, then removes duplicates.
def generate_HPBI_charge_poln_diagrams(order=1, save_name='charge_poln_diagrams.npz', draw=True):
    # There are 2(n - 1) vertices to the generating vacuum diagrams if n is the polarization diagram order
    n_verts_vacuum = 2 * (order - 1)
    n_verts_poln = 2 * order

    # Hard-coded diagram generation for the zeroth order bubble diagram
    # (since there are no vacuum diagrams with zero interaction lines)
    if n_verts_vacuum == 0:
        # There is one diagram, and one fermion loop in the diagram
        n_diags_HPBI = 1
        n_loops = np.array([1.0])
        loops = np.array([[0, 1]])
        # Simple polarization bubble is G * G
        psi = np.array([[1, 0]], dtype=int)
        # No internal boson lines (the external vertex flag is -1)
        phi = np.array([[-1, -1]], dtype=int)
        # Make the graph in the adjacency list representations
        distinct_poln_graphs_HPBI = defaultdict(lambda: defaultdict(list))
        distinct_poln_graphs_HPBI[0][1].append('f')
        distinct_poln_graphs_HPBI[1][0].append('f')
        distinct_poln_graphs_HPBI = [distinct_poln_graphs_HPBI]
        # Save the graph and relevant loop/combinatorial information
        save_contents = {'n_loops': n_loops, 'loops': loops}
        # Add this set of diagram connections to the .npz file
        save_contents['psi_0'] = psi
        save_contents['phi_0'] = phi
        # Save the graph information to a .npz file
        np.savez(save_name, **save_contents)
        with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
            info_block = (
                'Number of vacuum vertices: ' + str(n_verts_vacuum) + '\n'
                'Number of polarization vertices: ' +
                str(n_verts_poln) + '\n'
                'Diagram order (number of interaction lines plus one): ' +
                str(order) + '\n'
                '\nTotal number of disconnected EG diagrams: 0' + '\n'
                'Number of topologically distinct bold HFI (Ext. Hub.) polarization diagrams: 1' + '\n'
            )
            diagram_file.write(info_block)
    # Otherwise, there are 2(n - 1) vertices to the base
    # vacuum diagrams if n is the polarization diagram order;
    # derive the polarization graphs from these by gluing two legs
    else:
        # Generate all fermionic connections
        psi_all = get_feyn_vacuum_perms(n_verts_vacuum)

        # Build all the naive vacuum graphs
        all_vacuum_graphs = []
        for i in range(psi_all.shape[0]):
            all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
        all_vacuum_graphs = np.asarray(all_vacuum_graphs)

        # Get all distinct (1BI) vacuum graphs
        distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
        distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)

        # Now, get all distinct (PBI) polarization graphs using the distinct 1BI vacuum graphs
        distinct_poln_graphs_1BI = rem_top_equiv_al(get_poln_graphs(distinct_vacuum_graphs_1BI), fixed_pts=[0, 1])
        distinct_poln_graphs_HPBI = get_PBI_subset(distinct_poln_graphs_1BI, is_self_en=False, legs=[0, 1])
        
        # Get the number of fermion loops in each polarization graph
        n_diags_HPBI = len(distinct_poln_graphs_HPBI)
        n_loops = np.zeros(len(distinct_poln_graphs_HPBI), dtype=int)
        loops = np.zeros(len(distinct_poln_graphs_HPBI), dtype=object)
        for i in range(len(distinct_poln_graphs_HPBI)):
            loops[i] = get_cycles(distinct_poln_graphs_HPBI[i])
            n_loops[i] = len(loops[i])
        # Unmake the graphs to get numpy arrays for \psi and \phi
        # (i.e., to get the permutation group representations of all graphs)
        save_contents = {'n_loops': n_loops, 'loops': loops}
        for i in range(n_diags_HPBI):
            psi, phi = graph_al_to_pg(distinct_poln_graphs_HPBI[i])
            # Add this set of diagram connections to the .npz file
            save_contents['psi_' + str(i)] = psi
            save_contents['phi_' + str(i)] = phi
        # Save the graph information to a .npz file
        np.savez(save_name, **save_contents)

        with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
            info_block = (
                'Number of vacuum vertices: ' + str(n_verts_vacuum) + '\n'
                'Number of polarization vertices: ' +
                str(n_verts_poln) + '\n'
                'Diagram order (number of interaction lines plus one): ' +
                str(order) + '\n'
                '\nTotal number of disconnected EG diagrams: ' +
                str(math.factorial(n_verts_vacuum)) + '\n'
                'Number of topologically distinct disconnected EG vacuum diagrams: ' +
                str(len(distinct_vacuum_graphs)) + '\n'
                'Number of topologically distinct 1BI (G0, V) vacuum diagrams: ' + str(
                    len(distinct_vacuum_graphs_1BI)) + '\n'
                '\nNumber of topologically distinct 1BI (G0, V) polarization diagrams: ' + str(
                    len(distinct_poln_graphs_1BI)) + '\n'
                'Number of topologically distinct 1BI and PBI (G0, W0) polarization diagrams: ' + str(
                    n_diags_HPBI) + '\n'
            )
            diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_poln_graphs_HPBI, n_legs=2,
                         savename='charge_poln_n='+str(order)+'_HPBI.tex')
    # Return the number of diagrams generated, and the
    # number of vertices in each (including external ones)
    return (n_diags_HPBI, n_verts_poln)


# Generate all nth-order charge polarization diagrams with bold (and bold Hartree)
# (H + GW) irreducibility rules. Attaches legs to the (n-1)th-order
# vacuum diagrams in all possible ways, then removes duplicates.
def generate_bold_charge_poln_diagrams(order=1, use_hhz=True, draw=True, save_name='charge_poln_diagrams.npz'):
    # There are 2(n - 1) vertices to the generating vacuum diagrams if n is the polarization diagram order
    n_verts_vac_feyn = 2 * (order - 1)
    n_verts_poln = 2 * order
    # Hard-coded diagram generation for the zeroth order bubble diagram
    # (since there are no vacuum diagrams with zero interaction lines)
    if n_verts_vac_feyn == 0:
        # There is one diagram, and one fermion loop in the diagram
        n_loops = np.array([1.0])
        loops = np.array([[0, 1]])
        # Simple polarization bubble is G * G
        psi = np.array([[1, 0]], dtype=int)
        # No internal boson lines (the external vertex flag is -1)
        phi = np.array([[-1, -1]], dtype=int)
        # Make the graph in the adjacency list representations
        distinct_poln_graphs_bold = defaultdict(lambda: defaultdict(list))
        distinct_poln_graphs_bold[0][1].append('f')
        distinct_poln_graphs_bold[1][0].append('f')
        distinct_poln_graphs_bold = [distinct_poln_graphs_bold]
        # Save the graph and relevant loop/combinatorial information
        save_contents = {'n_loops': n_loops, 'loops': loops}
        # Add this set of diagram connections to the .npz file
        save_contents['psi_0'] = psi
        save_contents['phi_0'] = phi
        # Save the graph information to a .npz file
        np.savez(save_name, **save_contents)
        with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
            info_block = (
                'Number of vacuum vertices: ' + str(n_verts_vac_feyn) + '\n'
                'Number of polarization vertices: ' +
                str(n_verts_poln) + '\n'
                'Diagram order (number of interaction lines plus one): ' +
                str(order) + '\n'
                '\nTotal number of disconnected EG diagrams: 0' + '\n'
                'Number of topologically distinct 1BI and bold (H + GW) charge polarization diagrams: 1' + '\n'
            )
            diagram_file.write(info_block)
    # Otherwise, there are 2(n - 1) vertices to the base
    # vacuum diagrams if n is the polarization diagram order;
    # derive the polarization graphs from these by gluing two legs
    else:
        if use_hhz:
            n_verts_vac_hhz = int(n_verts_vac_feyn / 2)
            print('Generating Hugenholtz diagrams...')
            distinct_hhz_graphs = rem_top_equiv_al(get_connected_subset(get_naive_vacuum_hhz_diags(n_verts_vac_hhz)))
            print('Done!\n')
            print('Unwrapping Hugenholtz diagrams with 1BI rules...')
            distinct_vacuum_graphs_1BI = []
            for graph in distinct_hhz_graphs:
                # First, we double all vertex labels in the graph; this makes room for the new vertex labels
                # in such a manner as to preserve our pairwise convention for the bosonic connections
                g_shifted = map_vertices_defaultdict(
                    graph, vmap=map(lambda x: 2*x, graph.keys()))
                # Then, we recursively unwrap the Hugenholtz diagrams into the set of all contained Feynman diagrams
                expanded_graphs = unwrap_hhz_to_feyn_with_irred_rule(
                    g_shifted, n_verts_vac_feyn, is_irred=is_1BI)
                distinct_vacuum_graphs_1BI.extend(
                    rem_top_equiv_al(expanded_graphs, n_verts_vac_feyn))
            print('Done!\n')
        else:
            # Generate all fermionic connections
            psi_all = get_feyn_vacuum_perms(n_verts_vac_feyn)
            # Build all the naive vacuum graphs
            all_vacuum_graphs = []
            for i in range(psi_all.shape[0]):
                all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
            all_vacuum_graphs = np.asarray(all_vacuum_graphs)
            # Get all distinct (1BI) vacuum graphs
            distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
            distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)
        # Now, get all naive (PBI) polarization graphs using the distinct 1BI vacuum graphs
        poln_graphs_1BI = get_poln_graphs(distinct_vacuum_graphs_1BI)
        # Get all naive 1BI + 2BI + 2FI (bold) Feynman diagrams
        poln_graphs_bold = get_bold_subset(poln_graphs_1BI, diag_type='poln', legs=[0, 1])
        # Finally, get the distinct subset of these graphs
        distinct_poln_graphs_bold = list(rem_top_equiv_al(poln_graphs_bold, fixed_pts=[0, 1]))
        
        # Get the number of fermion loops in each polarization graph
        n_loops = np.zeros(len(distinct_poln_graphs_bold), dtype=int)
        loops = np.zeros(
            len(distinct_poln_graphs_bold), dtype=object)
        for i in range(len(distinct_poln_graphs_bold)):
            loops[i] = get_cycles(distinct_poln_graphs_bold[i])
            n_loops[i] = len(loops[i])
        # Unmake the graphs to get numpy arrays for \psi and \phi
        # (i.e., to get the permutation group representations of all graphs)
        save_contents = {'n_loops': n_loops, 'loops': loops}
        for i in range(len(distinct_poln_graphs_bold)):
            psi, phi = graph_al_to_pg(distinct_poln_graphs_bold[i])
            # Add this set of diagram connections to the .npz file
            save_contents['psi_' + str(i)] = psi
            save_contents['phi_' + str(i)] = phi
        # Save the graph information to a .npz file
        np.savez(save_name, **save_contents)

        with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
            info_block = (
                'Number of vacuum vertices: ' + str(n_verts_vac_feyn) + '\n'
                'Number of polarization vertices: ' +
                str(n_verts_poln) + '\n'
                'Diagram order (number of interaction lines plus one): ' +
                str(order) + '\n'
                '\nTotal number of disconnected EG diagrams: ' +
                str(math.factorial(n_verts_vac_feyn)) + '\n'
                'Number of topologically distinct 1BI (G0, V) vacuum diagrams: ' + str(
                    len(distinct_vacuum_graphs_1BI)) + '\n'
                'Number of topologically distinct 1BI and bold (H + GW) polarization diagrams: ' + str(
                    len(distinct_poln_graphs_bold)) + '\n'
            )
            diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_poln_graphs_bold, n_legs=2,
                         savename='charge_poln_n='+str(order)+'_bold.tex')
    # Return the number of diagrams generated, and the
    # number of vertices in each (including external ones)
    return (len(distinct_poln_graphs_bold), n_verts_poln)


# Generate all nth-order corrections (i.e. (n+1)th-order diagrams) to the
# spin polarization with RPA irreducibility rules. Attaches legs to the
# nth-order vacuum diagrams in all possible ways, then removes duplicates.
def generate_HPBI_spin_poln_diagrams(order=1, save_name='spin_poln_diagrams.npz', draw=True):
    # There are 2(n - 1) vertices to the diagram diagrams if n is the polarization diagram order
    n_verts_vacuum = 2 * (order - 1)
    n_verts_poln = 2 * order
    # Hard-coded diagram generation for the zeroth order bubble diagram
    # (since there are no vacuum diagrams with zero interaction lines)
    if n_verts_vacuum == 0:
        print('No graphs to draw!')
        return
    # Otherwise, there are 2(n - 1) vertices to the base
    # vacuum diagrams if n is the polarization diagram order;
    # derive the polarization graphs from these by gluing two legs
    else:
        # Generate all fermionic connections
        psi_all = get_feyn_vacuum_perms(n_verts_vacuum)

        # Build all the naive vacuum graphs
        all_vacuum_graphs = []
        for i in range(psi_all.shape[0]):
            all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
        all_vacuum_graphs = np.asarray(all_vacuum_graphs)

        # Get all distinct vacuum graphs
        distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
        distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)

        # Do the same, but also remove topological equivalents
        distinct_poln_graphs_1BI = rem_top_equiv_al(get_poln_graphs(distinct_vacuum_graphs_1BI), fixed_pts=[0, 1])
        distinct_poln_graphs_HPBI = get_PBI_subset(distinct_poln_graphs_1BI, is_self_en=False, legs=[0, 1])

        # Throw out diagrams that vanish due to mixed-operator correlation functions,
        # i.e., those for which the two external vertices are not connected.
        distinct_spin_poln_graphs_HPBI = get_spin_poln_subset_from_charge_poln(distinct_poln_graphs_HPBI, legs=[0, 1])

        # Get the number of fermion loops in each polarization graph
        n_loops = np.zeros(len(distinct_spin_poln_graphs_HPBI), dtype=int)
        loops = np.zeros(len(distinct_spin_poln_graphs_HPBI), dtype=object)
        for i in range(len(distinct_spin_poln_graphs_HPBI)):
            loops[i] = get_cycles(distinct_spin_poln_graphs_HPBI[i])
            n_loops[i] = len(loops[i])
        # Unmake the graphs to get numpy arrays for \psi and \phi
        # (i.e., to get the permutation group representations of all graphs)
        save_contents = {'n_loops': n_loops, 'loops': loops}
        for i in range(len(distinct_spin_poln_graphs_HPBI)):
            psi, phi = graph_al_to_pg(distinct_spin_poln_graphs_HPBI[i])
            # Add this set of diagram connections to the .npz file
            save_contents['psi_' + str(i)] = psi
            save_contents['phi_' + str(i)] = phi
        # Save the graph information to a .npz file
        np.savez(save_name, **save_contents)

        with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
            info_block = (
                'Number of vertices: ' + str(n_verts_vacuum) + '\n'
                'Diagram order (number of interaction lines plus one): ' +
                str(order) + '\n'
                '\nTotal number of disconnected EG diagrams: ' +
                str(math.factorial(n_verts_vacuum)) + '\n'
                'Number of topologically distinct disconnected EG vacuum diagrams: ' +
                str(len(distinct_vacuum_graphs)) + '\n'
                'Number of topologically distinct 1BI (G0, V) vacuum diagrams: ' + str(
                    len(distinct_vacuum_graphs_1BI)) + '\n'
                '\nNumber of topologically distinct 1BI (G0, V) charge polarization diagrams: ' + str(
                    len(distinct_poln_graphs_1BI)) + '\n'
                'Number of topologically distinct 1BI and PBI (G0, W0) charge polarization diagrams: ' + str(
                    len(distinct_poln_graphs_HPBI)) + '\n'
                'Number of topologically distinct 1BI and PBI (G0, W0) spin polarization diagrams: ' + str(
                    len(distinct_spin_poln_graphs_HPBI)) + '\n'
            )
            diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_spin_poln_graphs_HPBI, n_legs=2,
                         savename='spin_poln_n='+str(order)+'_HPBI.tex')
    # Return the number of diagrams generated, and the
    # number of vertices in each (including external ones)
    return (len(distinct_spin_poln_graphs_HPBI), n_verts_poln)


# Generate all nth-order self-energy diagrams with bold Hartree
# irreducibility rules. Deletes a Green's function from nth-order
# vacuum diagrams in all possible ways, then removes duplicates.
def generate_bHI_self_energy_diagrams(order=1, save_name='self_energy_diagrams.npz', draw=True):
    # There are 2*n vertices to a diagram, if n is the self-energy diagram order
    n_verts = 2 * order

    # Generate all fermionic connections
    psi_all = get_feyn_vacuum_perms(n_verts)

    # Build all the naive vacuum graphs
    all_vacuum_graphs = []
    for i in range(psi_all.shape[0]):
        all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
    all_vacuum_graphs = np.asarray(all_vacuum_graphs)

    # Get all distinct 1BI vacuum graphs
    distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
    distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)

    # Now, make all naive self-energy diagrams
    self_energy_graphs_1BI = get_self_energy_graphs(distinct_vacuum_graphs_1BI)

    # Do the same, but also remove topological equivalents
    distinct_self_energy_graphs_1BI = list(rem_top_equiv_al(self_energy_graphs_1BI, fixed_pts=[0, 1]))
    
    # Get the number of fermion loops in each self-energy graph
    n_loops = np.zeros(len(distinct_self_energy_graphs_1BI), dtype=int)
    loops = np.zeros(len(distinct_self_energy_graphs_1BI), dtype=object)
    for i in range(len(distinct_self_energy_graphs_1BI)):
        loops[i] = get_cycles(distinct_self_energy_graphs_1BI[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_self_energy_graphs_1BI)):
        psi, phi = graph_al_to_pg(distinct_self_energy_graphs_1BI[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)

    with open('n_'+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Self-energy diagram order (number of interaction lines): ' + str(
                order) + '\n'
            '\nTotal number of disconnected EG diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct disconnected EG vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs)) + '\n'
            'Number of topologically distinct 1BI vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs_1BI)) + '\n'
            '\nNumber of naive 1BI self-energy diagrams: ' +
            str(len(self_energy_graphs_1BI)) + '\n'
            '\nNumber of topologically distinct 1BI (bHI) self-energy diagrams: ' + str(
                len(distinct_self_energy_graphs_1BI)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_self_energy_graphs_1BI, n_legs=2,
                         savename='self_energy_n='+str(order)+'_bHI.tex')
    # Return the number of diagrams generated, and the
    # number of vertices in each (including external ones)
    return (len(distinct_self_energy_graphs_1BI), n_verts)


# Generate all nth-order self energy diagrams with HF irreducibility rules;
# since the bold Hatree term is trivially obtained, by HFI we mean bHI + FI, i.e.,
# no Hartree terms at all, and nested Hartree-Fock subdiagrams only. Deletes a Green's
# function from nth-order vacuum diagrams in all possible ways, then removes duplicates.
def generate_HFI_self_energy_diagrams(order=1, save_name='self_energy_diagrams.npz', draw=True):
    # There are 2*n vertices to a diagram, if n is the self-energy diagram order
    n_verts = 2 * order

    # Generate all fermionic connections
    psi_all = get_feyn_vacuum_perms(n_verts)

    # Build all the naive vacuum graphs
    all_vacuum_graphs = []
    for i in range(psi_all.shape[0]):
        all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
    all_vacuum_graphs = np.asarray(all_vacuum_graphs)

    # Get all distinct 1BI vacuum graphs
    distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
    distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)

    # Now, make all naive self-energy diagrams
    self_energy_graphs_1BI = get_self_energy_graphs(distinct_vacuum_graphs_1BI)
    self_energy_graphs_HFI = get_FI_subset(self_energy_graphs_1BI, legs=[0, 1])

    # Do the same, but also remove topological equivalents
    distinct_self_energy_graphs_1BI = rem_top_equiv_al(self_energy_graphs_1BI, fixed_pts=[0, 1])
    distinct_self_energy_graphs_HFI = get_FI_subset(distinct_self_energy_graphs_1BI, legs=[0, 1])
    
    # Get the number of fermion loops in each self-energy graph
    n_loops = np.zeros(len(distinct_self_energy_graphs_HFI), dtype=int)
    loops = np.zeros(len(distinct_self_energy_graphs_HFI), dtype=object)
    for i in range(len(distinct_self_energy_graphs_HFI)):
        loops[i] = get_cycles(distinct_self_energy_graphs_HFI[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_self_energy_graphs_HFI)):
        psi, phi = graph_al_to_pg(distinct_self_energy_graphs_HFI[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)

    with open('n_'+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Self-energy diagram order (number of interaction lines): ' + str(
                order) + '\n'
            '\nTotal number of disconnected EG diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct disconnected EG vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs)) + '\n'
            'Number of topologically distinct 1BI vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs_1BI)) + '\n'
            '\nNumber of naive 1BI self-energy diagrams: ' +
            str(len(self_energy_graphs_1BI)) + '\n'
            'Number of naive bold HFI self-energy diagrams: ' +
            str(len(self_energy_graphs_HFI)) + '\n'
            '\nNumber of topologically distinct 1BI self-energy diagrams: ' +
            str(len(distinct_self_energy_graphs_1BI)) + '\n'
            'Number of topologically distinct HFI self-energy diagrams: ' +
            str(len(distinct_self_energy_graphs_HFI)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_self_energy_graphs_HFI, n_legs=2,
                         savename='self_energy_n='+str(order)+'_HFI.tex')
    # Return the number of diagrams generated, and the
    # number of vertices in each (including external ones)
    return (len(distinct_self_energy_graphs_HFI), n_verts)


# Generate all nth-order self-energy diagrams with HF
# irreducibility rules. Deletes a Green's function from nth-order
# vacuum diagrams in all possible ways, then removes duplicates.
def generate_bHFI_self_energy_diagrams(order=1, save_name='self_energy_diagrams.npz', draw=True):
    # There are 2*n vertices to a diagram, if n is the self-energy diagram order
    n_verts = 2 * order

    # Generate all fermionic connections
    psi_all = get_feyn_vacuum_perms(n_verts)

    # Build all the naive vacuum graphs
    all_vacuum_graphs = []
    for i in range(psi_all.shape[0]):
        all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
    all_vacuum_graphs = np.asarray(all_vacuum_graphs)

    # Get all distinct 1BI vacuum graphs
    distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
    distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)

    # Now, make all naive self-energy diagrams
    self_energy_graphs_1BI = get_self_energy_graphs(distinct_vacuum_graphs_1BI)

    # Do the same, but also remove topological equivalents
    distinct_self_energy_graphs_1BI = rem_top_equiv_al(self_energy_graphs_1BI, fixed_pts=[0, 1])
    distinct_self_energy_graphs_bHFI = get_bFI_subset(distinct_self_energy_graphs_1BI, diag_type='self_en', legs=[0, 1])

    # Get the number of fermion loops in each self-energy graph
    n_loops = np.zeros(len(distinct_self_energy_graphs_bHFI), dtype=int)
    loops = np.zeros(len(distinct_self_energy_graphs_bHFI), dtype=object)
    for i in range(len(distinct_self_energy_graphs_bHFI)):
        loops[i] = get_cycles(distinct_self_energy_graphs_bHFI[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_self_energy_graphs_bHFI)):
        psi, phi = graph_al_to_pg(distinct_self_energy_graphs_bHFI[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)

    with open('n_'+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Self-energy diagram order (number of interaction lines): ' + str(
                order) + '\n'
            '\nTotal number of disconnected EG diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct disconnected EG vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs)) + '\n'
            'Number of topologically distinct 1BI vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs_1BI)) + '\n'
            '\nNumber of naive 1BI self-energy diagrams: ' +
            str(len(self_energy_graphs_1BI)) + '\n'
            '\nNumber of topologically distinct 1BI self-energy diagrams: ' +
            str(len(distinct_self_energy_graphs_1BI)) + '\n'
            'Number of topologically distinct bold HFI self-energy diagrams: ' +
            str(len(distinct_self_energy_graphs_bHFI)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_self_energy_graphs_bHFI, n_legs=2,
                         savename='self_energy_n='+str(order)+'_bHFI.tex')
    # Return the number of diagrams generated, and the
    # number of vertices in each (including external ones)
    return (len(distinct_self_energy_graphs_bHFI), n_verts)


# Generate all nth-order self-energy diagrams with H + RPA (G0W0)
# irreducibility rules. Deletes a Green's function from nth-order
# vacuum diagrams in all possible ways, then removes duplicates.
def generate_HPBI_self_energy_diagrams(order=1, save_name='self_energy_diagrams.npz', 
                                       spinless=False, draw=True):
    # There are 2*n vertices to a diagram, if n is the self-energy diagram order
    n_verts = 2 * order

    # Generate all fermionic connections
    psi_all = get_feyn_vacuum_perms(n_verts)

    # Build all the naive vacuum graphs
    all_vacuum_graphs = []
    for i in range(psi_all.shape[0]):
        all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
    all_vacuum_graphs = np.asarray(all_vacuum_graphs)

    # Get all distinct 1BI vacuum graphs
    distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
    distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)

    # Now, make all naive self-energy diagrams
    self_energy_graphs_1BI = get_self_energy_graphs(distinct_vacuum_graphs_1BI)
    self_energy_graphs_HPBI = get_PBI_subset(self_energy_graphs_1BI, is_self_en=True, legs=[0, 1])

    if spinless:
        self_energy_graphs_HPBI = get_loop_bipartite_subset(self_energy_graphs_HPBI)

    # Finally, remove any topologically equivalent graphs
    distinct_self_energy_graphs_HPBI = list(rem_top_equiv_al(self_energy_graphs_1BI, fixed_pts=[0, 1]))

    # Get the number of fermion loops in each self-energy graph
    n_loops = np.zeros(len(distinct_self_energy_graphs_HPBI), dtype=int)
    loops = np.zeros(len(distinct_self_energy_graphs_HPBI), dtype=object)
    for i in range(len(distinct_self_energy_graphs_HPBI)):
        loops[i] = get_cycles(distinct_self_energy_graphs_HPBI[i])
        n_loops[i] = len(loops[i])
        
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_self_energy_graphs_HPBI)):
        psi, phi = graph_al_to_pg(distinct_self_energy_graphs_HPBI[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)

    with open('n_'+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Self-energy correction order (number of interaction lines minus one): ' + str(
                order) + '\n'
            '\nTotal number of disconnected EG diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct disconnected EG vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs)) + '\n'
            'Number of topologically distinct 1BI vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs_1BI)) + '\n'
            '\nNumber of naive 1BI self-energy diagrams: ' +
            str(len(self_energy_graphs_1BI)) + '\n'
            'Number of topologically distinct 1BI and PBI self-energy diagrams: ' +
            str(len(distinct_self_energy_graphs_HPBI)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_self_energy_graphs_HPBI, n_legs=2,
                         savename='self_energy_n='+str(order)+'_HPBI.tex')
    # Return the number of diagrams generated, and the
    # number of vertices in each (including external ones)
    return (len(distinct_self_energy_graphs_HPBI), n_verts)


# Generate all nth-order self-energy diagrams with bold HF + RPA
# irreducibility rules. Deletes a Green's function from nth-order
# vacuum diagrams in all possible ways, then removes duplicates.
def generate_bHFPBI_self_energy_diagrams(order=1, use_hhz=True, draw=True, save_name='self_energy_diagrams.npz'):
    # There are 2*n vertices to a diagram, if n is the self-energy diagram order
    n_verts = 2 * order

    # Generate all fermionic connections
    psi_all = get_feyn_vacuum_perms(n_verts)

    # Build all the naive vacuum graphs
    all_vacuum_graphs = []
    for i in range(psi_all.shape[0]):
        all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
    all_vacuum_graphs = np.asarray(all_vacuum_graphs)

    # Get all distinct 1BI vacuum graphs
    distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
    distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)

    # Now, make all naive self-energy diagrams
    self_energy_graphs_1BI = get_self_energy_graphs(distinct_vacuum_graphs_1BI)

    # Do the same, but also remove topological equivalents
    distinct_self_energy_graphs_1BI = rem_top_equiv_al(self_energy_graphs_1BI, fixed_pts=[0, 1])
    distinct_self_energy_graphs_HPBI = get_PBI_subset(distinct_self_energy_graphs_1BI, is_self_en=True, legs=[0, 1])
    distinct_self_energy_graphs_bHFPBI = get_bFI_subset(distinct_self_energy_graphs_HPBI, diag_type='self_en', legs=[0, 1])
    
    # Get the number of fermion loops in each self-energy graph
    n_loops = np.zeros(len(distinct_self_energy_graphs_bHFPBI), dtype=int)
    loops = np.zeros(len(distinct_self_energy_graphs_bHFPBI), dtype=object)
    for i in range(len(distinct_self_energy_graphs_bHFPBI)):
        loops[i] = get_cycles(distinct_self_energy_graphs_bHFPBI[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_self_energy_graphs_bHFPBI)):
        psi, phi = graph_al_to_pg(distinct_self_energy_graphs_bHFPBI[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)

    with open('n_'+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vertices: ' + str(n_verts) + '\n'
            'Self-energy correction order (number of interaction lines minus one): ' + str(
                order) + '\n'
            '\nTotal number of disconnected EG diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct disconnected EG vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs)) + '\n'
            'Number of topologically distinct 1BI vacuum diagrams: ' +
            str(len(distinct_vacuum_graphs_1BI)) + '\n'
            '\nNumber of naive 1BI self-energy diagrams: ' +
            str(len(self_energy_graphs_1BI)) + '\n'
            #   'Number of naive 1BI and PBI self-energy diagrams: ' + str(len(self_energy_graphs_HPBI)) + '\n'
            '\nNumber of topologically distinct 1BI self-energy diagrams: ' + \
            str(len(distinct_self_energy_graphs_1BI)) + '\n'
            'Number of topologically distinct 1BI and PBI self-energy diagrams: ' + \
            str(len(distinct_self_energy_graphs_HPBI)) + '\n'
            'Number of topologically distinct 1BI, PBI, and bFI self-energy diagrams: ' + \
            str(len(distinct_self_energy_graphs_bHFPBI)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_self_energy_graphs_bHFPBI, n_legs=2,
                         savename='self_energy_n='+str(order)+'_bHFPBI.tex')
    # Return the number of diagrams generated, and the
    # number of vertices in each (including external ones)
    return (len(distinct_self_energy_graphs_bHFPBI), n_verts)


# Generate all nth-order self-energy diagrams with bold (and bold Hartree)
# (H + GW) irreducibility rules. Deletes a Green's function from nth-order
# vacuum diagrams in all possible ways, then removes duplicates.
def generate_bold_self_energy_diagrams(order=1, use_hhz=True, draw=True, save_name='self_energy_diagrams.npz'):
    # There are 2*n vertices to a diagram, if n is the self-energy diagram order
    n_verts = 2 * order
    if use_hhz:
        n_verts_hhz = order
        print('Generating Hugenholtz diagrams...')
        distinct_hhz_graphs = rem_top_equiv_al(get_connected_subset(get_naive_vacuum_hhz_diags(n_verts_hhz)))
        print('Done!\n')
        print('Unwrapping Hugenholtz diagrams with 1BI rules...')
        distinct_vacuum_graphs_1BI = []
        for graph in distinct_hhz_graphs:
            # First, we double all vertex labels in the graph; this makes room for the new vertex labels
            # in such a manner as to preserve our pairwise convention for the bosonic connections
            g_shifted = map_vertices_defaultdict(
                graph, vmap=map(lambda x: 2*x, graph.keys()))
            # Then, we recursively unwrap the Hugenholtz diagrams into the set of all contained Feynman diagrams
            expanded_graphs = unwrap_hhz_to_feyn_with_irred_rule(g_shifted, n_verts, is_irred=is_1BI)
            distinct_vacuum_graphs_1BI.extend(rem_top_equiv_al(expanded_graphs, n_verts))
        print('Done!\n')
    else:
        # Generate all fermionic connections
        psi_all = get_feyn_vacuum_perms(n_verts)
        # Build all the naive vacuum graphs
        all_vacuum_graphs = []
        for i in range(psi_all.shape[0]):
            all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
        all_vacuum_graphs = np.asarray(all_vacuum_graphs)
        # Get all distinct (1BI) vacuum graphs
        distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs)
        distinct_vacuum_graphs_1BI = get_1BI_subset(distinct_vacuum_graphs)
    # Now, get all naive (PBI) self-energy graphs using the distinct 1BI vacuum graphs
    self_en_graphs_1BI = get_self_energy_graphs(distinct_vacuum_graphs_1BI)
    # Get all naive 1BI + 2BI + 2FI (bold) Feynman diagrams
    self_en_graphs_bold = get_bold_subset(self_en_graphs_1BI, diag_type='self_en', legs=[0, 1])
    # Finally, get the distinct subset of these graphs
    distinct_self_en_graphs_bold = list(rem_top_equiv_al(self_en_graphs_bold, fixed_pts=[0, 1]))

    # Get the number of fermion loops in each self-energy graph
    n_loops = np.zeros(len(distinct_self_en_graphs_bold), dtype=int)
    loops = np.zeros(len(distinct_self_en_graphs_bold), dtype=object)
    for i in range(len(distinct_self_en_graphs_bold)):
        loops[i] = get_cycles(distinct_self_en_graphs_bold[i])
        n_loops[i] = len(loops[i])
    # Unmake the graphs to get numpy arrays for \psi and \phi
    # (i.e., to get the permutation group representations of all graphs)
    save_contents = {'n_loops': n_loops, 'loops': loops}
    for i in range(len(distinct_self_en_graphs_bold)):
        psi, phi = graph_al_to_pg(distinct_self_en_graphs_bold[i])
        # Add this set of diagram connections to the .npz file
        save_contents['psi_' + str(i)] = psi
        save_contents['phi_' + str(i)] = phi
    # Save the graph information to a .npz file
    np.savez(save_name, **save_contents)

    with open('n='+str(order)+'_diagram_counts.out', 'w+') as diagram_file:
        info_block = (
            'Number of vacuum vertices: ' + str(n_verts) + '\n'
            'Number of self-energy vertices: ' + str(n_verts) + '\n'
            'Diagram order (number of interaction lines plus one): ' +
            str(order) + '\n'
            '\nTotal number of disconnected EG diagrams: ' +
            str(math.factorial(n_verts)) + '\n'
            'Number of topologically distinct 1BI (G0, V) vacuum diagrams: ' + str(
                len(distinct_vacuum_graphs_1BI)) + '\n'
            'Number of topologically distinct 1BI and bold (H + GW) self-energy diagrams: ' + str(
                len(distinct_self_en_graphs_bold)) + '\n'
        )
        diagram_file.write(info_block)
    # Optionally draw the diagrams
    if draw:
        draw_bulk_feynmp(distinct_self_en_graphs_bold,
                         n_legs=2, savename='self_en_n='+str(order)+'_bold.tex')
    # Return the number of diagrams generated, and the
    # number of vertices in each (including external ones)
    return (len(distinct_self_en_graphs_bold), n_verts)
