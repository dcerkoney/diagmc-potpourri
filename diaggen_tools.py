#!/usr/bin/env python3
from collections import defaultdict
import numpy as np
import itertools
import math
import copy
import warnings


# Draw a batch of feynman diagrams to a bulk .tex file; this will generate a .pdf
# with all of them in a grid (optionally: labeled vertices), as well as standalone
# postscript .1 files.
def draw_bulk_feynmp(graphs, n_legs=0, fside='left', savename='diagrams.tex'):
    if graphs == []:
        print('No graphs to draw!')
        return
    if fside not in ['left', 'right']:
        raise ValueError("Fermion line side preference for drawing must be either 'left' or 'right'!")
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
    # Draw each possible nth order diagram in a separate file
    for i in range(len(graphs)):
        n_vertices = len(graphs[i])
        # Get back the psi and phi lists corresponding to this graph
        psi_list, phi_list = graph_al_to_pg(graphs[i])
        # print(psi_list, phi_list)
        # print('Graph #'+str(i)+':\n'+''.join(str(k)+': '+str(v)+'\n' for k, v in graphs[i].items()))

        # Draw vertices; there are 2n
        diagram_str = (
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
                                +str(j + 1)+'],v['
                                +str(psi_list[j] + 1)+']}\n')
        # Freeze the diagram shape after drawing Fermion lines
        # diagram_str += ('\n'+r'\fmffreeze'+'\n')
        # Draw interaction lines; there are n
        for j in range(n_vertices):
            # If the vertex is an external one, don't
            # label it, and add a small bosonic leg
            if phi_list[j] == -1:
                # Separately label external vertices
                diagram_str += (
                                r'\fmflabel{$'+str(j + 1)+r'_{\mathrm{ext}}$}{v['+str(j + 1)+']}'+'\n'
                                r'\fmfv{d.sh=circle, d.fi=full, d.si=2thick, f=(1,,0,,0)}{v['+str(j + 1)+']}'+'\n'
                               )
                continue
            else:
                diagram_str += (r'\fmflabel{$'+str(j + 1)+r'$}{v['+str(j + 1)+']}'+'\n')
            # Ignoring mirrored connections, add the bosonic lines
            if phi_list[j] != -2:
                diagram_str += (r'\fmf{photon}{v['
                                +str(j + 1)+'],v['
                                +str(phi_list[j] + 1)+']}\n')
            # # Label all external vertices 'ext'; according to the new convention, they come first in the list
            # if j < n_legs:
            #     diagram_str += (
            #                     r'\fmflabel{$'+str(j + 1)+r'_{\mathrm{ext}}$}{v['+str(j + 1)+']}'+'\n'
            #                     r'\fmfv{d.sh=circle, d.fi=full, d.si=2thick, f=(1,,0,,0)}{v['+str(j + 1)+']}'+'\n'
            #                    )
        # Find and label the incoming/outgoing self-energy external vertices
        for j in range(n_vertices):
            if psi_list[j] == -1:
                # Found the outgoing vertex; label it!
                diagram_str += (
                                r'\fmflabel{$'+str(j + 1)+r'_{\mathrm{ext}}$}{v['+str(j + 1)+']}'+'\n'
                                r'\fmfv{d.sh=circle, d.fi=full, d.si=2thick, f=(1,,0,,0)}{v['+str(j + 1)+']}'+'\n'
                               )
                continue
            # Found the incoming vertex; label it!
            if j not in psi_list:
                diagram_str += (
                                r'\fmflabel{$'+str(j + 1)+r'_{\mathrm{ext}}$}{v['+str(j + 1)+']}'+'\n'
                                r'\fmfv{d.sh=circle, d.fi=full, d.si=2thick, f=(1,,0,,0)}{v['+str(j + 1)+']}'+'\n'
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
        # Write feynmf diagram contents to file
        myfile.write(diagram_str)
    # Write end of doc statements to file
    eod = '\n'+r'\end{document}'+'\n'
    myfile.write(eod)
    # Done writing; close the file and return
    myfile.close()
    # End of diagram creation loop
    return


# Draw a batch of feynman diagrams to a bulk .tex file; this will generate a .pdf
# with all of them in a grid (optionally: labeled vertices), as well as standalone
# postscript .1 files.
def draw_bulk_feynmp_vacuum(graphs, savename='diagrams.tex'):
    if graphs == []:
        print('No graphs to draw!')
        return 0
    n_vertices = len(graphs[0])
    # Make the new .tex file
    myfile = open(str(savename), 'w+')
    # Preamble
    preamble = (
                r'\documentclass[varwidth=\maxdimen,border=20pt]{standalone}'+'\n'
                r'\usepackage[force]{feynmp-auto}'+'\n'
                '\n'
                r'\begin{document}'+'\n'
                '\n'
               )
    myfile.write(preamble)
    # Draw each possible nth order diagram in a separate file
    for i in range(len(graphs)):
        # Get back the phi and psi lists corresponding to this graph
        psi_list, _ = graph_al_to_pg(graphs[i])

        print('Graph #'+str(i)+':\n'+''.join(str(k)+': '+str(v)+'\n' for k, v in graphs[i].items()))

        # Draw vertices; there are 2n
        diagram_str = (
                       r'\begin{fmffile}{diagram_'+str(i)+'}'+'\n'
                       r'\begin{fmfgraph*}(100,100)'+'\n'
                       r'\fmfsurroundn{v}{'+str(n_vertices)+'}'+'\n'
                       r'\fmfdotn{v}{'+str(n_vertices)+'}'+'\n'
                      )
        # Label vertices
        for j in range(n_vertices):
            diagram_str += (r'\fmflabel{$'+str(j+1)+r'$}{v['+str(j+1)+']}'+'\n')
        # Draw Fermion lines; there are 2n
        for j in range(n_vertices):
            diagram_str += (r'\fmf{fermion,right=0.3}{v['
                            +str(j + 1)+'],v['
                            +str(psi_list[j] + 1)+']}\n')
        # Freeze the diagram shape after drawing Fermion lines
        # diagram_str += ('\n'+r'\fmffreeze'+'\n')
        # Draw interaction lines; there are n
        for j in range(0, n_vertices, 2):
            diagram_str += (r'\fmf{photon}{v['
                            +str(j + 1)+'],v['
                            +str(j + 2)+']}\n')
        # End of this feynmf file
        diagram_str += (
                        r'\end{fmfgraph*}'+'\n'
                        r'\end{fmffile}'+'\n'
                        r'\hspace{3ex}'+'\n'
                       )
        # Spaces the drawings as nearly as possible to a square grid
        if ((i + 1) % int(math.ceil(math.sqrt(len(graphs))))) == 0:
            # diagram_str += '\n'+r'\vspace{4ex}'+'\n'
            diagram_str += '\n'+r'\vspace{8ex}'+'\n'
        # Write feynmf diagram contents to file
        myfile.write(diagram_str)
    # Write end of doc statements to file
    eod = '\n'+r'\end{document}'+'\n'
    myfile.write(eod)
    # Done writing; close the file and return
    myfile.close()
    # End of diagram creation loop
    return


# Given a list of vacuum graphs, generate the corresponding
# polarization graphs by adding two external vertices
def get_poln_graphs(vacuum_graphs):
    poln_graphs = []
    # For each vacuum graph
    for vacuum_graph in vacuum_graphs:
        # For all possible pairs of the fermion lines, with replacement
        for iline_pair in list(itertools.product(list(vacuum_graph.keys()), list(vacuum_graph.keys()) + [len(vacuum_graph)])):
            # Make a deep copy of the vacuum graph; we will modify it to make a polarization graph
            new_poln_graph = copy.deepcopy(vacuum_graph)
            # Add two boson legs to form the polarization graph
            for idx, iline in enumerate(iline_pair):
                for dest, color in new_poln_graph[iline].items():
                    if 'f' in color:
                        # Remove the original fermion line destination
                        del_conn_al(graph=new_poln_graph, v1=iline, v2=dest, line_type='f')
                        # Modify the fermion line destination to be the new external vertex 'E0'
                        new_poln_graph[iline][len(vacuum_graph) + idx].append('f')
                        # Make a new source (number len(vacuum_graph) + idx) with the old fermion line destination
                        new_poln_graph[len(vacuum_graph) + idx][dest].append('f')
                        # There is only one outgoing fermion lines per
                        # vertex, so if we found it, exit the inner loop
                        break
            # Add to the list of polarization graphs if
            # the current graph is not already included
            if new_poln_graph not in poln_graphs:
                poln_graphs.append(new_poln_graph)
    # Note: this is the list of all naive polarization graphs,
    #       i.e., including topologically equivalent diagrams.
    return poln_graphs


# Applies a 2-rotation to the graph so that the external leg convention is
# changed from {I = n_v - 2, O = n_v - 1} to {I = 0, O = 1} (back to front)
def shift_legs_back_to_front(graphs):
    vmap_C_2 = lambda keys: dict(zip(list(keys), np.roll(list(keys), -2)))
    # Rotate the vertex labels so that the incoming and outgoing vertices are first in the list (0 and 1)
    graphs_shifted = []
    for g in graphs:
        g_c2 = map_vertices_defaultdict(g, vmap=vmap_C_2(g.keys()))
        graphs_shifted.append(g_c2)
    return graphs_shifted


# NOTE: Assumes that the external vertices are either first or 
#       last in the list (depending on new/old conventions).
#       Default behaviour assumes the new convention!
def enforce_self_en_ext_pair_convention(graphs, ext_convn='new'):
    graphs_relabeled = copy.deepcopy(graphs)
    # Get the number of vertices from the first graph
    n_verts = len(graphs_relabeled[0])
    # Make sure the batch of graphs is at constant order
    assert all(len(g) == n_verts for g in graphs_relabeled[1:])
    if ext_convn not in ['new', 'old']:
        raise ValueError("Choose a valid external leg convention type ('new' or 'old')!")
    # New convention: external vertices first in the list
    dv = 2
    if ext_convn == 'new':
        v_i1, v_o1 = 0, 1
    # Old convention: external vertices last in the list
    else:
        v_i1, v_o1 = n_verts - 1, n_verts - 2
        dv = -2
    # First, do the swaps, if necessary, to enforce the external connections
    # (v_i1, v_i2 := v_i1 + dv), (v_o1, v_o2 := v_o1 + dv)
    for i, g in enumerate(graphs_relabeled):
        curr_ext_links = [0, 0]
        for i_start, v_start in enumerate([v_i1, v_o1]):
            for v_end in g[v_start]:
                if 'b' in g[v_start][v_end]:
                    curr_ext_links[i_start] = v_end
                    break
        v_i2_curr, v_o2_curr = curr_ext_links
        print("Found the following existing external pairs:\n" +
              "(I = {}, I' = {}), (O = {}, O' = {})".format(v_i1, v_i2_curr, v_o1, v_o2_curr))
        # Perform all necessary swaps on the external pair vertices
        # No swaps to perform
        if (v_i2_curr == v_i1 + dv) and (v_o2_curr == v_o1 + dv):
            continue
        # Only need to swap the current I mate
        elif v_o2_curr == v_o1 + dv:
            swap_i = g.keys()
            swap_i[v_i2_curr], swap_i[v_i1 + dv] = swap_i[v_i1 + dv], swap_i[v_i2_curr]
            graphs_relabeled[i] = map_vertices_defaultdict(g, vmap=swap_i)
        # Only need to swap the current O mate
        elif v_i2_curr == v_i1 + dv:
            swap_o = g.keys()
            swap_o[v_o2_curr], swap_o[v_o1 + dv] = swap_o[v_o1 + dv], swap_o[v_o2_curr]
            graphs_relabeled[i] = map_vertices_defaultdict(g, vmap=swap_o)
        # Need to exchange the two current I/O mates
        elif (v_i2_curr == v_o1 + dv) and (v_o2_curr == v_i1 + dv):
            exch_io = g.keys()
            exch_io[v_i2_curr], exch_io[v_o2_curr] = exch_io[v_o2_curr], exch_io[v_i2_curr]
            graphs_relabeled[i] = map_vertices_defaultdict(g, vmap=exch_io)
        # Need to swap both the I and O mates 
        else:
            swap_b = g.keys()
            swap_b[v_i2_curr], swap_b[v_i1 + dv] = swap_b[v_i1 + dv], swap_b[v_i2_curr]
            swap_b[v_o2_curr], swap_b[v_o1 + dv] = swap_b[v_o1 + dv], swap_b[v_o2_curr]
            graphs_relabeled[i] = map_vertices_defaultdict(g, vmap=swap_b)
    return graphs_relabeled


def enforce_alternating_pair_convention(graphs, ext_convn='new'):
    # To be implemented later (unnecessary up to n = 3)!
    raise NotImplementedError
    # graphs_relabeled = copy.deepcopy(graphs)
    # # Get the number of vertices from the first graph
    # n_verts = len(graphs_relabeled[0])
    # # Make sure the batch of graphs is at constant order
    # assert all(len(g) == n_verts for g in graphs_relabeled[1:])
    # if ext_convn not in ['new', 'old']:
    #     raise ValueError("Choose a valid external leg convention type ('new' or 'old')!") 
    # # Now, we swap all even internal vertex labels to fit with their bosonic
    # # mates (i, i + 1); this takes at most (n - 1) two-vertex swaps
    # for i, g in enumerate(graphs_relabeled):
    #     pass
    # return graphs_relabeled


# Given a list of vacuum graphs, generate the corresponding
# self-energy graphs by deleting a Green's function line.
# This function does not modify the graphs to conform to
# vertex pairing and/or external vertex conventions!
def get_self_energy_graphs_ext_arb(vacuum_graphs):
    self_energy_graphs = []
    # For each vacuum graph
    for vacuum_graph in vacuum_graphs:
        # For all possible fermion lines
        for vert_out in vacuum_graph.keys():
            # Make a deep copy of the vacuum graph; we will modify it to make a self-energy graph
            new_self_energy_graph = copy.deepcopy(vacuum_graph)
            # Remove the Green's function to form a self-energy graph
            for vert_in, color in new_self_energy_graph[vert_out].items():
                if 'f' in color:
                    # # Delete the fermion line destination, thereby removing it from the graph
                    del_conn_al(graph=new_self_energy_graph, v1=vert_out, v2=vert_in, line_type='f')
                    # There is only one outgoing fermion line per
                    # vertex, so if we found it, exit the inner loop
                    break
            # Add to the list of self-energy graphs if the current graph is not
            # already included and one-fermion irreducible (a proper self-energy)
            if is_1FI(new_self_energy_graph) and (new_self_energy_graph not in self_energy_graphs):
                self_energy_graphs.append(new_self_energy_graph)
    # Note: this is the list of all naive self-energy graphs,
    #       i.e., including topologically equivalent diagrams.
    return self_energy_graphs


# Given a list of vacuum graphs, generate the corresponding
# self-energy graphs by deleting a Green's function line.
# Modifies the generated graphs to conform to vertex
# pairing and/or external vertex conventions!
def get_self_energy_graphs(vacuum_graphs):
    self_energy_graphs = []
    # For each vacuum graph
    for vacuum_graph in vacuum_graphs:
        # For all possible fermion lines
        for v_out in vacuum_graph.keys():
            # Make a deep copy of the vacuum graph; we will modify it to make a self-energy graph
            new_self_energy_graph = copy.deepcopy(vacuum_graph)
            # Remove the Green's function to form a self-energy graph
            for v_in, color in new_self_energy_graph[v_out].items():
                if 'f' in color:
                    # Delete the fermion line destination, thereby removing it from the graph
                    del_conn_al(graph=new_self_energy_graph, v1=v_out, v2=v_in, line_type='f')
                    # Make a permutation (swap) list on the graph vertices
                    # so that the external vertices are last in the list
                    swap = np.asarray(new_self_energy_graph.keys())
                    v_io_curr = np.array([v_in, v_out])
                    # The new in/out vertices are the last two in the list of vertices
                    v_out_new = swap[-1]
                    v_in_new = swap[-2]
                    # Switch the current and new out vertices in the swap array
                    idx_curr = (swap==v_io_curr[1])
                    idx_new = (swap==v_out_new)
                    swap[idx_curr], swap[idx_new] = swap[idx_new], swap[idx_curr]
                    # If the new out vertex was the old in vertex, swap
                    # them in v_io_curr in order to track both changes
                    if v_out_new == v_io_curr[0]:
                        v_io_curr[0], v_io_curr[1] = v_io_curr[1], v_io_curr[0]
                    # Otherwise, just update the out vertex in v_io_curr
                    else:
                        v_io_curr[1] = v_out_new
                    # The new in vertex is the second to last in the list of vertices
                    # Switch the current and new in vertices in the swap array
                    idx_curr = (swap==v_io_curr[0])
                    idx_new = (swap==v_in_new)
                    swap[idx_curr], swap[idx_new] = swap[idx_new], swap[idx_curr]
                    # If the new in vertex was the old out vertex, swap
                    # them in v_io_curr in order to track both changes
                    if v_in_new == v_io_curr[1]:
                        v_io_curr[0], v_io_curr[1] = v_io_curr[1], v_io_curr[0]
                    # Otherwise, just update the in vertex in v_io_curr
                    else:
                        v_io_curr[0] = v_in_new
                    # Now, perform the swaps on the graph
                    new_self_energy_graph = map_vertices_defaultdict(new_self_energy_graph, swap)
                    # There is only one outgoing fermion line per
                    # vertex, so if we found it, exit the inner loop
                    break
            # Add to the list of self-energy graphs if the current graph is not
            # already included and one-fermion irreducible (a proper self-energy)
            if is_1FI(new_self_energy_graph) and (new_self_energy_graph not in self_energy_graphs):
                self_energy_graphs.append(new_self_energy_graph)
    # Note: this is the list of all naive self-energy graphs,
    #       i.e., including topologically equivalent diagrams.
    return self_energy_graphs


# n is the diagram order
def get_feyn_vacuum_perms(n_verts):
    # Ordered list of outgoing Fermion lines
    psi_all = np.array(list(itertools.permutations(range(n_verts))), dtype=int)
    return psi_all


# Get all (indistinct) vacuum Hugenholtz diagrams
def get_naive_vacuum_hhz_diags(n_verts):
    # Naive list of all possible outgoing multi-connections
    base_hf_hhz_mc = np.repeat(range(n_verts), 2) # represents a collection of n_verts disconnected Hartree-Fock diagrams
    psis = np.unique(list(itertools.permutations(base_hf_hhz_mc)), axis=0)
    # O(N^2), but faster than np.unique --- can we do better?
    graphs = []
    for g in (make_vacuum_hhz_graph_al_dd(psi) for psi in psis):
        if g not in graphs:
            graphs.append(g)
    return graphs


# Make an adjacency list representation of the Hugenholtz graph, where we account for the
# multigraph nature explicitly by adding a weight 'f' to the connection list once per edge
def make_vacuum_hhz_graph_al_dd(psi):
    graph_al_dd = defaultdict(lambda: defaultdict(list))
    for idx_start in range(len(psi)):
        # Fermionic connections, obtained from the pairings (psi_dagger_{j}, psi_{j}); we divide the starting
        # vertex by two to convert to the actual vertex labels (as there are two connections in psi per vertex)
        graph_al_dd[idx_start // 2][psi[idx_start]].append('f')
    return graph_al_dd


# Expands one vertex in a given hhz or mixed hhz/feynman graph from Hugenholtz
# to Feynman type, thus producing two (possibly indistinct) expanded graphs
def expand_hhz_vert(graph, v_exp):
    # Now, expand the vertex and shift one outgoing connection from v_exp accordingly
    moved = list(graph[v_exp].keys())[0]
    graph[v_exp + 1][moved].append(pop_conn_al(graph=graph, v1=v_exp, v2=moved, line_type='f'))
    # Insert the boson line
    graph[v_exp][v_exp + 1].insert(0, 'b')
    graph[v_exp + 1][v_exp].insert(0, 'b')
    # Finally, shift the incoming connections to v_exp in all possible (at most 2) ways
    expanded_graphs = []
    for k in graph:
        # Loops over the two possible crossing types, and generates a new graph for each
        if v_exp in graph[k]:
            if 'f' not in graph[k][v_exp]: continue
            # Create a new graph with the current crossing type
            g_expanded = copy.deepcopy(graph)
            # Adjust one of the two incoming connections to v_exp accordingly
            g_expanded[k][v_exp + 1].append(pop_conn_al(graph=g_expanded, v1=k, v2=v_exp, line_type='f'))
            # Add to the list of expanded graphs
            expanded_graphs.append(g_expanded)
    return expanded_graphs


# Recursively expands all Hugenholtz vertices in a graph to return the full
# list of Feynman diagrams represented by it (there are at most 2^n of them)
# NOTE: Assumes the initial input Hugenholtz diagram has even
#       numbered vertices only, i.e., has already been relabeled
def unwrap_hhz_to_feyn(graph, n_verts_feyn, v_exp=0):
    # Break case: no more Hugenholtz vertices to expand, so this is a Feynman diagram!
    if v_exp == n_verts_feyn:
        return [graph]
    # Otherwise, it is a Hugenholtz or mixed type diagram, so continue to expand
    expanded_graphs = expand_hhz_vert(graph=graph, v_exp=v_exp)
    # Do the recursive step for each of the expanded graphs (maximum of two branches per graph)
    feynman_graphs = []
    for child_graph in expanded_graphs:
        feynman_graphs.extend(unwrap_hhz_to_feyn(
                                                 graph=child_graph,
                                                 n_verts_feyn=n_verts_feyn,
                                                 v_exp=v_exp + 2
                                                ))
    return np.asarray(feynman_graphs).flatten()


# Expands one vertex in a given hhz or mixed hhz/feynman graph from Hugenholtz
# to Feynman type, thus producing two (possibly indistinct) expanded graphs.
# is_irred is a function implementing some irreducibility rule(s) on a graph;
# by default, no rule is implemented, i.e., it always returns True.
def expand_hhz_vert_with_irred_rule(graph, v_exp, is_irred=lambda g: True):
    # Now, expand the vertex and shift one outgoing connection from v_exp accordingly
    moved = list(graph[v_exp].keys())[0]
    graph[v_exp + 1][moved].append(pop_conn_al(graph=graph, v1=v_exp, v2=moved, line_type='f'))
    # Insert the boson line
    graph[v_exp][v_exp + 1].insert(0, 'b')
    graph[v_exp + 1][v_exp].insert(0, 'b')
    # Finally, shift the incoming connections to v_exp in all possible (at most 2) ways
    expanded_graphs = []
    for k in graph:
        # Loops over the two possible crossing types, and generates a new graph for each
        if v_exp in graph[k]:
            if 'f' not in graph[k][v_exp]: continue
            # Create a new graph with the current crossing type
            g_expanded = copy.deepcopy(graph)
            # Adjust one of the two incoming connections to v_exp accordingly
            g_expanded[k][v_exp + 1].append(pop_conn_al(graph=g_expanded, v1=k, v2=v_exp, line_type='f'))
            # Add to the list of expanded graphs if the graph is
            # irreducible (according to whichever specified rule(s))
            if is_irred(g_expanded):
                expanded_graphs.append(g_expanded)
    return expanded_graphs


# Recursively expands all Hugenholtz vertices in a graph to return the full
# list of Feynman diagrams represented by it (there are at most 2^n of them)
# NOTE: Assumes the initial input Hugenholtz diagram has even
#       numbered vertices only, i.e., has already been relabeled
# NOTE: is_irred must be an irreducibility rule function implemented for
#       mixed-type diagrams in order to be applied at every recursive step!
def unwrap_hhz_to_feyn_with_irred_rule(graph, n_verts_feyn, is_irred=lambda g: True, v_exp=0):
    # Break case: no more Hugenholtz vertices to expand, so this is a Feynman diagram!
    if v_exp == n_verts_feyn:
        return [graph]
    # Otherwise, it is a Hugenholtz or mixed type diagram, so continue to expand
    expanded_graphs = expand_hhz_vert_with_irred_rule(graph=graph, v_exp=v_exp, is_irred=is_irred)
    # Do the recursive step for each of the expanded graphs (maximum of two branches per graph)
    feynman_graphs = []
    for child_graph in expanded_graphs:
        feynman_graphs.extend(unwrap_hhz_to_feyn_with_irred_rule(
                                                                 graph=child_graph,
                                                                 n_verts_feyn=n_verts_feyn,
                                                                 is_irred=is_irred,
                                                                 v_exp=v_exp + 2
                                                                ))
    return np.asarray(feynman_graphs).flatten()


# Draw a batch of Hugenholtz diagrams to a bulk .tex file; this will generate a .pdf
# with all of them in a grid (optionally: labeled vertices), as well as standalone
# postscript .1 files.
def draw_bulk_hhz_feynmp(hhz_graphs, fside='left', savename='hhz_diagrams.tex'):
    if hhz_graphs == []:
        print('No hhz_graphs to draw!')
        return
    if fside not in ['left', 'right']:
        raise ValueError("Fermion line side preference for drawing must be either 'left' or 'right'!")
    n_vertices = len(hhz_graphs[0])
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
    # Draw each possible nth order diagram in a separate file
    for i in range(len(hhz_graphs)):
        this_graph = hhz_graphs[i]
        # Draw vertices; there are n
        diagram_str = (
                       r'\begin{fmffile}{hhz_diagram_'+str(i)+'}'+'\n'
                       r'\begin{fmfgraph*}(100,100)'+'\n'
                       r'\fmfsurroundn{v}{'+str(n_vertices)+'}'+'\n'
                       r'\fmfdotn{v}{'+str(n_vertices)+'}'+'\n'
                      )
        # Draw the fermion lines; there are 2n for vacuum graphs, plus
        # 2 for polarization graphs after the external leg addition
        phantom_string = r'\fmf{phantom}{v['+str(n_vertices)+'],'
        for v_start in this_graph:
            phantom_string += 'v['+str(v_start + 1)+'],'
            diagram_str += r'\fmflabel{$'+str(v_start + 1)+r'$}{v['+str(v_start + 1)+']}'+'\n'
            for v_end in this_graph[v_start]:
                # Mark any external vertices (where the connection
                # is either (i, -1) or (-1, i)) by color and label
                if v_end == -1:
                    diagram_str += (
                                    r'\fmflabel{$'+str(v_start + 1)+r'_{\mathrm{ext}}$}{v['+str(v_start + 1)+']}'+'\n'
                                    r'\fmfv{d.sh=circle, d.fi=full, d.si=2thick, f=(1,,0,,0)}{v['+str(v_start + 1)+']}'+'\n'
                                   )
                else:
                    if v_start == v_end:
                        tension = 2
                        diagram_str += (
                                        r'\fmf{fermion, tension='+str(tension)+'}{v['+str(v_start + 1)+'],v['+str(v_end + 1)+']}\n'
                                       )
                    else:
                        for i_line, _ in enumerate(this_graph[v_start][v_end]):
                            arc_angle = 0.3 + (0.2 * i_line)
                            diagram_str += (
                                            r'\fmf{fermion,'+fside+'='+str(arc_angle)+'}{v['+str(v_start + 1)+'],v['+str(v_end + 1)+']}\n'
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
        # Write feynmf diagram contents to file
        myfile.write(diagram_str)
    # Write end of doc statements to file
    eod = '\n'+r'\end{document}'+'\n'
    myfile.write(eod)
    # Done writing; close the file and return
    myfile.close()
    # End of diagram creation loop
    return


# Go from a permutation group to an adjacency list representation
def graph_pg_to_al(graph_pg):
    graph_al = defaultdict(lambda: defaultdict(list))
    for iloc in range(graph_pg.shape[1]):
        # Fermionic connections
        psi_iloc = graph_pg[0, iloc]
        # Skip external vertex; no internal fermionic connection to add!
        if psi_iloc < 0:
            continue
        graph_al[iloc][psi_iloc].append('f')
        # Bosonic connections (mirrored in the adjacency list, as they are undirected)
        phi_iloc = graph_pg[1, iloc]
        # Skip external vertex; no internal fermionic connection to add!
        if phi_iloc < 0:
            continue
        # We insert at the beginning to ensure alphabetical ordering of the connection lists
        graph_al[iloc][phi_iloc].insert(0, 'b')
        graph_al[phi_iloc][iloc].insert(0, 'b')
    return graph_al


# Go from an adjacency list to permutation group representation
def graph_al_to_pg(graph_al, flag_mirrored=True):
    # Catching initialization errors with vertex flag None
    psi_list = len(graph_al) * [None]
    phi_list = len(graph_al) * [None]
    for i in graph_al:
        ferm_ext_flag = True
        bos_ext_flag = True
        for j in graph_al[i]:
            if 'f' in graph_al[i][j]:
                ferm_ext_flag = False
                psi_list[i] = j
            if ('b' in graph_al[i][j]):
                # If there is a bosonic connection, this is not an external vertex
                bos_ext_flag = False
                # Add the bosonic connection only if i < j, if we are flagging mirrored connections
                if flag_mirrored:
                    if i < j:
                        phi_list[i] = j
                    # Exclude the mirrored bosonic connection where i > j; mark this with a -2 flag
                    else:
                        phi_list[i] = -2
                # Otherwise, just add the connection
                else:
                    phi_list[i] = j
        # The outgoing external vertex has no fermionic connections, so we mark it with a -1 flag
        if ferm_ext_flag:
            psi_list[i] = -1
        # If there were no bosonic connections, this is a bosonic external vertex
        if bos_ext_flag:
            phi_list[i] = -1
    # Recast to stacked numpy array and return
    graph_pg = np.array([psi_list, phi_list])
    if any((None in l) for l in [psi_list, phi_list]):
        raise RuntimeError('Permutation group graph initialization failed!\n\n' +
                           f'graph_al was\n{graph_al}\n' +
                           f'but graph_pg computed as\n{graph_pg}')
    return graph_pg


def graph_al_to_el(graph_al, sort=True):
    """Convert from an adjacency list to an edge list representation.

    We represent the graph as a list of bosonic/fermionic edges,
    where the (arbitrary) ordering can be viewed as a choice of
    edge labels.

    Args:
        graph_al (GraphAdjList): Graph in the adjacency list representation
        sort (bool): Sorts the list with bosonic edges first if true.
    Returns:
        GraphEdgeList: Graph in the edge list representation
    """
    graph_el = []
    for v in graph_al:
        for w in graph_al[v]:
            for kind in graph_al[v][w]:
                # Avoid double counting bosonic edges
                if kind == 'b' and [w, v, 'b'] in graph_el:
                    continue
                graph_el.append([v, w, kind])
    if sort:
        graph_el.sort(key=lambda edge: edge[-1])
    return graph_el


def graph_el_to_al(graph_el):
    """Go from an edge list to an adjacency list representation of the graph.

    Args:
        graph_el (GraphEdgeList): Graph in the edge list representation

    Returns:
        GraphAdjList: Graph in the adjacency list representation
    """
    graph_al = defaultdict(lambda: defaultdict(list))
    for v, w, kind in graph_el:
        graph_al[v][w].append(kind)
        # No mirroring for fermionic edges
        if kind == 'f':
            continue
        # If the edge is bosonic, we need to swap it to front of the 'kind' list
        graph_al[v][w].sort()
        # Mirror the bosonic connection
        graph_al[w][v].insert(0, kind)
    return graph_al


# Go from an edge list to a split bosonic/fermionic edge 
# list representation of the graph {'b': el_b, 'f': el_f}
def graph_el_to_split_el(graph_el):
    graph_sel = {'b': [], 'f': []}
    for v1, v2, linetype in graph_el:
        if linetype == 'b':
            graph_sel['b'].append([v1, v2])
        else:
            graph_sel['f'].append([v1, v2])
    return graph_sel


# TODO: Impplement directly
def graph_al_to_split_el(graph_al, sort=True):
    return graph_el_to_split_el(graph_al_to_el(graph_al, sort))


# TODO: Implement directly
def graph_el_to_pg(graph_el):
    return graph_al_to_pg(graph_el_to_al(graph_el))


# TODO: Implement directly
def graph_pg_to_el(graph_pg):
    return graph_al_to_el(graph_pg_to_al(graph_pg))


def make_vacuum_graph_al(psi_list):
    """
    Make an adjacency list representation of the graph, where we mark edge types explicitly
    by adding a weight corresponding to the edge color: w='f' (fermion lines), w='b' (boson lines)

    Args:
        psi_list (list[int]): a list [l_0, l_1, ..., l_2n] implying 
        fermionic connections [(0, l_0), (1, l_1), ..., (2n, l_2n)]

    Returns:
        GraphAdjList: a copy of the adjacency list graph representation
    """
    graph_al = defaultdict(lambda: defaultdict(list))
    for iloc in range(len(psi_list)):
        # Fermionic connections, obtained from the pairings (psi_dagger_{j}, psi_{j})
        graph_al[iloc][psi_list[iloc]].append('f')
        # Bosonic connections, which are fixed by convention as (phi_{j}, phi_{j+1}) for j=0,2,...
        if iloc % 2 == 0:
            # Mirror the connections, as they are undirected; we
            # insert at the beginning to ensure alphabetical ordering
            graph_al[iloc][iloc+1].insert(0, 'b')
            graph_al[iloc+1][iloc].insert(0, 'b')
        graph_al[iloc][psi_list[iloc]].sort()
    return graph_al


# Breadth-first search function for a generic directed graph, represented as an adjacency list;
# thus, expects that all undirected edges in graph are represented by mirrored directed edges.
def bfs(graph, start=0):
    # Create a queue for BFS
    queue = []
    # Marks all the vertices as unvisited by default
    visited = dict(zip(graph.keys(), [False] * len(graph)))
    # Enqueue the source node and mark it as visited
    queue.append(start)
    visited[start] = True
    while queue:
        # Dequeue a vertex from queue
        loc = queue.pop(0)
        # Get all adjacent vertices of the dequeued vertex s. If an adjacent
        # vertex has not been visited, then mark it visited and enqueue it.
        for iadj in graph[loc]:
            if not visited[iadj]:
                queue.append(iadj)
                visited[iadj] = True
    # Necessary to return the boolean visitation array in sorted order wrt the vertex list
    return np.asarray([v for k, v in sorted(visited.items())])


# Constructs a depth-first spanning tree (DFST) rooted at v_start,
# prioritizing either fermionic or bosonic connections
def build_dfst(graph, v_start, prefer='f'):
    # Revert to the default if an invalid edge type preference is given
    if prefer not in ['b', 'f']:
        prefer = 'f'
    visited = set()
    spanning_tree = defaultdict(lambda: defaultdict(list))
    loop_basis_edges = defaultdict(lambda: defaultdict(list))
    # DFS helper recursion function
    def do_dfs(v_curr):
        # Mark the current vertex as visisted
        visited.add(v_curr)
        if visited == range(len(graph)):
            return
        if prefer == 'f':
            # Resort to priotitize fermionic neighbors
            neighbors = [k for (k, v) in sorted(
                graph[v_curr].items(), key=lambda edge: edge[1], reverse=True)]
        else:
            neighbors = graph[v_curr]
        # Now, recurse through the (downstream) neighbors
        for v_adj in neighbors:
            edge_type = graph[v_curr][v_adj][-1]
            if v_adj not in visited:
                # Add the edge between v_curr and v_adj to the spanning tree;
                # we pick the last edge type from the list, if applicable, to
                # prioritize fermionic connections for the spanning tree.
                spanning_tree[v_curr][v_adj].append(edge_type)
                do_dfs(v_adj)
            # If v_adj WAS already visisted, the edge from v_curr to v_adj should
            # be added to the loop basis (complement of the spanning tree),
            # excluding mirrored bosonic connections.
            elif not (conn_exists_al(spanning_tree, v_adj, v_curr, 'b')
                      or conn_exists_al(loop_basis_edges, v_adj, v_curr, 'b')):
                # print(f'Adding ({v_curr}->{v_adj}, {edge_type}) edge to loop basis...')
                print(graph_al_defaultdict_to_dict(spanning_tree), '\n')
                loop_basis_edges[v_curr][v_adj].append(edge_type)
        return
    do_dfs(v_start)
    return spanning_tree, loop_basis_edges


# Returns the loop momenta in terms of the fundamental cycle basis
# constructed via the (oriented) BFS spanning tree with root at
# vertex v_start (by default v_start = 0, which is upstream
# for self-energy graphs by construction)
def get_momentum_loops(graph, v_start=0, diag_type='vacuum', lb_prefer='b'):
    # This algorithm is only sensible for self energy graphs
    # when started at the upstream (incoming external) vertex
    if 'self_en' in diag_type:
        assert v_start == 0

    # Either the opposite of the loop basis edge type
    # preference, or the default ('f') if not applicable
    st_prefer = (set(['b', 'f']) - set(lb_prefer)).pop()

    # Include the external loop for correlation functions
    # by temporarily adding a ficticious edge to the graph
    if any(dt in diag_type for dt in ['poln', 'self_en']):
        # Insert the fake external edge at list index
        # 0 if bosonic, and index 1 if fermionic
        ext_type = ('b' if 'poln' in diag_type else 'f')
        graph[1][0].insert((ext_type == 'b'), ext_type)

    # The number of momentum loops is equivalent
    # to the cycle space dimension (basis size)
    n_vert = len(graph)
    n_mom = (n_vert // 2) + 1

    # Convert the graph to edge list representation,
    # sorting the bosonic edges first by convention
    graph_el = graph_al_to_el(graph)
    n_edges = len(graph_el)

    # Build the DFS spanning tree (and its complement)
    spanning_tree, loop_basis_edges = build_dfst(graph, v_start, st_prefer)
    spanning_tree_el = graph_al_to_el(spanning_tree)
    loop_basis_edges_el = graph_al_to_el(loop_basis_edges)
    loop_basis_edges_sel = graph_el_to_split_el(loop_basis_edges_el)

    # Make sure the size of the calculated loop basis
    # matches the cycle space dimension for this graph
    assert len(loop_basis_edges_el) == n_mom

    # Make sure the bosonic edges are ordered the same in
    # the graph, loop basis, and spanning tree edge lists
    for i, e1 in enumerate(graph_el):
        if e1[2] == 'f':
            continue
        for e2 in spanning_tree_el + loop_basis_edges_el:
            if set(e1) == set(e2):
                graph_el[i] = e2
    print(f'graph_el (orig):\n{graph_el}')
    # Convert the graph to split edge list representation
    graph_sel = graph_el_to_split_el(graph_el)

    # Now, build the momentum loops by connecting each loop basis edge
    # (i, j) to the path in the spanning tree from i to j; since this path
    # is unique by construction, we can simply loop thru it downstream
    assert n_edges == 3 * n_vert // 2
    momentum_loops = np.zeros((n_mom, n_edges), dtype=int)
    for i_loop, edge_list in enumerate(momentum_loops):
        this_basis_edge = loop_basis_edges_el[i_loop]
        v_stop, v_curr = this_basis_edge[:2]
        # We reverse the direction of flow if the basis edge is bosonic
        # and we would be attempting to travel upstream (since we may
        # freely choose the direction of flow for bosonic edges)
        if this_basis_edge[2] == 'b' and not any('f' in v for v in spanning_tree[v_curr].values()):
            # print(f'Swapping basis edge flow direction for loop {i_loop}...')
            v_stop, v_curr = v_curr, v_stop
            for edge in graph_el:
                if edge == this_basis_edge:
                    edge[0], edge[1] = edge[1], edge[0]
            this_basis_edge[0], this_basis_edge[1] = this_basis_edge[1], this_basis_edge[0]
        # Build the loop
        # print(f'\nBuilding loop {i_loop} from basis edge {this_basis_edge}:')
        loop = [graph_el.index(this_basis_edge)]
        flow_signs = [1]
        visited = []
        while True:
            visited.append(v_curr)
            if v_curr == v_stop:
                break
            # NOTE: If v_downstream is not in the spanning tree adjacency
            #       list, then it is a dead end, and we should skip it!
            v_downstream = list(spanning_tree[v_curr].keys())[0]
            if v_curr in spanning_tree and v_downstream not in visited:
                # If the downstream edge is a dead end but is not the
                # final path vertex, skip it and travel upstream instead
                linetype_downstream = list(
                    spanning_tree[v_curr].values())[0][0]
                this_edge = [v_curr, v_downstream, linetype_downstream]
                if v_downstream != v_stop and v_downstream not in spanning_tree:
                    # print(f'Skipping dead end downstream edge {this_edge}...')
                    pass
                else:
                    # Add this edge to the loop and advance downstream
                    # print(f'Adding downstream edge {this_edge}...')
                    loop.append(graph_el.index(this_edge))
                    flow_signs.append(1)
                    # Advance downstream
                    v_curr = v_downstream
                    continue
            # Otherwise, the momentum is flowing opposite the direction of a loop edge
            for v_upstream in spanning_tree:
                # Be sure we are not backtracking along the current loop path
                if v_curr in spanning_tree[v_upstream] and v_upstream not in visited:
                    # Add the edge to the loop
                    linetype_upstream = spanning_tree[v_upstream][v_curr][0]
                    this_edge = [v_upstream, v_curr, linetype_upstream]
                    # print(f'Adding upstream edge {this_edge}...')
                    loop.append(graph_el.index(this_edge))
                    # Minus sign in the loop matrix to represent opposite
                    # flow direction to the basis edge (i.e., -\mathbb{p}_i)
                    flow_signs.append(-1)
                    # Advance upstream
                    v_curr = v_upstream
        # Add this loop to the loop matrix
        assert len(loop) == len(flow_signs)
        assert len(loop) == len(visited)
        edge_list[loop] = flow_signs
        # print('Done!')

    print('\ngraph\n', graph_el, '\n', graph_sel, '\n',
          graph_al_defaultdict_to_dict(graph), '\n')
    
    print('spanning_tree\n', graph_al_to_el(spanning_tree), '\n',
          graph_al_defaultdict_to_dict(spanning_tree), '\n')
    
    print('loop_basis_edges\n', loop_basis_edges_el, '\n', loop_basis_edges_sel, '\n',
          graph_al_defaultdict_to_dict(loop_basis_edges), '\n')
    
    print('momentum_loops\n', momentum_loops, '\n')

    # Remove the ficticious external momentum flow edge from the graph
    if any(dt in diag_type for dt in ['poln', 'self_en']):
        del_conn_al(graph=graph, v1=1, v2=0, line_type=ext_type)
        # print(f'Fixed graph: {graph_al_defaultdict_to_dict(graph)}')

    return momentum_loops, graph_sel, loop_basis_edges_sel


# Determine the total number of fermion loops (i.e., fermion-connected cycles) in the graph
def num_cycles(graph):
    n_cycles = 0
    # Mark all the vertices as unvisited
    visited = [False] * (len(graph))
    # For all possible starting locations, do a bfs
    for start_loc in graph:
        # Create a queue for BFS
        queue = []
        # Mark the source node as visited and enqueue it
        queue.append(start_loc)
        visited[start_loc] = True
        while queue:
            # Dequeue a vertex from queue
            curr_loc = queue.pop(0)
            # Get all adjacent vertices of the dequeued vertex s. If an adjacent
            # vertex has not been visited, then mark it visited and enqueue it.
            for iadj in graph[curr_loc]:
                # Only consider fermionic connections to determine n_cycles
                if 'f' not in graph[curr_loc][iadj]:
                    continue
                # If we haven't been here yet, we are continuing the cycle
                if not visited[iadj]:
                    queue.append(iadj)
                    visited[iadj] = True
                # If we return to starting loc (cycle ends), update the n_cycles count
                if iadj == start_loc:
                    n_cycles += 1
    return n_cycles


# We add the external fermionic path in self-energy terms to the loops array for spin assignment;
# assumes two external legs, and that these are last in the list by convention.
def self_en_ext_ferm_path(graph):
    # Mark all the vertices as initially unvisited
    visited = [False] * (len(graph))
    # Check for fermionic connectivity between all
    # the external legs, starting with the first one
    start_loc = len(graph) - 2
    # Create a queue for BFS
    queue = []
    ext_ferm_path = []
    # Mark the source node as visited and enqueue it
    queue.append(start_loc)
    ext_ferm_path.append(start_loc)
    visited[start_loc] = True
    while queue:
        # Dequeue a vertex from queue
        curr_loc = queue.pop(0)
        # Get all adjacent vertices of the dequeued vertex s. If an adjacent
        # vertex has not been visited, then mark it visited and enqueue it.
        for iadj in graph[curr_loc]:
            # Only consider fermionic connections to determine n_cycles
            if 'f' not in graph[curr_loc][iadj]:
                continue
            # If we haven't been here yet, we are continuing the cycle
            if not visited[iadj]:
                queue.append(iadj)
                ext_ferm_path.append(iadj)
                visited[iadj] = True
            # If we reach the outgoing external vertex, the path is finished
            if iadj == len(graph) - 1:
                return ext_ferm_path
    # Return the truth value on the connectivity of all external vertices
    return []


# Returns all fermion-connected cycles in the graph as a list of lists, as well
# as the number of cycles (i.e., number of fermion loops F for Feynman rules)
def get_cycles(graph, self_en=False):
    n_cycles = 0
    cycles = []
    # Mark all the vertices as unvisited
    visited = [False] * (len(graph))
    # For all possible starting locations, do a bfs
    for start_loc in graph:
        # Create a queue for BFS, as well as a list holding all children of this
        # start_loc; if we traverse a new cycle, we will append this list of
        # vertices to the list of cycles.
        queue = []
        this_traversal = []
        # Mark the source node as visited and enqueue it
        queue.append(start_loc)
        this_traversal.append(start_loc)
        visited[start_loc] = True
        while queue:
            # Dequeue a vertex from queue
            curr_loc = queue.pop(0)
            # Get all adjacent vertices of the dequeued vertex s. If an adjacent
            # vertex has not been visited, then mark it visited and enqueue it.
            for iadj in graph[curr_loc]:
                # Only consider fermionic connections to determine n_cycles
                if 'f' not in graph[curr_loc][iadj]:
                    continue
                # If we haven't been here yet, we are continuing the cycle
                if not visited[iadj]:
                    queue.append(iadj)
                    this_traversal.append(iadj)
                    visited[iadj] = True
                # If we return to starting loc (cycle ends), update the n_cycles count
                if iadj == start_loc:
                    n_cycles += 1
                    # Only add to the list of cycles if an
                    # equivalent cycle is not already present
                    if set(this_traversal) not in set([frozenset(c) for c in cycles]):
                        cycles.append(this_traversal)
    # # Add in the external fermion path to the list of cycles if this is a self-energy term
    # if self_en:
    #     cycles.append(self_en_ext_ferm_path(graph))
    #     n_cycles += 1
    return n_cycles, cycles


# Determine the length of the cycle starting from loc; if there is no cycle, returns 0
def len_cycle(graph, start):
    # Mark all the vertices as unvisited
    visited = [False] * (len(graph))
    num_edges = 0
    cycle_length = 0
    # Create a queue for BFS
    queue = []
    # Mark the source node as visited and enqueue it
    queue.append(start)
    visited[start] = True
    while queue:
        # Dequeue a vertex from queue
        curr_loc = queue.pop(0)
        # Get all adjacent vertices of the dequeued vertex s. If an adjacent
        # vertex has not been visited, then mark it visited and enqueue it.
        for iadj in graph[curr_loc]:
            # Only consider fermionic connections to determine cycle length
            if 'f' not in graph[curr_loc][iadj]:
                continue
            # Update the edge count
            num_edges += 1
            # If we haven't been here yet, we are continuing the cycle
            if not visited[iadj]:
                queue.append(iadj)
                visited[iadj] = True
            # If we return to starting loc (cycle ends), update the cycle length
            if iadj == start:
                cycle_length = num_edges
                break
    return cycle_length


# Get all valid spin polarization diagrams (a subset of all valid charge polarization diagrams)
# from the charge polarization diagrams; if n_legs > 2, the subset taken corresponds only to
# the connected parts of these higher order correlation functions!
def get_valid_spin_poln_from_charge_poln(original_graphs, n_legs=0):
    if n_legs == 0:
        return
    valid_graphs = []
    for graph in original_graphs:
        if external_vertices_connected(graph, n_legs=n_legs):
            valid_graphs.append(graph)
    return valid_graphs


# Checks whether a certain connection is in the graph without initializing it if not;
# since the adjacency list is represented as a defaultdict of defaultdict of lists,
# some care must be taken!
def has_connection(graph, v_start, v_end, line_type):
    return (v_start in graph) and (v_end in graph[v_start]) and (line_type in graph[v_start][v_end])


# The external vertices are the last n_legs vertices in the graph; if they are
# not all connected, this means this is a vanishing spin polarization term
def external_vertices_connected(graph, n_legs=0):
    if n_legs == 0:
        return
    # Mark all the vertices as initially unvisited
    visited = [False] * (len(graph))
    # Check for fermionic connectivity between all
    # the external legs, starting with the first one
    start_loc = list(graph.keys())[-n_legs]
    # Create a queue for BFS
    queue = []
    # Mark the source node as visited and enqueue it
    queue.append(start_loc)
    visited[start_loc] = True
    while queue:
        # Dequeue a vertex from queue
        curr_loc = queue.pop(0)
        # Get all adjacent vertices of the dequeued vertex s. If an adjacent
        # vertex has not been visited, then mark it visited and enqueue it.
        for iadj in graph[curr_loc]:
            # Only consider fermionic connections to determine n_cycles
            if 'f' not in graph[curr_loc][iadj]:
                continue
            # If we haven't been here yet, we are continuing the cycle
            if not visited[iadj]:
                queue.append(iadj)
                visited[iadj] = True
    # Return the truth value on the connectivity of all external vertices
    return np.all(visited[-n_legs:])


# Checks for a bold fermionic connection (i.e., indirect fermion
# connected with no boson line leakage) from v_start to v_end
def bold_ferm_connected(graph, v_start, v_end, diag_type):
    # Mark all the vertices as unvisited
    ferm_visited = np.zeros(len(graph))
    int_bos_visited = np.zeros(len(graph))
    # Create a queue for BFS
    queue = []
    # Mark the source node as visited and enqueue it
    queue.append(v_start)
    ferm_visited[v_start] = 1
    while queue:
        # Dequeue a vertex from queue
        curr_loc = queue.pop(0)
        # Get all adjacent vertices of the dequeued vertex s. If an adjacent
        # vertex has not been visited, then mark it visited and enqueue it.
        for iadj in graph[curr_loc]:
            # Require indirect fermionic connectivity
            if 'f' not in graph[curr_loc][iadj]:
                continue
            # If we visit v_end, the vertices are indirect fermion-connected
            if (iadj == v_end):
                # If there was no internal boson line leakage, i.e., all
                # visited bosonic connections were along the fermionic
                # path, then the vertices are bold fermion-connected
                no_bos_leakage = np.all((ferm_visited - int_bos_visited) >= 0)
                # For polarization diagrams, indirect fermion connections don't
                # count if we pass through an exeternal vertex along the path,
                # as this is regarded as external boson line leakage!
                if ('poln' in diag_type) and np.any(ferm_visited[-2:]):
                    no_bos_leakage = False
                if no_bos_leakage:
                    return True
            # If we haven't been here yet, continue traversing the fermionic path
            if ferm_visited[iadj] == 0:
                queue.append(iadj)
                ferm_visited[iadj] = 1
            # Also mark internal vertices with outgoing bosonic
            # connections in order to check for bold connectivity
            for v_out in graph[iadj]:
                if 'b' in graph[iadj][v_out]:
                    int_bos_visited[iadj] = 1
                    int_bos_visited[v_out] = 1
    return False


# Returns the undirected version of the input directed-undirected
# multigraph by mirroring all fermionic connections
def get_undirected(graph):
    undirected_graph = copy.deepcopy(graph)
    for iloc in undirected_graph:
        for jloc in undirected_graph[iloc]:
            if ('f' in undirected_graph[iloc][jloc]
                    and 'f' not in undirected_graph[jloc][iloc]):
                undirected_graph[jloc][iloc].append('f')
    return undirected_graph


# Returns a boolean array indicating all undirected neighbors of start in graph
def undirected_neighbors(graph, v_start):
    return bfs(get_undirected(graph), start=v_start)


# Returns the list of all nearest neighbors for the base vertex, in the undirected sense.
# If the neighbors are to be 'signed', appends a negative sign to the vertex ID for any
# non-directed neighbor (i.e., an edge which flows into v_base, but not out of it).
def undirected_nearest_neighbors(graph, v_base, sort=True, signed=True):
    if sort:
        undir_nns = sorted(list(get_undirected(graph)[v_base].keys()))
    else:
        undir_nns = list(get_undirected(graph)[v_base].keys())
    if signed:
        for i, n in enumerate(undir_nns):
            if n not in list(graph[v_base].keys()):
                undir_nns[i] = -n
    return undir_nns


# Returns the list of all nearest neighbors 
# for the base vertex, in the undirected sense
def get_nearest_neighbor_list(graph, sort=True, signed=True):
    return [undirected_nearest_neighbors(graph, v, sort, signed) for v in sorted(graph.keys())]


# Do a breadth-first search to check for UNDIRECTED graph connectivity;
# the starting location is hence unimportant. This means we must convert
# the directed-undirected input multigraph to a fully undirected graph.
# For Hugenholtz diagrams, this simplifies to a regular directed bfs.
def is_connected(graph):
    return np.all(bfs(graph, start=0))


# Do a breadth-first search to check for UNDIRECTED graph connectivity;
# the starting location is hence unimportant. This means we must convert
# the directed-undirected input multigraph to a fully undirected graph.
def is_connected_undirected(graph):
    return np.all(bfs(get_undirected(graph), start=0))


# Get the connected subset of a list of Feynman or Hugenholtz graphs
def get_connected_graphs(original_graphs):
    connected_graphs = []
    for graph in original_graphs:
        if is_connected(graph):
            connected_graphs.append(graph)
    return connected_graphs


# Check for one-boson irreducibility (implies both connectivity and Hartree irreducibility)
def is_1BI(graph):
    # Make sure we are looking through all bosonic edges
    for iloc in range(0, max(graph.keys()) + 1, 2):
        # Create a deep copy of the graph
        cut_graph = copy.deepcopy(graph)
        # Remove the current boson line, if it exists (a necessary check for mixed-type diagrams)
        del_conn_if_exists_al(graph=cut_graph, v1=iloc  , v2=iloc+1, line_type='b')
        del_conn_if_exists_al(graph=cut_graph, v1=iloc+1, v2=iloc  , line_type='b')
        # If we ever get a disconnected graph as a result, the graph is not 1BI
        if not is_connected_undirected(cut_graph): return False
    # If we made it this far, the graph is 1BI
    return True


# Check for two-boson irreducibility (for exclusion of all polarization subdiagrams)
# NOTE: Perhaps counterintuitively we define a graph with less than two
#       bosonic connections as 2BI (as it has no polarization insertions)
def is_2BI(graph, n_legs=0, diag_type='vacuum'):
    if len(graph) < 4: return True
    # Get all pairs of bosonic connections in the graph; note that this
    # definition is sufficiently general to work for mixed-type graphs!
    n_verts = len(graph)
    boson_lines = [(i, j) for i in graph for j in graph[i] if ('b' in graph[i][j]) and (j > i)]
    # print(boson_lines)
    # Now, consider the bosonic external vertices as connected for purposes of 2BI for polarization diagrams;
    # we achieve this by adding in an artificial bosonic connection (not counted in list of real boson lines)
    if ('poln' in diag_type):
        g = copy.deepcopy(graph)
        g[n_verts - n_legs][n_verts - n_legs + 1].insert(0, 'b')
        g[n_verts - n_legs + 1][n_verts - n_legs].insert(0, 'b')
    # For other diagram types, the 2BI condition is the same as for vacuum graphs
    else: g = graph
    # Iterate over all pairs of boson edges, so that we can randomly delete two at a time
    for boson_line_pair in itertools.combinations(boson_lines, 2):
        # Create a deep copy of the (possibly modified) graph g
        cut_graph = copy.deepcopy(g)
        # Delete the two boson lines
        for line in boson_line_pair:
            # Remove the current boson line if it exists (a necessary check for mixed-type diagrams)
            del_conn_if_exists_al(graph=cut_graph, v1=line[0], v2=line[1], line_type='b')
            del_conn_if_exists_al(graph=cut_graph, v1=line[1], v2=line[0], line_type='b')
            # If we ever get a disconnected result, the graph is not 2BI
            if not is_connected_undirected(cut_graph): return False
    # If we made it this far, the graph is 2BI
    return True


# Check for one-fermion irreducibility
# (to obtain proper self-energy graphs)
def is_1FI(graph):
    # Look through all fermionic edges
    for iloc in graph:
        # Create a deep copy of the graph
        cut_graph = copy.deepcopy(graph)
        # Remove the current fermion line
        for jloc in cut_graph[iloc]:
            if 'f' in cut_graph[iloc][jloc]:
                del_conn_al(graph=cut_graph, v1=iloc, v2=jloc, line_type='f')
                break
        # If we ever get a disconnected result, the graph is not 1BI
        if not is_connected_undirected(cut_graph): return False
    # If we made it this far, the graph is 1FI
    return True


# Check for two-fermion irreducibility (for exclusion of all self-energy subdiagrams)
# # of vacuum diagrams (NOTE: assumes n_legs = 0!)
# NOTE: this function works for all diagram types (Feynman, Hugenholtz, and mixed)
# by assuming that ALL non-Feynman diagrams are 2FI (to avoid false negatives)!
def is_2FI(graph, n_legs=0, diag_type='vacuum'):
    # First, deduce whether this is a Feynman-type vacuum diagram
    # by counting the boson lines and total vertex number
    n_verts = len(graph)
    n_boson_lines = 0
    for i in graph:
        for j in graph[i]:
            if ('b' in graph[i][j]) and (j > i):
                n_boson_lines += 1
    # If the diagram is not Feynman-type, define 2FI as true to avoid false negatives
    if n_verts != 2 * (n_boson_lines + ('poln' in diag_type)): return True
    # Now, consider the bosonic external vertices as connected for purposes of 2FI for polarization diagrams; we
    # achieve this by adding in an artificial bosonic connection (not included in the number of real boson lines above)
    if ('poln' in diag_type):
        g = copy.deepcopy(graph)
        g[n_verts - n_legs][n_verts - n_legs + 1].insert(0, 'b')
        g[n_verts - n_legs + 1][n_verts - n_legs].insert(0, 'b')
    else: g = graph
    # For other diagram types, the 2FI condition is the same as for vacuum graphs
    # Iterate over all pairs of vertices, so that we can randomly delete two
    # fermion lines (originating from the specified vertices) at a time.
    for iloc_pair in itertools.combinations(range(len(graph)), 2):
        # Create a deep copy of the (possibly modified) graph g
        cut_graph = copy.deepcopy(g)
        # Delete the two fermion lines
        for iloc in iloc_pair:
            # Remove the current fermion line
            for jloc in cut_graph[iloc]:
                if 'f' in cut_graph[iloc][jloc]:
                    del_conn_al(graph=cut_graph, v1=iloc, v2=jloc, line_type='f')
                    break
            # If we ever get a disconnected result, the graph is not 2FI
            if not is_connected_undirected(cut_graph): return False
    # If we made it this far, the graph is 2FI
    return True


# Checks if a diagram is 1BI + 2BI + 2FI. For vacuum diagrams, this is equivalent
# to a check for whether it is in the bold series. For correlation functions, the same
# is true, with the caviat that polarization incoming/outgoing connections be regarded as
# virtually connected (for consistency with the definition via functional differentiation)
def is_bold(graph, n_legs=0, diag_type='vacuum'):
    return (is_1BI(graph) and is_2BI(graph, n_legs, diag_type) and is_2FI(graph, n_legs, diag_type))


# Check for polarization bubble (P0) irreducibility
# (for exclusion of all tadpole terms and first order (P0) boson self-energy subdiagrams)
def is_PBI(graph, n_legs=0, is_self_en=False):
    # If this is a trivial graph, i.e., Fock free- or self- energies,
    # it is PBI for free (only one interaction line => n_verts < 4)
    if len(graph) < 4: return True
    # Get all pairs of bosonic connections in the graph; note that this
    # definition is sufficiently general to work for mixed-type graphs!
    boson_lines = [(i, j) for i in graph for j in graph[i] if ('b' in graph[i][j]) and (j > i)]
    # If there are less than two boson lines in the graph, it cannot be PBR!
    if (len(boson_lines) < 2): return True
    # Iterate over all pairs of boson edges, so that we can randomly delete two at a time;
    # if the leg type is bosonic, we must ignore the last n_leg vertices,
    # which are external and thus have no explicit bosonic connections
    for boson_line_pair in itertools.combinations(boson_lines, 2):
        # Create a deep copy of the graph
        cut_graph = copy.deepcopy(graph)
        # Delete the two boson lines
        for boson_line in boson_line_pair:
            v_out, v_in = boson_line[0], boson_line[1]
            del_conn_al(graph=cut_graph, v1=v_out, v2=v_in , line_type='b')
            del_conn_al(graph=cut_graph, v1=v_in , v2=v_out, line_type='b')
        # Check for connectivity now that they have been deleted
        subresult = is_connected_undirected(cut_graph)
        # If we find a disconnected (2BR) result, check for PBR before discarding;
        # there are at most two subgraphs created by this deletion!
        if not subresult:
            # Size of the whole cut graph
            size = len(cut_graph)
            # wlog, we define subgraph 'a' to include the vertex (n_verts - n_legs),
            # i.e., the incoming vertex for correlation functions, and
            # subgraph 'b' is consequently the set complement; get the
            # vertices in subgraph 'a' via an undirected neighbor search
            if n_legs == 0:
                start_a = 0
            else:
                start_a = size - n_legs
            subgraph_a = undirected_neighbors(cut_graph, v_start=start_a)
            # subsize_a is the number of undirected neighbors,
            # and subsize_b is then size - subsize_a; a cyclic
            # subgraph of size 2 implies a bubble insertion!
            subsize_a = np.sum(subgraph_a)
            subsize_b = size - subsize_a
            # To get the length of subgraph b, we need to choose
            # a starting point that wasn't in subgraph a.
            # start_b = np.where(subgraph_a == False)[0][0]
            # In the case of a self-energy term, a cyclic subgraph of size 2 is
            # possible only for subgraph b not containing the external vertices
            if is_self_en:
                is_PBR = (subsize_b == 2)
            # Otherwise, the graph is PBR if either one of the subgraphs is of size 2
            else:
                is_PBR = (subsize_a == 2) or (subsize_b == 2)
            # If the 2BR graph isn't PBR, re-mark it for inclusion (present in W0 series)
            if not is_PBR:
                subresult = True
        # If the graph is PBR, don't drink it!
        if not subresult: return False
    # If we made it this far, the graph is PBI
    return True


# Brute-force check for self-consistent Fock irreducibility by searching all vertex pairs for a bare
# Fock insertion; a recursive algorithm would scale better, but this is simpler, and good enough for low orders.
def is_FI_simple(graph, n_legs):
    n_verts = len(graph)
    # Check if this graph is the bare Fock self-energy term assumes the two
    # external legs (if any) are ordered last in the vertex numbering scheme.
    if (n_legs == 2) and all(has_connection(graph, n_verts - 2, n_verts - 1, line_type) for line_type in ['b', 'f']):
        return False
    # Check the internal connections only; the only external-leg base case is handled above
    for v1, v2 in itertools.combinations(range(len(graph) - n_legs), 2):
        n_ferm_connections = has_connection(graph, v1, v2, 'f') + has_connection(graph, v2, v1, 'f')
        n_bos_connections = has_connection(graph, v1, v2, 'b')
        # Catches the Fock free-energy edge case where n_ferm_connections = 2
        if (n_ferm_connections >= 1) and (n_bos_connections == 1):
            return False
    return True


# Brute-force check for bold Fock irreducibility by searching all vertex pairs for a bold Fock term;
# a recursive algorithm would scale better, but this is simpler, and good enough for low orders.
def is_bFI_simple(graph, n_legs, diag_type):
    n_verts = len(graph)
    # Check if this graph is itself a bold Fock self-energy term (external vertices
    # boson-connected); assumes the external legs are last in the vertex numbering scheme.
    if (n_legs == 2) and has_connection(graph, n_verts - 2, n_verts - 1, 'b'):
        return False
    # Now check the internal connections for any bold Fock terms
    for v1, v2 in itertools.combinations(range(len(graph) - n_legs), 2):
        n_indirect_ferm_connections = (bold_ferm_connected(graph, v_start=v1, v_end=v2, diag_type=diag_type)
                                       + bold_ferm_connected(graph, v_start=v2, v_end=v1, diag_type=diag_type))
        n_bos_connections = has_connection(graph, v1, v2, 'b')
        # Catches the Fock free-energy edge case where n_indirect_ferm_connections = 2
        if (n_indirect_ferm_connections >= 1) and (n_bos_connections == 1):
            return False
    return True


# TODO: remove bug by checking for internal BOLD Fock terms, as in is_bFI_simple
# Check whether this self energy graph is bold Fock irreducible
# NOTE: assumes two legs and 1FI (proper self energy), and that the last two vertices are the external ones
def is_bFI(self_energy_graph):
    n_verts = len(self_energy_graph)
    # If this self energy graph is itself a Fock term, it is not FI
    if 'b' in self_energy_graph[n_verts - 2][n_verts - 1]:
        return False
    # Unmake the graph to easily get the pairs of all fermionic connections between vertices
    psi_list = graph_al_to_pg(self_energy_graph)[0]
    ferm_lines = [(i, psi_list[i]) for i in range(len(self_energy_graph)) if (psi_list[i] >= 0)]
    # Now, check whether the graph contains any Fock insertions; if so, it is FR
    for ferm_line_pair in itertools.combinations(ferm_lines, 2):
        # Ceate a deep copy of the graph
        cut_graph = copy.deepcopy(self_energy_graph)
        # Record the four vertices at the cut lines
        v_cuts = []
        # Delete the two fermion lines specified by the sources iloc_pair = (iloc_1, iloc_2)
        # for iline, ferm_line in enumerate(ferm_line_pair):
        for ferm_line in ferm_line_pair:
            v_out, v_in = ferm_line[0], ferm_line[1]
            # Add this pair of vertices to the list of ones occuring at the cut lines
            v_cuts.extend([v_out, v_in])
            # Remove the current fermion line
            del_conn_al(graph=cut_graph, v1=v_out, v2=v_in, line_type='f')
        # If we find a disconnected (2FR) result, check for FR before discarding;
        # there are at most two subgraphs created by this deletion!
        if not is_connected_undirected(cut_graph):
            # Size of the whole cut graph
            size = len(cut_graph)
            # We define subgraph A to include the incoming external vertex #(len(cut_graph) - 2)
            start_a = size - 2
            subgraph_a = undirected_neighbors(cut_graph, v_start=start_a)
            # Subgraph A is the connected part of the self-energy graph; it must
            # include the other external vertex as the original graph was 1FI
            assert subgraph_a[size - 1]
            # Determine which pair of vertices corresponding
            # to the cut lines is in subgraph A
            v_cuts_a = []
            v_cuts_b = []
            for v in v_cuts:
                if subgraph_a[v]:
                    v_cuts_a.append(v)
                else:
                    v_cuts_b.append(v)
            if (len(v_cuts_a) != 2) or (len(v_cuts_b) != 2):
                print(cut_graph, '\n', subgraph_a, '\n', v_cuts_a, '\n', v_cuts_b)
                raise ValueError('Bad number of v_cuts in subgraph a/b! (!= 2)')
            # Now, check if the disconnected piece is a Fock insertion
            if 'b' in cut_graph[v_cuts_b[0]][v_cuts_b[1]]:
                # If so, the graph is not FI
                return False
    # Otherwise, the graph is FI
    return True


# Check for Fock and polarization bubble irreducibility (FPBI)
def is_HFPBI(graph, n_legs, diag_type):
    return is_FI_simple(graph, n_legs) and is_PBI(graph, n_legs, ('self_en' in diag_type))


def get_bold_graphs(original_graphs, n_legs=0, diag_type='vacuum'):
    bold_graphs = []
    for graph in original_graphs:
        if is_bold(graph, n_legs, diag_type):
            bold_graphs.append(graph)
    return bold_graphs


# Note: original_graphs can be either a list or generator of graph objects!
def get_1BI_graphs(original_graphs):
    graphs_1BI = []
    for graph in original_graphs:
        if is_1BI(graph):
            graphs_1BI.append(graph)
    return graphs_1BI


# Note: original_graphs can be either a list or generator of graph objects!
def get_PBI_graphs(original_graphs, n_legs, diag_type):
    graphs_PBI = []
    for graph in original_graphs:
        if is_PBI(graph, n_legs, ('self_en' in diag_type)):
            graphs_PBI.append(graph)
    return graphs_PBI


def get_HFPBI_graphs(original_graphs, n_legs, diag_type):
    graphs_HFPBI = []
    for graph in original_graphs:
        if is_HFPBI(graph, n_legs, diag_type):
            graphs_HFPBI.append(graph)
    return graphs_HFPBI


# Iteratively generates all nth-order self-consistent GW graphs in the bare series.
# To do so, we use the bare GW self-energy and polarization as base cases. Duplicate
# topologies are generated in the process and are removed in a post-processing step.
def get_C1_graphs(order, n_legs=0, diag_type='vacuum'):
    graphs_m = []
    # Free energy base case graph \Psi_1 = G W G
    if 'vacuum' in diag_type:
        n_legs = 0
        free_en_1 = graph_al_dict_to_defaultdict({
                                                  0: {1: ['b', 'f']},
                                                  1: {0: ['b', 'f']},
                                                 })
        graphs_m.append(free_en_1)
    # Self energy base case graph \Sigma_1 = G W
    elif 'self_en' in diag_type:
        n_legs = 2
        self_en_1 = graph_al_dict_to_defaultdict({
                                                  0: {1: ['b', 'f']},
                                                  1: {0: ['b']},
                                                 })
        graphs_m.append(self_en_1)
    # Polarization base case graph P_1 = G G
    else:
        assert 'poln' in diag_type
        n_legs = 2
        poln_1 = graph_al_dict_to_defaultdict({
                                               0: {1: ['f']},
                                               1: {0: ['f']},
                                              })
        graphs_m.append(poln_1)
    # Stop at base case if n = 1
    if order == 1:
        return graphs_m
    # Otherwise, build up order-by-order from the base case,
    # m = 1, ..., n - 1, by inserting Fock and polarization
    # bubble terms in all possible ways
    graphs_mp1 = []
    for m in range(1, order):
        n_verts_m = 2 * m
        graphs_mp1 = []
        # For every graph at the previous order
        for graph_m in graphs_m:
            # Insert a fermionic self-energy in all possible ways
            for f_start in range(0, n_verts_m):
                graphs_mp1.append(insert_fock(graph_m, n_verts_m, f_start, diag_type))
            # Insert a bosonic self-energy in all possible ways
            for b_start in range(0, n_verts_m, 2):
                graphs_mp1.append(insert_bubble(graph_m, n_verts_m, b_start, diag_type))
        # Move up one diagram order and repeat until m + 1 = n
        graphs_m = graphs_mp1
    # return rem_top_equiv_al(graphs_mp1, 2 * order, n_legs)
    assert graphs_m == graphs_mp1 # type: ignore
    return rem_top_equiv_al(graphs_m, 2 * order, n_legs)


# Iteratively generates all nth-order self-consistent C2 graphs in the bare series.
# We hard-code the bare C2 self-energy and polarization terms as base cases. Duplicate
# topologies are generated in the process and are removed in a post-processing step.
def get_C2_graphs(order, n_legs=0, diag_type='vacuum'):
    # Free energy base case graphs
    free_en_1 = graph_al_dict_to_defaultdict({
                                              0: {1: ['b', 'f']},
                                              1: {0: ['b', 'f']},
                                             })
    free_en_2 = graph_al_dict_to_defaultdict({
                                              0: {1: ['b'], 2: ['f']},
                                              1: {0: ['b'], 3: ['f']},
                                              2: {1: ['f'], 3: ['b']},
                                              3: {0: ['f'], 2: ['b']},
                                             })
    free_en_base_set = [free_en_1, free_en_2]
    # Self energy base case graphs
    self_en_1 = graph_al_dict_to_defaultdict({
                                              0: {1: ['b', 'f']},
                                              1: {0: ['b']},
                                             })
    self_en_2 = graph_al_dict_to_defaultdict({
                                              0: {1: ['b'], 2: ['f']},
                                              1: {0: ['b'], 3: ['f']},
                                              2: {1: ['f'], 3: ['b']},
                                              3: {2: ['b']},
                                             })
    self_en_base_set = [self_en_1, self_en_2]
    # Polarization base case graphs
    poln_1 = graph_al_dict_to_defaultdict({
                                           0: {1: ['f']},
                                           1: {0: ['f']},
                                          })
    poln_2 = graph_al_dict_to_defaultdict({
                                           0: {1: ['b'], 2: ['f']},
                                           1: {0: ['b'], 3: ['f']},
                                           2: {1: ['f']},
                                           3: {0: ['f']},
                                          })
    poln_base_set = [poln_1, poln_2]
    # Now build the set of re-expanded diagrams according to the specified diag_type
    graphs_m = []
    if 'vacuum' in diag_type:
        n_legs = 0
        graphs_m.extend(free_en_base_set)
    elif 'self_en' in diag_type:
        n_legs = 2
        graphs_m.extend(self_en_base_set)
    else:
        assert 'poln' in diag_type
        n_legs = 2
        graphs_m.extend(poln_base_set)
    # Stop at base case if n = 1
    if order == 1:
        print('Done!')
        return graphs_m[:1]
    # Otherwise, build up order-by-order from the base case,
    # m = 1, ..., n - 1, by inserting Fock and polarization
    # bubble terms in all possible ways
    reexpanded_graphs = []
    for _ in range(1, order + 1):
        # For every graph at the mth iteration
        graphs_mp1 = []
        for curr_graph in graphs_m:
            # Determine the number of vertices in the current graph (it is variable!)
            n_verts_m = len(curr_graph)
            # If this graph is of order N, it is in the final set of reexpanded graphs
            if n_verts_m == 2 * order:
                reexpanded_graphs.append(curr_graph)
                continue
            # Otherwise, insert a fermionic self-energy in all possible ways
            for f_start in range(0, n_verts_m):
                # Consider every possible base-case insertion that results in an nth-order diagram
                for self_en in self_en_base_set:
                    if n_verts_m + len(self_en) <= 2 * order:
                        graphs_mp1.append(insert_self_en(curr_graph, self_en, n_verts_m, f_start, diag_type))
            # Then, insert a bosonic self-energy in all possible ways
            for b_start in range(0, n_verts_m, 2):
                # Consider every possible base-case insertion that results in an nth-order diagram
                for poln in poln_base_set:
                    if n_verts_m + len(poln) <= 2 * order:
                        graphs_mp1.append(insert_poln(curr_graph, poln, n_verts_m, b_start, diag_type))
        # Move up one diagram order and repeat until m + 1 = n
        graphs_m = graphs_mp1
    # Double-check that all re-expanded graphs are indeed at the
    # specified order n, then return the topologically distinct graphs
    assert np.all(np.asarray(map(len, reexpanded_graphs)) == 2 * order)
    return rem_top_equiv_al(reexpanded_graphs, 2*order, n_legs)


# Iteratively generates all nth-order self-consistent BSE2 graphs in the bold series.
# We hard-code the 2nd-order approximation to the Bethe-Salpeter kernel (irreducible 4-point vertex function)
# as a base case. Duplicate topologies are generated in the process and are removed in a post-processing step.
#
# if diag_type == 'self_en':
#     n_rots = gamma3_base_orders
#     for ig, g in gamma3_base_set:
#         vmap_C_n = lambda keys: dict(zip(keys, np.roll(keys, n_rots[ig])))
#         gamma3_base_set[ig] = lambda i: map_vertices_defaultdict(g(i), vmap=vmap_C_n(g(i).keys()))
#
# TODO: Implement this function for a general 3-point vertex insertion, not
#       just at the outgoing external vertex (v = 1) for correlation functions!
#
def get_BSE2_graphs(order, n_legs=2, diag_type='poln'):
    if (n_legs != 2) or all(t not in diag_type for t in ['poln', 'self_en']):
        raise ValueError('The Bethe-Salpeter based approximation has only been implemented'
                         + 'for two-leg correlation functions (polarization and self-energy diagrams) so far!')
    # Set up polarization type BSE2 insertions
    if 'poln' in diag_type:
        # Bethe-Salpeter three-point vertex function using theta kernel base case graphs
        n_verts = 2 * order
        def gamma3_1(i): return graph_al_dict_to_defaultdict({
            1: {(i + 1): ['f']},
            i: {1: ['f'], (i + 1): ['b']},
            (i + 1): {i: ['b']},
            })
        def gamma3_21(i): return graph_al_dict_to_defaultdict({
            1: {(i + 3): ['f']},
            i: {(i + 1): ['b'], (i + 2): ['f']},
            (i + 1): {1: ['f'], i: ['b']},
            (i + 2): {(i + 3): ['b']},
            (i + 3): {(i + 1): ['f'], (i + 2): ['b']},
            })
        def gamma3_22(i): return graph_al_dict_to_defaultdict({
            1: {(i + 1): ['f']},
            i: {(i + 1): ['b'], (i + 2): ['f']},
            (i + 1): {i: ['b'], (i + 3): ['f']},
            (i + 2): {(i + 3): ['b']},
            (i + 3): {1: ['f'], (i + 2): ['b']},
            })
        def gamma3_1_vlinks(i): return [i, i + 1]
        def gamma3_21_vlinks(i): return [i, i + 2]
        def gamma3_22_vlinks(i): return [i, i + 2]
        # Build the base set lists
        gamma3_base_set = [gamma3_1, gamma3_21, gamma3_22]
        vlinks = [gamma3_1_vlinks, gamma3_21_vlinks, gamma3_22_vlinks]
        gamma3_base_orders = list(map(lambda g: len(g(0)) / 2, gamma3_base_set))
        assert gamma3_base_orders == [1, 2, 2]
        # Derived polarization base case graphs
        poln_base_graph = graph_al_dict_to_defaultdict({ 0: {1: ['f']}, 1: {0: ['f']} })
        graphs_m = unwrap_BSA(
            graph=poln_base_graph, 
            gamma3_base_set=gamma3_base_set, 
            vlinks=vlinks, 
            gamma3_base_orders=gamma3_base_orders, 
            max_order=order,
            diag_type='poln',
        )
    # Set up self-energy type BSE2 insertions
    else:
        # Bethe-Salpeter three-point vertex function using theta kernel base case graphs
        n_verts = 2 * order
        def gamma3_1(i): return graph_al_dict_to_defaultdict({
            1: {(i + 1): ['b']},
            i: {1: ['f']},
            (i + 1): {1: ['b'], i: ['f']},
            })
        def gamma3_21(i): return graph_al_dict_to_defaultdict({
            1: {(i + 1): ['b']},
            i: {(i + 1): ['f']},
            (i + 1): {1: ['b'], (i + 2): ['f']},
            (i + 2): {i: ['f'], (i + 3): ['b']},
            (i + 3): {1: ['f'], (i + 2): ['b']},
            })
        def gamma3_22(i): return graph_al_dict_to_defaultdict({
            1: {(i + 1): ['b']},
            i: {(i + 2): ['f']},
            (i + 1): {1: ['b'], i: ['f']},
            (i + 2): {(i + 1): ['f'], (i + 3): ['b']},
            (i + 3): {1: ['f'], (i + 2): ['b']},
            })
        def gamma3_1_vlinks(i): return [i, i + 1]
        def gamma3_21_vlinks(i): return [i, i + 3]
        def gamma3_22_vlinks(i): return [i, i + 3]
        # Build the base set lists
        gamma3_base_set = [gamma3_1, gamma3_21, gamma3_22]
        vlinks = [gamma3_1_vlinks, gamma3_21_vlinks, gamma3_22_vlinks]
        gamma3_base_orders = list(map(lambda g: len(g(0)) / 2, gamma3_base_set))
        assert gamma3_base_orders == [1, 2, 2]
        # Derived self-energy base case graphs
        self_en_base_graph = graph_al_dict_to_defaultdict({ 0: {1: ['b', 'f']}, 1: {0: ['b']} })
        graphs_m = unwrap_BSA(
            graph=self_en_base_graph, 
            gamma3_base_set=gamma3_base_set, 
            vlinks=vlinks, 
            gamma3_base_orders=gamma3_base_orders, 
            max_order=order,
            diag_type='self_en',
        )
    return rem_top_equiv_al(graphs_m, n_verts, n_legs)


def unwrap_BSA(graph, gamma3_base_set, vlinks, gamma3_base_orders, max_order, diag_type):
    n_verts = len(graph)
    # Break case 1: the current graph is greater than the desired maximum order, so ignore it
    if n_verts > 2 * max_order:
        return []
    # Break case 2: the current graph is of the desired maximum order, so we return it
    if n_verts == 2 * max_order:
        return [graph]
    # Otherwise, keep iterating using the approximate base set for theta
    iterated_graphs = []
    for i in range(len(gamma3_base_set)):
        iterated_graphs.append(
            insert_gamma3(
                diag_type=diag_type,
                graph=graph,
                gamma3=gamma3_base_set[i](n_verts),
                vlinks=vlinks[i](n_verts),
            )
        )
    unwrapped_graphs = []
    for child_graph in iterated_graphs:
        unwrapped_graphs.extend(
            unwrap_BSA(
                graph=child_graph,
                gamma3_base_set=gamma3_base_set,
                vlinks=vlinks,
                gamma3_base_orders=gamma3_base_orders,
                max_order=max_order,
                diag_type=diag_type,
            )
        )
    return np.asarray(unwrapped_graphs).flatten()


# TODO: the external leg convention must be maintained for self-energy diagrams! (partially resolved, at low orders)
def insert_gamma3(diag_type, graph, gamma3, vlinks, v_insert=1):
    # Add the gamma3 term into the graph
    # print(vlinks)
    # n_vert_orig = len(graph)
    new_graph = copy.deepcopy(graph)
    # gamma = copy.deepcopy(gamma3)
    # update_graph_al(g1=new_graph, g2=gamma)
    update_graph_al(g1=new_graph, g2=gamma3)
    ilink = 0
    if 'poln' in diag_type:
        # Adjust the incoming fermionic connection to v_insert
        for v1 in graph:
            if v_insert in new_graph[v1]:
                move_conn_al(
                    graph=new_graph,
                    conn_old=(v1, v_insert),
                    conn_new=(v1, vlinks[0]),
                    line_type='f',
                )
                ilink += 1
                break
        # Adjust the outgoing fermionic connection from v_insert
        for v2 in graph[v_insert]:
            if 'f' in new_graph[v_insert][v2]:
                move_conn_al(
                    graph=new_graph,
                    conn_old=(v_insert, v2),
                    conn_new=(vlinks[1], v2),
                    line_type='f',
                )
                ilink += 1
                break
    else:
        # Adjust the incoming bosonic connection to v_insert
        for v1 in graph:
            if v_insert in new_graph[v1]:
                if 'b' in new_graph[v1][v_insert]:
                    move_conn_al(
                        graph=new_graph,
                        conn_old=(v1, v_insert),
                        conn_new=(v1, vlinks[0]),
                        line_type='b',
                    )
                    ilink += 1
                    break
        # Adjust the incoming fermionic connection to v_insert
        for v1 in graph:
            if v_insert in new_graph[v1]:
                if 'f' in new_graph[v1][v_insert]:
                    move_conn_al(
                        graph=new_graph,
                        conn_old=(v1, v_insert),
                        conn_new=(v1, vlinks[1]),
                        line_type='f',
                    )
                    ilink += 1
                    break
    assert ilink == 2
    return new_graph


# Inserts the given self-energy term into a fermion line
# in the graph specified by the source vertex v_start.
# NOTE: In order to preserve the bosonic convention, the
# self-energy insertion should be defined accordingly,
# i.e., the external leg convention should not be enforced!
def insert_self_en(graph, self_en, n_verts, v_start, diag_type):
    if 'self_en' in diag_type:
        raise ValueError('Function insert_self_en() not yet implemented for self energy '
                         +'diagrams, where the external leg convention must be maintained!')
    # Merge the original and self-energy graphs into a single (disconnected) graph;
    # in the process, we need to shift all vertex labels in the self-energy term to
    # begin at n_verts
    graph_with_insertion = copy.deepcopy(graph)
    self_en_shifted = map_vertices_defaultdict(self_en, vmap=np.asarray(self_en.keys()) + n_verts)
    graph_with_insertion.update(self_en_shifted)
    # Search for the outgoing fermionic connection
    for v_end in graph[v_start]:
        # When we find it:
        if 'f' in graph[v_start][v_end]:
            # Remove the original fermion line f(i -> j)
            del_conn_al(graph=graph_with_insertion, v1=v_start, v2=v_end, line_type='f')
            # Connect up the original and self-energy subgraphs, i.e.,
            # replace the original fermion line with (G_0 \Sigma G_0)
            graph_with_insertion[v_start][n_verts] = ['f']
            graph_with_insertion[max(self_en_shifted)][v_end] = ['f']
            break
    return graph_with_insertion


# Inserts the given polarization term into a boson line
# in the graph specified by the source vertex v_start
def insert_poln(graph, poln, n_verts, v_start, diag_type):
    if 'self_en' in diag_type:
        raise ValueError('Function insert_poln() not yet implemented for self energy '
                         +'diagrams, where the external leg convention must be maintained!')
    # Merge the original and self-energy graphs into a single (disconnected) graph;
    # in the process, we need to shift all vertex labels in the self-energy term to
    # begin at n_verts
    graph_with_insertion = copy.deepcopy(graph)
    poln_shifted = map_vertices_defaultdict(poln, vmap=np.asarray(poln.keys()) + n_verts)
    graph_with_insertion.update(poln_shifted)
    # Search for the outgoing bosonic connection
    v_end = None
    for v_end in graph[v_start]:
        # When we find it:
        if 'b' in graph[v_start][v_end]:
            # Remove the original boson line b(i -> j)
            del_conn_al(graph=graph_with_insertion, v1=v_start, v2=v_end  , line_type='b')
            del_conn_al(graph=graph_with_insertion, v1=v_end  , v2=v_start, line_type='b')
            # Connect up the original and self-energy subgraphs, i.e.,
            # replace the original fermion line with (G_0 \Sigma G_0)
            graph_with_insertion[v_start][max(poln_shifted) - 1] = ['b']
            graph_with_insertion[max(poln_shifted) - 1][v_start] = ['b']
            graph_with_insertion[max(poln_shifted)][v_end] = ['b']
            graph_with_insertion[v_end][max(poln_shifted)] = ['b']
            break
    if ('vacuum' in diag_type) or ('poln' in diag_type):
        # Swap the vetices i' and j; this is needed to maintain the
        # alternating basis convention chosen for bosonic lines in
        # vacuum and polarization diagrams
        swap = graph_with_insertion.keys()
        swap[v_end], swap[max(poln_shifted) - 1] = swap[max(poln_shifted) - 1], swap[v_end]
        graph_with_insertion = map_vertices_defaultdict(graph_with_insertion, vmap=swap)
    return graph_with_insertion


# Inserts a bare Fock self-energy term into a fermion line
# in the graph specified by the source vertex v_start
def insert_fock(graph, n_verts, v_start, diag_type):
    if 'self_en' in diag_type:
        raise ValueError('Function insert_fock() not yet implemented for self energy '
                         +'diagrams, where the external leg convention must be maintained!')
    graph_p_fock = copy.deepcopy(graph)
    # Search for the outgoing fermionic connection
    for v_end in graph_p_fock[v_start]:
        # When we find it:
        if 'f' in graph_p_fock[v_start][v_end]:
            # Replace f(i -> j) |--> f(i -> i')
            graph_p_fock[v_start][n_verts].append(pop_conn_al(graph_p_fock, v1=v_start, v2=v_end, line_type='f'))
            # Add b(i' -> j'), f(i' -> j') and f(j' -> j)
            graph_p_fock[n_verts][n_verts + 1] = ['b', 'f']
            graph_p_fock[n_verts + 1][n_verts] = ['b']
            graph_p_fock[n_verts + 1][v_end] = ['f']
            break
    return graph_p_fock


# Inserts a bare polarization bubble term into a boson line
# in the graph specified by the source vertex v_start
def insert_bubble(graph, n_verts, v_start, diag_type):
    if 'self_en' in diag_type:
        raise ValueError('Function insert_bubble() not yet implemented for self energy '
                         +'diagrams, where the external leg convention must be maintained!')
    graph_p_bubble = copy.deepcopy(graph)
    # Search for the outgoing bosonic connection
    v_end = None
    for v_end in graph_p_bubble[v_start]:
        # When we find it:
        if 'b' in graph_p_bubble[v_start][v_end]:
            # Replace b(i -> j) |--> b(i -> i')
            graph_p_bubble[v_start][n_verts].append(pop_conn_al(graph_p_bubble, v1=v_start, v2=v_end  , line_type='b'))
            graph_p_bubble[n_verts][v_start].append(pop_conn_al(graph_p_bubble, v1=v_end  , v2=v_start, line_type='b'))
            # Add b(i' -> j'), f(i' -> j') and f(j' -> j)
            graph_p_bubble[n_verts][n_verts + 1] = ['f']
            graph_p_bubble[n_verts + 1][n_verts] = ['f']
            graph_p_bubble[n_verts + 1][v_end] = ['b']
            graph_p_bubble[v_end][n_verts + 1] = ['b']
            break
    if v_end is None:
        raise ValueError('Outgoing bosonic connection not found! (v_end == None)')
    if ('vacuum' in diag_type) or ('poln' in diag_type):
        # Swap the vetices i' and j; this is needed to maintain the
        # alternating basis convention chosen for bosonic lines in
        # vacuum and polarization diagrams
        swap = np.arange(n_verts + 2, dtype=int)
        swap[v_end], swap[n_verts] = swap[n_verts], swap[v_end]
        graph_p_bubble = map_vertices_defaultdict(graph_p_bubble, vmap=swap)
    return graph_p_bubble


# Get the bold Fock-irreducible self energies
def get_bFI_fancy(original_self_energy_graphs):
    graphs_bFI = []
    for graph in original_self_energy_graphs:
        if is_bFI(graph):
            graphs_bFI.append(graph)
    return graphs_bFI


# Get the bold Fock-irreducible subset of the input graphs
def get_bFI_simple(original_graphs, n_legs, diag_type):
    graphs_bFI = []
    for graph in original_graphs:
        if is_bFI_simple(graph, n_legs, diag_type=diag_type):
            graphs_bFI.append(graph)
    return graphs_bFI


# Get the self-consistent Fock-irreducible subset of the input graphs
def get_FI_simple(original_graphs, n_legs):
    graphs_FI = []
    for graph in original_graphs:
        if is_FI_simple(graph, n_legs):
            graphs_FI.append(graph)
    return graphs_FI


# Tests the simplified form of the bold Fock-irreducibility checker function
# against the more complicated base function written for self-energy diagrams
def test_bFI_self_energy_simple(max_correction_order = 2):
    for order in range(max_correction_order + 1):
        try:
            # There are 2*(n + 1) vertices to a diagram, if n is the self-energy correction order
            n_verts = 2 * (order + 1)
            # Generate all fermionic connections
            psi_all = get_feyn_vacuum_perms(n_verts)
            # Build all the naive vacuum graphs
            all_vacuum_graphs = []
            for i in range(psi_all.shape[0]):
                all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
            all_vacuum_graphs = np.asarray(all_vacuum_graphs)
            # Get all distinct 1BI vacuum graphs
            distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs, n_verts, n_legs=0)
            distinct_vacuum_graphs_1BI = get_1BI_graphs(distinct_vacuum_graphs)
            # Now, make all naive self-energy diagrams
            self_energy_graphs_1BI = get_self_energy_graphs(distinct_vacuum_graphs_1BI)
            # Check that the simplified Fock-irreducibility checker agrees with the complicated self-energy one
            assert get_bFI_simple(self_energy_graphs_1BI, n_legs = 2, diag_type='self_en') == get_bFI_fancy(self_energy_graphs_1BI)
        except:
            return False
    return True


# Profiles the simplified form of the bold Fock-irreducibility checker function
def profile_FI_self_energy_simple(max_correction_order = 2):
    self_energy_graphs_1BI = None
    self_energy_graphs_HFI = None
    for order in range(max_correction_order + 1):
        # There are 2*(n + 1) vertices to a diagram, if n is the self-energy correction order
        n_verts = 2 * (order + 1)
        # Generate all fermionic connections
        psi_all = get_feyn_vacuum_perms(n_verts)
        # Build all the naive vacuum graphs
        all_vacuum_graphs = []
        for i in range(psi_all.shape[0]):
            all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
        all_vacuum_graphs = np.asarray(all_vacuum_graphs)
        # Get all distinct 1BI vacuum graphs
        distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs, n_verts, n_legs=0)
        distinct_vacuum_graphs_1BI = get_1BI_graphs(distinct_vacuum_graphs)
        # Now, make all naive self-energy diagrams
        self_energy_graphs_1BI = get_self_energy_graphs(distinct_vacuum_graphs_1BI)
        self_energy_graphs_HFI = get_bFI_simple(self_energy_graphs_1BI, n_legs = 2, diag_type='self_en')
    return self_energy_graphs_1BI, self_energy_graphs_HFI


# Profiles the fancy form of the Fock-irreducibility checker function
def profile_FI_self_energy_fancy(max_correction_order = 2):
    self_energy_graphs_1BI = None
    self_energy_graphs_HFI = None
    for order in range(max_correction_order + 1):
        # There are 2*(n + 1) vertices to a diagram, if n is the self-energy correction order
        n_verts = 2 * (order + 1)
        # Generate all fermionic connections
        psi_all = get_feyn_vacuum_perms(n_verts)
        # Build all the naive vacuum graphs
        all_vacuum_graphs = []
        for i in range(psi_all.shape[0]):
            all_vacuum_graphs.append(make_vacuum_graph_al(psi_all[i, :]))
        all_vacuum_graphs = np.asarray(all_vacuum_graphs)
        # Get all distinct 1BI vacuum graphs
        distinct_vacuum_graphs = rem_top_equiv_al(all_vacuum_graphs, n_verts, n_legs=0)
        distinct_vacuum_graphs_1BI = get_1BI_graphs(distinct_vacuum_graphs)
        # Now, make all naive self-energy diagrams
        self_energy_graphs_1BI = get_self_energy_graphs(distinct_vacuum_graphs_1BI)
        self_energy_graphs_HFI = get_bFI_fancy(self_energy_graphs_1BI)
    return self_energy_graphs_1BI, self_energy_graphs_HFI


# Uses recursion to update graph g1 by g2 in the adjacency list
# representation, which requires updating the connection
# defaultdicts at both the first and second levels.
def update_graph_al(g1, g2):
    for k, v in g2.items():
        if isinstance(v, defaultdict):
            tmp = update_graph_al(g1.get(k, defaultdict(list)), v)
            g1[k] = tmp
        elif isinstance(v, list):
            g1[k] = (g1.get(k, []) + v)
            # Puts the list of line types in lexigraphical order
            g1[k].sort()
        else:
            g1[k] = g2[k]
    return g1


# Pop a specified connection from a graph in the adjacency list (defaultdict) representation;
# defaults to removal of the last line type in the list, i.e., of the fermionic connection
def pop_conn_al(graph, v1, v2, line_type):
    graph[v1][v2].remove(line_type)
    # Prune the defaultdict key if it no longer contains any values
    if not graph[v1][v2]:
        del graph[v1][v2]
    return line_type


# Delete a specified connection from a graph in
# the adjacency list (defaultdict) representation
def del_conn_al(graph, v1, v2, line_type):
    graph[v1][v2].remove(line_type)
    # Prune the defaultdict key if it no longer contains any values
    if not graph[v1][v2]:
        del graph[v1][v2]
    return


# Move a specified connection in a graph in the adjacency list 
# (defaultdict) representation from (v1_old -> v2_old) to (v1 -> v2_new)
# NOTE: conn_old, conn_new are connections pairs (v_i, v_j) 
def move_conn_al(graph, conn_old, conn_new, line_type):
    v1_old, v2_old = conn_old
    v1_new, v2_new = conn_new
    if line_type == 'f':
        # Delete the original connection
        graph[v1_old][v2_old].remove('f')
        # Prune the defaultdict key if it no longer contains any values
        if not graph[v1_old][v2_old]:
            del graph[v1_old][v2_old]
        # Add the updated connection to the graph
        graph[v1_new][v2_new].append('f')
    else:
        # Delete the original connection
        graph[v1_old][v2_old].remove('b')
        graph[v2_old][v1_old].remove('b')
        # Prune the defaultdict keys if they no longer contains any values
        if not graph[v1_old][v2_old]:
            del graph[v1_old][v2_old]
        if not graph[v2_old][v1_old]:
            del graph[v2_old][v1_old]
        # Add the updated connection(s) to the graph
        graph[v1_new][v2_new].insert(0, 'b')
        graph[v2_new][v1_new].insert(0, 'b')
    return


# Check if a specified connection exists in a graph 
# in the adjacency list (defaultdict) representation,
# and delete it if so
def del_conn_if_exists_al(graph, v1, v2, line_type):
    if line_type in graph[v1][v2]:
        graph[v1][v2].remove(line_type)
        # Prune the defaultdict key if it no longer contains any values
        if not graph[v1][v2]:
            del graph[v1][v2]
    return


# Peeks into the nested defauldict adjacency list structure to
# check if an edge exists without adding any default keys if not.
def conn_exists_al(graph, v1, v2, line_type):
    if v1 in graph and v2 in graph[v1] and line_type in graph[v1][v2]:
        return True
    return False


# Convert a graph from nested defaultdicts to dicts; since we only use the defaultdict
# feature for convenient initialization and modification, we can convert them to a more
# lightweight representation after they are fully initialized, and/or when performing
# comparisons (e.g., in the function rem_top_equiv_al)
def graph_al_defaultdict_to_dict(graph_al_dd):
    return dict(zip(graph_al_dd.keys(), map(dict, graph_al_dd.values())))


# Convert a graph from nested dicts to defaultdicts
def graph_al_dict_to_defaultdict(graph_al_d):
    graph_al_dd_inner = dict(zip(graph_al_d.keys(), map(lambda l: defaultdict(list, l), graph_al_d.values())))
    return defaultdict(lambda: defaultdict(list), graph_al_dd_inner)


# Here, vmap is a tuple containing the permutation or mapping on the graph vertices
# (w.r.t. sorted order), e.g., orig = (0, 1, 2) |--> vmap = (2, 0, 1), vmap = (0, 2, 4), etc.
# Because the original set of vertices is sorted, the function mapping it to vmap is simply vmap itself: orig |--> vmap(orig).
# NOTE 1: this does not swap the vertices in place; a new graph is made and returned.
# NOTE 2: assumes that the inner list is in alphabetical order (enforced on initialization / modification)
#
# TODO: change instances of vmaps to dicts, i.e., 'real maps', so that
#       we needn't assume the original vertex labels are range(n_verts)
#
def map_vertices_defaultdict(graph, vmap):
    # Swap vertices at the upper level
    swapped_graph = {vmap[k] : v for k, v in graph.items()}
    # Swap vertices at the lower level
    for key in swapped_graph:
        swapped_graph[key] = defaultdict(list, {vmap[k] : v for k, v in swapped_graph[key].items()})
    # Retype back to a defaultdict at the outer level
    swapped_graph = defaultdict(lambda: defaultdict(list), swapped_graph)
    return swapped_graph


# Uses lighter-weight nested dict representation
# Note 1: this does not swap the vertices in place; a new graph is made and returned.
# Note 2: assumes that the inner list is in alphabetical order (enforced on initialization / modification)
def map_vertices_dict(graph, vmap):
    # Swap vertices at the upper level
    swapped_graph = {vmap[k] : v for k, v in graph.items()}
    # Swap vertices at the lower level
    for key in swapped_graph:
        swapped_graph[key] = {vmap[k] : v for k, v in swapped_graph[key].items()}
    return swapped_graph


# Returns the set difference of two diagram lists up to topological
# equivalence, [L1] - [L2], where [X] := {[x] | x \in X} and
# [x] denotes a single topological equivalence class.
def get_diag_set_diff(g_list, g_sublist, n_legs=0):
    # First, get the naive set difference (not considering topological equivalence)
    naive_set_diff = [diag for diag in g_list if diag not in g_sublist]
    # Now, additionally remove any diagrams which are
    # topologically equivalent to an element in g_sublist
    return [diag for diag in naive_set_diff if not any(d_swap in g_sublist for d_swap in all_swaps_al_dd(diag, n_legs))]


# Generate all possible vertex permutations (i.e., topological equivalents) for a given graph;
# by convention, all external vertices come numerically last; these are treated as fixed points
def all_swaps_al_dd(graph, n_legs=0):
    v_list = list(graph.keys())
    n_verts = len(v_list)
    perms = itertools.permutations(v_list[: (n_verts - n_legs)])
    # If the permutation is identity (i.e., the first in the list), skip it
    next(perms)
    # Otherwise, yield the swapped graphs
    for perm in perms:
        swap = perm + tuple(v_list[(n_verts - n_legs):])
        yield map_vertices_defaultdict(graph, swap)


# Check for topological equivalence by permuting the vertices of every diagram,
# and checking whether or not the result is in the naive list of diagrams; if it is,
# do not add it to the list of distinct diagrams.
def rem_top_equiv_al(naive_graphs_al_dd, n_verts, n_legs=0, verbose=False):
    n_graphs = len(naive_graphs_al_dd)
    distincts = np.ones(n_graphs, dtype=bool)
    duplicates = np.zeros(n_graphs, dtype=bool)
    # Convert from nested defaultdicts to dicts for
    # lightweight dictcomps (and easy C++ conversion)
    if isinstance(naive_graphs_al_dd[0], defaultdict):
        naive_graphs_al_d = list(map(graph_al_defaultdict_to_dict, naive_graphs_al_dd))
    else:
        naive_graphs_al_d = list(map(lambda g: copy.deepcopy(g), naive_graphs_al_dd))
    # For each diagram in the naive list of graphs
    visited_count = 0
    for i in range(n_graphs):
        # Skip finding topological equivalents of this diagram if it is a duplicate itself
        if duplicates[i]:
            continue
        for j in range(i + 1, n_graphs):
            # Skip finding topological equivalents of this diagram if it is a duplicate itself
            if duplicates[j]:
                continue
            # Build a list of vertex permutations
            perms = itertools.permutations(range(n_verts - n_legs))
            # Skip the identity permutation (first in the list)
            next(perms)
            # Then iterate over all vertex permutations of graph_j
            for perm in perms:
                visited_count += 1
                # Add the fixed points (external vertices), if any
                perm += tuple(range(n_verts - n_legs, n_verts))
                # Get the current swap of graph j
                swapped_graph_j = map_vertices_dict(naive_graphs_al_d[j], perm)
                # Compare it to graph_i; if they are equivalent, then graph_j
                # is a duplicate, and hence, is topologically non-distinct
                # if cmp(naive_graphs_al_d[i], swapped_graph_j) == 0:
                if naive_graphs_al_d[i] == swapped_graph_j:
                    distincts[j] = False
                    duplicates[j] = True
    if verbose:
        print('Diagram visitations:', visited_count)
    return np.asarray(naive_graphs_al_dd)[distincts]


# TODO: Double-check that this is truly a worse algorithm; should the C+
#       implementation really be a factor of 2 slower than rem_top_equiv_al?!
#
# Check for topological equivalence by permuting the vertices of every diagram,
# and checking whether or not the result is in the naive list of diagrams; if it is,
# do not add it to the list of distinct diagrams.
def rem_top_equiv_pg(naive_graphs_al, n_verts, n_legs=0):
    # Give a deprecation warning
    warnings.warn("This function is deprecated in favor of 'rem_top_equiv_al'!", DeprecationWarning)

    n_graphs = len(naive_graphs_al)
    distincts = np.ones(n_graphs, dtype=bool)
    duplicates = np.zeros(n_graphs, dtype=bool)
    # Make a single temp object hold all swapped graph copies for comparison
    swapped_graph_j = np.zeros((2, n_verts), dtype=int)
    # Get the naive graphs in the permutation group representation; for the purposes of doing simple swaps,
    # we do not put a flag on mirrored bosonic connections, and simply include them in the connection lists.
    # External vertices are fixed points, so we do not worry about the possible inclusion of the ext_flag (-1).
    naive_graphs_pg = [graph_al_to_pg(graph_al, flag_mirrored=False) for graph_al in naive_graphs_al]
    # For each diagram in the naive list of graphs
    visited_count = 0
    for i in range(n_graphs):
        # Skip finding topological equivalents of this diagram if it is a duplicate itself
        if duplicates[i]:
            continue
        for j in range(i + 1, n_graphs):
            # Skip finding topological equivalents of this diagram if it is a duplicate itself
            if duplicates[j]:
                continue
            # Build a list of internal vertex permutations
            internal_perms = itertools.permutations(range(n_verts - n_legs))
            # Skip the identity permutation (first in the list)
            next(internal_perms)
            # Then iterate over all vertex permutations of graph_j
            for internal_perm in internal_perms:
                visited_count += 1
                # Add the fixed points (external vertices), if any
                # and convert to numpy array for advanced indexing
                perm = np.asarray(internal_perm + tuple(range(n_verts - n_legs, n_verts)))
                # Calculate the inverse of the permutation P
                perm_inv = np.argsort(perm)
                # Get the current permutation of graph_j
                swapped_graph_j[0] = perm[(naive_graphs_pg[j][0])[perm_inv]]
                swapped_graph_j[1] = perm[(naive_graphs_pg[j][1])[perm_inv]]
                # We manually enforce P[G_j == -1] = -1 for (external) vertices with no fermionic
                # or bosonic connections, as these are indeed fixed points in that sense.
                swapped_graph_j[naive_graphs_pg[j] == -1] = -1
                # Compare it to graph_i; if they are equivalent, then graph_j
                # is a duplicate, and hence, is topologically non-distinct
                if np.array_equal(naive_graphs_pg[i], swapped_graph_j):
                    distincts[j] = False
                    duplicates[j] = True
    print('Diagram visitations:', visited_count)
    return np.asarray(naive_graphs_al)[distincts]


####################
# Unused functions #
####################


# Get all permutations on a list / array with fixed points, supplied as a list of indices
def permutations_with_fixed_points(iterable, fixed_pts=[]):
    for perm in itertools.permutations([v for i, v in enumerate(iterable) if i not in fixed_pts]):
        perm = list(perm)
        # Insert fixed vertices at their respective indices
        for fp_idx in fixed_pts:
            perm.insert(fp_idx, iterable[fp_idx])
        yield perm


# Generate all possible vertex permutations (i.e., topological equivalents) for a given graph
def all_swaps_arb_legs(graph, legs=[]):
    if not legs:
        perms = itertools.permutations(graph.keys())
    else:
        # First get all permutations of internal vertices with fixed legs
        internal_perms = permutations_with_fixed_points(
            iterable=graph.keys(), fixed_pts=legs)
        # Then, also include all leg permutations with fixed internal vertices (to remove hole self-energies)
        internal_verts = [vert for vert in graph.keys() if vert not in legs]
        external_perms = permutations_with_fixed_points(
            iterable=graph.keys(), fixed_pts=internal_verts)
        perms = itertools.chain(internal_perms, external_perms)
    # If the permutation is identity (i.e., the first in the list), skip it
    next(perms)
    # Otherwise, yield the swapped graphs
    for perm in perms:
        yield map_vertices_defaultdict(graph, perm)


