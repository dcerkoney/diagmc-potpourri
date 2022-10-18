#pragma once
#include "diagmc_includes.hpp"

/* cmath constant definitions */
#define _USE_MATH_DEFINES

// Abstract representation of a Hugenholtz graph.
// Internally, both adjacency matrix and list representations
// are stored for efficient access and algorithmic manipulations.
template <typename T>
class hhz_graph {
 public:
  T id;
  int n_verts;
  int n_edges = 2 * n_verts;  // all HHZ edges are fermionic
  hhz_graph(T id, int n_verts_) : id(id_), n_verts(n_verts_) {}
  // TODO...
};

// ostream overload for printing Hugenholtz graphs
template <typename T>
std::ostream& operator<<(std::ostream& os, const hhz_graph<T>& g) {
  os << "Hugenholtz graph" << std::endl;
  os << "------------------" << std::endl;
  os << "ID: " << g.id << std::endl;
  os << "n_verts: " << g.n_verts << std::endl;
  os << "n_edges: " << g.n_edges << std::endl;
  return os;
}

// Abstract representation of a Feynman graph
// Internally, both adjacency matrix and list representations
// are stored for efficient access and algorithmic manipulations.
template <typename T>
class feyn_graph {
 public:
  T id;
  int n_verts;
  int n_b_edges = n_verts;      // bosonic edges
  int n_f_edges = 2 * n_verts;  // fermionic edges
  feyn_graph(T id, int n_verts_) : id(id_), n_verts(n_verts_) {}
  // TODO...
};

// ostream overload for printing Feynman graphs
template <typename T>
std::ostream& operator<<(std::ostream& os, const feyn_graph<T>& g) {
  os << "Feynman graph:" << std::endl;
  os << "------------------" << std::endl;
  os << "ID: " << g.id << std::endl;
  os << "n_b_edges = " << g.n_b_edges << std::endl;
  os << "n_f_edges = " << g.n_f_edges << std::endl;
  return os;
}

// Macros to define JSON (de)serialization methods to_json/from_json
// for each mcmc config group (all children/parent structs) concisely
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(hhz_graph<int>, id, n_verts, n_edges)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(feyn_graph<int>, id, n_verts, n_b_edges, n_f_edges)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(hhz_graph<std::string>, id, n_verts, n_edges)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(feyn_graph<std::string>, id, n_verts, n_b_edges, n_f_edges)