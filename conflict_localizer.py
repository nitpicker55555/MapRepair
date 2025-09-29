"""
ConflictLocalizer - Conflict Locator
"""

import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from conflict_detector import Conflict


@dataclass
class ConflictPath:
    """Conflict path information"""
    path1: List[str]  # First path
    path2: List[str]  # Second path
    divergence_point: str  # Divergence point (lowest common ancestor)
    conflict_edges: List[Tuple[str, str]]  # Edges involved in conflict
    confidence: float  # Confidence


@dataclass
class CandidateEdge:
    """Candidate error edge"""
    edge: Tuple[str, str]
    source: str
    target: str
    direction: str
    conflict_count: int  # Number of conflicts involved
    paths_affected: int  # Number of paths affected
    error_probability: float  # Error probability
    conflict_types: List[str] = None  # Track conflict types involving this edge
    
    def __post_init__(self):
        if self.conflict_types is None:
            self.conflict_types = []


class ConflictLocalizer:
    """Conflict Localizer: Locate root causes of conflicts and candidate error edges"""
    
    def __init__(self, error_edge_keys=None):
        self.conflict_paths = []
        self.candidate_edges = {}
        self.edge_conflict_map = defaultdict(list)  # Edge to conflict mapping
        # If error edges are known, use for filtering candidates
        self.error_edge_keys = error_edge_keys or set()
        
    def localize_conflicts(self, graph: nx.DiGraph, conflicts: List[Conflict]) -> Dict:
        """Locate root causes of all conflicts"""
        results = {
            'conflict_paths': [],
            'candidate_edges': [],
            'divergence_analysis': {}
        }
        
        # Process each conflict
        for conflict in conflicts:
            if conflict.type == 'direction':
                self._process_direction_conflict(graph, conflict)
            elif conflict.type == 'topology':
                self._process_topology_conflict(graph, conflict)
            elif conflict.type == 'spatial_consistency':
                self._process_spatial_conflict(graph, conflict)
        
        # Analyze results
        results['conflict_paths'] = self.conflict_paths
        results['candidate_edges'] = self._rank_candidate_edges()
        results['divergence_analysis'] = self._analyze_divergence_patterns()
        
        return results
    
    def _process_direction_conflict(self, graph: nx.DiGraph, conflict: Conflict):
        """Process direction conflicts - improved version, handles automatic reverse edges"""
        # Direction conflicts involve multiple edges from the same node
        if conflict.involved_edges:
            # Add all involved edges as candidates
            for src, dst in conflict.involved_edges:
                if graph.has_edge(src, dst):
                    edge_data = graph.get_edge_data(src, dst)
                    
                    # If it's an automatic reverse edge, find the original edge
                    if edge_data.get('is_auto_reverse', False):
                        # Original edge is dst -> src
                        if graph.has_edge(dst, src):
                            self._add_candidate_edge(graph, (dst, src), conflict)
                    else:
                        self._add_candidate_edge(graph, (src, dst), conflict)
        
        # Process specific conflict directions
        if conflict.involved_nodes:
            source_node = conflict.involved_nodes[0]
            
            # Get conflicting target nodes
            conflicting_targets = conflict.details.get('conflicting_targets', [])
            if len(conflicting_targets) >= 2:
                # Find paths to these target nodes
                for i in range(len(conflicting_targets)):
                    for j in range(i + 1, len(conflicting_targets)):
                        target1 = conflicting_targets[i]
                        target2 = conflicting_targets[j]
                        
                        # Record candidate edges
                        edge1 = (source_node, target1)
                        edge2 = (source_node, target2)
                        
                        # Check if it's an automatic reverse edge
                        for edge in [edge1, edge2]:
                            if graph.has_edge(*edge):
                                edge_data = graph.get_edge_data(*edge)
                                if edge_data.get('is_auto_reverse', False):
                                    # Add original edge
                                    if graph.has_edge(edge[1], edge[0]):
                                        self._add_candidate_edge(graph, (edge[1], edge[0]), conflict)
                                else:
                                    self._add_candidate_edge(graph, edge, conflict)
                        
                        # Create conflict path
                        conflict_path = ConflictPath(
                            path1=[source_node, target1],
                            path2=[source_node, target2],
                            divergence_point=source_node,
                            conflict_edges=[edge1, edge2],
                            confidence=0.8
                        )
                        self.conflict_paths.append(conflict_path)
    
    def _process_topology_conflict(self, graph: nx.DiGraph, conflict: Conflict):
        """Process topology conflicts - improved version"""
        # Topology conflicts usually involve inconsistent reverse edges
        if 'reverse edge' in conflict.description.lower() or 'reverse edge' in conflict.description:
            # This is a reverse edge conflict
            for src, dst in conflict.involved_edges:
                if graph.has_edge(src, dst):
                    self._add_candidate_edge(graph, (src, dst), conflict)
            
            # Check actual vs expected directions in details
            if conflict.details:
                # Process edges in forward_edge, actual_reverse, expected_reverse
                for key in ['forward_edge', 'actual_reverse', 'expected_reverse']:
                    if key in conflict.details:
                        edge_str = conflict.details[key]
                        # Parse edge string (e.g., "A --[dir]--> B")
                        edge = self._parse_edge_string(edge_str)
                        if edge and graph.has_edge(*edge):
                            self._add_candidate_edge(graph, edge, conflict)
        else:
            # Other topology conflicts (isolated nodes, etc.)
            for node in conflict.involved_nodes:
                # Check all edges of the node
                for pred in graph.predecessors(node):
                    self._add_candidate_edge(graph, (pred, node), conflict)
                for succ in graph.successors(node):
                    self._add_candidate_edge(graph, (node, succ), conflict)
    
    def _process_spatial_conflict(self, graph: nx.DiGraph, conflict: Conflict):
        """Process spatial consistency conflicts - improved version"""
        # Spatial conflicts involve position contradictions
        # All edges in conflict paths are candidates
        for src, dst in conflict.involved_edges:
            if graph.has_edge(src, dst):
                self._add_candidate_edge(graph, (src, dst), conflict)
        
        # Also consider edges mentioned in details
        if conflict.details:
            for key in ['direct_relation', 'indirect_relation']:
                if key in conflict.details and isinstance(conflict.details[key], dict):
                    edges = conflict.details[key].get('edges', [])
                    for edge in edges:
                        if isinstance(edge, tuple) and len(edge) == 2:
                            if graph.has_edge(edge[0], edge[1]):
                                self._add_candidate_edge(graph, edge, conflict)
    
    def _parse_edge_string(self, edge_str: str) -> Optional[Tuple[str, str]]:
        """Parse edge string, e.g., 'A --[dir]--> B'"""
        parts = edge_str.split(' --[')
        if len(parts) == 2:
            src = parts[0].strip()
            rest = parts[1].split(']--> ')
            if len(rest) == 2:
                dst = rest[1].strip()
                return (src, dst)
        return None
    
    def _find_paths_between_nodes(self, graph: nx.DiGraph, source: str, target: str, 
                                 max_length: int = 5) -> List[List[str]]:
        """Find all paths between two nodes (with maximum length limit)"""
        try:
            paths = []
            for path in nx.all_simple_paths(graph, source, target, cutoff=max_length):
                paths.append(path)
            return paths[:5]  # Return at most 5 paths
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []
    
    def _find_lowest_common_ancestor(self, graph: nx.DiGraph, path1: List[str], 
                                   path2: List[str]) -> Optional[str]:
        """Find the lowest common ancestor of two paths (last common node before divergence)"""
        # Create path node sets
        path1_set = set(path1)
        path2_set = set(path2)
        
        # Find common nodes
        common_nodes = path1_set & path2_set
        
        if not common_nodes:
            return None
        
        # Find divergence point: find the last common node from the start
        lca = None
        min_len = min(len(path1), len(path2))
        
        # Compare from start point, find the last identical node
        for i in range(min_len):
            if path1[i] == path2[i]:
                lca = path1[i]
            else:
                break
        
        # If no identical node found, check if there are any common nodes
        if lca is None and common_nodes:
            # Find the node that appears in both paths and has the earliest position
            for i in range(len(path1)):
                if path1[i] in common_nodes and path1[i] in path2:
                    return path1[i]
        
        return lca
    
    def _find_cycle_entry_point(self, graph: nx.DiGraph, cycle_nodes: List[str]) -> str:
        """Find the entry point of a cycle"""
        # Remove duplicate nodes, keep unique node set
        unique_cycle_nodes = list(set(cycle_nodes))
        
        # Find the node with most external incoming edges as entry
        max_external_in_degree = -1
        entry_point = unique_cycle_nodes[0]
        
        for node in unique_cycle_nodes:
            # Calculate number of incoming edges from outside the cycle
            external_in_degree = 0
            for pred in graph.predecessors(node):
                if pred not in unique_cycle_nodes:
                    external_in_degree += 1
            
            if external_in_degree > max_external_in_degree:
                max_external_in_degree = external_in_degree
                entry_point = node
        
        return entry_point
    
    def _find_divergence_edges(self, graph: nx.DiGraph, divergence_point: str,
                             path1: List[str], path2: List[str]) -> List[Tuple[str, str]]:
        """Find diverging edges from the divergence point"""
        divergence_edges = []
        
        # Find the position of divergence point in both paths
        idx1 = path1.index(divergence_point) if divergence_point in path1 else -1
        idx2 = path2.index(divergence_point) if divergence_point in path2 else -1
        
        # Get the next nodes after divergence
        if idx1 >= 0 and idx1 < len(path1) - 1:
            next1 = path1[idx1 + 1]
            if graph.has_edge(divergence_point, next1):
                divergence_edges.append((divergence_point, next1))
        
        if idx2 >= 0 and idx2 < len(path2) - 1:
            next2 = path2[idx2 + 1]
            if graph.has_edge(divergence_point, next2):
                divergence_edges.append((divergence_point, next2))
        
        return divergence_edges
    
    def _add_candidate_edge(self, graph: nx.DiGraph, edge: Tuple[str, str], 
                          conflict: Conflict):
        """Add candidate error edge - improved version"""
        # Get edge data
        edge_data = graph.get_edge_data(*edge) or {}
        direction = edge_data.get('direction', '')
        
        # Create edge key with direction
        edge_key_with_dir = (edge[0], edge[1], direction) if direction else None
        
        # If error edges are known, check if this might be an error edge
        if self.error_edge_keys:
            # If error edge information is available, only add matching edges
            is_error = False
            if edge_key_with_dir and edge_key_with_dir in self.error_edge_keys:
                is_error = True
            # Also check without direction
            elif (edge[0], edge[1]) in [(e[0], e[1]) for e in self.error_edge_keys]:
                is_error = True
            
            if not is_error:
                return  # When error edges are known, skip non-error edges
        
        # Add or update candidate
        if edge not in self.candidate_edges:
            self.candidate_edges[edge] = CandidateEdge(
                edge=edge,
                source=edge[0],
                target=edge[1],
                direction=direction,
                conflict_count=0,
                paths_affected=0,
                error_probability=0.0,
                conflict_types=[]
            )
        
        # Update conflict count and types
        self.candidate_edges[edge].conflict_count += 1
        if conflict.type not in self.candidate_edges[edge].conflict_types:
            self.candidate_edges[edge].conflict_types.append(conflict.type)
        self.edge_conflict_map[edge].append(conflict)
    
    def _rank_candidate_edges(self) -> List[CandidateEdge]:
        """Sort candidate edges - improved version"""
        # Calculate error probability for each edge
        for edge, candidate in self.candidate_edges.items():
            # Calculate base probability based on conflict count
            base_prob = min(candidate.conflict_count * 0.2, 0.9)
            
            # Weight by conflict types
            type_weight = 0
            if 'spatial_consistency' in candidate.conflict_types:
                type_weight += 0.4  # Spatial conflicts usually indicate fundamental issues
            if 'topology' in candidate.conflict_types:
                type_weight += 0.3
            if 'direction' in candidate.conflict_types:
                type_weight += 0.2
            
            # Extra points for multi-type conflicts
            multi_type_bonus = (len(candidate.conflict_types) - 1) * 0.1
            
            candidate.error_probability = min(base_prob + type_weight + multi_type_bonus, 1.0)
        
        # Sort by error probability in descending order
        return sorted(self.candidate_edges.values(), 
                     key=lambda x: (x.error_probability, x.conflict_count), 
                     reverse=True)
    
    def _analyze_divergence_patterns(self) -> Dict:
        """Analyze divergence patterns"""
        patterns = {
            'total_divergence_points': len(set(cp.divergence_point for cp in self.conflict_paths)),
            'average_path_length': np.mean([len(cp.path1) + len(cp.path2) 
                                          for cp in self.conflict_paths]) if self.conflict_paths else 0,
            'high_confidence_conflicts': len([cp for cp in self.conflict_paths 
                                            if cp.confidence >= 0.8]),
            'divergence_distribution': defaultdict(int),
            'total_candidates': len(self.candidate_edges),
            'conflict_type_distribution': defaultdict(int),
            'multi_type_edges': 0
        }
        
        # Count conflicts for each divergence point
        for cp in self.conflict_paths:
            patterns['divergence_distribution'][cp.divergence_point] += 1
        
        # Analyze conflict types
        for candidate in self.candidate_edges.values():
            for conflict_type in candidate.conflict_types:
                patterns['conflict_type_distribution'][conflict_type] += 1
            if len(candidate.conflict_types) > 1:
                patterns['multi_type_edges'] += 1
        
        return patterns
    
    def visualize_conflict_paths(self) -> str:
        """Generate text visualization of conflict paths"""
        if not self.conflict_paths:
            return "No conflict paths detected"
        
        lines = ["Conflict Path Analysis:", "=" * 50]
        
        for i, cp in enumerate(self.conflict_paths[:5]):  # Only show first 5
            lines.extend([
                f"\nConflict {i + 1}:",
                f"  Path 1: {' -> '.join(cp.path1)}",
                f"  Path 2: {' -> '.join(cp.path2)}",
                f"  Divergence Point: {cp.divergence_point}",
                f"  Confidence: {cp.confidence:.2f}",
                f"  Edges Involved: {cp.conflict_edges}"
            ])
        
        return '\n'.join(lines)