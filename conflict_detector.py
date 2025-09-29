import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Conflict:
    """Represents a detected conflict"""
    type: str  # 'naming', 'direction', 'topology'
    severity: str  # 'high', 'medium', 'low'
    description: str
    involved_nodes: List[str]
    involved_edges: List[Tuple[str, str]]
    step_num: int
    details: Dict


class ConflictDetector:
    """Detect direction conflicts and topology conflicts in navigation graph"""
    
    def __init__(self):
        self.conflicts = []
        self.direction_conflicts = []
        self.topology_conflicts = []
    
    def detect_all_conflicts(self, graph: nx.DiGraph, step_num: int, game_name: str = "") -> List[Conflict]:
        """Detect all types of conflicts: direction conflicts, topology conflicts, and spatial consistency conflicts"""
        self.conflicts = []
        
        # Detect various conflicts
        self.conflicts.extend(self._detect_direction_conflicts(graph, step_num))
        self.conflicts.extend(self._detect_topology_conflicts(graph, step_num, game_name))
        self.conflicts.extend(self._detect_reverse_edge_conflicts(graph, step_num))
        self.conflicts.extend(self._detect_spatial_consistency_conflicts(graph, step_num))
        
        return self.conflicts
    
    
    def _detect_direction_conflicts(self, graph: nx.DiGraph, step_num: int) -> List[Conflict]:
        """Detect direction conflicts: a node has multiple edges pointing to the same direction"""
        conflicts = []
        
        for node in graph.nodes():
            # Collect all directions from this node
            directions = defaultdict(list)
            for _, target, data in graph.out_edges(node, data=True):
                direction = data.get('direction', '')
                if direction:
                    directions[direction.lower()].append(target)
            
            # Check if there are same directions pointing to different locations
            for direction, targets in directions.items():
                if len(targets) > 1:
                    conflict = Conflict(
                        type='direction',
                        severity='high',
                        description=f"Direction conflict: From '{node}' the '{direction}' direction points to multiple locations",
                        involved_nodes=[node] + targets,
                        involved_edges=[(node, target) for target in targets],
                        step_num=step_num,
                        details={
                            'conflicting_direction': direction,
                            'conflicting_targets': targets
                        }
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_topology_conflicts(self, graph: nx.DiGraph, step_num: int, game_name: str = "") -> List[Conflict]:
        """Detect topology conflicts: unreachable nodes, over-connected areas"""
        conflicts = []
        
        # Detect unreachable nodes (cannot be reached from any other node)
        if graph.number_of_nodes() > 1:
            # Find weakly connected components
            components = list(nx.weakly_connected_components(graph))
            if len(components) > 1:
                # Find the largest component
                main_component = max(components, key=len)
                for component in components:
                    if component != main_component and len(component) == 1:
                        isolated_node = list(component)[0]
                        conflict = Conflict(
                            type='topology',
                            severity='high',
                            description=f"Isolated node: '{isolated_node}' is disconnected from the main graph",
                            involved_nodes=[isolated_node],
                            involved_edges=[],
                            step_num=step_num,
                            details={
                                'component_size': 1,
                                'main_component_size': len(main_component)
                            }
                        )
                        conflicts.append(conflict)
        
        # 3. Detect over-connected nodes (only log, not as conflicts)
        degree_threshold = 8  # Log if more than 8 connections
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            total_degree = in_degree + out_degree
            
            if total_degree > degree_threshold:
                # Get detailed information for all in-edges and out-edges
                in_edges_detail = []
                for src, dst in graph.in_edges(node):
                    edge_data = graph[src][dst]
                    in_edges_detail.append({
                        'from': src,
                        'to': dst,
                        'direction': edge_data.get('direction', 'unknown')
                    })
                
                out_edges_detail = []
                for src, dst in graph.out_edges(node):
                    edge_data = graph[src][dst]
                    out_edges_detail.append({
                        'from': src,
                        'to': dst,
                        'direction': edge_data.get('direction', 'unknown')
                    })
                
                # Print detailed connection information
                # game_prefix = f"[{game_name}] " if game_name else ""
                # print(f"\n{game_prefix}[INFO] Node over-connected - '{node}' has {total_degree} connections:")
                # print(f"  In edges ({in_degree}):")
                # for edge in in_edges_detail:
                #     print(f"    {edge['from']} --[{edge['direction']}]--> {edge['to']}")
                # print(f"  Out edges ({out_degree}):")
                # for edge in out_edges_detail:
                #     print(f"    {edge['from']} --[{edge['direction']}]--> {edge['to']}")
                #
        return conflicts
    
    def _detect_reverse_edge_conflicts(self, graph: nx.DiGraph, step_num: int) -> List[Conflict]:
        """Detect reverse edge consistency conflicts: only check if bidirectional edges with classic topological directions match"""
        conflicts = []
        
        # Define reverse mapping for classic topological directions
        classic_reverse_lookup = {
            'north': 'south', 'south': 'north',
            'east': 'west', 'west': 'east',
            'up': 'down', 'down': 'up',
            'northeast': 'southwest', 'southwest': 'northeast',
            'northwest': 'southeast', 'southeast': 'northwest',
            'enter': 'exit', 'exit': 'enter'
        }
        
        # Record checked edge pairs to avoid duplicate detection
        checked_pairs = set()
        
        for u, v, data in graph.edges(data=True):
            # Skip auto-generated reverse edges
            if data.get('is_auto_reverse', False):
                continue
                
            # Avoid duplicate checking of the same pair of nodes
            if (v, u) in checked_pairs:
                continue
            checked_pairs.add((u, v))
            
            direction = data.get('direction', '').lower()
            expected_reverse = classic_reverse_lookup.get(direction)
            
            # Only detect classic topological directions
            if expected_reverse and graph.has_edge(v, u):
                reverse_data = graph[v][u]
                actual_reverse_dir = reverse_data.get('direction', '').lower()
                
                # Only detect when the reverse edge is also a classic topological direction
                if actual_reverse_dir in classic_reverse_lookup:
                    # Check if the reverse edge direction is correct
                    if actual_reverse_dir != expected_reverse:
                        conflict = Conflict(
                            type='topology',
                            severity='high',
                            description=f"Reverse edge direction inconsistent: {v} to {u} should be '{expected_reverse}', but is actually '{actual_reverse_dir}'",
                            involved_nodes=[u, v],
                            involved_edges=[(u, v), (v, u)],
                            step_num=step_num,
                            details={
                                'forward_edge': f"{u} --[{direction}]--> {v}",
                                'expected_reverse': f"{v} --[{expected_reverse}]--> {u}",
                                'actual_reverse': f"{v} --[{actual_reverse_dir}]--> {u}",
                                'is_auto_reverse_forward': data.get('is_auto_reverse', False),
                                'is_auto_reverse_backward': reverse_data.get('is_auto_reverse', False)
                            }
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    def _detect_spatial_consistency_conflicts(self, graph: nx.DiGraph, step_num: int) -> List[Conflict]:
        """Detect spatial consistency conflicts: same location appears in different relative positions"""
        conflicts = []
        
        # For each node, calculate its spatial relationship relative to other nodes
        for node_a in graph.nodes():
            for node_c in graph.nodes():
                if node_a == node_c:
                    continue
                    
                # Find all possible paths and relationships between A and C
                direct_relations = self._get_direct_spatial_relations(graph, node_a, node_c)
                indirect_relations = self._get_indirect_spatial_relations(graph, node_a, node_c)
                
                # Check if direct and indirect relationships conflict
                if direct_relations and indirect_relations:
                    for direct_rel in direct_relations:
                        for indirect_rel in indirect_relations:
                            if self._are_spatial_relations_conflicting(direct_rel, indirect_rel):
                                conflict = Conflict(
                                    type='spatial_consistency',
                                    severity='high',
                                    description=f"Spatial consistency conflict: {node_c}'s position relative to {node_a} is contradictory",
                                    involved_nodes=[node_a, node_c] + indirect_rel['path'],
                                    involved_edges=direct_rel['edges'] + indirect_rel['edges'],
                                    step_num=step_num,
                                    details={
                                        'direct_relation': direct_rel,
                                        'indirect_relation': indirect_rel,
                                        'conflict_explanation': f"{node_c} is directly in the {direct_rel.get('direction', direct_rel.get('inferred_direction', '?'))} direction from {node_a}, but inferred to be in the {indirect_rel['inferred_direction']} direction through path {' -> '.join(indirect_rel['path'])}"
                                    }
                                )
                                conflicts.append(conflict)
        
        return conflicts
    
    def _get_direct_spatial_relations(self, graph: nx.DiGraph, node_a: str, node_c: str) -> List[Dict]:
        """Get direct spatial relationship between two nodes"""
        relations = []
        
        # Check direct connection from A to C
        if graph.has_edge(node_a, node_c):
            edge_data = graph[node_a][node_c]
            direction = edge_data.get('direction', '')
            if direction:
                relations.append({
                    'direction': direction,
                    'edges': [(node_a, node_c)],
                    'type': 'direct'
                })
        
        # Check direct connection from C to A (reverse)
        if graph.has_edge(node_c, node_a):
            edge_data = graph[node_c][node_a]
            direction = edge_data.get('direction', '')
            if direction:
                # Convert to direction from A to C
                reverse_direction = self._get_opposite_direction(direction)
                if reverse_direction:
                    relations.append({
                        'direction': reverse_direction,
                        'edges': [(node_c, node_a)],
                        'type': 'reverse_direct'
                    })
        
        return relations
    
    def _get_indirect_spatial_relations(self, graph: nx.DiGraph, node_a: str, node_c: str) -> List[Dict]:
        """Get indirect spatial relationship between two nodes (through third-party nodes)"""
        relations = []
        
        # Find intermediate nodes that both A and C connect to
        for intermediate in graph.nodes():
            if intermediate == node_a or intermediate == node_c:
                continue
            
            # Check path A -> intermediate -> C
            a_to_int = None
            int_to_c = None
            
            if graph.has_edge(node_a, intermediate):
                a_to_int = graph[node_a][intermediate].get('direction', '')
            
            if graph.has_edge(intermediate, node_c):
                int_to_c = graph[intermediate][node_c].get('direction', '')
            
            # Check path C -> intermediate -> A  
            c_to_int = None
            int_to_a = None
            
            if graph.has_edge(node_c, intermediate):
                c_to_int = graph[node_c][intermediate].get('direction', '')
            
            if graph.has_edge(intermediate, node_a):
                int_to_a = graph[intermediate][node_a].get('direction', '')
            
            # Analyze these relationships
            if a_to_int and int_to_c:
                # A -> intermediate -> C
                inferred_direction = self._infer_direction_through_path(a_to_int, int_to_c)
                if inferred_direction:
                    relations.append({
                        'inferred_direction': inferred_direction,
                        'path': [node_a, intermediate, node_c],
                        'edges': [(node_a, intermediate), (intermediate, node_c)],
                        'directions': [a_to_int, int_to_c],
                        'type': 'indirect'
                    })
            
            if c_to_int and int_to_a:
                # C -> intermediate -> A, convert to direction from A to C
                path_direction = self._infer_direction_through_path(c_to_int, int_to_a)
                if path_direction:
                    reverse_direction = self._get_opposite_direction(path_direction)
                    if reverse_direction:
                        relations.append({
                            'inferred_direction': reverse_direction,
                            'path': [node_c, intermediate, node_a],
                            'edges': [(node_c, intermediate), (intermediate, node_a)],
                            'directions': [c_to_int, int_to_a],
                            'type': 'reverse_indirect'
                        })
        
        return relations
    
    def _infer_direction_through_path(self, dir1: str, dir2: str) -> Optional[str]:
        """Infer overall direction through two path segments"""
        # Simplified direction inference logic
        # If two directions are the same, the overall direction is the same
        if dir1.lower() == dir2.lower():
            return dir1.lower()
        
        # If two directions are opposite, may return to origin (no net direction)
        if self._get_opposite_direction(dir1) == dir2.lower():
            return None  # Round trip, net displacement is 0
        
        # Return None for other complex cases for now
        return None
    
    def _are_spatial_relations_conflicting(self, relation1: Dict, relation2: Dict) -> bool:
        """Check if two spatial relationships conflict"""
        dir1 = relation1.get('direction') or relation1.get('inferred_direction')
        dir2 = relation2.get('direction') or relation2.get('inferred_direction')
        
        if not dir1 or not dir2:
            return False
        
        dir1_lower = dir1.lower()
        dir2_lower = dir2.lower()
        
        # If two directions are the same, no conflict
        if dir1_lower == dir2_lower:
            return False
            
        # Check if they are incompatible direction combinations
        return self._are_directions_incompatible(dir1_lower, dir2_lower)
    
    def _are_directions_incompatible(self, dir1: str, dir2: str) -> bool:
        """Check if two directions are incompatible (conflicting)"""
        # Define incompatible direction combinations
        incompatible_pairs = {
            # Basic direction conflicts
            ('north', 'south'), ('south', 'north'),
            ('east', 'west'), ('west', 'east'),
            ('up', 'down'), ('down', 'up'),
            # Vertical direction conflicts - this is key!
            ('north', 'east'), ('north', 'west'), 
            ('south', 'east'), ('south', 'west'),
            ('east', 'north'), ('east', 'south'),
            ('west', 'north'), ('west', 'south'),
            # Other incompatible combinations
            ('in', 'out'), ('out', 'in'),
            ('enter', 'exit'), ('exit', 'enter'),
        }
        
        return (dir1, dir2) in incompatible_pairs
    
    def _get_opposite_direction(self, direction: str) -> Optional[str]:
        """Get reverse direction"""
        opposites = {
            'north': 'south', 'south': 'north',
            'east': 'west', 'west': 'east',
            'up': 'down', 'down': 'up',
            'northeast': 'southwest', 'southwest': 'northeast',
            'northwest': 'southeast', 'southeast': 'northwest',
            'in': 'out', 'out': 'in',
            'enter': 'exit', 'exit': 'enter'
        }
        return opposites.get(direction.lower(), None)
    
    
    def _are_locations_connected(self, graph: nx.DiGraph, node1: str, node2: str, 
                                max_distance: int) -> bool:
        """Check if two positions are connected within specified distance"""
        try:
            # Check bidirectional shortest path
            path_length = nx.shortest_path_length(graph.to_undirected(), node1, node2)
            return path_length <= max_distance
        except nx.NetworkXNoPath:
            return False
    
    def _get_shortest_path_length(self, graph: nx.DiGraph, node1: str, node2: str) -> Optional[int]:
        """Get shortest path length between two points"""
        try:
            return nx.shortest_path_length(graph.to_undirected(), node1, node2)
        except nx.NetworkXNoPath:
            return None
    
    
    def get_conflict_summary(self) -> Dict:
        """Get conflict summary"""
        summary = {
            'total_conflicts': len(self.conflicts),
            'by_type': {
                'direction': len([c for c in self.conflicts if c.type == 'direction']),
                'topology': len([c for c in self.conflicts if c.type == 'topology']),
                'spatial_consistency': len([c for c in self.conflicts if c.type == 'spatial_consistency'])
            },
            'by_severity': {
                'high': len([c for c in self.conflicts if c.severity == 'high']),
                'medium': len([c for c in self.conflicts if c.severity == 'medium']),
                'low': len([c for c in self.conflicts if c.severity == 'low'])
            }
        }
        return summary