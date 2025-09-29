"""
Edge Impact Scorer
Evaluates the importance and impact of edges in the navigation graph
"""

import networkx as nx
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter


@dataclass
class EdgeScore:
    """Edge scoring information"""
    edge: Tuple[str, str]
    structural_score: float = 0.0
    usage_frequency: int = 0
    conflict_impact: float = 0.0
    reachability_count: int = 0
    betweenness: float = 0.0
    pagerank_score: float = 0.0
    total_score: float = 0.0


class EdgeImpactScorer:
    """Scores edges based on their structural importance and conflict impact"""
    
    def __init__(self):
        self.edge_scores: Dict[Tuple[str, str], EdgeScore] = {}
        self.conflict_edges: Set[Tuple[str, str]] = set()
        
    def score_edges(self, graph: nx.DiGraph, conflicts: List, 
                   walkthrough_path: List[Tuple[str, str]]) -> Dict[Tuple[str, str], EdgeScore]:
        """Score all edges based on multiple factors"""
        self.edge_scores.clear()
        
        # Extract conflict edges
        self._extract_conflict_edges(conflicts)
        
        # Initialize scores for all edges
        for edge in graph.edges():
            self.edge_scores[edge] = EdgeScore(edge=edge)
        
        # Calculate various scoring components
        self._calculate_structural_scores(graph)
        self._calculate_usage_frequency(walkthrough_path)
        self._calculate_conflict_impact(graph, conflicts)
        self._calculate_reachability_scores(graph)
        self._calculate_centrality_scores(graph)
        
        # Combine scores
        self._calculate_total_scores()
        
        return self.edge_scores
    
    def _extract_conflict_edges(self, conflicts: List):
        """Extract edges involved in conflicts"""
        self.conflict_edges.clear()
        for conflict in conflicts:
            for edge in conflict.involved_edges:
                if isinstance(edge, tuple) and len(edge) == 2:
                    self.conflict_edges.add(edge)
    
    def _calculate_structural_scores(self, graph: nx.DiGraph):
        """Calculate structural importance of edges"""
        # Check if edge removal would disconnect the graph
        for edge in graph.edges():
            score = EdgeScore(edge=edge)
            
            # Test graph connectivity after edge removal
            test_graph = graph.copy()
            test_graph.remove_edge(*edge)
            
            # Check if removal creates more components
            original_components = nx.number_weakly_connected_components(graph)
            new_components = nx.number_weakly_connected_components(test_graph)
            
            if new_components > original_components:
                score.structural_score = 1.0  # Critical edge
            else:
                # Score based on degree of connected nodes
                from_degree = graph.out_degree(edge[0]) + graph.in_degree(edge[0])
                to_degree = graph.out_degree(edge[1]) + graph.in_degree(edge[1])
                score.structural_score = 2.0 / (from_degree + to_degree)
            
            self.edge_scores[edge] = score
    
    def _calculate_usage_frequency(self, walkthrough_path: List[Tuple[str, str]]):
        """Calculate how often each edge is used in the walkthrough"""
        edge_usage = Counter(walkthrough_path)
        
        for edge, count in edge_usage.items():
            if edge in self.edge_scores:
                self.edge_scores[edge].usage_frequency = count
    
    def _calculate_conflict_impact(self, graph: nx.DiGraph, conflicts: List):
        """Calculate edge impact on conflicts"""
        for edge in self.conflict_edges:
            if edge in self.edge_scores:
                # Base conflict score
                self.edge_scores[edge].conflict_impact = 1.0
                
                # Additional score based on conflict severity
                high_severity_count = sum(1 for c in conflicts 
                                        if edge in c.involved_edges and c.severity == 'high')
                self.edge_scores[edge].conflict_impact += high_severity_count * 0.5
    
    def _calculate_reachability_scores(self, graph: nx.DiGraph):
        """Calculate how many nodes become unreachable if edge is removed"""
        for edge in graph.edges():
            test_graph = graph.copy()
            test_graph.remove_edge(*edge)
            
            # Count reachable nodes from common starting points
            reachability_impact = 0
            
            # Find potential starting nodes (nodes with only outgoing edges)
            starting_nodes = [n for n in graph.nodes() 
                            if graph.in_degree(n) == 0 and graph.out_degree(n) > 0]
            
            if not starting_nodes and graph.number_of_nodes() > 0:
                # Use first node as starting point if no clear start
                starting_nodes = [list(graph.nodes())[0]]
            
            for start in starting_nodes:
                original_reachable = len(nx.descendants(graph, start)) if start in graph else 0
                new_reachable = len(nx.descendants(test_graph, start)) if start in test_graph else 0
                reachability_impact += original_reachable - new_reachable
            
            if edge in self.edge_scores:
                self.edge_scores[edge].reachability_count = reachability_impact
    
    def _calculate_centrality_scores(self, graph: nx.DiGraph):
        """Calculate centrality-based scores"""
        # Betweenness centrality for edges
        if graph.number_of_edges() > 0:
            try:
                edge_betweenness = nx.edge_betweenness_centrality(graph, normalized=True)
                for edge, betweenness in edge_betweenness.items():
                    if edge in self.edge_scores:
                        self.edge_scores[edge].betweenness = betweenness
            except:
                pass  # Handle graphs where betweenness cannot be calculated
        
        # PageRank-based scoring
        if graph.number_of_nodes() > 0:
            try:
                pagerank = nx.pagerank(graph)
                for edge in graph.edges():
                    if edge in self.edge_scores:
                        # Edge importance based on PageRank of connected nodes
                        self.edge_scores[edge].pagerank_score = (
                            pagerank.get(edge[0], 0) + pagerank.get(edge[1], 0)
                        ) / 2
            except:
                pass  # Handle graphs where PageRank cannot be calculated
    
    def _calculate_total_scores(self):
        """Combine all scores into a total score"""
        for edge_score in self.edge_scores.values():
            edge_score.total_score = (
                edge_score.structural_score * 2.0 +
                edge_score.usage_frequency * 0.1 +
                edge_score.conflict_impact * 3.0 +
                edge_score.reachability_count * 0.05 +
                edge_score.betweenness * 1.5 +
                edge_score.pagerank_score * 1.0
            )
    
    def get_top_impact_edges(self, n: int = 10) -> List[EdgeScore]:
        """Get top n edges by total score"""
        sorted_edges = sorted(self.edge_scores.values(), 
                            key=lambda x: x.total_score, 
                            reverse=True)
        return sorted_edges[:n]
    
    def get_high_risk_edges(self, threshold: float = 2.0) -> List[EdgeScore]:
        """Get edges with high risk scores"""
        return [score for score in self.edge_scores.values() 
                if score.conflict_impact >= threshold or score.total_score >= 5.0]
    
    def visualize_edge_rankings(self, top_n: int = 10) -> str:
        """Generate text visualization of edge rankings"""
        lines = ["Edge Impact Rankings:"]
        lines.append("-" * 80)
        lines.append(f"{'Edge':<30} {'Total':<8} {'Struct':<8} {'Usage':<8} {'Conflict':<10} {'Between':<10}")
        lines.append("-" * 80)
        
        for edge_score in self.get_top_impact_edges(top_n):
            edge_str = f"{edge_score.edge[0]} -> {edge_score.edge[1]}"
            lines.append(
                f"{edge_str:<30} "
                f"{edge_score.total_score:<8.2f} "
                f"{edge_score.structural_score:<8.2f} "
                f"{edge_score.usage_frequency:<8} "
                f"{edge_score.conflict_impact:<10.2f} "
                f"{edge_score.betweenness:<10.3f}"
            )
        
        return '\n'.join(lines)