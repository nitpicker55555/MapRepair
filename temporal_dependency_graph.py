"""
Temporal Dependency Graph
Tracks the evolution history of navigation graphs
"""

import networkx as nx
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from dataclasses import dataclass, field


@dataclass
class GraphVersion:
    """Graph version information"""
    version_id: str
    step_num: int
    timestamp: datetime
    graph_snapshot: nx.DiGraph
    changes: Dict[str, List]  # added_nodes, added_edges, removed_nodes, removed_edges
    metadata: Dict[str, Any] = field(default_factory=dict)
    conflicts: List[Dict] = field(default_factory=list)


class TemporalDependencyGraph:
    """Temporal dependency graph: tracks the evolution history of navigation graphs"""
    
    def __init__(self):
        self.versions: Dict[str, GraphVersion] = {}
        self.version_order: List[str] = []
        self.current_version_id: Optional[str] = None
        
    def create_version(self, graph: nx.DiGraph, step_num: int, 
                      changes: Dict[str, List], observation: str,
                      conflicts: List[Dict]) -> str:
        """Create a new version"""
        version_id = f"v{step_num}"
        
        # Create a deep copy of the graph as a snapshot
        graph_snapshot = graph.copy()
        
        version = GraphVersion(
            version_id=version_id,
            step_num=step_num,
            timestamp=datetime.now(),
            graph_snapshot=graph_snapshot,
            changes=changes,
            metadata={'observation': observation[:200]},  # Truncate long observations
            conflicts=conflicts
        )
        
        self.versions[version_id] = version
        self.version_order.append(version_id)
        self.current_version_id = version_id
        
        return version_id
    
    def get_version(self, version_id: str) -> Optional[GraphVersion]:
        """Get specified version"""
        return self.versions.get(version_id)
    
    def get_version_at_step(self, step_num: int) -> Optional[GraphVersion]:
        """Get version at specified step"""
        version_id = f"v{step_num}"
        return self.get_version(version_id)
    
    def get_changes_between(self, from_step: int, to_step: int) -> Dict:
        """Get changes between two steps"""
        changes = {
            'added_nodes': set(),
            'added_edges': set(),
            'removed_nodes': set(),
            'removed_edges': set()
        }
        
        for step in range(from_step + 1, to_step + 1):
            version = self.get_version_at_step(step)
            if version:
                for node in version.changes.get('added_nodes', []):
                    changes['added_nodes'].add(node)
                for edge in version.changes.get('added_edges', []):
                    if isinstance(edge, tuple):
                        changes['added_edges'].add(edge[:2])  # Keep only from and to
                    elif isinstance(edge, dict):
                        changes['added_edges'].add((edge['from'], edge['to']))
        
        return {k: list(v) for k, v in changes.items()}
    
    def analyze_evolution(self) -> Dict:
        """Analyze graph evolution patterns"""
        if not self.versions:
            return {}
        
        analysis = {
            'total_versions': len(self.versions),
            'growth_pattern': [],
            'conflict_evolution': [],
            'major_changes': []
        }
        
        for i, version_id in enumerate(self.version_order):
            version = self.versions[version_id]
            
            # Growth pattern
            growth_info = {
                'step': version.step_num,
                'nodes': version.graph_snapshot.number_of_nodes(),
                'edges': version.graph_snapshot.number_of_edges()
            }
            analysis['growth_pattern'].append(growth_info)
            
            # Conflict evolution
            if version.conflicts:
                conflict_info = {
                    'step': version.step_num,
                    'conflict_count': len(version.conflicts),
                    'conflict_types': list(set(c.get('type', 'unknown') for c in version.conflicts))
                }
                analysis['conflict_evolution'].append(conflict_info)
            
            # Major changes
            changes = version.changes
            total_changes = (len(changes.get('added_nodes', [])) + 
                           len(changes.get('added_edges', [])))
            
            if total_changes >= 3:  # 3 or more changes considered as major
                analysis['major_changes'].append({
                    'step': version.step_num,
                    'changes': total_changes,
                    'details': changes
                })
        
        return analysis
    
    def export_timeline(self) -> List[Dict]:
        """Export timeline data"""
        timeline = []
        
        for version_id in self.version_order:
            version = self.versions[version_id]
            timeline_entry = {
                'version_id': version.version_id,
                'step_num': version.step_num,
                'timestamp': version.timestamp.isoformat(),
                'graph_stats': {
                    'nodes': version.graph_snapshot.number_of_nodes(),
                    'edges': version.graph_snapshot.number_of_edges()
                },
                'changes': version.changes,
                'conflict_count': len(version.conflicts),
                'metadata': version.metadata
            }
            timeline.append(timeline_entry)
        
        return timeline
    
    def visualize_version_tree(self) -> str:
        """Generate text visualization of version tree"""
        lines = ["Temporal Dependency Graph:"]
        lines.append(f"Total versions: {len(self.versions)}")
        lines.append("\nVersion Timeline:")
        
        for i, version_id in enumerate(self.version_order[-10:]):  # Show only last 10 versions
            version = self.versions[version_id]
            prefix = "└──" if i == len(self.version_order[-10:]) - 1 else "├──"
            
            conflict_info = f" [Conflicts: {len(version.conflicts)}]" if version.conflicts else ""
            lines.append(f"{prefix} Step {version.step_num}: "
                        f"Nodes={version.graph_snapshot.number_of_nodes()}, "
                        f"Edges={version.graph_snapshot.number_of_edges()}"
                        f"{conflict_info}")
        
        if len(self.version_order) > 10:
            lines.append(f"... and {len(self.version_order) - 10} more versions")
        
        return '\n'.join(lines)