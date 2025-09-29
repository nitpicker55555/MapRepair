"""
MANGO Map SLAM System
Main system integrating all components
"""

import json
from typing import Dict, List, Optional
import os

from mango_dataset import MANGODataset, WalkthroughStep
from navigation_graph import NavigationGraph
from conflict_detector import ConflictDetector
from temporal_dependency_graph import TemporalDependencyGraph
from conflict_localizer import ConflictLocalizer
from edge_impact_scorer import EdgeImpactScorer


class MapSLAMSystem:
    """Map SLAM System: Integrates dataset, navigation graph, conflict detection and temporal dependency graph"""
    
    def __init__(self, data_dir: str = "/Users/puzhen/Desktop/mango/data", 
                 model: str = "gpt-4o"):
        self.dataset = MANGODataset(data_dir)
        self.nav_graph = NavigationGraph(model)  # Keep as is for now, update in process_game
        self.conflict_detector = ConflictDetector()
        self.tdg = TemporalDependencyGraph()
        self.conflict_localizer = ConflictLocalizer()
        self.edge_scorer = EdgeImpactScorer()
        
        self.current_game = None
        self.processed_steps = 0
        self.walkthrough_edges = []  # Record edge sequence in walkthrough
        self.model = model  # Save model name
        
    def process_game(self, game_name: str, max_steps: Optional[int] = None,
                     verbose: bool = True) -> Dict:
        """Process the entire game walkthrough"""
        
        if game_name not in self.dataset.games:
            raise ValueError(f"Game '{game_name}' does not exist")
        
        self.current_game = game_name
        self.processed_steps = 0
        
        # Create new NavigationGraph instance for current game
        self.nav_graph = NavigationGraph(self.model, game_name, self.dataset)
        
        # Get walkthrough steps
        steps = self.dataset.load_walkthrough(game_name)
        if max_steps:
            steps = steps[:max_steps]
        
        results = {
            'game': game_name,
            'total_steps': len(steps),
            'processed_steps': 0,
            'final_graph_stats': {},
            'conflicts_summary': {},
            'conflict_localization': {},
            'edge_impact_analysis': {},
            'processing_log': []
        }
        
        previous_step = None
        
        for i, current_step in enumerate(steps):
            if verbose:
                print(f"Processing step {current_step.step_num}/{len(steps)-1}..."
            
            # Process single step
            step_result = self._process_single_step(current_step, previous_step)
            results['processing_log'].append(step_result)
            
            previous_step = current_step
            self.processed_steps += 1
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1} steps, current graph: {self.nav_graph.get_graph_stats()}"
        
        # Final statistics
        results['processed_steps'] = self.processed_steps
        results['final_graph_stats'] = self.nav_graph.get_graph_stats()
        results['conflicts_summary'] = self.conflict_detector.get_conflict_summary()
        
        # Execute conflict localization analysis
        if self.conflict_detector.conflicts:
            localization_results = self.conflict_localizer.localize_conflicts(
                self.nav_graph.graph, self.conflict_detector.conflicts
            )
            results['conflict_localization'] = localization_results
        
        # Execute edge impact scoring
        edge_scores = self.edge_scorer.score_edges(
            self.nav_graph.graph, 
            self.conflict_detector.conflicts,
            self.walkthrough_edges
        )
        results['edge_impact_analysis'] = {
            'top_impact_edges': [self._edge_score_to_dict(e) 
                               for e in self.edge_scorer.get_top_impact_edges(10)],
            'high_risk_edges': [self._edge_score_to_dict(e) 
                              for e in self.edge_scorer.get_high_risk_edges()],
            'total_edges_scored': len(edge_scores)
        }
        
        return results
    
    def _process_single_step(self, current_step: WalkthroughStep, 
                            previous_step: Optional[WalkthroughStep]) -> Dict:
        """Process single step"""
        
        # 1. Update navigation graph
        nav_result = self.nav_graph.process_step(current_step)
        
        # Record newly added edges to walkthrough path
        for edge_info in nav_result.get('added_edges', []):
            if isinstance(edge_info, dict):
                edge = (edge_info['from'], edge_info['to'])
            else:
                edge = edge_info
            self.walkthrough_edges.append(edge)
        
        # 2. Detect conflicts
        conflicts = self.conflict_detector.detect_all_conflicts(
            self.nav_graph.graph, current_step.step_num
        )
        
        # 3. Create TDG version
        changes = {
            'added_nodes': nav_result['added_nodes'],
            'added_edges': nav_result['added_edges']
        }
        
        version_id = self.tdg.create_version(
            self.nav_graph.graph,
            current_step.step_num,
            changes,
            current_step.observation,
            [self._conflict_to_dict(c) for c in conflicts]
        )
        
        return {
            'step_num': current_step.step_num,
            'action': current_step.action,
            'current_location': nav_result['current_location'],
            'changes': changes,
            'conflicts_detected': len(conflicts),
            'version_id': version_id
        }
    
    def _conflict_to_dict(self, conflict) -> Dict:
        """Convert Conflict object to dictionary"""
        return {
            'type': conflict.type,
            'severity': conflict.severity,
            'description': conflict.description,
            'involved_nodes': conflict.involved_nodes,
            'involved_edges': conflict.involved_edges,
            'details': conflict.details
        }
    
    def _edge_score_to_dict(self, edge_score) -> Dict:
        """Convert EdgeScore object to dictionary"""
        return {
            'edge': edge_score.edge,
            'total_score': edge_score.total_score,
            'structural_score': edge_score.structural_score,
            'usage_frequency': edge_score.usage_frequency,
            'conflict_impact': edge_score.conflict_impact,
            'reachability_count': edge_score.reachability_count,
            'betweenness': edge_score.betweenness,
            'pagerank_score': edge_score.pagerank_score
        }
    
    def save_results(self, output_dir: str = "./output"):
        """Save processing results"""
        if not self.current_game:
            print("No game processed")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        game_output_dir = os.path.join(output_dir, self.current_game)
        os.makedirs(game_output_dir, exist_ok=True)
        
        # Save navigation graph
        graph_data = self.nav_graph.export_to_json()
        with open(os.path.join(game_output_dir, "navigation_graph.json"), 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Save conflict report
        conflicts_data = {
            'summary': self.conflict_detector.get_conflict_summary(),
            'conflicts': [self._conflict_to_dict(c) for c in self.conflict_detector.conflicts]
        }
        with open(os.path.join(game_output_dir, "conflicts.json"), 'w') as f:
            json.dump(conflicts_data, f, indent=2)
        
        # Save timeline
        timeline = self.tdg.export_timeline()
        with open(os.path.join(game_output_dir, "timeline.json"), 'w') as f:
            json.dump(timeline, f, indent=2)
        
        # Save evolution analysis
        evolution = self.tdg.analyze_evolution()
        with open(os.path.join(game_output_dir, "evolution_analysis.json"), 'w') as f:
            json.dump(evolution, f, indent=2)
        
        # Save conflict localization analysis
        if self.conflict_localizer.conflict_paths:
            localization_data = {
                'conflict_paths': [self._conflict_path_to_dict(cp) 
                                 for cp in self.conflict_localizer.conflict_paths],
                'candidate_edges': [self._candidate_edge_to_dict(ce) 
                                  for ce in self.conflict_localizer._rank_candidate_edges()],
                'divergence_analysis': self.conflict_localizer._analyze_divergence_patterns()
            }
            with open(os.path.join(game_output_dir, "conflict_localization.json"), 'w') as f:
                json.dump(localization_data, f, indent=2)
        
        # Save edge impact analysis
        edge_impact_data = {
            'top_impact_edges': [self._edge_score_to_dict(e) 
                               for e in self.edge_scorer.get_top_impact_edges(20)],
            'high_risk_edges': [self._edge_score_to_dict(e) 
                              for e in self.edge_scorer.get_high_risk_edges()],
            'edge_rankings': self.edge_scorer.visualize_edge_rankings(15)
        }
        with open(os.path.join(game_output_dir, "edge_impact_analysis.json"), 'w') as f:
            json.dump(edge_impact_data, f, indent=2)
        
        # Save text report
        report = self._generate_report()
        with open(os.path.join(game_output_dir, "report.txt"), 'w') as f:
            f.write(report)
        
        print(f"Results saved to: {game_output_dir}")
    
    def _conflict_path_to_dict(self, conflict_path) -> Dict:
        """Convert ConflictPath object to dictionary"""
        return {
            'path1': conflict_path.path1,
            'path2': conflict_path.path2,
            'divergence_point': conflict_path.divergence_point,
            'conflict_edges': conflict_path.conflict_edges,
            'confidence': conflict_path.confidence
        }
    
    def _candidate_edge_to_dict(self, candidate_edge) -> Dict:
        """Convert CandidateEdge object to dictionary"""
        return {
            'edge': candidate_edge.edge,
            'source': candidate_edge.source,
            'target': candidate_edge.target,
            'direction': candidate_edge.direction,
            'conflict_count': candidate_edge.conflict_count,
            'paths_affected': candidate_edge.paths_affected,
            'error_probability': candidate_edge.error_probability
        }
    
    def _generate_report(self) -> str:
        """Generate text report"""
        lines = [
            f"MANGO Map SLAM Processing Report",
            f"=" * 50,
            f"Game: {self.current_game}",
            f"Processed steps: {self.processed_steps}",
            "",
            "Final graph statistics:",
            f"  Number of nodes: {self.nav_graph.graph.number_of_nodes()}",
            f"  Number of edges: {self.nav_graph.graph.number_of_edges()}",
            "",
            "Conflict summary:",
            json.dumps(self.conflict_detector.get_conflict_summary(), indent=2),
            "",
            "Conflict localization analysis:",
            self.conflict_localizer.visualize_conflict_paths() if self.conflict_localizer.conflict_paths else "  No conflict paths",
            "",
            "Edge impact scoring:",
            self.edge_scorer.visualize_edge_rankings(10) if self.edge_scorer.edge_scores else "  No edge scoring data",
            "",
            self.nav_graph.visualize_graph(),
            "",
            self.tdg.visualize_version_tree()
        ]
        
        return '\n'.join(lines)
    
    def interactive_explore(self, game_name: str):
        """Interactive exploration mode"""
        print(f"Starting interactive exploration of game: {game_name}")
        
        steps = self.dataset.load_walkthrough(game_name)
        previous_step = None
        
        for i, current_step in enumerate(steps):
            print(f"\n{'='*60}")
            print(f"Step {current_step.step_num}")
            print(f"Action: {current_step.action}")
            print(f"Observation: {current_step.observation}")
            
            # Process step
            nav_result = self.nav_graph.process_step(current_step, previous_step)
            conflicts = self.conflict_detector.detect_all_conflicts(
                self.nav_graph.graph, current_step.step_num
            )
            
            print(f"\nCurrent location: {nav_result['current_location']}")
            if nav_result['added_nodes']:
                print(f"New nodes: {nav_result['added_nodes']}")
            if nav_result['added_edges']:
                print(f"New edges: {nav_result['added_edges']}")
            if conflicts:
                print(f"Detected {len(conflicts)} conflicts:")
                for c in conflicts[:3]:  # Show only first 3
                    print(f"  - [{c.severity}] {c.description}")
            
            # Wait for user input
            user_input = input("\nPress Enter to continue, 'q' to quit, 'stats' to view statistics: ").strip()
            
            if user_input == 'q':
                break
            elif user_input == 'stats':
                print("\nCurrent graph statistics:")
                print(json.dumps(self.nav_graph.get_graph_stats(), indent=2))
                input("Press Enter to continue...")
            
            previous_step = current_step
        
        print("\nExploration ended")


def main():
    """Example usage"""
    # Create system instance
    slam = MapSLAMSystem()
    
    # List available games
    print("Available games:")
    for i, game in enumerate(slam.dataset.games[:10]):
        print(f"  {i+1}. {game}")
    print("  ...")
    
    # Process a small game as example
    game_name = "zork1"
    print(f"\nProcessing game: {game_name}")
    
    # Process first 20 steps as demo
    results = slam.process_game(game_name, max_steps=20, verbose=True)
    
    # Print result summary
    print(f"\nProcessing complete!")
    print(f"Processed steps: {results['processed_steps']}")
    print(f"Final graph stats: {results['final_graph_stats']}")
    print(f"Conflict summary: {results['conflicts_summary']}")
    
    # Save results
    slam.save_results()
    
    # Optional: Interactive exploration
    # slam.interactive_explore("zork1")


if __name__ == "__main__":
    main()