"""
MapRepair: end-to-end MANGO Map SLAM system

Glues together the modules described in the paper:

  * NavigationGraph     - LLM-driven incremental graph construction
                          with unit-distance position tracking
  * ConflictDetector    - three structural conflicts (Section 2.2)
  * VersionControl      - versioned reasoning history (Section 2.4)
  * ConflictLocalizer   - LCA on the Reasoning History Tree (Section 2.3)
  * EdgeImpactScorer    - EIS = R_hat + C_hat + U_hat (Eq. 1)

Each walkthrough step is processed in a single pipeline pass:

    1. NavigationGraph.process_step  -> add nodes/edges, compute positions
    2. VersionControl.commit         -> log the change with observation
    3. ConflictDetector.detect_all   -> three structural checks
    4. If conflicts: ConflictLocalizer + EdgeImpactScorer to produce
       a ranked candidate edge list ready for an LLM repair prompt.
"""

import json
import os
from typing import Dict, List, Optional

from mango_dataset import MANGODataset, WalkthroughStep
from navigation_graph import NavigationGraph
from conflict_detector import ConflictDetector, Conflict
from version_control import VersionControl, TRIGGER_OBSERVATION, TRIGGER_INIT
from conflict_localizer import ConflictLocalizer
from edge_impact_scorer import EdgeImpactScorer


class MapSLAMSystem:
    """End-to-end MapRepair pipeline."""

    def __init__(self,
                 data_dir: str = "/Users/puzhen/Desktop/mango/data",
                 model: str = "gpt-4o"):
        self.dataset = MANGODataset(data_dir)
        self.model = model

        # Per-game state created in process_game()
        self.nav_graph: Optional[NavigationGraph] = None
        self.conflict_detector = ConflictDetector()
        self.version_control = VersionControl()
        self.conflict_localizer = ConflictLocalizer()
        self.edge_scorer = EdgeImpactScorer()

        self.current_game: Optional[str] = None
        self.processed_steps = 0
        self.walkthrough_edges: List = []

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------
    def process_game(self,
                     game_name: str,
                     max_steps: Optional[int] = None,
                     verbose: bool = True) -> Dict:
        """Process a complete walkthrough and return per-step results."""
        if game_name not in self.dataset.games:
            raise ValueError(f"Game '{game_name}' does not exist")

        self.current_game = game_name
        self.processed_steps = 0
        self.walkthrough_edges = []

        # Fresh per-game modules so different runs don't share state
        self.nav_graph = NavigationGraph(self.model, game_name, self.dataset)
        self.conflict_detector = ConflictDetector()
        self.version_control = VersionControl()
        self.conflict_localizer = ConflictLocalizer()
        self.edge_scorer = EdgeImpactScorer()

        # Initial commit so the version chain has a defined starting point
        self.version_control.commit(
            step_id=0,
            trigger_event=TRIGGER_INIT,
            observation="",
            analysis="empty graph",
        )

        steps = self.dataset.load_walkthrough(game_name)
        if max_steps:
            steps = steps[:max_steps]

        results = {
            "game": game_name,
            "total_steps": len(steps),
            "processed_steps": 0,
            "final_graph_stats": {},
            "conflicts_summary": {},
            "conflict_localization": {},
            "edge_impact_analysis": {},
            "processing_log": [],
        }

        for i, current_step in enumerate(steps):
            if verbose:
                print(f"Processing step {current_step.step_num}/{len(steps)-1}...")
            step_result = self._process_single_step(current_step)
            results["processing_log"].append(step_result)
            self.processed_steps += 1
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1} steps, "
                      f"current graph: {self.nav_graph.get_graph_stats()}")

        # Final aggregation
        results["processed_steps"] = self.processed_steps
        results["final_graph_stats"] = self.nav_graph.get_graph_stats()
        results["conflicts_summary"] = self.conflict_detector.get_conflict_summary()

        if self.conflict_detector.conflicts:
            tree, tau = self.version_control.reasoning_history_tree()
            localization = self.conflict_localizer.localize_conflicts(
                self.nav_graph.graph,
                self.conflict_detector.conflicts,
                tree, tau,
            )
            results["conflict_localization"] = localization

            # Score the LCA-filtered candidates with EIS (paper Eq. 1)
            candidates = self.conflict_localizer.get_candidate_edges()
            self.edge_scorer.score_edges(
                self.nav_graph.graph,
                self.conflict_detector.conflicts,
                self.walkthrough_edges,
                candidate_edges=candidates,
            )
            results["edge_impact_analysis"] = {
                "ranked_candidates": [self._edge_score_to_dict(e)
                                      for e in self.edge_scorer.rank()],
                "total_candidates": len(candidates),
            }

        return results

    # ------------------------------------------------------------------
    # Per-step pipeline
    # ------------------------------------------------------------------
    def _process_single_step(self, current_step: WalkthroughStep) -> Dict:
        # 1. Update navigation graph
        nav_result = self.nav_graph.process_step(current_step)

        # Track the walkthrough path for the EIS usage factor
        for edge_info in nav_result.get("added_edges", []):
            if isinstance(edge_info, tuple) and len(edge_info) >= 2:
                self.walkthrough_edges.append((edge_info[0], edge_info[1]))

        # 2. Commit to Version Control with full observation/analysis
        added_edges_for_vc = []
        for e in nav_result.get("added_edges", []):
            if isinstance(e, tuple):
                if len(e) == 3:
                    added_edges_for_vc.append((e[0], e[1], e[2]))
                elif len(e) == 2:
                    added_edges_for_vc.append((e[0], e[1], ""))
        version_id = self.version_control.commit(
            step_id=current_step.step_num,
            added_nodes=list(nav_result.get("added_nodes", [])),
            added_edges=added_edges_for_vc,
            trigger_event=TRIGGER_OBSERVATION,
            observation=current_step.observation,
            analysis=nav_result.get("analysis", ""),
        )

        # 3. Detect conflicts using current positions for the topology check
        conflicts = self.conflict_detector.detect_all_conflicts(
            self.nav_graph.graph,
            current_step.step_num,
            node_positions=self.nav_graph.get_node_positions(),
        )

        return {
            "step_num": current_step.step_num,
            "action": current_step.action,
            "current_location": nav_result["current_location"],
            "added_nodes": nav_result.get("added_nodes", []),
            "added_edges": nav_result.get("added_edges", []),
            "conflicts_detected": len(conflicts),
            "version_id": version_id,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_results(self, output_dir: str = "./output"):
        if not self.current_game:
            print("No game processed")
            return
        os.makedirs(output_dir, exist_ok=True)
        game_dir = os.path.join(output_dir, self.current_game)
        os.makedirs(game_dir, exist_ok=True)

        with open(os.path.join(game_dir, "navigation_graph.json"), "w") as f:
            json.dump(self.nav_graph.export_to_json(), f, indent=2)

        with open(os.path.join(game_dir, "conflicts.json"), "w") as f:
            json.dump({
                "summary": self.conflict_detector.get_conflict_summary(),
                "conflicts": [self._conflict_to_dict(c)
                              for c in self.conflict_detector.conflicts],
            }, f, indent=2)

        with open(os.path.join(game_dir, "version_history.json"), "w") as f:
            json.dump(self.version_control.export_timeline(), f, indent=2)

        if self.conflict_detector.conflicts and self.edge_scorer.edge_scores:
            with open(os.path.join(game_dir, "edge_impact.json"), "w") as f:
                json.dump([self._edge_score_to_dict(e)
                           for e in self.edge_scorer.rank()], f, indent=2)

        with open(os.path.join(game_dir, "report.txt"), "w") as f:
            f.write(self._generate_report())

        print(f"Results saved to: {game_dir}")

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def _generate_report(self) -> str:
        lines = [
            "MapRepair Processing Report",
            "=" * 50,
            f"Game: {self.current_game}",
            f"Processed steps: {self.processed_steps}",
            "",
            "Final graph statistics:",
            f"  Nodes: {self.nav_graph.graph.number_of_nodes()}",
            f"  Edges: {self.nav_graph.graph.number_of_edges()}",
            "",
            "Conflict summary:",
            json.dumps(self.conflict_detector.get_conflict_summary(), indent=2),
            "",
        ]
        if self.edge_scorer.edge_scores:
            lines.append("Top EIS-ranked candidates:")
            lines.append(self.edge_scorer.visualize_edge_rankings(10))
            lines.append("")
        lines.append(self.nav_graph.visualize_graph())
        return "\n".join(lines)

    @staticmethod
    def _conflict_to_dict(conflict: Conflict) -> Dict:
        return {
            "type": conflict.type,
            "description": conflict.description,
            "involved_nodes": conflict.involved_nodes,
            "involved_edges": conflict.involved_edges,
            "step_num": conflict.step_num,
            "details": conflict.details,
        }

    @staticmethod
    def _edge_score_to_dict(edge_score) -> Dict:
        return {
            "edge": list(edge_score.edge),
            "reach": edge_score.reach,
            "conflict_count": edge_score.conflict_count,
            "usage": edge_score.usage,
            "r_hat": edge_score.r_hat,
            "c_hat": edge_score.c_hat,
            "u_hat": edge_score.u_hat,
            "score": edge_score.score,
        }


def main():
    """Smoke test on a small slice of zork1."""
    slam = MapSLAMSystem()
    if "zork1" in slam.dataset.games:
        results = slam.process_game("zork1", max_steps=20, verbose=True)
        print("Final graph stats:", results["final_graph_stats"])
        print("Conflict summary:", results["conflicts_summary"])
        slam.save_results()


if __name__ == "__main__":
    main()
