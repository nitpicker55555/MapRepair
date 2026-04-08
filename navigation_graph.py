"""
Navigation Graph

Builds the spatial navigation graph incrementally from walkthrough steps.

Key responsibilities:
  * call the LLM to interpret each step (see llm_agent.py)
  * add nodes / edges to a NetworkX DiGraph
  * track unit-distance positions for every node, so the topological
    conflict detector can detect physical exclusivity violations
  * automatically insert reverse edges for cardinal directions
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from llm_agent import chat_single, message_template
from mango_dataset import MANGODataset, WalkthroughStep


# Unit-distance offsets used by the topological conflict check.
# Each cardinal action is assumed to displace the agent by exactly one unit
# along (x, y); vertical actions move along z but are projected to a 2D
# coordinate for the unit-distance check.
DIRECTION_OFFSETS: Dict[str, Tuple[int, int]] = {
    "north":     (0,  1),
    "south":     (0, -1),
    "east":      (1,  0),
    "west":     (-1,  0),
    "northeast": (1,  1),
    "northwest": (-1, 1),
    "southeast": (1, -1),
    "southwest": (-1, -1),
    # Vertical and portal-style movements do not change planar position.
    "up":   (0, 0),
    "down": (0, 0),
    "in":   (0, 0),
    "out":  (0, 0),
    "enter": (0, 0),
    "exit":  (0, 0),
}

REVERSE_DIRECTIONS: Dict[str, str] = {
    "north": "south", "south": "north",
    "east":  "west",  "west":  "east",
    "northeast": "southwest", "southwest": "northeast",
    "northwest": "southeast", "southeast": "northwest",
    "up":   "down",  "down": "up",
    "in":   "out",   "out":  "in",
    "enter": "exit", "exit": "enter",
}


@dataclass
class Location:
    name: str
    description: str = ""
    first_seen_step: int = -1
    aliases: Set[str] = field(default_factory=set)


class NavigationGraph:
    """Incrementally constructed navigation graph with unit-distance positions."""

    def __init__(self, model: str = "gpt-4o",
                 game_name: Optional[str] = None,
                 dataset: Optional[MANGODataset] = None):
        self.graph = nx.DiGraph()
        self.model = model
        self.game_name = game_name
        self.dataset = dataset
        self.current_location: Optional[str] = None
        self.previous_location: Optional[str] = None
        self.step_history: List[Dict] = []
        self.location_registry: Dict[str, str] = {}
        self.pending_steps: List[WalkthroughStep] = []
        self.last_graph_update_step = -1
        self.llm_history: List[str] = []
        self.game_locations: List[str] = []
        self.node_positions: Dict[str, Tuple[int, int]] = {}

        if self.game_name and self.dataset:
            self.game_locations = self.dataset.load_locations(self.game_name)

    # ------------------------------------------------------------------
    # LLM-driven step processing
    # ------------------------------------------------------------------
    def process_step(self, current_step: WalkthroughStep) -> Dict:
        """Process a single walkthrough step.

        Returns a dict containing added nodes/edges and the LLM interaction
        record so the caller can persist it for Version Control.
        """
        self.pending_steps.append(current_step)
        if len(self.pending_steps) > 3:
            self.pending_steps = self.pending_steps[-3:]

        nav_info, llm_interaction = self._extract_navigation_info(self.pending_steps)

        result = {
            "step_num": current_step.step_num,
            "added_nodes": [],
            "added_edges": [],
            "current_location": self.current_location,
            "llm_interaction": llm_interaction,
            "analysis": nav_info.get("analysis", ""),
        }

        # Add new locations
        for location_info in nav_info.get("new_locations", []):
            loc_name = self._standardize_location_name(location_info["name"])
            if loc_name not in self.graph:
                self.graph.add_node(loc_name, data=Location(
                    name=loc_name,
                    description=location_info.get("description", ""),
                    first_seen_step=location_info.get("first_seen_step", current_step.step_num),
                ))
                result["added_nodes"].append(loc_name)

        # Add new edges (and assign coordinates)
        for edge_info in nav_info.get("new_edges", []):
            from_loc = self._standardize_location_name(edge_info["from_location"])
            to_loc = self._standardize_location_name(edge_info["to_location"])
            direction = (edge_info.get("direction") or "").lower().strip()
            step_num = edge_info.get("step_num", current_step.step_num)

            for loc in (from_loc, to_loc):
                if loc not in self.graph:
                    self.graph.add_node(loc, data=Location(
                        name=loc, first_seen_step=step_num))
                    if loc not in result["added_nodes"]:
                        result["added_nodes"].append(loc)

            if not self.graph.has_edge(from_loc, to_loc):
                self.graph.add_edge(from_loc, to_loc,
                                    direction=direction, step_num=step_num)
                result["added_edges"].append((from_loc, to_loc, direction))

            # Assign unit-distance position to the destination if needed
            self._propagate_position(from_loc, to_loc, direction)

            # Auto-insert reverse edge for cardinal directions
            reverse_dir = (edge_info.get("reverse_direction")
                           or REVERSE_DIRECTIONS.get(direction))
            if reverse_dir and not self.graph.has_edge(to_loc, from_loc):
                self.graph.add_edge(to_loc, from_loc,
                                    direction=reverse_dir, step_num=step_num,
                                    is_auto_reverse=True)
                result["added_edges"].append((to_loc, from_loc, reverse_dir))

        if nav_info.get("current_location"):
            self.current_location = self._standardize_location_name(
                nav_info["current_location"])
            result["current_location"] = self.current_location

        if result["added_nodes"] or result["added_edges"]:
            self.last_graph_update_step = current_step.step_num
            self.pending_steps = []

        self.step_history.append({
            "step_num": current_step.step_num,
            "action": current_step.action,
            "observation": current_step.observation,
        })
        return result

    # ------------------------------------------------------------------
    # Position handling for the topological conflict check
    # ------------------------------------------------------------------
    def _propagate_position(self,
                            from_loc: str,
                            to_loc: str,
                            direction: str):
        """Assign a unit-distance position to `to_loc` relative to `from_loc`.

        The first node added is anchored at the origin (0, 0). Every
        subsequent node receives a position equal to its predecessor's
        position plus the unit offset for `direction`. Existing positions
        are preserved (so two paths converging on the same node do not
        clobber its coordinate).
        """
        if from_loc not in self.node_positions and not self.node_positions:
            self.node_positions[from_loc] = (0, 0)
        if from_loc not in self.node_positions:
            # Anchor disconnected fragments at the origin too
            self.node_positions[from_loc] = (0, 0)

        if to_loc in self.node_positions:
            return

        offset = DIRECTION_OFFSETS.get(direction, (0, 0))
        src_pos = self.node_positions[from_loc]
        self.node_positions[to_loc] = (src_pos[0] + offset[0],
                                       src_pos[1] + offset[1])

    def get_node_positions(self) -> Dict[str, Tuple[int, int]]:
        """Snapshot of the unit-distance positions for the topological check."""
        return dict(self.node_positions)

    # ------------------------------------------------------------------
    # Naming utilities
    # ------------------------------------------------------------------
    def _standardize_location_name(self, location_name: str) -> str:
        """Standardize names; reuse existing canonical names where possible."""
        if location_name in self.location_registry:
            return self.location_registry[location_name]

        for std_name in self.graph.nodes():
            if location_name.lower() == std_name.lower():
                self.location_registry[location_name] = std_name
                return std_name

        self.location_registry[location_name] = location_name
        return location_name

    @staticmethod
    def _opposite(direction: str) -> Optional[str]:
        return REVERSE_DIRECTIONS.get(direction.lower())

    # ------------------------------------------------------------------
    # LLM prompt construction
    # ------------------------------------------------------------------
    def _extract_navigation_info(self,
                                 pending_steps: List[WalkthroughStep]) -> Tuple[Dict, Dict]:
        locations_info = ""
        if self.game_locations:
            locations_info = (
                f"\nList of all known locations in the current game: "
                f"{self.game_locations}\n"
            )

        system_prompt = """You are a text game navigation analysis expert. Analyze the player's sequence of actions and observations to extract location and movement information and update the navigation graph.
""" + locations_info + """
You need to decide whether new location nodes or connection edges need to be created in the graph. Often the player may just be performing actions at the same location (taking items, looking, talking to NPCs, etc.), in which case no new nodes or edges need to be created.

Return JSON of the form:
{
    "new_locations": [
        {"name": "location name", "description": "location description", "first_seen_step": step_number}
    ],
    "new_edges": [
        {"from_location": "starting location", "to_location": "target location",
         "direction": "movement direction", "step_num": step_number,
         "reverse_direction": "reverse direction (optional)"}
    ],
    "current_location": "current location name",
    "analysis": "brief explanation of the decision"
}

Important principles:
1. Only add to new_locations when new locations are actually discovered.
2. Only add to new_edges when movement between locations actually occurs.
3. A new edge can only be created when the action is one of: "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "up", "down", "in", "out", "enter", "exit".
4. If there are no new discoveries, new_locations and new_edges should be empty arrays [].
5. Location names usually appear in the first line of the observation or in obvious titles.
6. Direction words usually indicate movement.
7. Carefully analyze observation text changes to determine if a new location has truly been reached.
8. Results should only pertain to the current step.
9. from_location and to_location cannot be the same.
"""

        user_prompt = "Analyze the following sequence of steps to identify new locations and connections:\n\n"
        for step in pending_steps:
            user_prompt += (f"Step {step.step_num}:\n"
                            f"Action: {step.action}\n"
                            f"Observation: {step.observation}\n\n")

        if self.llm_history:
            user_prompt += "Previous analysis results:\n"
            for i, hist in enumerate(self.llm_history[-3:]):
                user_prompt += f"Historical analysis {i + 1}: {hist}\n"
            user_prompt += "\n"

        user_prompt += (f"Current known graph state:\n"
                        f"Number of nodes: {self.graph.number_of_nodes()}\n"
                        f"Number of edges: {self.graph.number_of_edges()}\n")
        if self.graph.nodes():
            user_prompt += f"Existing locations: {list(self.graph.nodes())}\n"

        messages = [
            message_template("system", system_prompt),
            message_template("user", user_prompt),
        ]

        llm_interaction = {
            "step_range": (f"{pending_steps[0].step_num}-{pending_steps[-1].step_num}"
                           if pending_steps else ""),
            "input": {"system_prompt": system_prompt,
                      "user_prompt": user_prompt,
                      "messages": messages},
            "output": None,
        }

        response = chat_single(messages, mode="json", model=self.model)
        llm_interaction["output"] = response
        self.llm_history.append(response)
        return json.loads(response), llm_interaction

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def get_graph_stats(self) -> Dict:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "connected_components": nx.number_weakly_connected_components(self.graph),
            "current_location": self.current_location,
            "locations": list(self.graph.nodes()),
        }

    def export_to_json(self) -> Dict:
        nodes = []
        for node, data in self.graph.nodes(data=True):
            loc_data = data.get("data")
            nodes.append({
                "id": node,
                "name": node,
                "position": list(self.node_positions.get(node, (0, 0))),
                "data": {
                    "description": loc_data.description if loc_data else "",
                    "first_seen_step": loc_data.first_seen_step if loc_data else -1,
                },
            })
        edges = []
        for src, dst, data in self.graph.edges(data=True):
            edges.append({
                "from": src,
                "to": dst,
                "direction": data.get("direction", ""),
                "step_num": data.get("step_num", -1),
                "is_auto_reverse": data.get("is_auto_reverse", False),
            })
        return {"nodes": nodes, "edges": edges, "stats": self.get_graph_stats()}

    def visualize_graph(self) -> str:
        lines = ["Navigation Graph Structure:",
                 f"Number of nodes: {self.graph.number_of_nodes()}",
                 f"Number of edges: {self.graph.number_of_edges()}",
                 f"Current location: {self.current_location}",
                 "",
                 "Connections:"]
        for node in sorted(self.graph.nodes()):
            edges = []
            for _, target, data in self.graph.out_edges(node, data=True):
                direction = data.get("direction", "?")
                edges.append(f"{direction} -> {target}")
            if edges:
                lines.append(f"{node}:")
                for e in edges:
                    lines.append(f"  {e}")
        return "\n".join(lines)
