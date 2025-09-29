import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import json
from datetime import datetime

from llm_agent import chat_single, message_template
from mango_dataset import WalkthroughStep, MANGODataset


@dataclass
class Location:
    """Represents a location node in the graph"""
    name: str
    description: str = ""
    first_seen_step: int = -1
    aliases: Set[str] = field(default_factory=set)


@dataclass
class Edge:
    """Represents an edge between locations"""
    from_location: str
    to_location: str
    direction: str
    step_num: int
    reverse_direction: Optional[str] = None


class NavigationGraph:
    """Class for incrementally building navigation graphs"""
    
    def __init__(self, model: str = "gpt-4o", game_name: str = None, dataset=None):
        self.graph = nx.DiGraph()
        self.model = model
        self.game_name = game_name
        self.dataset = dataset
        self.current_location = None
        self.previous_location = None
        self.step_history = []
        self.location_registry = {}  # Mapping from location names to standard names
        self.pending_steps = []  # Buffer for pending steps
        self.last_graph_update_step = -1  # Step number of last graph update
        self.llm_history = []  # LLM output history
        self.game_locations = []  # List of all locations in the game
        
        # If game name and dataset are provided, load location list
        if self.game_name and self.dataset:
            self.game_locations = self.dataset.load_locations(self.game_name)
        
    def _extract_navigation_info(self, pending_steps: List[WalkthroughStep]) -> Tuple[Dict, Dict]:
        """Extract navigation information using LLM based on accumulated step history"""
        
        # Build system prompt, including game location list (if available)
        locations_info = ""
        if self.game_locations:
            locations_info = f"\nList of all known locations in the current game: {self.game_locations}\n"
        
        system_prompt = """You are a text game navigation analysis expert. Analyze the player's sequence of actions and observations to extract location and movement information and update the navigation graph.
""" + locations_info + """
You need to decide whether new location nodes or connection edges need to be created in the graph. Often the player may just be performing actions at the same location (e.g., taking items, looking, talking to NPCs, etc.), in which case no new nodes or edges need to be created.

Return in JSON format:
{
    "new_locations": [
        {
            "name": "location name",
            "description": "location description",
            "first_seen_step": step_number
        }
    ],
    "new_edges": [
        {
            "from_location": "starting location",
            "to_location": "target location", 
            "direction": "movement direction",
            "step_num": step_number,
            "reverse_direction": "reverse direction (optional)"
        }
    ],
    "current_location": "current location name",
    "analysis": "analysis explanation of why new nodes/edges were added or not added"
}

Important principles:
1. Only add to new_locations when new locations are actually discovered
2. Only add to new_edges when movement between locations actually occurs
3. A new edge can only be created when the action is one of: "north", "south", "east", "west","northeast", "southwest", "northwest", "southeast","up", "down", "exit", "enter", "in", or "out".
4. If there are no new discoveries, new_locations and new_edges should be empty arrays []
5. Location names are usually in the first line of the observation or in obvious titles
6. Direction words (north, south, east, west, up, down, etc.) usually indicate movement
7. Carefully analyze changes in observation text to determine if a new location has truly been reached
8. The results should only pertain to the current step
9. from_location and to_location cannot be the same, you can add parenthetical descriptions after locations with the same name to distinguish different parts
"""

        # Build user input containing step sequence
        user_prompt = f"Analyze the following sequence of steps to identify new locations and connections:\n\n"
        
        for i, step in enumerate(pending_steps):
            user_prompt += f"Step {step.step_num}:\n"
            user_prompt += f"Action: {step.action}\n"
            user_prompt += f"Observation: {step.observation}\n\n"
        
        # Add historical LLM output as context
        if self.llm_history:
            user_prompt += "Previous analysis results:\n"
            for i, hist in enumerate(self.llm_history[-3:]):  # Only include the last 3
                user_prompt += f"Historical analysis {i+1}: {hist}\n"
            user_prompt += "\n"
        
        user_prompt += f"Current known graph state:\n"
        user_prompt += f"Number of nodes: {self.graph.number_of_nodes()}\n"
        user_prompt += f"Number of edges: {self.graph.number_of_edges()}\n"
        if self.graph.nodes():
            user_prompt += f"Existing locations: {list(self.graph.nodes())}\n"
        
        messages = [
            message_template('system', system_prompt),
            message_template('user', user_prompt)
        ]
        
        # Save LLM input and output
        llm_interaction = {
            'step_range': f"{pending_steps[0].step_num}-{pending_steps[-1].step_num}",
            'input': {
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'messages': messages
            },
            'output': None
        }
        
        response = chat_single(messages, mode="json", model=self.model)
        llm_interaction['output'] = response
        
        # Save to history
        self.llm_history.append(response)
        
        return json.loads(response), llm_interaction
    
    def _standardize_location_name(self, location_name: str) -> str:
        """Standardize location names, handle aliases and detect naming conflicts"""
        # Check if standard name already exists
        if location_name in self.location_registry:
            return self.location_registry[location_name]
        
        # Check if it's exactly the same as an existing location (potential naming conflict)
        for std_name in self.graph.nodes():
            if location_name.lower() == std_name.lower():
                # Check if current location is different from existing location with same name
                # Can judge based on context (e.g., previous location)
                current_context = getattr(self, 'current_location', None)
                
                # If current location has no direct connection to existing location with same name, might be naming conflict
                if (current_context and current_context in self.graph and 
                    std_name in self.graph and 
                    not self.graph.has_edge(current_context, std_name) and
                    not self.graph.has_edge(std_name, current_context)):
                    
                    # Create unique name with suffix
                    counter = 2
                    unique_name = f"{location_name} ({counter})"
                    while unique_name in self.graph:
                        counter += 1
                        unique_name = f"{location_name} ({counter})"
                    
                    self.location_registry[location_name] = unique_name
                    return unique_name
                else:
                    # Consider it the same location
                    self.location_registry[location_name] = std_name
                    return std_name
        
        # New location, use original name as standard name
        self.location_registry[location_name] = location_name
        return location_name
    
    def _get_opposite_direction(self, direction: str) -> str:
        """Get opposite direction"""
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
    
    def process_step(self, current_step: WalkthroughStep) -> Dict:
        """Process a single step, call LLM for each step but let LLM decide whether to create new nodes/edges"""
        
        # Add current step to pending queue
        self.pending_steps.append(current_step)
        
        # Keep only the last 3 steps for context
        if len(self.pending_steps) > 3:
            self.pending_steps = self.pending_steps[-3:]
        
        # Call LLM analysis for each step
        nav_info, llm_interaction = self._extract_navigation_info(self.pending_steps)
        
        result = {
            'step_num': current_step.step_num,
            'added_nodes': [],
            'added_edges': [],
            'current_location': self.current_location,
            'llm_interaction': llm_interaction
        }
        
        # Process new locations
        for location_info in nav_info.get('new_locations', []):
            loc_name = self._standardize_location_name(location_info['name'])
            if loc_name not in self.graph:
                location_data = Location(
                    name=loc_name,
                    description=location_info.get('description', ''),
                    first_seen_step=location_info.get('first_seen_step', current_step.step_num)
                )
                self.graph.add_node(loc_name, data=location_data)
                result['added_nodes'].append(loc_name)
        
        # Process new edges
        for edge_info in nav_info.get('new_edges', []):
            from_loc = self._standardize_location_name(edge_info['from_location'])
            to_loc = self._standardize_location_name(edge_info['to_location'])
            direction = edge_info.get('direction', '').lower()
            step_num = edge_info.get('step_num', current_step.step_num)
            
            # Ensure nodes exist
            for loc in [from_loc, to_loc]:
                if loc not in self.graph:
                    self.graph.add_node(loc, data=Location(
                        name=loc,
                        first_seen_step=step_num
                    ))
                    if loc not in result['added_nodes']:
                        result['added_nodes'].append(loc)
            
            # Add edge (if it doesn't exist)
            if not self.graph.has_edge(from_loc, to_loc):
                edge_data = {
                    'direction': direction,
                    'step_num': step_num
                }
                self.graph.add_edge(from_loc, to_loc, **edge_data)
                result['added_edges'].append((from_loc, to_loc, direction))
            
            # Add reverse edge
            reverse_dir = edge_info.get('reverse_direction') or self._get_opposite_direction(direction)
            if reverse_dir and not self.graph.has_edge(to_loc, from_loc):
                self.graph.add_edge(to_loc, from_loc, 
                                  direction=reverse_dir,
                                  step_num=step_num)
                result['added_edges'].append((to_loc, from_loc, reverse_dir))
        
        # Update current location
        if nav_info.get('current_location'):
            self.current_location = self._standardize_location_name(nav_info['current_location'])
            result['current_location'] = self.current_location
        
        # If there are new nodes or edges, clear pending queue; otherwise keep for next analysis
        if result['added_nodes'] or result['added_edges']:
            self.last_graph_update_step = current_step.step_num
            self.pending_steps = []  # Clear after successful update
        
        # Record step history
        self.step_history.append({
            'step_num': current_step.step_num,
            'action': current_step.action,
            'observation': current_step.observation
        })
        
        return result
    
    
    def get_graph_stats(self) -> Dict:
        """Get graph statistics"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'current_location': self.current_location,
            'locations': list(self.graph.nodes())
        }
    
    def export_to_json(self) -> Dict:
        """Export graph to JSON format"""
        nodes = []
        for node, data in self.graph.nodes(data=True):
            node_data = {
                'id': node,
                'name': node,
                'data': {
                    'description': data['data'].description if 'data' in data else '',
                    'first_seen_step': data['data'].first_seen_step if 'data' in data else -1
                }
            }
            nodes.append(node_data)
        
        edges = []
        for from_node, to_node, data in self.graph.edges(data=True):
            edge_data = {
                'from': from_node,
                'to': to_node,
                'direction': data.get('direction', ''),
                'step_num': data.get('step_num', -1)
            }
            edges.append(edge_data)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'stats': self.get_graph_stats()
        }
    
    def visualize_graph(self) -> str:
        """Generate text visualization of the graph"""
        lines = ["Navigation Graph Structure:"]
        lines.append(f"Number of nodes: {self.graph.number_of_nodes()}")
        lines.append(f"Number of edges: {self.graph.number_of_edges()}")
        lines.append(f"Current location: {self.current_location}")
        lines.append("\nConnection relationships:")
        
        for node in sorted(self.graph.nodes()):
            edges = []
            for _, target, data in self.graph.out_edges(node, data=True):
                direction = data.get('direction', '?')
                edges.append(f"{direction} -> {target}")
            
            if edges:
                lines.append(f"{node}:")
                for edge in edges:
                    lines.append(f"  {edge}")
        
        return '\n'.join(lines)