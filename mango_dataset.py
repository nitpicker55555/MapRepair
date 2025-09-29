import os
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class WalkthroughStep:
    """Represents a step in the walkthrough"""
    step_num: int
    action: str
    observation: str


class MANGODataset:
    """MANGO dataset interface for loading and parsing walkthrough data"""
    
    def __init__(self, data_dir: str = "/Users/puzhen/Desktop/mango/data"):
        self.data_dir = data_dir
        self.games = self._discover_games()
    
    def _discover_games(self) -> List[str]:
        """Discover all available games"""
        games = []
        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.isdir(folder_path):
                walkthrough_file = os.path.join(folder_path, f"{folder}.walkthrough")
                if os.path.exists(walkthrough_file):
                    games.append(folder)
        return sorted(games)
    
    def load_walkthrough(self, game_name: str) -> List[WalkthroughStep]:
        """Load walkthrough data for specified game"""
        if game_name not in self.games:
            raise ValueError(f"Game '{game_name}' not found. Available games: {self.games}")
        
        walkthrough_path = os.path.join(self.data_dir, game_name, f"{game_name}.walkthrough")
        steps = []
        
        with open(walkthrough_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by steps
        step_blocks = content.split("===========\n")
        
        for block in step_blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            step_num = None
            action = None
            observation_lines = []
            
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.startswith("==>STEP NUM:"):
                    step_num = int(line.split(":")[1].strip())
                elif line.startswith("==>ACT:"):
                    action = line.split(":", 1)[1].strip()
                elif line.startswith("==>OBSERVATION:"):
                    # Collect observation content (may span multiple lines)
                    obs_first_line = line.split(":", 1)[1].strip()
                    if obs_first_line:
                        observation_lines.append(obs_first_line)
                    i += 1
                    while i < len(lines) and not lines[i].startswith("==>"):
                        observation_lines.append(lines[i])
                        i += 1
                    i -= 1  # Go back one line
                i += 1
            
            if step_num is not None and action is not None:
                observation = '\n'.join(observation_lines)
                steps.append(WalkthroughStep(step_num, action, observation))
        
        return steps
    
    def load_locations(self, game_name: str) -> List[str]:
        """Load location list for the game"""
        locations_path = os.path.join(self.data_dir, game_name, f"{game_name}.locations.json")
        if os.path.exists(locations_path):
            with open(locations_path, 'r') as f:
                data = json.load(f)
                # locations.json directly contains a list
                if isinstance(data, list):
                    return data
                # If it's a dictionary format, try to get the locations key
                elif isinstance(data, dict):
                    return data.get('locations', [])
        return []
    
    def load_actions(self, game_name: str) -> List[str]:
        """Load action list for the game"""
        actions_path = os.path.join(self.data_dir, game_name, f"{game_name}.actions.json")
        if os.path.exists(actions_path):
            with open(actions_path, 'r') as f:
                data = json.load(f)
                return data.get('actions', [])
        return []
    
    def get_step_iterator(self, game_name: str):
        """返回一个迭代器，逐步返回walkthrough步骤"""
        steps = self.load_walkthrough(game_name)
        for step in steps:
            yield step


if __name__ == "__main__":
    # 测试代码
    dataset = MANGODataset()
    print(f"发现的游戏: {dataset.games[:5]}...")  # 显示前5个
    
    # 测试加载walkthrough
    game = "zork1"
    steps = dataset.load_walkthrough(game)
    print(f"\n{game}的前3个步骤:")
    for step in steps[:3]:
        print(f"步骤 {step.step_num}: {step.action}")
        print(f"观察: {step.observation[:100]}...")
        print()