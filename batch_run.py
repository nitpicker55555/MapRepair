"""
Batch run MAP-SLAM system, process multiple games and record all conflicts
Ignore over-connected conflicts
Support test_modify_tdg mode: graph construction with conflict repair functionality
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
import time
from collections import defaultdict
import pickle
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import copy
from datetime import datetime


def convert_to_serializable(obj, depth=0):
    """Convert objects to JSON serializable format"""
    if depth > 10:  # Prevent infinite recursion
        return f"<Max depth reached for {type(obj).__name__}>"
    
    try:
        # Try direct serialization first
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass
    
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        result = []
        for i, item in enumerate(obj):
            try:
                result.append(convert_to_serializable(item, depth + 1))
            except Exception as e:
                result.append(f"<Error serializing item {i}: {str(e)}>")
        return result
    elif isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            try:
                result[str(key)] = convert_to_serializable(value, depth + 1)
            except Exception as e:
                result[str(key)] = f"<Error serializing value: {str(e)}>"
        return result
    elif isinstance(obj, set):
        return [convert_to_serializable(item, depth + 1) for item in obj]
    elif hasattr(obj, '__dict__'):
        # If it's a custom object, return its attribute dictionary
        result = {'__class__': obj.__class__.__name__}
        for key, value in obj.__dict__.items():
            try:
                result[key] = convert_to_serializable(value, depth + 1)
            except Exception as e:
                result[key] = f"<Error: {str(e)}>"
        return result
    else:
        # Convert other types to string
        return f"<{type(obj).__name__}: {str(obj)}>"

# Add parent directory to Python path to import mango modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from map_slam_system import MapSLAMSystem
from conflict_detector import ConflictDetector
from navigation_graph import NavigationGraph
from temporal_dependency_graph import TemporalDependencyGraph
from conflict_localizer import ConflictLocalizer
from edge_impact_scorer import EdgeImpactScorer


def should_ignore_conflict(conflict) -> bool:
    """Determine if a conflict should be ignored"""
    # Ignore over-connected conflicts
    if conflict.type == 'topology' and 'over-connected' in conflict.description:
        return True
    return False


def get_conflict_id(conflict) -> str:
    """Generate unique identifier for a conflict"""
    # Use conflict type, description and involved edges to generate unique ID
    involved_edges_str = str(sorted(conflict.involved_edges)) if conflict.involved_edges else ""
    involved_nodes_str = str(sorted(conflict.involved_nodes)) if conflict.involved_nodes else ""
    
    # Combine various fields to create unique identifier
    conflict_id = f"{conflict.type}_{conflict.severity}_{conflict.description}_{involved_nodes_str}_{involved_edges_str}"
    return conflict_id


def process_game(game_name: str, data_dir: str, model: str, output_dir: str, max_steps: int = None) -> Dict:
    """Process a single game, return conflict statistics"""
    print(f"\n{'='*60}")
    print(f"Processing game: {game_name}")
    print(f"Data directory: {data_dir}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}")
    
    # Check if game folder exists
    game_path = os.path.join(data_dir, game_name)
    if not os.path.exists(game_path):
        print(f"Error: Game folder does not exist: {game_path}")
        return {"error": f"Game folder does not exist: {game_path}"}
    
    try:
        print(f"Creating MapSLAMSystem instance...")
        # Create system instance
        slam = MapSLAMSystem(data_dir=data_dir, model=model)
        print(f"MapSLAMSystem instance created successfully")
        
        # Load walkthrough steps
        steps = slam.dataset.load_walkthrough(game_name)
        total_steps = len(steps)
        
        if max_steps:
            process_steps = min(max_steps, total_steps)
        else:
            process_steps = total_steps
            
        print(f"Total steps: {total_steps}, Will process: {process_steps} steps")
        
        # Create game-specific JSONL log file
        game_log_file = os.path.join(output_dir, f"{game_name}_detailed.jsonl")
        
        # Write initial game info
        with open(game_log_file, 'w', encoding='utf-8') as log_file:
            game_info = {
                "type": "game_info",
                "timestamp": datetime.now().isoformat(),
                "game": game_name,
                "total_steps": total_steps,
                "processed_steps": process_steps,
                "model": model
            }
            log_file.write(json.dumps(game_info, ensure_ascii=False) + '\n')
        
        # 处理每一步
        slam.current_game = game_name
        # Recreate NavigationGraph instance for current game, including locations info
        slam.nav_graph = NavigationGraph(slam.model, game_name, slam.dataset)
        all_conflicts = []
        unique_conflicts = {}  # 用于存储唯一冲突，key为冲突的唯一标识
        conflict_counts = defaultdict(int)
        detailed_steps = []
        
        for i, current_step in enumerate(steps[:process_steps]):
            # 简单进度显示
            if i % 10 == 0 or i == process_steps - 1:
                print(f"\r处理进度: {i+1}/{process_steps}", end="", flush=True)

            # 处理单步
            nav_result = slam.nav_graph.process_step(current_step)
            
            # 检测冲突，传入游戏名称
            conflicts = slam.conflict_detector.detect_all_conflicts(
                slam.nav_graph.graph, current_step.step_num, game_name
            )
            
            # 过滤掉需要忽略的冲突
            filtered_conflicts = [c for c in conflicts if not should_ignore_conflict(c)]
            
            # 记录详细步骤信息
            step_info = {
                'step_num': convert_to_serializable(current_step.step_num),
                'action': convert_to_serializable(current_step.action),
                'observation': convert_to_serializable(current_step.observation),
                'llm_interaction': convert_to_serializable(nav_result.get('llm_interaction')),
                'added_nodes': convert_to_serializable(nav_result.get('added_nodes', [])),
                'added_edges': convert_to_serializable(nav_result.get('added_edges', [])),
                'conflicts': []
            }
            
            # 记录冲突信息
            if filtered_conflicts:
                for conflict in filtered_conflicts:
                    # 生成冲突的唯一标识符
                    conflict_id = get_conflict_id(conflict)
                    
                    # 记录冲突详情
                    conflict_info = {
                        'game': game_name,
                        'step': convert_to_serializable(current_step.step_num),
                        'action': convert_to_serializable(current_step.action),
                        'type': conflict.type,
                        'severity': conflict.severity,
                        'description': conflict.description,
                        'involved_nodes': convert_to_serializable(conflict.involved_nodes),
                        'involved_edges': convert_to_serializable(conflict.involved_edges),
                        'details': convert_to_serializable(conflict.details),
                        'conflict_id': conflict_id
                    }
                    
                    # 只有首次遇到这个冲突时才添加到unique_conflicts
                    if conflict_id not in unique_conflicts:
                        unique_conflicts[conflict_id] = conflict_info
                        conflict_counts[f"{conflict.type}_{conflict.severity}"] += 1
                        all_conflicts.append(conflict_info)
                    else:
                        # 更新最后出现的步骤信息
                        unique_conflicts[conflict_id]['last_seen_step'] = convert_to_serializable(current_step.step_num)
                    
                    step_info['conflicts'].append(conflict_info)
            
            # Write JSONL records for steps with LLM interaction or conflicts
            with open(game_log_file, 'a', encoding='utf-8') as log_file:
                # Write step record
                step_record = {
                    "type": "step",
                    "step_num": current_step.step_num,
                    "action": current_step.action,
                    "observation": current_step.observation
                }
                log_file.write(json.dumps(step_record, ensure_ascii=False) + '\n')
                
                # Write LLM input/output if exists
                if nav_result.get('llm_interaction'):
                    llm_data = nav_result['llm_interaction']
                    
                    # Write LLM input
                    if llm_data.get('input'):
                        llm_input_record = {
                            "type": "llm_input",
                            "step_num": current_step.step_num,
                            "prompt": llm_data.get('input'),
                            "system_prompt": llm_data.get('system_prompt', '')
                        }
                        log_file.write(json.dumps(llm_input_record, ensure_ascii=False) + '\n')
                    
                    # Write LLM output
                    llm_output_record = {
                        "type": "llm_output",
                        "step_num": current_step.step_num,
                        "response": llm_data.get('output'),
                        "tokens_used": llm_data.get('tokens_used', 0),
                        "model": model
                    }
                    try:
                        llm_output_record["response"] = json.loads(llm_data['output'])
                    except:
                        pass
                    log_file.write(json.dumps(llm_output_record, ensure_ascii=False) + '\n')
                
                # Write new conflicts (only first occurrence)
                for conflict in filtered_conflicts:
                    conflict_id = get_conflict_id(conflict)
                    if conflict_id not in unique_conflicts:
                        # Collect involved steps
                        involved_steps = []
                        edge_details = []
                        for edge in (conflict.involved_edges or []):
                            source, target = edge
                            if slam.nav_graph.graph.has_edge(source, target):
                                edge_data = slam.nav_graph.graph.get_edge_data(source, target)
                                edge_step = edge_data.get('step_num', 'unknown')
                                if edge_step != 'unknown':
                                    involved_steps.append(edge_step)
                                edge_details.append({
                                    "edge": [source, target],
                                    "step_num": edge_step,
                                    "direction": edge_data.get('direction', 'unknown')
                                })
                        
                        conflict_record = {
                            "type": "new_conflict",
                            "step_num": current_step.step_num,
                            "conflict_id": conflict_id,
                            "conflict_type": conflict.type,
                            "severity": conflict.severity,
                            "description": conflict.description,
                            "involved_nodes": list(conflict.involved_nodes) if conflict.involved_nodes else [],
                            "involved_edges": list(conflict.involved_edges) if conflict.involved_edges else [],
                            "involved_steps": sorted(set(involved_steps)),
                            "edge_details": edge_details,
                            "details": convert_to_serializable(conflict.details),
                            "first_occurrence": True
                        }
                        log_file.write(json.dumps(conflict_record, ensure_ascii=False) + '\n')
                
                # Write graph changes
                for node in nav_result.get('added_nodes', []):
                    graph_change_record = {
                        "type": "graph_change",
                        "step_num": current_step.step_num,
                        "operation": "add_node",
                        "node": node,
                        "attributes": {}
                    }
                    log_file.write(json.dumps(graph_change_record, ensure_ascii=False) + '\n')
                
                for edge in nav_result.get('added_edges', []):
                    if len(edge) >= 2:
                        graph_change_record = {
                            "type": "graph_change",
                            "step_num": current_step.step_num,
                            "operation": "add_edge",
                            "source": edge[0],
                            "target": edge[1],
                            "attributes": edge[2] if len(edge) > 2 else {}
                        }
                        log_file.write(json.dumps(graph_change_record, ensure_ascii=False) + '\n')
            
            # Only record to detailed_steps if there's LLM interaction or conflicts
            if nav_result.get('llm_interaction') or filtered_conflicts:
                detailed_steps.append(step_info)
            
            slam.processed_steps += 1
        
        print(f"\nProcessing complete!")
        
        # Execute final edge impact scoring
        edge_scores = slam.edge_scorer.score_edges(
            slam.nav_graph.graph, 
            slam.conflict_detector.conflicts,
            slam.walkthrough_edges
        )
        
        # Save NetworkX graph data
        graph_file = os.path.join(output_dir, f"{game_name}_graph.pkl")
        with open(graph_file, 'wb') as f:
            pickle.dump(slam.nav_graph.graph, f)
            f.flush()  # Flush buffer immediately
            os.fsync(f.fileno())  # Force write to disk
        print(f"Saved graph data to: {graph_file}")
        
        # Save graph in JSON format (for viewing)
        graph_json_file = os.path.join(output_dir, f"{game_name}_graph.json")
        graph_data = {
            'nodes': convert_to_serializable(list(slam.nav_graph.graph.nodes(data=True))),
            'edges': convert_to_serializable(list(slam.nav_graph.graph.edges(data=True)))
        }
        with open(graph_json_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
            f.flush()  # Flush buffer immediately
            os.fsync(f.fileno())  # Force write to disk
        print(f"Saved JSON graph data to: {graph_json_file}")
        
        # Statistics
        final_stats = slam.nav_graph.get_graph_stats()
        conflict_summary = slam.conflict_detector.get_conflict_summary()
        
        # Write final statistics to JSONL log
        with open(game_log_file, 'a', encoding='utf-8') as log_file:
            final_stats_record = {
                "type": "final_stats",
                "graph_stats": {
                    "num_nodes": final_stats['num_nodes'],
                    "num_edges": final_stats['num_edges'],
                    "connected_components": final_stats['connected_components']
                },
                "unique_conflicts_count": {
                    "total": len(all_conflicts),
                    "direction": len([c for c in all_conflicts if c['type'] == 'direction']),
                    "topology": len([c for c in all_conflicts if c['type'] == 'topology']),
                    "spatial_consistency": len([c for c in all_conflicts if c['type'] == 'spatial_consistency'])
                }
            }
            log_file.write(json.dumps(final_stats_record, ensure_ascii=False) + '\n')
        
        result = {
            'game': game_name,
            'total_steps': total_steps,
            'processed_steps': process_steps,
            'graph_stats': convert_to_serializable(final_stats),
            'conflict_summary': convert_to_serializable(conflict_summary),
            'conflict_counts': dict(conflict_counts),
            'all_conflicts': convert_to_serializable(all_conflicts),
            'total_conflicts': len(all_conflicts),
            'detailed_steps': convert_to_serializable(detailed_steps),
            'graph_file': graph_file,
            'graph_json_file': graph_json_file,
            'log_file': game_log_file
        }
        
        print(f"  - Graph nodes: {final_stats['num_nodes']}")
        print(f"  - Graph edges: {final_stats['num_edges']}")
        print(f"  - Unique conflicts: {len(all_conflicts)}")
        print(f"  - Direction conflicts: {len([c for c in all_conflicts if c['type'] == 'direction'])}")
        print(f"  - Topology conflicts: {len([c for c in all_conflicts if c['type'] == 'topology'])}")
        print(f"  - Spatial consistency conflicts: {len([c for c in all_conflicts if c['type'] == 'spatial_consistency'])}")
        
        return result
        
    except Exception as e:
        print(f"Processing game {game_name} error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to save existing graph data even on error
        try:
            if 'slam' in locals() and hasattr(slam, 'nav_graph') and slam.nav_graph.graph:
                graph_file = os.path.join(output_dir, f"{game_name}_graph.pkl")
                with open(graph_file, 'wb') as f:
                    pickle.dump(slam.nav_graph.graph, f)
                print(f"已保存部分图数据到: {graph_file}")
        except Exception as save_error:
            print(f"保存图数据时也出错: {save_error}")
        
        return {"error": str(e), "game": game_name}


def discover_games(data_dir: str) -> List[str]:
    """自动发现Data directory下的所有游戏文件夹"""
    games = []
    if not os.path.exists(data_dir):
        print(f"错误：Data directory不存在: {data_dir}")
        return games
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            games.append(item)
    
    return sorted(games)


def get_processed_games(output_dir: str) -> Set[str]:
    """获取已经处理过的游戏列表"""
    processed_games = set()
    
    if not os.path.exists(output_dir):
        return processed_games
    
    # 查找所有已处理的游戏（通过查找_graph.json, _detailed.log或_detailed.jsonl文件）
    for filename in os.listdir(output_dir):
        if (filename.endswith('_graph.json') or 
            filename.endswith('_detailed.log') or 
            filename.endswith('_detailed.jsonl')):
            # 提取游戏名称（第一个下划线之前的部分）
            game_name = filename.split('_')[0]
            processed_games.add(game_name)
    
    return processed_games


def process_game_wrapper(args_tuple):
    """包装函数，用于并行处理"""
    try:
        game, data_dir, model, output_dir, max_steps = args_tuple
        print(f"Processing game包装器: {game}")
        result = process_game(game, data_dir, model, output_dir, max_steps)
        print(f"完成Processing game包装器: {game}")
        return result
    except Exception as e:
        print(f"游戏包装器异常 {game}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"包装器异常: {str(e)}", "game": game}


def call_repair_llm_no_tdg(model: str, conflict_info: Dict, candidate_edges: List,
                          current_step: int, repair_history: List[Dict]) -> Dict:
    """简化版修复LLM调用，不提供TDG查询能力和边权重信息"""
    try:
        import openai
    except ImportError:
        print("错误：需要安装openai库")
        return {"action": "skip_conflict", "reason": "openai library not installed"}
    
    # 准备系统提示（简化版）
    system_prompt = """You are a graph repair expert. You will analyze conflicts in a navigation graph and decide how to fix them.

You will receive:
1. Conflict information including type, description, and involved edges
2. A list of candidate error edges (WITHOUT probability scores)
3. History of previous repair attempts in this session

You must respond with a JSON object containing one of these actions:
- {"action": "modify_edge", "edge": [source, target], "new_direction": <direction>, "reason": <reason>} - Modify an edge direction
- {"action": "skip_conflict", "reason": <reason>} - Skip this conflict

CRITICAL CONSTRAINTS:
- You CANNOT remove edges from the graph. Edge removal is completely disabled.
- You CANNOT query historical information or rollback to previous versions.
- You only have basic conflict information and candidate edges list.
- Focus on modifying edge directions to resolve direction conflicts.
- Direction conflicts are usually resolved by changing the direction of one edge to make it unique.

Consider the history of previous repairs to avoid repeating failed attempts.
Analyze the conflict carefully and choose the most appropriate action."""

    # 构建消息列表
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加历史修复记录（简化版）
    if repair_history:
        history_prompt = "Previous repair attempts in this session:\n"
        for idx, history_item in enumerate(repair_history[-3:]):  # 只显示最近3次
            history_prompt += f"\n{idx+1}. Conflict: {history_item['conflict_type']}"
            history_prompt += f"\n   Action: {history_item['action']['action']}"
            history_prompt += f"\n   Success: {history_item['success']}"
        messages.append({"role": "user", "content": history_prompt})
        messages.append({"role": "assistant", "content": "I understand the repair history. Now analyzing the current conflict."})
    
    # 准备当前冲突提示（不包含权重信息）
    user_prompt = f"""Current conflict:
Type: {conflict_info.get('type')}
Severity: {conflict_info.get('severity')}
Description: {conflict_info.get('description')}
Involved edges: {conflict_info.get('involved_edges')}
Details: {json.dumps(conflict_info.get('details', {}), indent=2)}

Candidate error edges:"""
    
    for i, edge in enumerate(candidate_edges[:5]):  # 只显示前5个候选边，无权重信息
        user_prompt += f"\n{i+1}. Edge: {edge.source} -> {edge.target}"
        user_prompt += f"\n   Direction: {edge.direction}"
    
    user_prompt += f"\n\nCurrent step: {current_step}"
    user_prompt += "\n\nWhat action should be taken to fix this conflict?"
    
    messages.append({"role": "user", "content": user_prompt})
    
    # 调用LLM
    client = openai.OpenAI()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=300  # 减少token数量，因为选项更少
        )
        
        # 解析响应
        response_text = response.choices[0].message.content.strip()
        
        # 提取JSON部分
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            response_text = response_text[json_start:json_end].strip()
        elif '```' in response_text:
            json_start = response_text.find('```') + 3
            json_end = response_text.find('```', json_start)
            response_text = response_text[json_start:json_end].strip()
        
        action = json.loads(response_text)
        return action
        
    except Exception as e:
        print(f"修复LLM调用失败: {e}")
        return {"action": "skip_conflict", "reason": f"LLM error: {str(e)}"}


def call_repair_llm(model: str, conflict_info: Dict, candidate_edges: List,
                   tdg: TemporalDependencyGraph, current_step: int, 
                   repair_history: List[Dict]) -> Dict:
    """调用修复LLM来决定如何修复冲突，带历史上下文"""
    try:
        import openai
    except ImportError:
        print("错误：需要安装openai库")
        return {"action": "skip_conflict", "reason": "openai library not installed"}
    
    # 准备系统提示（优化版本）
    system_prompt = """You are a graph repair expert with advanced weighting system. You will analyze conflicts in a navigation graph and decide how to fix them efficiently.

You will receive:
1. Conflict information including type, description, and involved edges
2. HIGH PRIORITY candidate error edges with calculated error probabilities (higher = more likely to be wrong)
3. History of previous repair attempts in this session
4. Available commands to query and modify the graph

PRIORITY RULES:
- ALWAYS prioritize edges with higher error probabilities (>0.8 = very high priority)
- Focus on edges marked as "HIGH PRIORITY" - these have been pre-filtered for efficiency
- Direction conflicts with error probability >0.7 should be modified immediately

You must respond with a JSON object containing one of these actions:
- {"action": "modify_edge", "edge": [source, target], "new_direction": <direction>, "reason": <reason>} - Modify an edge direction (PREFERRED for high-probability edges)
- {"action": "query_step_thinking", "step_num": <step_number>} - Get the LLM thinking at a specific step (use sparingly)
- {"action": "rollback_to_version", "version_id": <version_id>} - Rollback graph to a specific version (for complex topology issues)

CRITICAL CONSTRAINTS:
- You CANNOT remove edges from the graph. Edge removal is completely disabled.
- PREFER modify_edge for efficiency - it's faster than rollback operations
- Direction conflicts are usually resolved by changing the direction of the highest probability error edge
- Use the error probability scores to make informed decisions quickly

Consider the history of previous repairs to avoid repeating failed attempts.
Prioritize high-probability error edges for maximum efficiency."""

    # 构建消息列表
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加历史修复记录
    if repair_history:
        history_prompt = "Previous repair attempts in this session:\n"
        for idx, history_item in enumerate(repair_history[-5:]):  # 只显示最近5次
            history_prompt += f"\n{idx+1}. Conflict: {history_item['conflict_type']}"
            history_prompt += f"\n   Action: {history_item['action']['action']}"
            history_prompt += f"\n   Success: {history_item['success']}"
            if history_item['action'].get('reason'):
                history_prompt += f"\n   Reason: {history_item['action']['reason']}"
        messages.append({"role": "user", "content": history_prompt})
        messages.append({"role": "assistant", "content": "I understand the repair history. Now analyzing the current conflict."})
    
    # 准备当前冲突提示（优化版本）
    user_prompt = f"""Current conflict:
Type: {conflict_info.get('type')}
Severity: {conflict_info.get('severity')}
Description: {conflict_info.get('description')}
Involved edges: {conflict_info.get('involved_edges')}
Details: {json.dumps(conflict_info.get('details', {}), indent=2)}

Candidate error edges (sorted by error probability, showing top candidates):"""
    
    # 分析conflicting_targets来选择不同的方向
    conflicting_targets = conflict_info.get('details', {}).get('conflicting_targets', [])
    conflicting_direction = conflict_info.get('details', {}).get('conflicting_direction', '')
    
    for i, edge in enumerate(candidate_edges[:3]):  # 只显示前3个最高权重候选边
        user_prompt += f"\n{i+1}. HIGH PRIORITY Edge: {edge.source} -> {edge.target}"
        user_prompt += f"\n   Current Direction: {edge.direction}"
        user_prompt += f"\n   Error probability: {edge.error_probability:.3f}"
        user_prompt += f"\n   Conflicts involved: {edge.conflict_count}"
        
        # 为不同的边建议不同的方向修改
        if len(conflicting_targets) >= 2:
            if edge.target == conflicting_targets[0]:
                user_prompt += f"\n   >>> SMART RECOMMENDATION: Change to 'west' (opposite of east) to resolve conflict"
            elif edge.target == conflicting_targets[1] and len(conflicting_targets) > 1:
                user_prompt += f"\n   >>> SMART RECOMMENDATION: Change to 'south' (perpendicular to north) to resolve conflict"
            else:
                user_prompt += f"\n   >>> RECOMMENDATION: Change to a unique direction different from '{conflicting_direction}'"
    
    user_prompt += f"\n\nCurrent step: {current_step}"
    user_prompt += "\n\nWhat action should be taken to fix this conflict?"
    
    messages.append({"role": "user", "content": user_prompt})
    
    # 调用LLM
    client = openai.OpenAI()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )
        
        # 解析响应
        response_text = response.choices[0].message.content.strip()
        
        # 提取JSON部分
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            response_text = response_text[json_start:json_end].strip()
        elif '```' in response_text:
            json_start = response_text.find('```') + 3
            json_end = response_text.find('```', json_start)
            response_text = response_text[json_start:json_end].strip()
        
        action = json.loads(response_text)
        return action
        
    except Exception as e:
        print(f"修复LLM调用失败: {e}")
        return {"action": "skip_conflict", "reason": f"LLM error: {str(e)}"}


def repair_conflict_loop(conflicts: List[Any], graph: nx.DiGraph, tdg: TemporalDependencyGraph,
                        conflict_localizer: ConflictLocalizer, conflict_detector: Any,
                        model: str, current_step: int, game_name: str,
                        log_file: Any, max_attempts: int = 5) -> Dict:
    """修复冲突的循环，返回修复结果"""
    repair_history = []  # 维护修复历史供LLM参考
    all_repair_logs = []
    total_success = 0
    remaining_conflicts = list(conflicts)  # 复制一份冲突列表
    
    attempt = 0
    while remaining_conflicts and attempt < max_attempts:
        attempt += 1
        current_conflict = remaining_conflicts[0]
        
        print(f"  修复尝试 {attempt}/{max_attempts}, 剩余冲突: {len(remaining_conflicts)}")
        
        # 写入日志并实时刷新
        log_file.write(f"\n{'='*40}\n")
        log_file.write(f"[修复LLM] 尝试 {attempt}, 处理冲突: {current_conflict.type}\n")
        log_file.write(f"描述: {current_conflict.description}\n")
        log_file.flush()  # 立即刷新到磁盘
        
        # 优化：缓存候选错误边定位结果
        localization_results = conflict_localizer.localize_conflicts(graph, [current_conflict])
        candidate_edges = localization_results.get('candidate_edges', [])
        
        # 优化：限制候选边数量以提高效率，优先处理高权重边
        candidate_edges = candidate_edges[:3]  # 只取前3个最高权重的边
        
        if not candidate_edges:
            repair_log = {
                'attempt': attempt,
                'conflict_type': current_conflict.type,
                'action': {'action': 'skip_conflict', 'reason': 'No candidate edges found'},
                'success': False
            }
            repair_history.append(repair_log)
            all_repair_logs.append(repair_log)
            remaining_conflicts.pop(0)
            continue
        
        # 准备冲突信息
        conflict_info = {
            'type': current_conflict.type,
            'severity': current_conflict.severity,
            'description': current_conflict.description,
            'involved_edges': current_conflict.involved_edges,
            'involved_nodes': current_conflict.involved_nodes,
            'details': current_conflict.details
        }
        
        # 调用修复LLM（带历史上下文）
        action = call_repair_llm(model, conflict_info, candidate_edges, tdg, 
                               current_step, repair_history)
        
        # 记录LLM输出到日志
        log_file.write(f"[修复LLM输出] {json.dumps(action, ensure_ascii=False)}\n")
        log_file.flush()  # 立即刷新到磁盘
        
        repair_success = False
        
        # 执行修复动作
        if action['action'] == 'query_step_thinking':
            # 获取指定步骤的思考内容
            step_num = action['step_num']
            version_id = tdg.find_version_by_step(step_num)
            if version_id is not None:
                version = tdg.get_version(version_id)
                if version:
                    print(f"    steps骤 {step_num} 的触发观察: {version.trigger_observation[:100]}...")
                    log_file.write(f"查询步骤 {step_num} 的观察: {version.trigger_observation}\n")
                    log_file.flush()  # 立即刷新到磁盘
        
        elif action['action'] == 'rollback_to_version':
            # 回滚到指定版本
            version_id = action['version_id']
            rolled_back_graph = tdg.rollback_to_version(version_id)
            if rolled_back_graph:
                graph.clear()
                graph.update(rolled_back_graph)
                print(f"    已回滚到版本 {version_id}")
                log_file.write(f"已回滚到版本 {version_id}\n")
                log_file.flush()  # 立即刷新到磁盘
                repair_success = True
        
        elif action['action'] == 'modify_edge':
            # 修改边
            edge = tuple(action['edge'])
            if graph.has_edge(*edge):
                old_direction = graph[edge[0]][edge[1]].get('direction', 'unknown')
                graph[edge[0]][edge[1]]['direction'] = action['new_direction']
                print(f"    已修改边 {edge[0]} -> {edge[1]} 方向: {old_direction} -> {action['new_direction']}")
                log_file.write(f"修改边 {edge[0]} -> {edge[1]} 方向: {old_direction} -> {action['new_direction']}\n")
                log_file.flush()  # 立即刷新到磁盘
                repair_success = True
        
        # 删除边的选项已被移除，保留原有的 elif 结构但不执行任何操作
        # elif action['action'] == 'remove_edge': 
        #     # 此选项已被禁用
        
        elif action['action'] == 'skip_conflict':
            # 跳过冲突
            print(f"    跳过冲突: {action.get('reason', 'Unknown reason')}")
            log_file.write(f"跳过冲突: {action.get('reason', 'Unknown reason')}\n")
            log_file.flush()  # 立即刷新到磁盘
            remaining_conflicts.pop(0)  # 移除当前冲突
        
        elif action['action'] == 'remove_edge':
            # 删除边的选项已被禁用
            print(f"    删除边选项已被禁用，跳过冲突")
            log_file.write(f"删除边选项已被禁用，跳过冲突\n")
            log_file.flush()  # 立即刷新到磁盘
            remaining_conflicts.pop(0)  # 移除当前冲突
        
        else:
            # 处理无效或未知的动作
            print(f"    无效动作: {action.get('action', 'unknown')}")
            log_file.write(f"无效动作: {action.get('action', 'unknown')}\n")
            log_file.flush()  # 立即刷新到磁盘
            remaining_conflicts.pop(0)  # 移除当前冲突
        
        # 记录修复历史
        repair_log = {
            'attempt': attempt,
            'conflict_type': current_conflict.type,
            'action': action,
            'success': repair_success
        }
        repair_history.append(repair_log)
        all_repair_logs.append(repair_log)
        
        # 如果修复成功，重新检测冲突
        if repair_success:
            total_success += 1
            log_file.write(f"修复成功，重新检测冲突...\n")
            log_file.flush()  # 立即刷新到磁盘
            
            # 重新检测所有冲突
            new_conflicts = conflict_detector.detect_all_conflicts(
                graph, current_step, game_name
            )
            
            # 过滤掉需要忽略的冲突
            new_conflicts = [c for c in new_conflicts if not should_ignore_conflict(c)]
            
            # 更新剩余冲突列表
            remaining_conflicts = new_conflicts
            
            if new_conflicts:
                print(f"    修复后仍有 {len(new_conflicts)} 个冲突")
                log_file.write(f"修复后检测到 {len(new_conflicts)} 个新冲突\n")
                log_file.flush()  # 立即刷新到磁盘
            else:
                print(f"    所有冲突已解决!")
                log_file.write(f"所有冲突已解决!\n")
                log_file.flush()  # 立即刷新到磁盘
                break
        else:
            # 如果没有成功修复且不是跳过，继续尝试
            if action['action'] != 'skip_conflict':
                continue
    
    return {
        'success': len(remaining_conflicts) == 0,
        'total_attempts': attempt,
        'successful_repairs': total_success,
        'remaining_conflicts': len(remaining_conflicts),
        'repair_logs': all_repair_logs
    }


def repair_conflict_loop_no_tdg(conflicts: List[Any], graph: nx.DiGraph,
                               conflict_localizer: ConflictLocalizer, conflict_detector: Any,
                               model: str, current_step: int, game_name: str,
                               log_file: Any, max_attempts: int = 5) -> Dict:
    """修复冲突的循环（不使用TDG），返回修复结果"""
    repair_history = []  # 维护修复历史供LLM参考
    all_repair_logs = []
    total_success = 0
    remaining_conflicts = list(conflicts)  # 复制一份冲突列表
    
    attempt = 0
    while remaining_conflicts and attempt < max_attempts:
        attempt += 1
        current_conflict = remaining_conflicts[0]
        
        print(f"  修复尝试 {attempt}/{max_attempts}, 剩余冲突: {len(remaining_conflicts)}")
        
        # 写入日志并实时刷新
        log_file.write(f"\n{'='*40}\n")
        log_file.write(f"[修复LLM-NoTDG] 尝试 {attempt}, 处理冲突: {current_conflict.type}\n")
        log_file.write(f"描述: {current_conflict.description}\n")
        log_file.flush()  # 立即刷新到磁盘
        
        # 优化：缓存候选错误边定位结果（no_tdg版本）
        localization_results = conflict_localizer.localize_conflicts(graph, [current_conflict])
        candidate_edges = localization_results.get('candidate_edges', [])
        
        # 优化：限制候选边数量以提高效率
        candidate_edges = candidate_edges[:3]  # 只取前3个最高权重的边
        
        if not candidate_edges:
            repair_log = {
                'attempt': attempt,
                'conflict_type': current_conflict.type,
                'action': {'action': 'skip_conflict', 'reason': 'No candidate edges found'},
                'success': False
            }
            repair_history.append(repair_log)
            all_repair_logs.append(repair_log)
            remaining_conflicts.pop(0)
            continue
        
        # 准备冲突信息
        conflict_info = {
            'type': current_conflict.type,
            'severity': current_conflict.severity,
            'description': current_conflict.description,
            'involved_edges': current_conflict.involved_edges,
            'involved_nodes': current_conflict.involved_nodes,
            'details': current_conflict.details
        }
        
        # 调用简化版修复LLM（不使用TDG，不提供边权重）
        action = call_repair_llm_no_tdg(model, conflict_info, candidate_edges, 
                                       current_step, repair_history)
        
        # 记录LLM输出到日志
        log_file.write(f"[修复LLM-NoTDG输出] {json.dumps(action, ensure_ascii=False)}\n")
        log_file.flush()  # 立即刷新到磁盘
        
        repair_success = False
        
        # 执行修复动作（仅支持修改边和跳过冲突）
        if action['action'] == 'modify_edge':
            # 修改边
            edge = tuple(action['edge'])
            if graph.has_edge(*edge):
                old_direction = graph[edge[0]][edge[1]].get('direction', 'unknown')
                graph[edge[0]][edge[1]]['direction'] = action['new_direction']
                print(f"    已修改边 {edge[0]} -> {edge[1]} 方向: {old_direction} -> {action['new_direction']}")
                log_file.write(f"修改边 {edge[0]} -> {edge[1]} 方向: {old_direction} -> {action['new_direction']}\n")
                log_file.flush()  # 立即刷新到磁盘
                repair_success = True
        
        elif action['action'] == 'skip_conflict':
            # 跳过冲突
            print(f"    跳过冲突: {action.get('reason', 'Unknown reason')}")
            log_file.write(f"跳过冲突: {action.get('reason', 'Unknown reason')}\n")
            log_file.flush()  # 立即刷新到磁盘
            remaining_conflicts.pop(0)  # 移除当前冲突
        
        else:
            # 处理无效或未知的动作
            print(f"    无效动作: {action.get('action', 'unknown')}")
            log_file.write(f"无效动作: {action.get('action', 'unknown')}\n")
            log_file.flush()  # 立即刷新到磁盘
            remaining_conflicts.pop(0)  # 移除当前冲突
        
        # 记录修复历史
        repair_log = {
            'attempt': attempt,
            'conflict_type': current_conflict.type,
            'action': action,
            'success': repair_success
        }
        repair_history.append(repair_log)
        all_repair_logs.append(repair_log)
        
        # 如果修复成功，重新检测冲突
        if repair_success:
            total_success += 1
            log_file.write(f"修复成功，重新检测冲突...\n")
            log_file.flush()  # 立即刷新到磁盘
            
            # 重新检测所有冲突
            new_conflicts = conflict_detector.detect_all_conflicts(
                graph, current_step, game_name
            )
            
            # 过滤掉需要忽略的冲突
            new_conflicts = [c for c in new_conflicts if not should_ignore_conflict(c)]
            
            # 更新剩余冲突列表
            remaining_conflicts = new_conflicts
            
            if new_conflicts:
                print(f"    修复后仍有 {len(new_conflicts)} 个冲突")
                log_file.write(f"修复后检测到 {len(new_conflicts)} 个新冲突\n")
                log_file.flush()  # 立即刷新到磁盘
            else:
                print(f"    所有冲突已解决!")
                log_file.write(f"所有冲突已解决!\n")
                log_file.flush()  # 立即刷新到磁盘
                break
        else:
            # 如果没有成功修复且不是跳过，继续尝试
            if action['action'] != 'skip_conflict':
                continue
    
    return {
        'success': len(remaining_conflicts) == 0,
        'total_attempts': attempt,
        'successful_repairs': total_success,
        'remaining_conflicts': len(remaining_conflicts),
        'repair_logs': all_repair_logs
    }


def call_repair_llm_weight_only(model: str, conflict_info: Dict, candidate_edges: List,
                               edge_scorer: Any, graph: nx.DiGraph, current_step: int, 
                               repair_history: List[Dict], current_conflict=None) -> Dict:
    """调用修复LLM，仅使用边权重信息（不使用TDG）"""
    try:
        import openai
    except ImportError:
        return {"action": "skip_conflict", "reason": "OpenAI library not installed"}
    
    # 准备系统提示
    system_prompt = """You are a map error correction LLM. When given a conflict in a game map, you need to analyze the conflict and decide how to fix it.

You will receive:
1. A description of the conflict
2. Candidate edges that might be causing the error with their weights
3. Previous repair history

You must output a JSON response with one of these actions:
- {"action": "modify_edge", "edge": ["source", "target"], "new_direction": "direction"}
- {"action": "skip_conflict", "reason": "explanation"}

Focus on edges with higher error probability scores - these are more likely to be the source of the conflict."""
    
    # 准备用户提示，包含边权重信息
    user_prompt = f"""Conflict detected:
Type: {conflict_info.get('type')}
Severity: {conflict_info.get('severity')}
Description: {conflict_info.get('description')}
Involved edges: {conflict_info.get('involved_edges')}
Details: {json.dumps(conflict_info.get('details', {}), indent=2)}

Candidate error edges with weights:"""
    
    # 计算边的权重得分
    # 如果提供了current_conflict对象，使用它；否则跳过边权重计算
    if current_conflict:
        edge_scores = edge_scorer.score_edges(graph, [current_conflict])
    else:
        edge_scores = {}
    
    for i, edge in enumerate(candidate_edges[:5]):  # 显示前5个候选边
        edge_key = (edge.source, edge.target)
        score = edge_scores.get(edge_key, None)
        if score:
            user_prompt += f"\n{i+1}. Edge: {edge.source} -> {edge.target}"
            user_prompt += f"\n   Direction: {edge.direction}"
            user_prompt += f"\n   Impact Score: {score.total_score:.3f}"
            user_prompt += f"\n   Structural: {score.structural_score:.3f}, Usage: {score.usage_frequency:.3f}, Conflict: {score.conflict_impact:.3f}"
    
    user_prompt += f"\n\nCurrent step: {current_step}"
    
    # 添加修复历史
    if repair_history:
        user_prompt += "\n\nPrevious repair attempts in this session:"
        for i, repair in enumerate(repair_history[-3:]):  # 只显示最近3次
            user_prompt += f"\n{i+1}. Conflict: {repair['conflict_type']}, Action: {repair['action']['action']}"
            if repair['success']:
                user_prompt += " (Success)"
            else:
                user_prompt += " (Failed)"
    
    user_prompt += "\n\nWhat action should be taken to fix this conflict?"
    
    # 调用LLM
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"修复LLM调用失败: {e}")
        return {"action": "skip_conflict", "reason": f"LLM error: {str(e)}"}


def call_repair_llm_tdg_only(model: str, conflict_info: Dict, candidate_edges: List,
                            tdg: TemporalDependencyGraph, current_step: int, 
                            repair_history: List[Dict]) -> Dict:
    """调用修复LLM，仅使用TDG信息（不使用边权重）"""
    try:
        import openai
    except ImportError:
        return {"action": "skip_conflict", "reason": "OpenAI library not installed"}
    
    # 准备系统提示
    system_prompt = """You are a map error correction LLM with temporal dependency analysis. When given a conflict in a game map, you need to analyze the conflict and decide how to fix it.

You will receive:
1. A description of the conflict
2. Candidate edges that might be causing the error
3. Temporal dependency information showing when edges were added
4. Previous repair history

Available actions:
- {"action": "modify_edge", "edge": ["source", "target"], "new_direction": "direction"}
- {"action": "rollback_to_version", "version_id": version_number, "reason": "explanation"}
- {"action": "skip_conflict", "reason": "explanation"}

Use temporal information to identify when errors were introduced. Consider rolling back to earlier versions when multiple related edges were added together."""
    
    # 准备用户提示，包含TDG信息
    user_prompt = f"""Conflict detected:
Type: {conflict_info.get('type')}
Severity: {conflict_info.get('severity')}
Description: {conflict_info.get('description')}
Involved edges: {conflict_info.get('involved_edges')}
Details: {json.dumps(conflict_info.get('details', {}), indent=2)}

Candidate error edges:"""
    
    for i, edge in enumerate(candidate_edges[:5]):  # 显示前5个候选边
        user_prompt += f"\n{i+1}. Edge: {edge.source} -> {edge.target}"
        user_prompt += f"\n   Direction: {edge.direction}"
        
        # 查找边在TDG中的历史
        edge_tuple = (edge.source, edge.target)
        edge_history = tdg.get_element_history(edge_tuple)
        if edge_history:
            # 找到添加这条边的版本
            for change in edge_history:
                if change.change_type == 'add_edge' and edge_tuple in change.affected_elements:
                    user_prompt += f"\n   Added in version: {change.version_id}"
                    user_prompt += f"\n   Step: {tdg.versions[change.version_id].step_num if change.version_id < len(tdg.versions) else 'Unknown'}"
                    break
    
    # 添加TDG版本信息
    current_version_id = tdg.current_version_id
    user_prompt += f"\n\nCurrent version: {current_version_id}"
    user_prompt += f"\nTotal versions: {len(tdg.versions)}"
    
    # 显示最近的版本历史
    recent_versions = tdg.versions[-3:] if len(tdg.versions) >= 3 else tdg.versions
    if recent_versions:
        user_prompt += "\n\nRecent version history:"
        for v in recent_versions:
            user_prompt += f"\n- Version {v.version_id}: {v.trigger_observation}"
    
    user_prompt += f"\n\nCurrent step: {current_step}"
    
    # 添加修复历史
    if repair_history:
        user_prompt += "\n\nPrevious repair attempts in this session:"
        for i, repair in enumerate(repair_history[-3:]):  # 只显示最近3次
            user_prompt += f"\n{i+1}. Conflict: {repair['conflict_type']}, Action: {repair['action']['action']}"
            if repair['success']:
                user_prompt += " (Success)"
            else:
                user_prompt += " (Failed)"
    
    user_prompt += "\n\nWhat action should be taken to fix this conflict?"
    
    # 调用LLM
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"修复LLM调用失败: {e}")
        return {"action": "skip_conflict", "reason": f"LLM error: {str(e)}"}


def repair_conflict_loop_weight_only(conflicts: List[Any], graph: nx.DiGraph,
                                   conflict_localizer: ConflictLocalizer, conflict_detector: Any,
                                   edge_scorer: Any, model: str, current_step: int, game_name: str,
                                   log_file: Any, max_attempts: int = 5) -> Dict:
    """修复冲突的循环（仅使用边权重），返回修复结果"""
    repair_history = []
    all_repair_logs = []
    total_success = 0
    remaining_conflicts = list(conflicts)
    
    attempt = 0
    while remaining_conflicts and attempt < max_attempts:
        attempt += 1
        current_conflict = remaining_conflicts[0]
        
        print(f"  修复尝试 {attempt}/{max_attempts}, 剩余冲突: {len(remaining_conflicts)}")
        
        # 写入日志
        log_file.write(f"\n{'='*40}\n")
        log_file.write(f"[修复LLM-WeightOnly] 尝试 {attempt}, 处理冲突: {current_conflict.type}\n")
        log_file.write(f"描述: {current_conflict.description}\n")
        log_file.flush()
        
        # 定位候选错误边
        localization_results = conflict_localizer.localize_conflicts(graph, [current_conflict])
        candidate_edges = localization_results.get('candidate_edges', [])
        candidate_edges = candidate_edges[:5]  # 限制候选边数量
        
        if not candidate_edges:
            repair_log = {
                'attempt': attempt,
                'conflict_type': current_conflict.type,
                'action': {'action': 'skip_conflict', 'reason': 'No candidate edges found'},
                'success': False
            }
            repair_history.append(repair_log)
            all_repair_logs.append(repair_log)
            remaining_conflicts.pop(0)
            continue
        
        # 准备冲突信息
        conflict_info = {
            'type': current_conflict.type,
            'severity': current_conflict.severity,
            'description': current_conflict.description,
            'involved_edges': current_conflict.involved_edges,
            'involved_nodes': current_conflict.involved_nodes,
            'details': current_conflict.details
        }
        
        # 调用修复LLM（使用边权重）
        action = call_repair_llm_weight_only(model, conflict_info, candidate_edges, 
                                           edge_scorer, graph, current_step, repair_history, current_conflict)
        
        # 记录LLM输出
        log_file.write(f"[修复LLM-WeightOnly输出] {json.dumps(action, ensure_ascii=False)}\n")
        log_file.flush()
        
        repair_success = False
        
        # 执行修复动作
        if action['action'] == 'modify_edge':
            edge = tuple(action['edge'])
            if graph.has_edge(*edge):
                old_direction = graph[edge[0]][edge[1]].get('direction', 'unknown')
                graph[edge[0]][edge[1]]['direction'] = action['new_direction']
                print(f"    已修改边 {edge[0]} -> {edge[1]} 方向: {old_direction} -> {action['new_direction']}")
                log_file.write(f"修改边 {edge[0]} -> {edge[1]} 方向: {old_direction} -> {action['new_direction']}\n")
                log_file.flush()
                repair_success = True
        
        elif action['action'] == 'skip_conflict':
            print(f"    跳过冲突: {action.get('reason', 'Unknown reason')}")
            log_file.write(f"跳过冲突: {action.get('reason', 'Unknown reason')}\n")
            log_file.flush()
            remaining_conflicts.pop(0)
        
        else:
            print(f"    无效动作: {action.get('action', 'unknown')}")
            log_file.write(f"无效动作: {action.get('action', 'unknown')}\n")
            log_file.flush()
            remaining_conflicts.pop(0)
        
        # 记录修复历史
        repair_log = {
            'attempt': attempt,
            'conflict_type': current_conflict.type,
            'action': action,
            'success': repair_success
        }
        repair_history.append(repair_log)
        all_repair_logs.append(repair_log)
        
        # 如果修复成功，重新检测冲突
        if repair_success:
            total_success += 1
            log_file.write(f"修复成功，重新检测冲突...\n")
            log_file.flush()
            
            # 重新检测所有冲突
            new_conflicts = conflict_detector.detect_all_conflicts(
                graph, current_step, game_name
            )
            
            # 过滤掉需要忽略的冲突
            new_conflicts = [c for c in new_conflicts if not should_ignore_conflict(c)]
            
            # 更新剩余冲突列表
            remaining_conflicts = new_conflicts
            
            if new_conflicts:
                print(f"    修复后仍有 {len(new_conflicts)} 个冲突")
                log_file.write(f"修复后检测到 {len(new_conflicts)} 个新冲突\n")
                log_file.flush()
            else:
                print(f"    所有冲突已解决!")
                log_file.write(f"所有冲突已解决!\n")
                log_file.flush()
                break
        else:
            if action['action'] != 'skip_conflict':
                continue
    
    return {
        'success': len(remaining_conflicts) == 0,
        'total_attempts': attempt,
        'successful_repairs': total_success,
        'remaining_conflicts': len(remaining_conflicts),
        'repair_logs': all_repair_logs
    }


def repair_conflict_loop_tdg_only(conflicts: List[Any], graph: nx.DiGraph,
                                tdg: TemporalDependencyGraph, conflict_localizer: ConflictLocalizer,
                                conflict_detector: Any, model: str, current_step: int, game_name: str,
                                log_file: Any, max_attempts: int = 5) -> Dict:
    """修复冲突的循环（仅使用TDG），返回修复结果"""
    repair_history = []
    all_repair_logs = []
    total_success = 0
    remaining_conflicts = list(conflicts)
    
    attempt = 0
    while remaining_conflicts and attempt < max_attempts:
        attempt += 1
        current_conflict = remaining_conflicts[0]
        
        print(f"  修复尝试 {attempt}/{max_attempts}, 剩余冲突: {len(remaining_conflicts)}")
        
        # 写入日志
        log_file.write(f"\n{'='*40}\n")
        log_file.write(f"[修复LLM-TDGOnly] 尝试 {attempt}, 处理冲突: {current_conflict.type}\n")
        log_file.write(f"描述: {current_conflict.description}\n")
        log_file.flush()
        
        # 定位候选错误边
        localization_results = conflict_localizer.localize_conflicts(graph, [current_conflict])
        candidate_edges = localization_results.get('candidate_edges', [])
        candidate_edges = candidate_edges[:5]  # 限制候选边数量
        
        if not candidate_edges:
            repair_log = {
                'attempt': attempt,
                'conflict_type': current_conflict.type,
                'action': {'action': 'skip_conflict', 'reason': 'No candidate edges found'},
                'success': False
            }
            repair_history.append(repair_log)
            all_repair_logs.append(repair_log)
            remaining_conflicts.pop(0)
            continue
        
        # 准备冲突信息
        conflict_info = {
            'type': current_conflict.type,
            'severity': current_conflict.severity,
            'description': current_conflict.description,
            'involved_edges': current_conflict.involved_edges,
            'involved_nodes': current_conflict.involved_nodes,
            'details': current_conflict.details
        }
        
        # 调用修复LLM（使用TDG）
        action = call_repair_llm_tdg_only(model, conflict_info, candidate_edges, 
                                        tdg, current_step, repair_history)
        
        # 记录LLM输出
        log_file.write(f"[修复LLM-TDGOnly输出] {json.dumps(action, ensure_ascii=False)}\n")
        log_file.flush()
        
        repair_success = False
        
        # 执行修复动作
        if action['action'] == 'modify_edge':
            edge = tuple(action['edge'])
            if graph.has_edge(*edge):
                old_direction = graph[edge[0]][edge[1]].get('direction', 'unknown')
                graph[edge[0]][edge[1]]['direction'] = action['new_direction']
                print(f"    已修改边 {edge[0]} -> {edge[1]} 方向: {old_direction} -> {action['new_direction']}")
                log_file.write(f"修改边 {edge[0]} -> {edge[1]} 方向: {old_direction} -> {action['new_direction']}\n")
                
                # 在TDG中创建新版本
                tdg.create_version(
                    graph, 
                    current_step, 
                    {'modified_edge': edge, 'old_direction': old_direction, 'new_direction': action['new_direction']},
                    f"Modified edge {edge[0]} -> {edge[1]} direction",
                    []
                )
                log_file.flush()
                repair_success = True
        
        elif action['action'] == 'rollback_to_version':
            version_id = action['version_id']
            reason = action.get('reason', 'No reason provided')
            print(f"    尝试回滚到版本 {version_id}: {reason}")
            log_file.write(f"尝试回滚到版本 {version_id}: {reason}\n")
            
            # 执行回滚
            rollback_graph = tdg.rollback_to_version(version_id)
            if rollback_graph is not None:
                # 更新当前图为回滚后的版本
                graph.clear()
                graph.add_nodes_from(rollback_graph.nodes(data=True))
                graph.add_edges_from(rollback_graph.edges(data=True))
                
                print(f"    成功回滚到版本 {version_id}")
                log_file.write(f"成功回滚到版本 {version_id}\n")
                log_file.flush()
                repair_success = True
            else:
                print(f"    回滚失败: 版本 {version_id} 不存在")
                log_file.write(f"回滚失败: 版本 {version_id} 不存在\n")
                log_file.flush()
        
        elif action['action'] == 'skip_conflict':
            print(f"    跳过冲突: {action.get('reason', 'Unknown reason')}")
            log_file.write(f"跳过冲突: {action.get('reason', 'Unknown reason')}\n")
            log_file.flush()
            remaining_conflicts.pop(0)
        
        else:
            print(f"    无效动作: {action.get('action', 'unknown')}")
            log_file.write(f"无效动作: {action.get('action', 'unknown')}\n")
            log_file.flush()
            remaining_conflicts.pop(0)
        
        # 记录修复历史
        repair_log = {
            'attempt': attempt,
            'conflict_type': current_conflict.type,
            'action': action,
            'success': repair_success
        }
        repair_history.append(repair_log)
        all_repair_logs.append(repair_log)
        
        # 如果修复成功，重新检测冲突
        if repair_success:
            total_success += 1
            log_file.write(f"修复成功，重新检测冲突...\n")
            log_file.flush()
            
            # 重新检测所有冲突
            new_conflicts = conflict_detector.detect_all_conflicts(
                graph, current_step, game_name
            )
            
            # 过滤掉需要忽略的冲突
            new_conflicts = [c for c in new_conflicts if not should_ignore_conflict(c)]
            
            # 更新剩余冲突列表
            remaining_conflicts = new_conflicts
            
            if new_conflicts:
                print(f"    修复后仍有 {len(new_conflicts)} 个冲突")
                log_file.write(f"修复后检测到 {len(new_conflicts)} 个新冲突\n")
                log_file.flush()
            else:
                print(f"    所有冲突已解决!")
                log_file.write(f"所有冲突已解决!\n")
                log_file.flush()
                break
        else:
            if action['action'] != 'skip_conflict':
                continue
    
    return {
        'success': len(remaining_conflicts) == 0,
        'total_attempts': attempt,
        'successful_repairs': total_success,
        'remaining_conflicts': len(remaining_conflicts),
        'repair_logs': all_repair_logs
    }


def process_game_test_edges(game_name: str, data_dir: str, model: str, 
                           output_dir: str, max_steps: int = None) -> Dict:
    """处理单个游戏，直接从edges.json加载图并开始冲突修复循环"""
    print(f"\n{'='*60}")
    print(f"Processing game(test_edges模式): {game_name}")
    print(f"Data directory: {data_dir}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Check if game folder exists
    game_path = os.path.join(data_dir, game_name)
    if not os.path.exists(game_path):
        print(f"Error: Game folder does not exist: {game_path}")
        return {"error": f"Game folder does not exist: {game_path}"}
    
    # 检查edges.json文件是否存在
    edges_file = os.path.join(game_path, f"{game_name}.edges.json")
    if not os.path.exists(edges_file):
        print(f"错误：edges.json文件不存在: {edges_file}")
        return {"error": f"edges.json文件不存在: {edges_file}"}
    
    try:
        print(f"正在从edges.json加载图结构...")
        
        # 直接从edges.json加载图
        with open(edges_file, 'r', encoding='utf-8') as f:
            edges_data = json.load(f)
        
        # 创建NetworkX图
        graph = nx.DiGraph()
        
        # 定义反向方向映射
        opposite_directions = {
            'north': 'south', 'south': 'north',
            'east': 'west', 'west': 'east',
            'northeast': 'southwest', 'southwest': 'northeast',
            'northwest': 'southeast', 'southeast': 'northwest',
            'up': 'down', 'down': 'up',
            'exit': 'enter', 'enter': 'exit',
            'in': 'out', 'out': 'in',
        }
        
        # 添加边到图中
        for edge in edges_data:
            if 'src_node' in edge and 'dst_node' in edge and 'action' in edge:
                from_node = edge['src_node']
                to_node = edge['dst_node']
                direction = edge['action']
                seen_in_forward = edge.get('seen_in_forward', 0)
                
                # 添加边及其属性
                graph.add_edge(from_node, to_node, 
                             direction=direction, 
                             action=direction,
                             step_num=seen_in_forward,
                             seen_in_forward=seen_in_forward)
        
        # 添加自动反向边
        edges_to_add = []
        for edge in edges_data:
            if 'src_node' in edge and 'dst_node' in edge and 'action' in edge:
                from_node = edge['src_node']
                to_node = edge['dst_node']
                direction = edge['action'].lower()
                
                if direction in opposite_directions:
                    reverse_direction = opposite_directions[direction]
                    
                    # 检查反向边是否已存在
                    if not graph.has_edge(to_node, from_node):
                        edges_to_add.append({
                            'from': to_node,
                            'to': from_node,
                            'direction': reverse_direction,
                            'action': reverse_direction,
                            'is_auto_reverse': True,
                            'step_num': edge.get('seen_in_forward', 0)
                        })
        
        # 添加反向边
        for edge in edges_to_add:
            graph.add_edge(edge['from'], edge['to'],
                          direction=edge['direction'], 
                          action=edge['action'],
                          is_auto_reverse=edge['is_auto_reverse'],
                          step_num=edge['step_num'])
        
        print(f"已加载图: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
        
        # 创建系统组件
        conflict_detector = ConflictDetector()
        tdg = TemporalDependencyGraph()
        conflict_localizer = ConflictLocalizer()
        
        # 创建初始版本
        initial_version_id = tdg.create_version(
            graph, 
            0, 
            {'loaded_from_file': True}, 
            f"Loaded from {edges_file}", 
            []
        )
        
        # Create game-specific log file
        game_log_file = os.path.join(output_dir, f"{game_name}_test_edges_detailed.log")
        
        with open(game_log_file, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Game: {game_name} (test_edges模式)\n")
            log_file.write(f"从文件加载: {edges_file}\n")
            log_file.write(f"初始图状态: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边\n")
            log_file.write("=" * 80 + "\n\n")
        
        # 检测初始冲突
        print(f"正在检测初始冲突...")
        conflicts = conflict_detector.detect_all_conflicts(graph, 0, game_name)
        
        # 过滤掉需要忽略的冲突
        filtered_conflicts = [c for c in conflicts if not should_ignore_conflict(c)]
        
        print(f"检测到 {len(filtered_conflicts)} 个冲突")
        
        all_conflicts = []
        unique_conflicts = {}
        repair_summary = []
        
        # 记录初始冲突
        for conflict in filtered_conflicts:
            conflict_id = get_conflict_id(conflict)
            conflict_info = {
                'game': game_name,
                'step': 0,
                'action': 'initial_load',
                'type': conflict.type,
                'severity': conflict.severity,
                'description': conflict.description,
                'involved_nodes': convert_to_serializable(conflict.involved_nodes),
                'involved_edges': convert_to_serializable(conflict.involved_edges),
                'details': convert_to_serializable(conflict.details),
                'conflict_id': conflict_id,
                'repair_attempted': False,
                'repair_success': False
            }
            unique_conflicts[conflict_id] = conflict_info
            all_conflicts.append(conflict_info)
        
        # 如果有冲突，开始修复循环
        if filtered_conflicts:
            print(f"开始冲突修复循环...")
            
            with open(game_log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"检测到 {len(filtered_conflicts)} 个初始冲突\n")
                log_file.write("开始修复循环...\n")
                log_file.flush()
                
                # 执行修复循环
                repair_result = repair_conflict_loop(
                    filtered_conflicts,
                    graph,
                    tdg,
                    conflict_localizer,
                    conflict_detector,
                    model,
                    0,  # step_num
                    game_name,
                    log_file,
                    max_attempts=10
                )
                
                repair_summary.append({
                    'step': 0,
                    'initial_conflicts': len(filtered_conflicts),
                    'repair_result': repair_result
                })
                
                if repair_result['success']:
                    print(f"所有冲突修复成功!")
                    log_file.write(f"\n修复结果: 成功解决所有冲突\n")
                    
                    # 更新冲突状态为已修复
                    for conflict_info in all_conflicts:
                        conflict_info['repair_attempted'] = True
                        conflict_info['repair_success'] = True
                else:
                    print(f"仍有 {repair_result['remaining_conflicts']} 个冲突未解决")
                    log_file.write(f"\n修复结果: 仍有 {repair_result['remaining_conflicts']} 个冲突未解决\n")
                    
                    # 重新检测剩余冲突
                    remaining_conflicts = conflict_detector.detect_all_conflicts(graph, 0, game_name)
                    remaining_conflicts = [c for c in remaining_conflicts if not should_ignore_conflict(c)]
                    
                    # 更新冲突状态
                    for conflict in remaining_conflicts:
                        conflict_id = get_conflict_id(conflict)
                        if conflict_id in unique_conflicts:
                            unique_conflicts[conflict_id]['repair_attempted'] = True
                            unique_conflicts[conflict_id]['repair_success'] = False
                
                log_file.flush()
        else:
            print(f"没有检测到冲突，无需修复")
        
        # 保存最终图数据
        graph_file = os.path.join(output_dir, f"{game_name}_test_edges_graph.pkl")
        with open(graph_file, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Saved graph data to: {graph_file}")
        
        # 保存图的JSON格式
        graph_json_file = os.path.join(output_dir, f"{game_name}_test_edges_graph.json")
        graph_data = {
            'nodes': convert_to_serializable(list(graph.nodes(data=True))),
            'edges': convert_to_serializable(list(graph.edges(data=True)))
        }
        with open(graph_json_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON graph data to: {graph_json_file}")
        
        # 保存TDG时间线
        timeline_file = os.path.join(output_dir, f"{game_name}_test_edges_timeline.json")
        with open(timeline_file, 'w', encoding='utf-8') as f:
            json.dump(tdg.export_timeline(), f, ensure_ascii=False, indent=2)
        print(f"已保存TDG时间线到: {timeline_file}")
        
        # Statistics
        final_stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'connected_components': nx.number_weakly_connected_components(graph)
        }
        
        conflict_summary = conflict_detector.get_conflict_summary()
        
        result = {
            'game': game_name,
            'mode': 'test_edges',
            'edges_file': edges_file,
            'graph_stats': convert_to_serializable(final_stats),
            'conflict_summary': convert_to_serializable(conflict_summary),
            'all_conflicts': convert_to_serializable(all_conflicts),
            'total_conflicts': len(all_conflicts),
            'repair_summary': convert_to_serializable(repair_summary),
            'total_repairs_attempted': len(repair_summary),
            'successful_repairs': len([r for r in repair_summary if r['repair_result']['success']]),
            'tdg_stats': convert_to_serializable(tdg.analyze_evolution()),
            'graph_file': graph_file,
            'graph_json_file': graph_json_file,
            'timeline_file': timeline_file,
            'log_file': game_log_file
        }
        
        print(f"  - Graph nodes: {final_stats['num_nodes']}")
        print(f"  - Graph edges: {final_stats['num_edges']}")
        print(f"  - 总冲突数: {len(all_conflicts)}")
        print(f"  - 修复尝试: {len(repair_summary)}")
        print(f"  - 成功修复: {result['successful_repairs']}")
        
        return result
        
    except Exception as e:
        print(f"Processing game {game_name} error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "game": game_name}


def process_game_test_edges_no_tdg(game_name: str, data_dir: str, model: str, 
                                  output_dir: str, max_steps: int = None) -> Dict:
    """处理单个游戏，直接从edges.json加载图并开始冲突修复循环（不使用TDG）"""
    print(f"\n{'='*60}")
    print(f"Processing game(test_edges_no_tdg模式): {game_name}")
    print(f"Data directory: {data_dir}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Check if game folder exists
    game_path = os.path.join(data_dir, game_name)
    if not os.path.exists(game_path):
        print(f"Error: Game folder does not exist: {game_path}")
        return {"error": f"Game folder does not exist: {game_path}"}
    
    # 检查edges.json文件是否存在
    edges_file = os.path.join(game_path, f"{game_name}.edges.json")
    if not os.path.exists(edges_file):
        print(f"错误：edges.json文件不存在: {edges_file}")
        return {"error": f"edges.json文件不存在: {edges_file}"}
    
    try:
        print(f"正在从edges.json加载图结构...")
        
        # 直接从edges.json加载图
        with open(edges_file, 'r', encoding='utf-8') as f:
            edges_data = json.load(f)
        
        # 创建NetworkX图
        graph = nx.DiGraph()
        
        # 定义反向方向映射
        opposite_directions = {
            'north': 'south', 'south': 'north',
            'east': 'west', 'west': 'east',
            'northeast': 'southwest', 'southwest': 'northeast',
            'northwest': 'southeast', 'southeast': 'northwest',
            'up': 'down', 'down': 'up',
            'exit': 'enter', 'enter': 'exit',
            'in': 'out', 'out': 'in',
        }
        
        # 添加边到图中
        for edge in edges_data:
            if 'src_node' in edge and 'dst_node' in edge and 'action' in edge:
                from_node = edge['src_node']
                to_node = edge['dst_node']
                direction = edge['action']
                seen_in_forward = edge.get('seen_in_forward', 0)
                
                # 添加边及其属性
                graph.add_edge(from_node, to_node, 
                             direction=direction, 
                             action=direction,
                             step_num=seen_in_forward,
                             seen_in_forward=seen_in_forward)
        
        # 添加自动反向边
        edges_to_add = []
        for edge in edges_data:
            if 'src_node' in edge and 'dst_node' in edge and 'action' in edge:
                from_node = edge['src_node']
                to_node = edge['dst_node']
                direction = edge['action'].lower()
                
                if direction in opposite_directions:
                    reverse_direction = opposite_directions[direction]
                    
                    # 检查反向边是否已存在
                    if not graph.has_edge(to_node, from_node):
                        edges_to_add.append({
                            'from': to_node,
                            'to': from_node,
                            'direction': reverse_direction,
                            'action': reverse_direction,
                            'is_auto_reverse': True,
                            'step_num': edge.get('seen_in_forward', 0)
                        })
        
        # 添加反向边
        for edge in edges_to_add:
            graph.add_edge(edge['from'], edge['to'],
                          direction=edge['direction'], 
                          action=edge['action'],
                          is_auto_reverse=edge['is_auto_reverse'],
                          step_num=edge['step_num'])
        
        print(f"已加载图: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
        
        # 创建系统组件（不使用TDG）
        conflict_detector = ConflictDetector()
        conflict_localizer = ConflictLocalizer()
        
        # Create game-specific log file
        game_log_file = os.path.join(output_dir, f"{game_name}_test_edges_no_tdg_detailed.log")
        
        with open(game_log_file, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Game: {game_name} (test_edges_no_tdg模式)\n")
            log_file.write(f"从文件加载: {edges_file}\n")
            log_file.write(f"初始图状态: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边\n")
            log_file.write("=" * 80 + "\n\n")
        
        # 检测初始冲突
        print(f"正在检测初始冲突...")
        conflicts = conflict_detector.detect_all_conflicts(graph, 0, game_name)
        
        # 过滤掉需要忽略的冲突
        filtered_conflicts = [c for c in conflicts if not should_ignore_conflict(c)]
        
        print(f"检测到 {len(filtered_conflicts)} 个冲突")
        
        all_conflicts = []
        unique_conflicts = {}
        repair_summary = []
        
        # 记录初始冲突
        for conflict in filtered_conflicts:
            conflict_id = get_conflict_id(conflict)
            conflict_info = {
                'game': game_name,
                'step': 0,
                'action': 'initial_load',
                'type': conflict.type,
                'severity': conflict.severity,
                'description': conflict.description,
                'involved_nodes': convert_to_serializable(conflict.involved_nodes),
                'involved_edges': convert_to_serializable(conflict.involved_edges),
                'details': convert_to_serializable(conflict.details),
                'conflict_id': conflict_id,
                'repair_attempted': False,
                'repair_success': False
            }
            unique_conflicts[conflict_id] = conflict_info
            all_conflicts.append(conflict_info)
        
        # 如果有冲突，开始修复循环
        if filtered_conflicts:
            print(f"开始冲突修复循环...")
            
            with open(game_log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"检测到 {len(filtered_conflicts)} 个初始冲突\n")
                log_file.write("开始修复循环...\n")
                log_file.flush()
                
                # 执行修复循环（不使用TDG）
                repair_result = repair_conflict_loop_no_tdg(
                    filtered_conflicts,
                    graph,
                    conflict_localizer,
                    conflict_detector,
                    model,
                    0,  # step_num
                    game_name,
                    log_file,
                    max_attempts=10
                )
                
                repair_summary.append({
                    'step': 0,
                    'initial_conflicts': len(filtered_conflicts),
                    'repair_result': repair_result
                })
                
                if repair_result['success']:
                    print(f"所有冲突修复成功!")
                    log_file.write(f"\n修复结果: 成功解决所有冲突\n")
                    
                    # 更新冲突状态为已修复
                    for conflict_info in all_conflicts:
                        conflict_info['repair_attempted'] = True
                        conflict_info['repair_success'] = True
                else:
                    print(f"仍有 {repair_result['remaining_conflicts']} 个冲突未解决")
                    log_file.write(f"\n修复结果: 仍有 {repair_result['remaining_conflicts']} 个冲突未解决\n")
                    
                    # 重新检测剩余冲突
                    remaining_conflicts = conflict_detector.detect_all_conflicts(graph, 0, game_name)
                    remaining_conflicts = [c for c in remaining_conflicts if not should_ignore_conflict(c)]
                    
                    # 更新冲突状态
                    for conflict in remaining_conflicts:
                        conflict_id = get_conflict_id(conflict)
                        if conflict_id in unique_conflicts:
                            unique_conflicts[conflict_id]['repair_attempted'] = True
                            unique_conflicts[conflict_id]['repair_success'] = False
                
                log_file.flush()
        else:
            print(f"没有检测到冲突，无需修复")
        
        # 保存最终图数据
        graph_file = os.path.join(output_dir, f"{game_name}_test_edges_no_tdg_graph.pkl")
        with open(graph_file, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Saved graph data to: {graph_file}")
        
        # 保存图的JSON格式
        graph_json_file = os.path.join(output_dir, f"{game_name}_test_edges_no_tdg_graph.json")
        graph_data = {
            'nodes': convert_to_serializable(list(graph.nodes(data=True))),
            'edges': convert_to_serializable(list(graph.edges(data=True)))
        }
        with open(graph_json_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON graph data to: {graph_json_file}")
        
        # Statistics
        final_stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'connected_components': nx.number_weakly_connected_components(graph)
        }
        
        conflict_summary = conflict_detector.get_conflict_summary()
        
        result = {
            'game': game_name,
            'mode': 'test_edges_no_tdg',
            'edges_file': edges_file,
            'graph_stats': convert_to_serializable(final_stats),
            'conflict_summary': convert_to_serializable(conflict_summary),
            'all_conflicts': convert_to_serializable(all_conflicts),
            'total_conflicts': len(all_conflicts),
            'repair_summary': convert_to_serializable(repair_summary),
            'total_repairs_attempted': len(repair_summary),
            'successful_repairs': len([r for r in repair_summary if r['repair_result']['success']]),
            'graph_file': graph_file,
            'graph_json_file': graph_json_file,
            'log_file': game_log_file
        }
        
        print(f"  - Graph nodes: {final_stats['num_nodes']}")
        print(f"  - Graph edges: {final_stats['num_edges']}")
        print(f"  - 总冲突数: {len(all_conflicts)}")
        print(f"  - 修复尝试: {len(repair_summary)}")
        print(f"  - 成功修复: {result['successful_repairs']}")
        
        return result
        
    except Exception as e:
        print(f"Processing game {game_name} error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "game": game_name}


def process_game_test_edges_weight_only(game_name: str, data_dir: str, model: str, 
                                       output_dir: str, max_steps: int = None) -> Dict:
    """处理单个游戏，仅使用边权重进行冲突修复（不使用TDG）"""
    print(f"\n{'='*60}")
    print(f"Processing game(test_edges_weight_only模式): {game_name}")
    print(f"Data directory: {data_dir}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Check if game folder exists
    game_path = os.path.join(data_dir, game_name)
    if not os.path.exists(game_path):
        print(f"Error: Game folder does not exist: {game_path}")
        return {"error": f"Game folder does not exist: {game_path}"}
    
    # 检查edges.json文件是否存在
    edges_file = os.path.join(game_path, f"{game_name}.edges.json")
    if not os.path.exists(edges_file):
        print(f"错误：edges.json文件不存在: {edges_file}")
        return {"error": f"edges.json文件不存在: {edges_file}"}
    
    try:
        print(f"正在从edges.json加载图结构...")
        
        # 直接从edges.json加载图
        with open(edges_file, 'r', encoding='utf-8') as f:
            edges_data = json.load(f)
        
        # 创建NetworkX图
        graph = nx.DiGraph()
        
        # 定义反向方向映射
        opposite_directions = {
            'north': 'south', 'south': 'north',
            'east': 'west', 'west': 'east',
            'northeast': 'southwest', 'southwest': 'northeast',
            'northwest': 'southeast', 'southeast': 'northwest',
            'up': 'down', 'down': 'up',
            'exit': 'enter', 'enter': 'exit',
            'in': 'out', 'out': 'in',
        }
        
        # 添加边到图中
        for edge in edges_data:
            if 'src_node' in edge and 'dst_node' in edge and 'action' in edge:
                from_node = edge['src_node']
                to_node = edge['dst_node']
                direction = edge['action']
                seen_in_forward = edge.get('seen_in_forward', 0)
                
                # 添加边及其属性
                graph.add_edge(from_node, to_node, 
                             direction=direction, 
                             action=direction,
                             step_num=seen_in_forward,
                             seen_in_forward=seen_in_forward)
        
        # 添加自动反向边
        edges_to_add = []
        for edge in edges_data:
            if 'src_node' in edge and 'dst_node' in edge and 'action' in edge:
                from_node = edge['src_node']
                to_node = edge['dst_node']
                direction = edge['action'].lower()
                
                if direction in opposite_directions:
                    reverse_direction = opposite_directions[direction]
                    
                    # 检查反向边是否已存在
                    if not graph.has_edge(to_node, from_node):
                        edges_to_add.append({
                            'from': to_node,
                            'to': from_node,
                            'direction': reverse_direction,
                            'action': reverse_direction,
                            'is_auto_reverse': True,
                            'step_num': edge.get('seen_in_forward', 0)
                        })
        
        # 添加反向边
        for edge in edges_to_add:
            graph.add_edge(edge['from'], edge['to'],
                          direction=edge['direction'], 
                          action=edge['action'],
                          is_auto_reverse=edge['is_auto_reverse'],
                          step_num=edge['step_num'])
        
        print(f"已加载图: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
        
        # 创建系统组件（使用边权重但不使用TDG）
        conflict_detector = ConflictDetector()
        conflict_localizer = ConflictLocalizer()
        edge_scorer = EdgeImpactScorer()
        
        # Create game-specific log file
        game_log_file = os.path.join(output_dir, f"{game_name}_test_edges_weight_only_detailed.log")
        
        with open(game_log_file, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Game: {game_name} (test_edges_weight_only模式)\n")
            log_file.write(f"从文件加载: {edges_file}\n")
            log_file.write(f"初始图状态: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边\n")
            log_file.write("=" * 80 + "\n\n")
        
        # 检测初始冲突
        print(f"正在检测初始冲突...")
        conflicts = conflict_detector.detect_all_conflicts(graph, 0, game_name)
        
        # 过滤掉需要忽略的冲突
        filtered_conflicts = [c for c in conflicts if not should_ignore_conflict(c)]
        
        print(f"检测到 {len(filtered_conflicts)} 个冲突")
        
        all_conflicts = []
        unique_conflicts = {}
        repair_summary = []
        
        # 记录初始冲突
        for conflict in filtered_conflicts:
            conflict_id = get_conflict_id(conflict)
            conflict_info = {
                'game': game_name,
                'step': 0,
                'action': 'initial_load',
                'type': conflict.type,
                'severity': conflict.severity,
                'description': conflict.description,
                'involved_nodes': convert_to_serializable(conflict.involved_nodes),
                'involved_edges': convert_to_serializable(conflict.involved_edges),
                'details': convert_to_serializable(conflict.details),
                'conflict_id': conflict_id,
                'repair_attempted': False,
                'repair_success': False
            }
            unique_conflicts[conflict_id] = conflict_info
            all_conflicts.append(conflict_info)
        
        # 如果有冲突，开始修复循环
        if filtered_conflicts:
            print(f"开始冲突修复循环（使用边权重）...")
            
            with open(game_log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"检测到 {len(filtered_conflicts)} 个初始冲突\n")
                log_file.write("开始修复循环（使用边权重）...\n")
                log_file.flush()
                
                # 执行修复循环（使用边权重）
                repair_result = repair_conflict_loop_weight_only(
                    filtered_conflicts,
                    graph,
                    conflict_localizer,
                    conflict_detector,
                    edge_scorer,
                    model,
                    0,  # step_num
                    game_name,
                    log_file,
                    max_attempts=10
                )
                
                repair_summary.append({
                    'step': 0,
                    'initial_conflicts': len(filtered_conflicts),
                    'repair_result': repair_result
                })
                
                if repair_result['success']:
                    print(f"所有冲突修复成功!")
                    log_file.write(f"\n修复结果: 成功解决所有冲突\n")
                    
                    # 更新冲突状态为已修复
                    for conflict_info in all_conflicts:
                        conflict_info['repair_attempted'] = True
                        conflict_info['repair_success'] = True
                else:
                    print(f"仍有 {repair_result['remaining_conflicts']} 个冲突未解决")
                    log_file.write(f"\n修复结果: 仍有 {repair_result['remaining_conflicts']} 个冲突未解决\n")
                    
                    # 重新检测剩余冲突
                    remaining_conflicts = conflict_detector.detect_all_conflicts(graph, 0, game_name)
                    remaining_conflicts = [c for c in remaining_conflicts if not should_ignore_conflict(c)]
                    
                    # 更新冲突状态
                    for conflict in remaining_conflicts:
                        conflict_id = get_conflict_id(conflict)
                        if conflict_id in unique_conflicts:
                            unique_conflicts[conflict_id]['repair_attempted'] = True
                            unique_conflicts[conflict_id]['repair_success'] = False
                
                log_file.flush()
        else:
            print(f"没有检测到冲突，无需修复")
        
        # 保存最终图数据
        graph_file = os.path.join(output_dir, f"{game_name}_test_edges_weight_only_graph.pkl")
        with open(graph_file, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Saved graph data to: {graph_file}")
        
        # 保存图的JSON格式
        graph_json_file = os.path.join(output_dir, f"{game_name}_test_edges_weight_only_graph.json")
        graph_data = {
            'nodes': convert_to_serializable(list(graph.nodes(data=True))),
            'edges': convert_to_serializable(list(graph.edges(data=True)))
        }
        with open(graph_json_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON graph data to: {graph_json_file}")
        
        # Statistics
        final_stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'connected_components': nx.number_weakly_connected_components(graph)
        }
        
        conflict_summary = conflict_detector.get_conflict_summary()
        
        result = {
            'game': game_name,
            'mode': 'test_edges_weight_only',
            'edges_file': edges_file,
            'graph_stats': convert_to_serializable(final_stats),
            'conflict_summary': convert_to_serializable(conflict_summary),
            'all_conflicts': convert_to_serializable(all_conflicts),
            'total_conflicts': len(all_conflicts),
            'repair_summary': convert_to_serializable(repair_summary),
            'total_repairs_attempted': len(repair_summary),
            'successful_repairs': len([r for r in repair_summary if r['repair_result']['success']]),
            'graph_file': graph_file,
            'graph_json_file': graph_json_file,
            'log_file': game_log_file
        }
        
        print(f"  - Graph nodes: {final_stats['num_nodes']}")
        print(f"  - Graph edges: {final_stats['num_edges']}")
        print(f"  - 总冲突数: {len(all_conflicts)}")
        print(f"  - 修复尝试: {len(repair_summary)}")
        print(f"  - 成功修复: {result['successful_repairs']}")
        
        return result
        
    except Exception as e:
        print(f"Processing game {game_name} error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "game": game_name}


def process_game_test_edges_tdg_only(game_name: str, data_dir: str, model: str, 
                                    output_dir: str, max_steps: int = None) -> Dict:
    """处理单个游戏，仅使用TDG进行冲突修复（不使用边权重）"""
    print(f"\n{'='*60}")
    print(f"Processing game(test_edges_tdg_only模式): {game_name}")
    print(f"Data directory: {data_dir}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Check if game folder exists
    game_path = os.path.join(data_dir, game_name)
    if not os.path.exists(game_path):
        print(f"Error: Game folder does not exist: {game_path}")
        return {"error": f"Game folder does not exist: {game_path}"}
    
    # 检查edges.json文件是否存在
    edges_file = os.path.join(game_path, f"{game_name}.edges.json")
    if not os.path.exists(edges_file):
        print(f"错误：edges.json文件不存在: {edges_file}")
        return {"error": f"edges.json文件不存在: {edges_file}"}
    
    try:
        print(f"正在从edges.json加载图结构...")
        
        # 直接从edges.json加载图
        with open(edges_file, 'r', encoding='utf-8') as f:
            edges_data = json.load(f)
        
        # 创建NetworkX图
        graph = nx.DiGraph()
        
        # 定义反向方向映射
        opposite_directions = {
            'north': 'south', 'south': 'north',
            'east': 'west', 'west': 'east',
            'northeast': 'southwest', 'southwest': 'northeast',
            'northwest': 'southeast', 'southeast': 'northwest',
            'up': 'down', 'down': 'up',
            'exit': 'enter', 'enter': 'exit',
            'in': 'out', 'out': 'in',
        }
        
        # 添加边到图中
        for edge in edges_data:
            if 'src_node' in edge and 'dst_node' in edge and 'action' in edge:
                from_node = edge['src_node']
                to_node = edge['dst_node']
                direction = edge['action']
                seen_in_forward = edge.get('seen_in_forward', 0)
                
                # 添加边及其属性
                graph.add_edge(from_node, to_node, 
                             direction=direction, 
                             action=direction,
                             step_num=seen_in_forward,
                             seen_in_forward=seen_in_forward)
        
        # 添加自动反向边
        edges_to_add = []
        for edge in edges_data:
            if 'src_node' in edge and 'dst_node' in edge and 'action' in edge:
                from_node = edge['src_node']
                to_node = edge['dst_node']
                direction = edge['action'].lower()
                
                if direction in opposite_directions:
                    reverse_direction = opposite_directions[direction]
                    
                    # 检查反向边是否已存在
                    if not graph.has_edge(to_node, from_node):
                        edges_to_add.append({
                            'from': to_node,
                            'to': from_node,
                            'direction': reverse_direction,
                            'action': reverse_direction,
                            'is_auto_reverse': True,
                            'step_num': edge.get('seen_in_forward', 0)
                        })
        
        # 添加反向边
        for edge in edges_to_add:
            graph.add_edge(edge['from'], edge['to'],
                          direction=edge['direction'], 
                          action=edge['action'],
                          is_auto_reverse=edge['is_auto_reverse'],
                          step_num=edge['step_num'])
        
        print(f"已加载图: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
        
        # 创建系统组件（使用TDG但不使用边权重）
        conflict_detector = ConflictDetector()
        tdg = TemporalDependencyGraph()
        conflict_localizer = ConflictLocalizer()
        
        # 创建初始版本
        initial_version_id = tdg.create_version(
            graph, 
            0, 
            {'loaded_from_file': True}, 
            f"Loaded from {edges_file}", 
            []
        )
        
        # Create game-specific log file
        game_log_file = os.path.join(output_dir, f"{game_name}_test_edges_tdg_only_detailed.log")
        
        with open(game_log_file, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Game: {game_name} (test_edges_tdg_only模式)\n")
            log_file.write(f"从文件加载: {edges_file}\n")
            log_file.write(f"初始图状态: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边\n")
            log_file.write("=" * 80 + "\n\n")
        
        # 检测初始冲突
        print(f"正在检测初始冲突...")
        conflicts = conflict_detector.detect_all_conflicts(graph, 0, game_name)
        
        # 过滤掉需要忽略的冲突
        filtered_conflicts = [c for c in conflicts if not should_ignore_conflict(c)]
        
        print(f"检测到 {len(filtered_conflicts)} 个冲突")
        
        all_conflicts = []
        unique_conflicts = {}
        repair_summary = []
        
        # 记录初始冲突
        for conflict in filtered_conflicts:
            conflict_id = get_conflict_id(conflict)
            conflict_info = {
                'game': game_name,
                'step': 0,
                'action': 'initial_load',
                'type': conflict.type,
                'severity': conflict.severity,
                'description': conflict.description,
                'involved_nodes': convert_to_serializable(conflict.involved_nodes),
                'involved_edges': convert_to_serializable(conflict.involved_edges),
                'details': convert_to_serializable(conflict.details),
                'conflict_id': conflict_id,
                'repair_attempted': False,
                'repair_success': False
            }
            unique_conflicts[conflict_id] = conflict_info
            all_conflicts.append(conflict_info)
        
        # 如果有冲突，开始修复循环
        if filtered_conflicts:
            print(f"开始冲突修复循环（仅使用TDG）...")
            
            with open(game_log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"检测到 {len(filtered_conflicts)} 个初始冲突\n")
                log_file.write("开始修复循环（仅使用TDG）...\n")
                log_file.flush()
                
                # 执行修复循环（仅使用TDG）
                repair_result = repair_conflict_loop_tdg_only(
                    filtered_conflicts,
                    graph,
                    tdg,
                    conflict_localizer,
                    conflict_detector,
                    model,
                    0,  # step_num
                    game_name,
                    log_file,
                    max_attempts=10
                )
                
                repair_summary.append({
                    'step': 0,
                    'initial_conflicts': len(filtered_conflicts),
                    'repair_result': repair_result
                })
                
                if repair_result['success']:
                    print(f"所有冲突修复成功!")
                    log_file.write(f"\n修复结果: 成功解决所有冲突\n")
                    
                    # 更新冲突状态为已修复
                    for conflict_info in all_conflicts:
                        conflict_info['repair_attempted'] = True
                        conflict_info['repair_success'] = True
                else:
                    print(f"仍有 {repair_result['remaining_conflicts']} 个冲突未解决")
                    log_file.write(f"\n修复结果: 仍有 {repair_result['remaining_conflicts']} 个冲突未解决\n")
                    
                    # 重新检测剩余冲突
                    remaining_conflicts = conflict_detector.detect_all_conflicts(graph, 0, game_name)
                    remaining_conflicts = [c for c in remaining_conflicts if not should_ignore_conflict(c)]
                    
                    # 更新冲突状态
                    for conflict in remaining_conflicts:
                        conflict_id = get_conflict_id(conflict)
                        if conflict_id in unique_conflicts:
                            unique_conflicts[conflict_id]['repair_attempted'] = True
                            unique_conflicts[conflict_id]['repair_success'] = False
                
                log_file.flush()
        else:
            print(f"没有检测到冲突，无需修复")
        
        # 保存最终图数据
        graph_file = os.path.join(output_dir, f"{game_name}_test_edges_tdg_only_graph.pkl")
        with open(graph_file, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Saved graph data to: {graph_file}")
        
        # 保存图的JSON格式
        graph_json_file = os.path.join(output_dir, f"{game_name}_test_edges_tdg_only_graph.json")
        graph_data = {
            'nodes': convert_to_serializable(list(graph.nodes(data=True))),
            'edges': convert_to_serializable(list(graph.edges(data=True)))
        }
        with open(graph_json_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON graph data to: {graph_json_file}")
        
        # 保存TDG时间线
        timeline_file = os.path.join(output_dir, f"{game_name}_test_edges_tdg_only_timeline.json")
        with open(timeline_file, 'w', encoding='utf-8') as f:
            json.dump(tdg.export_timeline(), f, ensure_ascii=False, indent=2)
        print(f"已保存TDG时间线到: {timeline_file}")
        
        # Statistics
        final_stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'connected_components': nx.number_weakly_connected_components(graph)
        }
        
        conflict_summary = conflict_detector.get_conflict_summary()
        
        result = {
            'game': game_name,
            'mode': 'test_edges_tdg_only',
            'edges_file': edges_file,
            'graph_stats': convert_to_serializable(final_stats),
            'conflict_summary': convert_to_serializable(conflict_summary),
            'all_conflicts': convert_to_serializable(all_conflicts),
            'total_conflicts': len(all_conflicts),
            'repair_summary': convert_to_serializable(repair_summary),
            'total_repairs_attempted': len(repair_summary),
            'successful_repairs': len([r for r in repair_summary if r['repair_result']['success']]),
            'graph_file': graph_file,
            'graph_json_file': graph_json_file,
            'timeline_file': timeline_file,
            'log_file': game_log_file
        }
        
        print(f"  - Graph nodes: {final_stats['num_nodes']}")
        print(f"  - Graph edges: {final_stats['num_edges']}")
        print(f"  - 总冲突数: {len(all_conflicts)}")
        print(f"  - 修复尝试: {len(repair_summary)}")
        print(f"  - 成功修复: {result['successful_repairs']}")
        
        return result
        
    except Exception as e:
        print(f"Processing game {game_name} error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "game": game_name}


def process_game_with_tdg(game_name: str, data_dir: str, model: str, 
                         output_dir: str, max_steps: int = None) -> Dict:
    """处理单个游戏，带有TDG版本控制和冲突修复功能"""
    print(f"\n{'='*60}")
    print(f"Processing game(TDG模式): {game_name}")
    print(f"Data directory: {data_dir}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}")
    
    # Check if game folder exists
    game_path = os.path.join(data_dir, game_name)
    if not os.path.exists(game_path):
        print(f"Error: Game folder does not exist: {game_path}")
        return {"error": f"Game folder does not exist: {game_path}"}
    
    try:
        print(f"Creating MapSLAMSystem instance...")
        # Create system instance
        slam = MapSLAMSystem(data_dir=data_dir, model=model)
        print(f"MapSLAMSystem instance created successfully")
        
        # 创建额外的组件
        tdg = TemporalDependencyGraph()
        conflict_localizer = ConflictLocalizer()
        
        # Load walkthrough steps
        steps = slam.dataset.load_walkthrough(game_name)
        total_steps = len(steps)
        
        if max_steps:
            process_steps = min(max_steps, total_steps)
        else:
            process_steps = total_steps
            
        print(f"Total steps: {total_steps}, Will process: {process_steps} steps")
        
        # Create game-specific log file
        game_log_file = os.path.join(output_dir, f"{game_name}_tdg_detailed.log")
        
        with open(game_log_file, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Game: {game_name} (TDG模式)\n")
            log_file.write(f"Total steps: {total_steps}, Processed steps: {process_steps}\n")
            log_file.write("=" * 80 + "\n\n")
        
        # 处理每一步
        slam.current_game = game_name
        slam.nav_graph = NavigationGraph(slam.model, game_name, slam.dataset)
        all_conflicts = []
        unique_conflicts = {}
        conflict_counts = defaultdict(int)
        detailed_steps = []
        repair_summary = []
        
        for i, current_step in enumerate(steps[:process_steps]):
            # 简单进度显示
            if i % 10 == 0 or i == process_steps - 1:
                print(f"\r处理进度: {i+1}/{process_steps}", end="", flush=True)

            # 处理单步
            nav_result = slam.nav_graph.process_step(current_step)
            
            # 记录版本到TDG
            changes = {
                'added_nodes': nav_result.get('added_nodes', []),
                'added_edges': nav_result.get('added_edges', [])
            }
            version_id = tdg.create_version(
                slam.nav_graph.graph,
                current_step.step_num,
                changes,
                current_step.observation,
                []
            )
            
            # 检测冲突
            conflicts = slam.conflict_detector.detect_all_conflicts(
                slam.nav_graph.graph, current_step.step_num, game_name
            )
            
            # 过滤掉需要忽略的冲突
            filtered_conflicts = [c for c in conflicts if not should_ignore_conflict(c)]
            
            # 如果有冲突，进入修复循环
            if filtered_conflicts:
                print(f"\n  steps骤 {current_step.step_num} 检测到 {len(filtered_conflicts)} 个冲突")
                
                # 写入日志
                with open(game_log_file, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"\n{'='*80}\n")
                    log_file.write(f"步骤 {current_step.step_num} - 冲突修复\n")
                    log_file.write(f"检测到 {len(filtered_conflicts)} 个冲突\n")
                    log_file.flush()  # 立即刷新到磁盘
                    
                    # 一次性处理所有冲突
                    repair_result = repair_conflict_loop(
                        filtered_conflicts,
                        slam.nav_graph.graph,
                        tdg,
                        conflict_localizer,
                        slam.conflict_detector,
                        model,
                        current_step.step_num,
                        game_name,
                        log_file,
                        max_attempts=10  # 给更多尝试机会
                    )
                    
                    repair_summary.append({
                        'step': current_step.step_num,
                        'initial_conflicts': len(filtered_conflicts),
                        'repair_result': repair_result
                    })
                    
                    if repair_result['success']:
                        print(f"    所有冲突修复成功!")
                        log_file.write(f"\n修复结果: 成功解决所有冲突\n")
                        log_file.flush()  # 立即刷新到磁盘
                        # 创建新版本记录修复
                        repair_version_id = tdg.create_version(
                            slam.nav_graph.graph,
                            current_step.step_num,
                            {'repair': True, 'conflicts_fixed': len(filtered_conflicts)},
                            f"Fixed all {len(filtered_conflicts)} conflicts",
                            []
                        )
                    else:
                        print(f"    仍有 {repair_result['remaining_conflicts']} 个冲突未解决")
                        log_file.write(f"\n修复结果: 仍有 {repair_result['remaining_conflicts']} 个冲突未解决\n")
                        log_file.flush()  # 立即刷新到磁盘
                        
                        # 记录未解决的冲突
                        remaining_conflicts = slam.conflict_detector.detect_all_conflicts(
                            slam.nav_graph.graph, current_step.step_num, game_name
                        )
                        remaining_conflicts = [c for c in remaining_conflicts if not should_ignore_conflict(c)]
                        
                        for conflict in remaining_conflicts:
                            conflict_id = get_conflict_id(conflict)
                            if conflict_id not in unique_conflicts:
                                conflict_info = {
                                    'game': game_name,
                                    'step': convert_to_serializable(current_step.step_num),
                                    'action': convert_to_serializable(current_step.action),
                                    'type': conflict.type,
                                    'severity': conflict.severity,
                                    'description': conflict.description,
                                    'involved_nodes': convert_to_serializable(conflict.involved_nodes),
                                    'involved_edges': convert_to_serializable(conflict.involved_edges),
                                    'details': convert_to_serializable(conflict.details),
                                    'conflict_id': conflict_id,
                                    'repair_attempted': True,
                                    'repair_success': False
                                }
                                unique_conflicts[conflict_id] = conflict_info
                                all_conflicts.append(conflict_info)
            
            # 记录详细步骤信息到日志文件
            if nav_result.get('llm_interaction') or filtered_conflicts:
                with open(game_log_file, 'a', encoding='utf-8') as log_file:
                    # 只在有LLM交互时写入建图LLM的日志
                    if nav_result.get('llm_interaction'):
                        log_file.write(f"\n{'='*80}\n")
                        log_file.write(f"步骤 {current_step.step_num} - 建图LLM\n")
                        log_file.write(f"动作: {current_step.action}\n")
                        log_file.write(f"观察: {current_step.observation}\n")
                        log_file.write(f"\n[建图LLM输出]\n")
                        try:
                            llm_output = json.loads(nav_result['llm_interaction']['output'])
                            log_file.write(json.dumps(llm_output, ensure_ascii=False, indent=2))
                        except:
                            log_file.write(nav_result['llm_interaction']['output'])
                        log_file.write("\n")
                        log_file.flush()  # 立即刷新到磁盘
                        
                        if nav_result.get('added_nodes'):
                            log_file.write(f"添加节点: {nav_result['added_nodes']}\n")
                            log_file.flush()  # 立即刷新到磁盘
                        if nav_result.get('added_edges'):
                            log_file.write(f"添加边: {nav_result['added_edges']}\n")
                            log_file.flush()  # 立即刷新到磁盘
                
                step_info = {
                    'step_num': convert_to_serializable(current_step.step_num),
                    'action': convert_to_serializable(current_step.action),
                    'observation': convert_to_serializable(current_step.observation),
                    'llm_interaction': convert_to_serializable(nav_result.get('llm_interaction')),
                    'added_nodes': convert_to_serializable(nav_result.get('added_nodes', [])),
                    'added_edges': convert_to_serializable(nav_result.get('added_edges', [])),
                    'conflicts': len(filtered_conflicts),
                    'version_id': version_id
                }
                detailed_steps.append(step_info)
            
            slam.processed_steps += 1
        
        print(f"\nProcessing complete!")
        
        # 保存最终图数据
        graph_file = os.path.join(output_dir, f"{game_name}_tdg_graph.pkl")
        with open(graph_file, 'wb') as f:
            pickle.dump(slam.nav_graph.graph, f)
        print(f"Saved graph data to: {graph_file}")
        
        # 保存TDG时间线
        timeline_file = os.path.join(output_dir, f"{game_name}_tdg_timeline.json")
        with open(timeline_file, 'w', encoding='utf-8') as f:
            json.dump(tdg.export_timeline(), f, ensure_ascii=False, indent=2)
        print(f"已保存TDG时间线到: {timeline_file}")
        
        # Statistics
        final_stats = slam.nav_graph.get_graph_stats()
        conflict_summary = slam.conflict_detector.get_conflict_summary()
        
        result = {
            'game': game_name,
            'mode': 'test_modify_tdg',
            'total_steps': total_steps,
            'processed_steps': process_steps,
            'graph_stats': convert_to_serializable(final_stats),
            'conflict_summary': convert_to_serializable(conflict_summary),
            'all_conflicts': convert_to_serializable(all_conflicts),
            'total_conflicts': len(all_conflicts),
            'repair_summary': convert_to_serializable(repair_summary),
            'total_repairs_attempted': len(repair_summary),
            'successful_repairs': len([r for r in repair_summary if r['repair_result']['success']]),
            'tdg_stats': convert_to_serializable(tdg.analyze_evolution()),
            'graph_file': graph_file,
            'timeline_file': timeline_file,
            'log_file': game_log_file
        }
        
        print(f"  - Graph nodes: {final_stats['num_nodes']}")
        print(f"  - Graph edges: {final_stats['num_edges']}")
        print(f"  - 总冲突数: {len(all_conflicts)}")
        print(f"  - 修复尝试: {len(repair_summary)}")
        print(f"  - 成功修复: {result['successful_repairs']}")
        
        return result
        
    except Exception as e:
        print(f"Processing game {game_name} error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "game": game_name}


def main():
    parser = argparse.ArgumentParser(description='批量运行MAP-SLAM系统处理多个游戏')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory路径')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='使用的LLMModel')
    parser.add_argument('--max-steps', type=int, default=70, help='每个游戏的最大Processed steps')
    parser.add_argument('--output-dir', type=str, default='./batch_output_fixed', help='Output directory')
    parser.add_argument('--games', type=str, help='指定要处理的游戏列表，用逗号分隔（如果不指定则自动发现所有游戏）')
    parser.add_argument('--workers', type=int, default=None, help='并行处理的进程数（默认为CPU核心数）')
    parser.add_argument('--sequential', action='store_true', help='顺序处理而不是并行处理')
    parser.add_argument('--mode', type=str, choices=['normal', 'test_modify_tdg', 'test_edges', 'test_edges_no_tdg', 'test_edges_weight_only', 'test_edges_tdg_only'], default='normal',
                        help='运行模式：normal（普通模式）、test_modify_tdg（带冲突修复的TDG模式）、test_edges（直接从edges.json加载并修复）、test_edges_no_tdg（从edges.json加载，不使用TDG和边权重修复）、test_edges_weight_only（仅使用边权重）、test_edges_tdg_only（仅使用TDG）')
    parser.add_argument('--unprocessed-only', action='store_true', help='只处理未在output-dir中的游戏')
    
    args = parser.parse_args()
    
    # Set default data directory if not provided
    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data_dir = os.path.join(script_dir, 'data_fixed')
    
    # 游戏列表
    if args.games:
        games = [g.strip() for g in args.games.split(',')]
    else:
        print(f"正在扫描Data directory: {args.data_dir}")
        games = discover_games(args.data_dir)
        if not games:
            print("未发现任何游戏文件夹")
            return 1
        print(f"发现 {len(games)} 个Game: {', '.join(games)}")
    
    # 如果设置了--unprocessed-only，过滤掉已处理的游戏
    if args.unprocessed_only:
        processed_games = get_processed_games(args.output_dir)
        original_count = len(games)
        games = [g for g in games if g not in processed_games]
        print(f"已处理的游戏: {len(processed_games)} 个")
        print(f"过滤后剩余: {len(games)} 个游戏")
        if not games:
            print("所有游戏都已处理完成！")
            return 0
    
    print(f"Will process {len(games)} 个游戏")
    print(f"每个游戏最多处理 {args.max_steps} steps")
    
    # 创建Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 批量处理结果
    batch_results = []
    total_conflicts_by_game = {}
    total_conflicts_by_type = defaultdict(int)
    
    start_time = time.time()
    
    if args.sequential:
        # 顺序处理
        print(f"使用顺序处理模式 (模式: {args.mode})")
        for i, game in enumerate(games):
            print(f"\n[{i+1}/{len(games)}] 处理Game: {game}")
            if args.mode == 'test_modify_tdg':
                result = process_game_with_tdg(game, args.data_dir, args.model, args.output_dir, args.max_steps)
            elif args.mode == 'test_edges':
                result = process_game_test_edges(game, args.data_dir, args.model, args.output_dir, args.max_steps)
            elif args.mode == 'test_edges_no_tdg':
                result = process_game_test_edges_no_tdg(game, args.data_dir, args.model, args.output_dir, args.max_steps)
            elif args.mode == 'test_edges_weight_only':
                result = process_game_test_edges_weight_only(game, args.data_dir, args.model, args.output_dir, args.max_steps)
            elif args.mode == 'test_edges_tdg_only':
                result = process_game_test_edges_tdg_only(game, args.data_dir, args.model, args.output_dir, args.max_steps)
            else:
                result = process_game(game, args.data_dir, args.model, args.output_dir, args.max_steps)
            batch_results.append(result)
            
            if 'error' not in result:
                total_conflicts_by_game[game] = result['total_conflicts']
                if 'conflict_summary' in result and 'by_type' in result['conflict_summary']:
                    for conflict_type, count in result['conflict_summary']['by_type'].items():
                        total_conflicts_by_type[conflict_type] += count
    else:
        # 并行处理
        if args.mode in ['test_modify_tdg', 'test_edges', 'test_edges_no_tdg', 'test_edges_weight_only', 'test_edges_tdg_only']:
            print(f"警告：{args.mode}模式不支持并行处理，切换到顺序处理")
            # 这些模式只能顺序处理
            for i, game in enumerate(games):
                print(f"\n[{i+1}/{len(games)}] 处理Game: {game}")
                if args.mode == 'test_modify_tdg':
                    result = process_game_with_tdg(game, args.data_dir, args.model, args.output_dir, args.max_steps)
                elif args.mode == 'test_edges':
                    result = process_game_test_edges(game, args.data_dir, args.model, args.output_dir, args.max_steps)
                elif args.mode == 'test_edges_no_tdg':
                    result = process_game_test_edges_no_tdg(game, args.data_dir, args.model, args.output_dir, args.max_steps)
                elif args.mode == 'test_edges_weight_only':
                    result = process_game_test_edges_weight_only(game, args.data_dir, args.model, args.output_dir, args.max_steps)
                elif args.mode == 'test_edges_tdg_only':
                    result = process_game_test_edges_tdg_only(game, args.data_dir, args.model, args.output_dir, args.max_steps)
                batch_results.append(result)
                
                if 'error' not in result:
                    total_conflicts_by_game[game] = result['total_conflicts']
                    if 'conflict_summary' in result and 'by_type' in result['conflict_summary']:
                        for conflict_type, count in result['conflict_summary']['by_type'].items():
                            total_conflicts_by_type[conflict_type] += count
        else:
            max_workers = args.workers or min(cpu_count(), len(games))
            print(f"使用并行处理模式，进程数: {max_workers}")
            
            # 准备参数
            game_args = [(game, args.data_dir, args.model, args.output_dir, args.max_steps) for game in games]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_game = {executor.submit(process_game_wrapper, game_arg): game_arg[0] for game_arg in game_args}
                
                # 处理完成的任务
                completed = 0
                for future in as_completed(future_to_game):
                    game = future_to_game[future]
                    completed += 1
                    try:
                        result = future.result()
                        print(f"\n[{completed}/{len(games)}] 完成Game: {game}")
                        batch_results.append(result)
                        
                        if 'error' not in result:
                            total_conflicts_by_game[game] = result['total_conflicts']
                            for conflict_type, count in result['conflict_summary']['by_type'].items():
                                total_conflicts_by_type[conflict_type] += count
                            print(f"  - 冲突数: {result['total_conflicts']}")
                        else:
                            print(f"  - 处理失败: {result['error']}")
                    except Exception as exc:
                        print(f"\n游戏 {game} 生成异常: {exc}")
                        batch_results.append({"error": str(exc), "game": game})
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 生成汇总报告
    summary_report = {
        'processing_info': {
            'mode': args.mode,
            'total_games': len(games),
            'successful_games': len([r for r in batch_results if 'error' not in r]),
            'failed_games': len([r for r in batch_results if 'error' in r]),
            'processing_time_seconds': processing_time,
            'max_steps_per_game': args.max_steps
        },
        'overall_conflict_stats': {
            'total_conflicts_by_type': dict(total_conflicts_by_type),
            'total_conflicts_by_game': total_conflicts_by_game,
            'total_conflicts_overall': sum(total_conflicts_by_game.values())
        },
        'detailed_results': batch_results
    }
    
    # 添加TDG和test_edges模式的额外统计
    if args.mode in ['test_modify_tdg', 'test_edges']:
        summary_report['repair_stats'] = {
            'total_repairs_attempted': sum(r.get('total_repairs_attempted', 0) for r in batch_results if 'error' not in r),
            'successful_repairs': sum(r.get('successful_repairs', 0) for r in batch_results if 'error' not in r),
            'games_with_repairs': len([r for r in batch_results if 'error' not in r and r.get('total_repairs_attempted', 0) > 0])
        }
    
    # 保存详细结果
    mode_suffix = '_tdg' if args.mode == 'test_modify_tdg' else '_test_edges' if args.mode == 'test_edges' else '_test_edges_no_tdg' if args.mode == 'test_edges_no_tdg' else '_weight_only' if args.mode == 'test_edges_weight_only' else '_tdg_only' if args.mode == 'test_edges_tdg_only' else ''
    output_file = os.path.join(args.output_dir, f'batch_conflict_analysis{mode_suffix}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(summary_report), f, ensure_ascii=False, indent=2)
    
    # 生成简化的冲突汇总文件
    conflicts_only = []
    for result in batch_results:
        if 'error' not in result and result.get('all_conflicts'):
            conflicts_only.extend(result['all_conflicts'])
    
    conflicts_file = os.path.join(args.output_dir, f'all_conflicts_summary{mode_suffix}.json')
    with open(conflicts_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(conflicts_only), f, ensure_ascii=False, indent=2)
    
    # 打印最终统计
    print(f"\n\n{'='*80}")
    print(f"批量Processing complete!(模式: {args.mode})")
    print(f"{'='*80}")
    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"成功处理: {summary_report['processing_info']['successful_games']} 个游戏")
    print(f"失败Game: {summary_report['processing_info']['failed_games']} 个")
    print(f"总冲突数: {summary_report['overall_conflict_stats']['total_conflicts_overall']}")
    
    if args.mode in ['test_modify_tdg', 'test_edges']:
        # TDG和test_edges模式的额外统计
        total_repairs = sum(r.get('total_repairs_attempted', 0) for r in batch_results if 'error' not in r)
        successful_repairs = sum(r.get('successful_repairs', 0) for r in batch_results if 'error' not in r)
        print(f"\n修复统计:")
        print(f"  - 总修复尝试: {total_repairs}")
        print(f"  - 成功修复: {successful_repairs}")
        if total_repairs > 0:
            print(f"  - 修复成功率: {successful_repairs/total_repairs*100:.1f}%")
    
    print(f"\n冲突类型统计:")
    for conflict_type, count in total_conflicts_by_type.items():
        print(f"  - {conflict_type}: {count}")
    
    print(f"\n各游戏冲突统计 (Top 10):")
    sorted_games = sorted(total_conflicts_by_game.items(), key=lambda x: x[1], reverse=True)
    for game, count in sorted_games[:10]:
        print(f"  - {game}: {count}")
    
    print(f"\n结果已保存到:")
    print(f"  - 详细报告: {output_file}")
    print(f"  - 冲突汇总: {conflicts_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())