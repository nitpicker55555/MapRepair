"""Clean reimplementation of the LLM-MapRepair framework.

Sub-modules:
  graph      - NavGraph data structure
  conflict   - Conflict dataclass and three detectors (direction, topology, naming)
  localizer  - LCA-based candidate filtering
  scoring    - Edge impact scoring
  history    - Versioned graph history (rollback / diff)
  synth      - Synthetic graph generator + error injector
  llm_client - Azure OpenAI wrapper
  agents     - RepairAgent implementations
"""

__version__ = "0.1.0"
