"""
State Space Definition for CBR+STM Framework

This module implements the state space and transition functions as defined in the paper.
"""

from typing import List, Dict, Optional
import numpy as np

class StateSpace:
    """
    State space definition with states and actions for predictive maintenance.
    
    Implements Definition 1 and 2 from the paper:
    - Definition 1: Problem State Space S = {s₀, s₁, ..., sₙ}
    - Definition 2: State Transition Function δ: S × A → S
    """
    
    def __init__(self, states: List[str], actions: List[str]):
        """
        Initialize state space.
        
        Args:
            states: List of state names representing equipment health conditions
            actions: List of available maintenance actions
        """
        self.states = states
        self.actions = actions
        self.transition_matrix = self._build_transition_matrix()
    
    def _build_transition_matrix(self) -> Dict[tuple, str]:
        """
        Build state transition matrix defining δ(s,a) for all valid (s,a) pairs.
        
        Returns:
            Dictionary mapping (state, action) tuples to resulting states
        """
        transitions = {}
        
        for state in self.states:
            for action in self.actions:
                next_state = self._calculate_transition(state, action)
                transitions[(state, action)] = next_state
        
        return transitions
    
    def _calculate_transition(self, state: str, action: str) -> str:
        """
        Calculate state transition based on maintenance domain logic.
        
        Args:
            state: Current health state
            action: Applied maintenance action
            
        Returns:
            Resulting health state after action application
        """
        state_idx = self.states.index(state)
        
        # Define transition logic based on action types
        if action == "Preventive_Maintenance":
            # Preventive maintenance can improve state by 1-2 levels
            new_idx = max(0, state_idx - np.random.randint(1, 3))
        
        elif action == "Corrective_Maintenance":
            # Corrective maintenance addresses current issues
            new_idx = max(0, state_idx - np.random.randint(0, 2))
        
        elif action == "Component_Replacement":
            # Component replacement significantly improves health
            new_idx = max(0, state_idx - np.random.randint(2, 4))
        
        elif action == "Condition_Monitoring":
            # Monitoring doesn't change state but provides information
            new_idx = state_idx
        
        elif action == "Emergency_Repair":
            # Emergency repair from failure state
            if state == self.states[-1]:  # Assuming last state is failure
                new_idx = len(self.states) - 2  # Move to second-to-last state
            else:
                new_idx = state_idx
        
        elif action == "Scheduled_Overhaul":
            # Overhaul restores to near-new condition
            new_idx = 0
        
        else:
            # Default: no change
            new_idx = state_idx
        
        # Ensure valid state index
        new_idx = max(0, min(new_idx, len(self.states) - 1))
        return self.states[new_idx]
    
    def transition(self, state: str, action: str) -> Optional[str]:
        """
        Apply state transition function δ(s,a).
        
        Args:
            state: Current state
            action: Action to apply
            
        Returns:
            Next state or None if transition is invalid
        """
        if state not in self.states or action not in self.actions:
            return None
        
        return self.transition_matrix.get((state, action))
    
    def get_valid_actions(self, state: str) -> List[str]:
        """
        Get valid actions for a given state.
        
        Args:
            state: Current state
            
        Returns:
            List of valid actions
        """
        if state not in self.states:
            return []
        
        valid_actions = []
        for action in self.actions:
            # All actions are generally valid, but some may be more appropriate
            if self._is_action_applicable(state, action):
                valid_actions.append(action)
        
        return valid_actions
    
    def _is_action_applicable(self, state: str, action: str) -> bool:
        """
        Check if action is applicable in current state.
        
        Args:
            state: Current state
            action: Action to check
            
        Returns:
            True if action is applicable
        """
        state_idx = self.states.index(state)
        
        # Define applicability rules
        if action == "Emergency_Repair":
            # Only applicable in failure state
            return state == self.states[-1]
        
        elif action == "Scheduled_Overhaul":
            # More appropriate for degraded states
            return state_idx > len(self.states) // 2
        
        elif action == "Preventive_Maintenance":
            # Most effective in early degradation stages
            return state_idx < len(self.states) - 1
        
        elif action == "Condition_Monitoring":
            # Always applicable
            return True
        
        else:
            # Default: action is applicable
            return True
    
    def get_state_features(self, state: str) -> Dict:
        """
        Get descriptive features for a state.
        
        Args:
            state: State name
            
        Returns:
            Dictionary of state features
        """
        if state not in self.states:
            return {}
        
        state_idx = self.states.index(state)
        total_states = len(self.states)
        
        return {
            'state_name': state,
            'state_index': state_idx,
            'health_score': (total_states - state_idx) / total_states,  # Higher is better
            'degradation_level': state_idx / total_states,  # 0 = new, 1 = failed
            'is_failure_state': state == self.states[-1],
            'is_healthy_state': state_idx < total_states // 3,
            'requires_attention': state_idx > total_states * 2 // 3
        }
    
    def visualize_transitions(self) -> Dict:
        """
        Create data structure for visualizing state transitions.
        
        Returns:
            Dictionary with nodes and edges for graph visualization
        """
        nodes = []
        edges = []
        
        # Create nodes for each state
        for i, state in enumerate(self.states):
            features = self.get_state_features(state)
            nodes.append({
                'id': state,
                'label': state,
                'health_score': features['health_score'],
                'position': i
            })
        
        # Create edges for each valid transition
        for (state, action), next_state in self.transition_matrix.items():
            if state != next_state:  # Only show actual transitions
                edges.append({
                    'source': state,
                    'target': next_state,
                    'action': action,
                    'weight': 1
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'total_states': len(self.states),
            'total_actions': len(self.actions)
        }
    
    def get_transition_probabilities(self, state: str) -> Dict[str, float]:
        """
        Get transition probabilities for stochastic transitions.
        
        Args:
            state: Current state
            
        Returns:
            Dictionary mapping next states to probabilities
        """
        probabilities = {}
        
        for action in self.actions:
            next_state = self.transition(state, action)
            if next_state:
                if next_state not in probabilities:
                    probabilities[next_state] = 0
                probabilities[next_state] += 1.0 / len(self.actions)
        
        return probabilities
