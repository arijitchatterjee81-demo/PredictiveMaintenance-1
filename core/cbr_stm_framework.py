"""
CBR+STM Framework Implementation

This module implements Algorithm 1 from the research paper:
A Theoretical Framework for Case-Based Reasoning with State Transition Mechanisms
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import heapq
from core.state_space import StateSpace
from core.heuristic import MultiObjectiveHeuristic
from core.case_base import CaseBase
from utils.similarity import SimilarityMeasure

class CBRSTMFramework:
    """
    Main framework implementing CBR+STM algorithm for predictive maintenance.
    
    This class implements Algorithm 1 from the paper, combining Case-Based Reasoning
    with State Transition Mechanisms for systematic maintenance decision making.
    """
    
    def __init__(self, state_space: StateSpace, heuristic: MultiObjectiveHeuristic, dataset: Dict):
        """
        Initialize the CBR+STM framework.
        
        Args:
            state_space: Defined state space with states and actions
            heuristic: Multi-objective heuristic function
            dataset: Training dataset for case base construction
        """
        self.state_space = state_space
        self.heuristic = heuristic
        self.case_base = CaseBase()
        self.similarity = SimilarityMeasure()
        
        # Build case base from dataset
        self._build_case_base(dataset)
    
    def _build_case_base(self, dataset: Dict):
        """
        Build case base from historical maintenance data.
        
        Args:
            dataset: Dictionary containing train/test data
        """
        train_data = dataset['train']
        
        # Group by unit to create cases
        for unit_id in train_data['unit'].unique():
            unit_data = train_data[train_data['unit'] == unit_id]
            
            # Create state transition sequence for this case
            case = self._create_case_from_unit_data(unit_data, unit_id)
            self.case_base.add_case(case)
    
    def _create_case_from_unit_data(self, unit_data: pd.DataFrame, unit_id: int) -> Dict:
        """
        Create a case from unit operational data.
        
        Args:
            unit_data: Operational data for a single unit
            unit_id: Unit identifier
            
        Returns:
            Case dictionary with state transition sequence
        """
        # Sort by cycle
        unit_data = unit_data.sort_values('cycle')
        
        # Create state sequence based on RUL progression
        state_sequence = []
        action_sequence = []
        
        for idx, row in unit_data.iterrows():
            # Map sensor readings to health state
            health_state = self._map_sensors_to_state(row)
            state_sequence.append(health_state)
            
            # Infer maintenance action (simplified for this implementation)
            if idx < len(unit_data) - 1:
                next_row = unit_data.iloc[idx + 1]
                action = self._infer_maintenance_action(row, next_row)
                action_sequence.append(action)
        
        # Calculate case outcomes
        initial_rul = len(unit_data)
        final_outcome = "Failure" if state_sequence[-1] == self.state_space.states[-1] else "Maintained"
        
        return {
            'id': f"unit_{unit_id}",
            'initial_state': state_sequence[0],
            'goal_state': state_sequence[-1],
            'state_sequence': state_sequence,
            'action_sequence': action_sequence,
            'initial_rul': initial_rul,
            'outcome': final_outcome,
            'sensor_features': unit_data.drop(['unit', 'cycle'], axis=1).iloc[0].to_dict()
        }
    
    def _map_sensors_to_state(self, sensor_data: pd.Series) -> str:
        """
        Map sensor readings to discrete health states.
        
        Args:
            sensor_data: Sensor readings for a single time step
            
        Returns:
            Health state string
        """
        # Simplified state mapping based on sensor degradation patterns
        # In practice, this would use domain expertise or learned mappings
        
        sensor_cols = [col for col in sensor_data.index if col.startswith('sensor')]
        if not sensor_cols:
            return self.state_space.states[0]  # Default to first state
        
        # Calculate aggregate health score
        sensor_values = sensor_data[sensor_cols].values
        health_score = np.mean(sensor_values)  # Simplified aggregation
        
        # Map to discrete states
        num_states = len(self.state_space.states)
        state_idx = min(int(health_score / 100 * num_states), num_states - 1)
        
        return self.state_space.states[state_idx]
    
    def _infer_maintenance_action(self, current_row: pd.Series, next_row: pd.Series) -> str:
        """
        Infer maintenance action from sensor data changes.
        
        Args:
            current_row: Current sensor readings
            next_row: Next time step sensor readings
            
        Returns:
            Inferred maintenance action
        """
        # Simplified action inference
        # In practice, this would analyze maintenance logs or sensor improvements
        
        available_actions = self.state_space.actions
        if not available_actions:
            return "Condition_Monitoring"
        
        # Default to first available action for simplification
        return available_actions[0]
    
    def map_to_state_space(self, problem_data: pd.Series) -> str:
        """
        Map current problem to state space (Step 1 of Algorithm 1).
        
        Args:
            problem_data: Current problem sensor readings
            
        Returns:
            Mapped state in the state space
        """
        return self._map_sensors_to_state(problem_data)
    
    def retrieve_similar_cases(self, current_state: str, threshold: float = 0.7) -> List[Dict]:
        """
        Retrieve similar historical cases (Steps 2-3 of Algorithm 1 Phase 1).
        
        Args:
            current_state: Current problem state
            threshold: Similarity threshold for case retrieval
            
        Returns:
            List of similar cases with similarity scores
        """
        similar_cases = []
        
        for case in self.case_base.cases:
            # Calculate similarity based on initial state and sensor features
            similarity_score = self.similarity.calculate_case_similarity(
                current_state, case, threshold
            )
            
            if similarity_score > threshold:
                case_copy = case.copy()
                case_copy['similarity'] = similarity_score
                similar_cases.append(case_copy)
        
        # Sort by similarity score (descending)
        similar_cases.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_cases
    
    def extract_patterns(self, similar_cases: List[Dict]) -> Dict[str, Any]:
        """
        Extract state transition patterns from retrieved cases.
        
        Args:
            similar_cases: List of similar cases
            
        Returns:
            Extracted patterns dictionary
        """
        patterns = {
            'common_transitions': {},
            'action_effectiveness': {},
            'temporal_patterns': {}
        }
        
        # Analyze common state transitions
        for case in similar_cases:
            state_seq = case['state_sequence']
            action_seq = case['action_sequence']
            
            # Track state transitions
            for i in range(len(state_seq) - 1):
                transition = (state_seq[i], state_seq[i + 1])
                if transition not in patterns['common_transitions']:
                    patterns['common_transitions'][transition] = 0
                patterns['common_transitions'][transition] += 1
            
            # Track action effectiveness
            for i, action in enumerate(action_seq):
                if action not in patterns['action_effectiveness']:
                    patterns['action_effectiveness'][action] = []
                
                # Simple effectiveness measure (improvement in next state)
                if i < len(state_seq) - 1:
                    effectiveness = self._calculate_action_effectiveness(
                        state_seq[i], action, state_seq[i + 1]
                    )
                    patterns['action_effectiveness'][action].append(effectiveness)
        
        # Calculate average effectiveness
        for action in patterns['action_effectiveness']:
            effectiveness_scores = patterns['action_effectiveness'][action]
            patterns['action_effectiveness'][action] = {
                'mean': np.mean(effectiveness_scores),
                'std': np.std(effectiveness_scores),
                'count': len(effectiveness_scores)
            }
        
        return patterns
    
    def _calculate_action_effectiveness(self, before_state: str, action: str, after_state: str) -> float:
        """
        Calculate effectiveness of an action based on state transition.
        
        Args:
            before_state: State before action
            action: Applied action
            after_state: State after action
            
        Returns:
            Effectiveness score (0-1)
        """
        # Simplified effectiveness calculation
        # Improvement if state index decreases (better health)
        before_idx = self.state_space.states.index(before_state)
        after_idx = self.state_space.states.index(after_state)
        
        if after_idx < before_idx:
            return 1.0  # Improvement
        elif after_idx == before_idx:
            return 0.5  # Maintained
        else:
            return 0.0  # Degradation
    
    def execute_algorithm(self, start_state: str, goal_state: str, max_iterations: int = 1000) -> Dict:
        """
        Execute Algorithm 1: CBR+STM Theoretical Framework.
        
        This implements the complete algorithm from the paper including:
        - Phase 1: State-Based Case Retrieval
        - Phase 2: Heuristic-Guided State Navigation
        
        Args:
            start_state: Initial problem state
            goal_state: Target goal state
            max_iterations: Maximum search iterations
            
        Returns:
            Solution dictionary with path, costs, and justifications
        """
        # Phase 1: State-Based Case Retrieval
        similar_cases = self.retrieve_similar_cases(start_state)
        patterns = self.extract_patterns(similar_cases)
        
        # Phase 2: Heuristic-Guided State Navigation (A* Search)
        return self._a_star_search(start_state, goal_state, patterns, max_iterations)
    
    def _a_star_search(self, start_state: str, goal_state: str, patterns: Dict, max_iterations: int) -> Dict:
        """
        Implement A* search for heuristic-guided state navigation.
        
        Args:
            start_state: Initial state
            goal_state: Target state
            patterns: Extracted patterns from similar cases
            max_iterations: Maximum iterations
            
        Returns:
            Search result with solution path and metadata
        """
        # Initialize search structures
        open_set = []
        heapq.heappush(open_set, (0, start_state, []))  # (f_score, state, path)
        
        g_score = {start_state: 0}  # Cost to reach each state
        f_score = {start_state: self._estimate_cost_to_goal(start_state, goal_state)}
        
        closed_set = set()
        iterations = 0
        
        justifications = []
        cost_breakdown = {
            'maintenance_cost': 0,
            'downtime_cost': 0,
            'reliability_improvement': 0,
            'urgency_cost': 0
        }
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get state with lowest f_score
            current_f, current_state, current_path = heapq.heappop(open_set)
            
            # Check if goal reached
            if current_state == goal_state:
                return {
                    'success': True,
                    'path': current_path + [{'state': current_state, 'cost': 0, 'heuristic': 0, 'total_cost': current_f}],
                    'total_cost': current_f,
                    'iterations': iterations,
                    'justifications': justifications,
                    'cost_breakdown': cost_breakdown,
                    'start_state': start_state,
                    'goal_state': goal_state
                }
            
            if current_state in closed_set:
                continue
            
            closed_set.add(current_state)
            
            # Generate applicable actions based on extracted patterns
            applicable_actions = self._get_applicable_actions(current_state, patterns)
            
            for action in applicable_actions:
                # Apply state transition
                next_state = self.state_space.transition(current_state, action)
                if next_state is None or next_state in closed_set:
                    continue
                
                # Calculate costs
                action_cost = self._calculate_action_cost(current_state, action, next_state)
                tentative_g = g_score[current_state] + action_cost
                
                # Calculate heuristic
                h_cost = self.heuristic.calculate(next_state, action, goal_state)
                f_cost = tentative_g + h_cost
                
                # Update costs if better path found
                if next_state not in g_score or tentative_g < g_score[next_state]:
                    g_score[next_state] = tentative_g
                    f_score[next_state] = f_cost
                    
                    # Create step information
                    step_info = {
                        'state': current_state,
                        'action': action,
                        'cost': action_cost,
                        'heuristic': h_cost,
                        'total_cost': tentative_g
                    }
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_cost, next_state, current_path + [step_info]))
                    
                    # Generate justification
                    justification = self._generate_step_justification(
                        current_state, action, next_state, action_cost, h_cost, patterns
                    )
                    justifications.append(justification)
                    
                    # Update cost breakdown
                    self._update_cost_breakdown(cost_breakdown, action, action_cost)
        
        # No solution found
        return {
            'success': False,
            'reason': f'No solution found within {max_iterations} iterations',
            'iterations': iterations,
            'start_state': start_state,
            'goal_state': goal_state
        }
    
    def _estimate_cost_to_goal(self, state: str, goal_state: str) -> float:
        """
        Estimate remaining cost to reach goal state.
        
        Args:
            state: Current state
            goal_state: Target goal state
            
        Returns:
            Estimated cost
        """
        # Simple distance-based estimate
        try:
            current_idx = self.state_space.states.index(state)
            goal_idx = self.state_space.states.index(goal_state)
            return abs(current_idx - goal_idx) * 10  # Base cost per state transition
        except ValueError:
            return 100  # Default high cost for unknown states
    
    def _get_applicable_actions(self, state: str, patterns: Dict) -> List[str]:
        """
        Get applicable actions for current state based on patterns.
        
        Args:
            state: Current state
            patterns: Extracted patterns from similar cases
            
        Returns:
            List of applicable actions
        """
        # Start with all available actions
        applicable = self.state_space.actions.copy()
        
        # Filter based on action effectiveness patterns
        if 'action_effectiveness' in patterns:
            # Prefer actions with higher effectiveness
            effectiveness_scores = []
            for action in applicable:
                if action in patterns['action_effectiveness']:
                    score = patterns['action_effectiveness'][action]['mean']
                    effectiveness_scores.append((action, score))
            
            # Sort by effectiveness and take top actions
            effectiveness_scores.sort(key=lambda x: x[1], reverse=True)
            if effectiveness_scores:
                # Take top 3 most effective actions or all if fewer
                top_actions = [action for action, _ in effectiveness_scores[:3]]
                return top_actions
        
        return applicable
    
    def _calculate_action_cost(self, state: str, action: str, next_state: str) -> float:
        """
        Calculate cost of applying action in current state.
        
        Args:
            state: Current state
            action: Applied action
            next_state: Resulting state
            
        Returns:
            Action cost
        """
        # Base costs for different action types
        base_costs = {
            'Preventive_Maintenance': 50,
            'Corrective_Maintenance': 100,
            'Component_Replacement': 200,
            'Condition_Monitoring': 10,
            'Emergency_Repair': 300,
            'Scheduled_Overhaul': 400
        }
        
        return base_costs.get(action, 75)  # Default cost
    
    def _generate_step_justification(self, current_state: str, action: str, next_state: str, 
                                   cost: float, heuristic: float, patterns: Dict) -> Dict:
        """
        Generate justification for a reasoning step.
        
        Args:
            current_state: Current state
            action: Applied action
            next_state: Resulting state
            cost: Step cost
            heuristic: Heuristic value
            patterns: Available patterns
            
        Returns:
            Justification dictionary
        """
        justification = {
            'step': f"{current_state} → {action} → {next_state}",
            'cost_components': {
                'direct_cost': cost,
                'heuristic_estimate': heuristic
            },
            'reasoning': f"Applied {action} in state {current_state} based on case patterns",
            'supporting_evidence': []
        }
        
        # Add pattern-based evidence
        if 'action_effectiveness' in patterns and action in patterns['action_effectiveness']:
            effectiveness = patterns['action_effectiveness'][action]
            justification['supporting_evidence'].append(
                f"Action effectiveness: {effectiveness['mean']:.2f} ± {effectiveness['std']:.2f} "
                f"(based on {effectiveness['count']} historical cases)"
            )
        
        # Add transition frequency evidence
        if 'common_transitions' in patterns:
            transition = (current_state, next_state)
            if transition in patterns['common_transitions']:
                count = patterns['common_transitions'][transition]
                justification['supporting_evidence'].append(
                    f"Transition {current_state}→{next_state} observed {count} times in similar cases"
                )
        
        return justification
    
    def _update_cost_breakdown(self, cost_breakdown: Dict, action: str, cost: float):
        """
        Update cost breakdown with new action cost.
        
        Args:
            cost_breakdown: Cost breakdown dictionary to update
            action: Applied action
            cost: Action cost
        """
        # Distribute cost across components based on action type
        if 'Maintenance' in action:
            cost_breakdown['maintenance_cost'] += cost * 0.6
            cost_breakdown['downtime_cost'] += cost * 0.3
            cost_breakdown['urgency_cost'] += cost * 0.1
        elif 'Replacement' in action:
            cost_breakdown['maintenance_cost'] += cost * 0.8
            cost_breakdown['downtime_cost'] += cost * 0.2
        elif 'Monitoring' in action:
            cost_breakdown['maintenance_cost'] += cost * 0.4
            cost_breakdown['reliability_improvement'] += cost * 0.6
        else:
            cost_breakdown['maintenance_cost'] += cost
