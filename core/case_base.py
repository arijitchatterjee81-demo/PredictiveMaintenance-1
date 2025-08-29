"""
Case Base Management for CBR+STM Framework

This module implements case storage, retrieval, and management functionality
as defined in Definition 3 of the paper.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

class CaseBase:
    """
    Case base for storing and managing historical maintenance cases.
    
    Implements Definition 3 from the paper:
    Each case c ∈ C corresponds to a state transition sequence π
    """
    
    def __init__(self):
        """Initialize empty case base."""
        self.cases = []
        self.case_index = {}  # For fast case lookup
        self._next_id = 1
    
    def add_case(self, case: Dict[str, Any]) -> str:
        """
        Add new case to the case base.
        
        Args:
            case: Case dictionary with state transition sequence
            
        Returns:
            Assigned case ID
        """
        # Validate case structure
        if not self._validate_case(case):
            raise ValueError("Invalid case structure")
        
        # Assign unique ID if not present
        if 'id' not in case:
            case['id'] = f"case_{self._next_id}"
            self._next_id += 1
        
        # Add metadata
        case['added_timestamp'] = datetime.now().isoformat()
        case['access_count'] = 0
        case['last_accessed'] = None
        
        # Add to case base
        self.cases.append(case)
        self._update_index(case)
        
        return case['id']
    
    def _validate_case(self, case: Dict[str, Any]) -> bool:
        """
        Validate case structure according to Definition 3.
        
        Args:
            case: Case to validate
            
        Returns:
            True if case is valid
        """
        required_fields = [
            'initial_state',
            'goal_state',
            'state_sequence',
            'outcome'
        ]
        
        for field in required_fields:
            if field not in case:
                return False
        
        # Validate state sequence is non-empty
        if not case['state_sequence'] or len(case['state_sequence']) == 0:
            return False
        
        # Validate sequence consistency
        if case['state_sequence'][0] != case['initial_state']:
            return False
        
        if case['state_sequence'][-1] != case['goal_state']:
            return False
        
        return True
    
    def _update_index(self, case: Dict[str, Any]):
        """
        Update case index for efficient retrieval.
        
        Args:
            case: Case to index
        """
        case_id = case['id']
        
        # Index by initial state
        initial_state = case['initial_state']
        if initial_state not in self.case_index:
            self.case_index[initial_state] = []
        self.case_index[initial_state].append(case_id)
        
        # Index by outcome
        outcome = case['outcome']
        outcome_key = f"outcome_{outcome}"
        if outcome_key not in self.case_index:
            self.case_index[outcome_key] = []
        self.case_index[outcome_key].append(case_id)
    
    def get_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve case by ID.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Case dictionary or None if not found
        """
        for case in self.cases:
            if case['id'] == case_id:
                # Update access statistics
                case['access_count'] += 1
                case['last_accessed'] = datetime.now().isoformat()
                return case
        
        return None
    
    def get_cases_by_initial_state(self, initial_state: str) -> List[Dict[str, Any]]:
        """
        Retrieve all cases with given initial state.
        
        Args:
            initial_state: Initial state to match
            
        Returns:
            List of matching cases
        """
        if initial_state not in self.case_index:
            return []
        
        case_ids = self.case_index[initial_state]
        cases = []
        
        for case_id in case_ids:
            case = self.get_case_by_id(case_id)
            if case:
                cases.append(case)
        
        return cases
    
    def get_cases_by_outcome(self, outcome: str) -> List[Dict[str, Any]]:
        """
        Retrieve all cases with given outcome.
        
        Args:
            outcome: Outcome to match
            
        Returns:
            List of matching cases
        """
        outcome_key = f"outcome_{outcome}"
        if outcome_key not in self.case_index:
            return []
        
        case_ids = self.case_index[outcome_key]
        cases = []
        
        for case_id in case_ids:
            case = self.get_case_by_id(case_id)
            if case:
                cases.append(case)
        
        return cases
    
    def find_similar_cases(self, target_features: Dict[str, Any], 
                          similarity_threshold: float = 0.7,
                          max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find cases similar to target features.
        
        Args:
            target_features: Features to match against
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            
        Returns:
            List of similar cases with similarity scores
        """
        similar_cases = []
        
        for case in self.cases:
            # Calculate similarity based on sensor features
            if 'sensor_features' in case:
                similarity = self._calculate_feature_similarity(
                    target_features, case['sensor_features']
                )
                
                if similarity >= similarity_threshold:
                    case_copy = case.copy()
                    case_copy['similarity_score'] = similarity
                    similar_cases.append(case_copy)
        
        # Sort by similarity score (descending)
        similar_cases.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_cases[:max_results]
    
    def _calculate_feature_similarity(self, features1: Dict[str, Any], 
                                    features2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two feature sets.
        
        Args:
            features1: First feature set
            features2: Second feature set
            
        Returns:
            Similarity score [0, 1]
        """
        # Get common features
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        # Calculate normalized differences
        differences = []
        for key in common_keys:
            try:
                val1 = float(features1[key])
                val2 = float(features2[key])
                
                # Normalize difference by max value to handle different scales
                max_val = max(abs(val1), abs(val2), 1e-6)
                normalized_diff = abs(val1 - val2) / max_val
                differences.append(normalized_diff)
                
            except (ValueError, TypeError):
                # Handle non-numeric features
                if features1[key] == features2[key]:
                    differences.append(0.0)
                else:
                    differences.append(1.0)
        
        # Calculate similarity as 1 - mean difference
        mean_difference = np.mean(differences)
        similarity = 1.0 - mean_difference
        
        return max(0.0, min(1.0, similarity))
    
    def get_case_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the case base.
        
        Returns:
            Statistics dictionary
        """
        if not self.cases:
            return {
                'total_cases': 0,
                'states_covered': 0,
                'outcomes_distribution': {},
                'average_sequence_length': 0
            }
        
        # Count outcomes
        outcomes = {}
        sequence_lengths = []
        states_covered = set()
        
        for case in self.cases:
            # Count outcome
            outcome = case['outcome']
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            
            # Track sequence length
            sequence_lengths.append(len(case['state_sequence']))
            
            # Track states covered
            states_covered.update(case['state_sequence'])
        
        return {
            'total_cases': len(self.cases),
            'states_covered': len(states_covered),
            'outcomes_distribution': outcomes,
            'average_sequence_length': np.mean(sequence_lengths),
            'min_sequence_length': min(sequence_lengths),
            'max_sequence_length': max(sequence_lengths),
            'most_common_outcome': max(outcomes.keys(), key=outcomes.get) if outcomes else None
        }
    
    def export_cases(self, filepath: str):
        """
        Export cases to JSON file.
        
        Args:
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump({
                'cases': self.cases,
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_cases': len(self.cases),
                    'next_id': self._next_id
                }
            }, f, indent=2)
    
    def import_cases(self, filepath: str):
        """
        Import cases from JSON file.
        
        Args:
            filepath: Input file path
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Import cases
        for case in data['cases']:
            if self._validate_case(case):
                self.cases.append(case)
                self._update_index(case)
        
        # Update ID counter
        if 'metadata' in data and 'next_id' in data['metadata']:
            self._next_id = max(self._next_id, data['metadata']['next_id'])
    
    def remove_case(self, case_id: str) -> bool:
        """
        Remove case from case base.
        
        Args:
            case_id: Case ID to remove
            
        Returns:
            True if case was removed
        """
        for i, case in enumerate(self.cases):
            if case['id'] == case_id:
                # Remove from main list
                removed_case = self.cases.pop(i)
                
                # Update index
                self._remove_from_index(removed_case)
                
                return True
        
        return False
    
    def _remove_from_index(self, case: Dict[str, Any]):
        """
        Remove case from index structures.
        
        Args:
            case: Case to remove from index
        """
        case_id = case['id']
        
        # Remove from initial state index
        initial_state = case['initial_state']
        if initial_state in self.case_index:
            self.case_index[initial_state] = [
                cid for cid in self.case_index[initial_state] if cid != case_id
            ]
            if not self.case_index[initial_state]:
                del self.case_index[initial_state]
        
        # Remove from outcome index
        outcome_key = f"outcome_{case['outcome']}"
        if outcome_key in self.case_index:
            self.case_index[outcome_key] = [
                cid for cid in self.case_index[outcome_key] if cid != case_id
            ]
            if not self.case_index[outcome_key]:
                del self.case_index[outcome_key]
    
    def clear_all_cases(self):
        """Clear all cases from the case base."""
        self.cases.clear()
        self.case_index.clear()
        self._next_id = 1
