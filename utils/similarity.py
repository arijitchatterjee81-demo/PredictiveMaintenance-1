"""
Similarity Measurement Utilities

This module implements similarity calculation methods for case-based reasoning
as referenced in the paper's similarity measurement section.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy.spatial.distance import euclidean, cosine, cityblock
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMeasure:
    """
    Similarity measurement utilities for CBR+STM framework.
    
    Implements weighted feature vector comparison for case retrieval
    as described in the paper.
    """
    
    def __init__(self):
        """Initialize similarity measurement with default weights."""
        self.feature_weights = {
            'static_features': 0.4,    # Equipment type, age, operating conditions
            'dynamic_features': 0.6    # Degradation rate, failure history patterns
        }
        
        # Distance metrics available
        self.distance_metrics = {
            'euclidean': euclidean,
            'manhattan': cityblock,
            'cosine': self._cosine_distance
        }
    
    def calculate_case_similarity(self, current_state: str, case: Dict[str, Any], 
                                threshold: float = 0.7) -> float:
        """
        Calculate similarity between current problem state and historical case.
        
        Args:
            current_state: Current problem state
            case: Historical case from case base
            threshold: Minimum similarity threshold
            
        Returns:
            Similarity score [0, 1]
        """
        # Extract features from current state and case
        current_features = self._extract_state_features(current_state)
        case_features = self._extract_case_features(case)
        
        # Calculate similarity components
        state_similarity = self._calculate_state_similarity(current_state, case['initial_state'])
        feature_similarity = self._calculate_feature_similarity(current_features, case_features)
        trajectory_similarity = self._calculate_trajectory_similarity(current_state, case)
        
        # Weighted combination
        total_similarity = (
            0.3 * state_similarity +
            0.4 * feature_similarity +
            0.3 * trajectory_similarity
        )
        
        return min(1.0, max(0.0, total_similarity))
    
    def _extract_state_features(self, state: str) -> Dict[str, float]:
        """
        Extract features from state representation.
        
        Args:
            state: State string
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract numeric components if present
        numeric_parts = [char for char in state if char.isdigit()]
        if numeric_parts:
            features['state_numeric'] = float(''.join(numeric_parts))
        else:
            features['state_numeric'] = 0.0
        
        # Map state names to numeric values
        state_mapping = {
            'healthy': 1.0,
            'good': 1.0,
            'normal': 1.0,
            'mild': 2.0,
            'low': 2.0,
            'moderate': 3.0,
            'medium': 3.0,
            'severe': 4.0,
            'high': 4.0,
            'critical': 5.0,
            'fail': 6.0,
            'failure': 6.0
        }
        
        state_lower = state.lower()
        features['degradation_level'] = 3.0  # Default
        for keyword, value in state_mapping.items():
            if keyword in state_lower:
                features['degradation_level'] = value
                break
        
        return features
    
    def _extract_case_features(self, case: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from case representation.
        
        Args:
            case: Case dictionary
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract from sensor features if available
        if 'sensor_features' in case:
            sensor_features = case['sensor_features']
            
            # Calculate statistical features
            sensor_values = [v for v in sensor_features.values() if isinstance(v, (int, float))]
            if sensor_values:
                features['sensor_mean'] = np.mean(sensor_values)
                features['sensor_std'] = np.std(sensor_values)
                features['sensor_min'] = np.min(sensor_values)
                features['sensor_max'] = np.max(sensor_values)
        
        # Extract from case metadata
        if 'initial_rul' in case:
            features['initial_rul'] = float(case['initial_rul'])
        
        # Extract sequence characteristics
        if 'state_sequence' in case:
            features['sequence_length'] = float(len(case['state_sequence']))
        
        if 'action_sequence' in case:
            features['num_actions'] = float(len(case['action_sequence']))
        
        return features
    
    def _calculate_state_similarity(self, state1: str, state2: str) -> float:
        """
        Calculate similarity between two states.
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            State similarity score [0, 1]
        """
        if state1 == state2:
            return 1.0
        
        # Extract features and compare
        features1 = self._extract_state_features(state1)
        features2 = self._extract_state_features(state2)
        
        # Calculate normalized difference in degradation levels
        deg1 = features1.get('degradation_level', 3.0)
        deg2 = features2.get('degradation_level', 3.0)
        
        max_deg_diff = 5.0  # Maximum possible difference
        deg_similarity = 1.0 - abs(deg1 - deg2) / max_deg_diff
        
        return max(0.0, deg_similarity)
    
    def _calculate_feature_similarity(self, features1: Dict[str, float], 
                                    features2: Dict[str, float]) -> float:
        """
        Calculate similarity between feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Feature similarity score [0, 1]
        """
        if not features1 or not features2:
            return 0.0
        
        # Get common features
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        # Calculate normalized differences
        similarities = []
        for key in common_keys:
            val1 = features1[key]
            val2 = features2[key]
            
            # Handle division by zero
            max_val = max(abs(val1), abs(val2), 1e-6)
            normalized_diff = abs(val1 - val2) / max_val
            similarity = 1.0 - min(normalized_diff, 1.0)
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_trajectory_similarity(self, current_state: str, case: Dict[str, Any]) -> float:
        """
        Calculate similarity based on expected trajectory patterns.
        
        Args:
            current_state: Current state
            case: Historical case
            
        Returns:
            Trajectory similarity score [0, 1]
        """
        if 'state_sequence' not in case:
            return 0.0
        
        state_sequence = case['state_sequence']
        if not state_sequence:
            return 0.0
        
        # Find if current state appears in the trajectory
        if current_state in state_sequence:
            # Calculate position similarity
            position = state_sequence.index(current_state)
            relative_position = position / len(state_sequence)
            
            # Prefer cases where current state appears early (more trajectory left)
            trajectory_similarity = 1.0 - relative_position
            return trajectory_similarity
        
        # If state doesn't appear exactly, find most similar state
        max_similarity = 0.0
        for seq_state in state_sequence:
            state_sim = self._calculate_state_similarity(current_state, seq_state)
            max_similarity = max(max_similarity, state_sim)
        
        return max_similarity * 0.5  # Reduce score for non-exact matches
    
    def calculate_weighted_similarity(self, target: Dict[str, Any], 
                                    candidates: List[Dict[str, Any]],
                                    weights: Dict[str, float] = None) -> List[Tuple[int, float]]:
        """
        Calculate weighted similarity for multiple candidates.
        
        Args:
            target: Target case/state for comparison
            candidates: List of candidate cases
            weights: Feature weights for similarity calculation
            
        Returns:
            List of (index, similarity_score) tuples sorted by similarity
        """
        if weights is None:
            weights = self.feature_weights
        
        similarities = []
        
        for i, candidate in enumerate(candidates):
            # Calculate various similarity components
            similarities_dict = {}
            
            # Feature-based similarity
            if 'sensor_features' in target and 'sensor_features' in candidate:
                feature_sim = self._calculate_sensor_similarity(
                    target['sensor_features'], 
                    candidate['sensor_features']
                )
                similarities_dict['sensor'] = feature_sim
            
            # State-based similarity
            if 'initial_state' in target and 'initial_state' in candidate:
                state_sim = self._calculate_state_similarity(
                    target['initial_state'],
                    candidate['initial_state']
                )
                similarities_dict['state'] = state_sim
            
            # Outcome similarity
            if 'outcome' in target and 'outcome' in candidate:
                outcome_sim = 1.0 if target['outcome'] == candidate['outcome'] else 0.0
                similarities_dict['outcome'] = outcome_sim
            
            # Weighted combination
            total_similarity = 0.0
            total_weight = 0.0
            
            for component, sim_score in similarities_dict.items():
                weight = weights.get(component, 1.0)
                total_similarity += weight * sim_score
                total_weight += weight
            
            if total_weight > 0:
                final_similarity = total_similarity / total_weight
            else:
                final_similarity = 0.0
            
            similarities.append((i, final_similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def _calculate_sensor_similarity(self, sensors1: Dict[str, float], 
                                   sensors2: Dict[str, float]) -> float:
        """
        Calculate similarity between sensor feature vectors.
        
        Args:
            sensors1: First sensor feature vector
            sensors2: Second sensor feature vector
            
        Returns:
            Sensor similarity score [0, 1]
        """
        # Get common sensor keys
        common_keys = set(sensors1.keys()) & set(sensors2.keys())
        if not common_keys:
            return 0.0
        
        # Extract values for common sensors
        values1 = np.array([sensors1[key] for key in common_keys])
        values2 = np.array([sensors2[key] for key in common_keys])
        
        # Calculate cosine similarity
        if np.linalg.norm(values1) == 0 or np.linalg.norm(values2) == 0:
            return 0.0
        
        cos_sim = np.dot(values1, values2) / (np.linalg.norm(values1) * np.linalg.norm(values2))
        
        # Convert to similarity score [0, 1]
        return (cos_sim + 1) / 2
    
    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine distance between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine distance
        """
        return 1.0 - cosine_similarity([vec1], [vec2])[0, 0]
    
    def find_k_most_similar(self, target: Dict[str, Any], 
                           candidates: List[Dict[str, Any]], 
                           k: int = 5) -> List[Dict[str, Any]]:
        """
        Find k most similar cases to target.
        
        Args:
            target: Target case for similarity matching
            candidates: List of candidate cases
            k: Number of most similar cases to return
            
        Returns:
            List of k most similar cases with similarity scores
        """
        # Calculate similarities
        similarities = self.calculate_weighted_similarity(target, candidates)
        
        # Return top k candidates
        top_k = similarities[:k]
        
        result = []
        for idx, similarity_score in top_k:
            case_copy = candidates[idx].copy()
            case_copy['similarity_score'] = similarity_score
            result.append(case_copy)
        
        return result
    
    def update_feature_weights(self, new_weights: Dict[str, float]):
        """
        Update feature weights for similarity calculation.
        
        Args:
            new_weights: New weight values
        """
        self.feature_weights.update(new_weights)
    
    def get_similarity_explanation(self, target: Dict[str, Any], 
                                 case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation for similarity calculation.
        
        Args:
            target: Target case
            case: Compared case
            
        Returns:
            Dictionary with similarity explanation
        """
        explanation = {
            'overall_similarity': self.calculate_case_similarity(
                target.get('initial_state', ''), case
            ),
            'components': {}
        }
        
        # State similarity
        if 'initial_state' in target and 'initial_state' in case:
            state_sim = self._calculate_state_similarity(
                target['initial_state'], case['initial_state']
            )
            explanation['components']['state_similarity'] = state_sim
        
        # Feature similarity
        if 'sensor_features' in target and 'sensor_features' in case:
            feature_sim = self._calculate_sensor_similarity(
                target['sensor_features'], case['sensor_features']
            )
            explanation['components']['sensor_similarity'] = feature_sim
        
        # Trajectory similarity
        trajectory_sim = self._calculate_trajectory_similarity(
            target.get('initial_state', ''), case
        )
        explanation['components']['trajectory_similarity'] = trajectory_sim
        
        return explanation
