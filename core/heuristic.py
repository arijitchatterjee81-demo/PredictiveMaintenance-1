"""
Multi-Objective Heuristic Function Implementation

This module implements Definition 4 from the paper:
Multi-Objective Heuristic Function h(s,a) with maintenance-specific components.
"""

import numpy as np
from typing import List, Dict, Any

class MultiObjectiveHeuristic:
    """
    Multi-objective heuristic function for predictive maintenance.
    
    Implements equation (1) from the paper:
    h(s,a) = w₁·C_maint(a) + w₂·C_downtime(s,a) - w₃·R_improvement(s,a) + w₄·T_urgency(s)
    """
    
    def __init__(self, weights: List[float]):
        """
        Initialize heuristic with weight coefficients.
        
        Args:
            weights: List of weights [w₁, w₂, w₃, w₄] for the four components
        """
        if len(weights) != 4:
            raise ValueError("Heuristic requires exactly 4 weight coefficients")
        
        self.w1, self.w2, self.w3, self.w4 = weights
        self._validate_weights()
        
        # Cost parameters for different maintenance actions
        self.maintenance_costs = {
            'Preventive_Maintenance': 50.0,
            'Corrective_Maintenance': 100.0,
            'Component_Replacement': 200.0,
            'Condition_Monitoring': 10.0,
            'Emergency_Repair': 300.0,
            'Scheduled_Overhaul': 400.0
        }
        
        # Downtime factors by action type
        self.downtime_factors = {
            'Preventive_Maintenance': 2.0,
            'Corrective_Maintenance': 4.0,
            'Component_Replacement': 8.0,
            'Condition_Monitoring': 0.5,
            'Emergency_Repair': 12.0,
            'Scheduled_Overhaul': 16.0
        }
        
        # Reliability improvement factors
        self.reliability_factors = {
            'Preventive_Maintenance': 15.0,
            'Corrective_Maintenance': 10.0,
            'Component_Replacement': 25.0,
            'Condition_Monitoring': 2.0,
            'Emergency_Repair': 8.0,
            'Scheduled_Overhaul': 30.0
        }
    
    def _validate_weights(self):
        """Validate weight coefficients are non-negative."""
        if any(w < 0 for w in [self.w1, self.w2, self.w3, self.w4]):
            raise ValueError("All weight coefficients must be non-negative")
    
    def calculate(self, state: str, action: str, goal_state: str = None) -> float:
        """
        Calculate heuristic value h(s,a) for given state-action pair.
        
        Args:
            state: Current equipment health state
            action: Proposed maintenance action
            goal_state: Target goal state (optional)
            
        Returns:
            Heuristic cost estimate
        """
        # Calculate individual components
        c_maint = self._calculate_maintenance_cost(action)
        c_downtime = self._calculate_downtime_cost(state, action)
        r_improvement = self._calculate_reliability_improvement(state, action)
        t_urgency = self._calculate_urgency_factor(state)
        
        # Apply multi-objective heuristic formula
        heuristic_value = (
            self.w1 * c_maint +
            self.w2 * c_downtime -
            self.w3 * r_improvement +
            self.w4 * t_urgency
        )
        
        return max(0, heuristic_value)  # Ensure non-negative
    
    def _calculate_maintenance_cost(self, action: str) -> float:
        """
        Calculate C_maint(a) - direct maintenance cost for action.
        
        Args:
            action: Maintenance action
            
        Returns:
            Direct maintenance cost
        """
        return self.maintenance_costs.get(action, 75.0)  # Default cost
    
    def _calculate_downtime_cost(self, state: str, action: str) -> float:
        """
        Calculate C_downtime(s,a) - expected downtime cost.
        
        Args:
            state: Current state
            action: Maintenance action
            
        Returns:
            Expected downtime cost
        """
        base_downtime = self.downtime_factors.get(action, 4.0)
        
        # Adjust based on current state severity
        state_severity = self._get_state_severity(state)
        
        # More severe states require longer downtime
        downtime_cost = base_downtime * (1 + state_severity)
        
        return downtime_cost
    
    def _calculate_reliability_improvement(self, state: str, action: str) -> float:
        """
        Calculate R_improvement(s,a) - projected reliability gain.
        
        Args:
            state: Current state
            action: Maintenance action
            
        Returns:
            Projected reliability improvement
        """
        base_improvement = self.reliability_factors.get(action, 10.0)
        
        # Improvement is higher for more degraded states
        state_severity = self._get_state_severity(state)
        
        # Reliability improvement scales with degradation level
        reliability_improvement = base_improvement * (1 + state_severity * 0.5)
        
        return reliability_improvement
    
    def _calculate_urgency_factor(self, state: str) -> float:
        """
        Calculate T_urgency(s) - time urgency factor for state.
        
        Args:
            state: Current equipment state
            
        Returns:
            Urgency factor (higher for more critical states)
        """
        state_severity = self._get_state_severity(state)
        
        # Exponential urgency increase for severe states
        urgency = 10.0 * np.exp(state_severity * 2)
        
        return urgency
    
    def _get_state_severity(self, state: str) -> float:
        """
        Map state name to severity score [0, 1].
        
        Args:
            state: State name
            
        Returns:
            Severity score (0 = healthy, 1 = failed)
        """
        # Extract severity from state name or use heuristic mapping
        if 'fail' in state.lower() or 'critical' in state.lower():
            return 1.0
        elif 'severe' in state.lower() or 'high' in state.lower():
            return 0.8
        elif 'moderate' in state.lower() or 'medium' in state.lower():
            return 0.6
        elif 'mild' in state.lower() or 'low' in state.lower():
            return 0.4
        elif 'good' in state.lower() or 'healthy' in state.lower():
            return 0.2
        else:
            # Try to extract numeric indicators
            if any(char.isdigit() for char in state):
                # Extract numbers and normalize
                numbers = [int(char) for char in state if char.isdigit()]
                if numbers:
                    return min(max(numbers) / 10.0, 1.0)
        
        return 0.5  # Default medium severity
    
    def get_component_breakdown(self, state: str, action: str) -> Dict[str, float]:
        """
        Get detailed breakdown of heuristic components.
        
        Args:
            state: Current state
            action: Maintenance action
            
        Returns:
            Dictionary with component values
        """
        c_maint = self._calculate_maintenance_cost(action)
        c_downtime = self._calculate_downtime_cost(state, action)
        r_improvement = self._calculate_reliability_improvement(state, action)
        t_urgency = self._calculate_urgency_factor(state)
        
        return {
            'maintenance_cost': c_maint,
            'downtime_cost': c_downtime,
            'reliability_improvement': r_improvement,
            'urgency_factor': t_urgency,
            'weighted_maintenance': self.w1 * c_maint,
            'weighted_downtime': self.w2 * c_downtime,
            'weighted_reliability': -self.w3 * r_improvement,  # Negative because it's subtracted
            'weighted_urgency': self.w4 * t_urgency,
            'total_heuristic': self.calculate(state, action)
        }
    
    def update_weights(self, new_weights: List[float]):
        """
        Update heuristic weights.
        
        Args:
            new_weights: New weight coefficients [w₁, w₂, w₃, w₄]
        """
        if len(new_weights) != 4:
            raise ValueError("Must provide exactly 4 weight coefficients")
        
        self.w1, self.w2, self.w3, self.w4 = new_weights
        self._validate_weights()
    
    def normalize_weights(self):
        """Normalize weights to sum to 1.0."""
        total = sum([self.w1, self.w2, self.w3, self.w4])
        if total > 0:
            self.w1 /= total
            self.w2 /= total
            self.w3 /= total
            self.w4 /= total
    
    def get_weight_sensitivity(self, state: str, action: str, perturbation: float = 0.1) -> Dict:
        """
        Analyze sensitivity to weight changes.
        
        Args:
            state: Current state
            action: Maintenance action
            perturbation: Perturbation magnitude for sensitivity analysis
            
        Returns:
            Sensitivity analysis results
        """
        base_value = self.calculate(state, action)
        original_weights = [self.w1, self.w2, self.w3, self.w4]
        sensitivities = {}
        
        for i, weight_name in enumerate(['w1', 'w2', 'w3', 'w4']):
            # Test positive perturbation
            perturbed_weights = original_weights.copy()
            perturbed_weights[i] += perturbation
            
            self.update_weights(perturbed_weights)
            pos_value = self.calculate(state, action)
            
            # Test negative perturbation
            perturbed_weights[i] = original_weights[i] - perturbation
            self.update_weights(perturbed_weights)
            neg_value = self.calculate(state, action)
            
            # Calculate sensitivity (derivative approximation)
            sensitivity = (pos_value - neg_value) / (2 * perturbation)
            sensitivities[weight_name] = sensitivity
        
        # Restore original weights
        self.update_weights(original_weights)
        
        return {
            'base_value': base_value,
            'sensitivities': sensitivities,
            'most_sensitive': max(sensitivities.keys(), key=lambda k: abs(sensitivities[k]))
        }
