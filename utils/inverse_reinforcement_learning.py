"""
Inverse Reinforcement Learning for Automatic Weight Calibration

This module implements Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)
to automatically calibrate the multi-objective heuristic weights from expert demonstrations
or historical maintenance decision data.

Based on Ziebart et al. (2008) "Maximum Entropy Inverse Reinforcement Learning"
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import minimize
from scipy.special import logsumexp
import logging

logger = logging.getLogger(__name__)

class ExpertDemonstration:
    """
    Container for expert maintenance decision demonstrations.
    
    Each demonstration contains:
    - Initial state
    - Action sequence taken by expert
    - Final state achieved
    - Context information (equipment type, urgency, etc.)
    """
    
    def __init__(self, trajectory: List[Dict[str, Any]], context: Dict[str, Any] = None):
        """
        Initialize expert demonstration.
        
        Args:
            trajectory: List of (state, action, next_state) tuples with costs
            context: Additional context information about the demonstration
        """
        self.trajectory = trajectory
        self.context = context or {}
        self.total_cost = sum(step.get('cost', 0) for step in trajectory)
        self.length = len(trajectory)
    
    def get_feature_vector(self, heuristic_func) -> np.ndarray:
        """
        Extract feature vector from demonstration trajectory.
        
        Args:
            heuristic_func: Multi-objective heuristic function
            
        Returns:
            Feature vector representing the demonstration
        """
        features = np.zeros(4)  # Four heuristic components
        
        for step in self.trajectory:
            state = step['state']
            action = step['action']
            
            # Extract individual heuristic components
            components = heuristic_func.get_components(state, action)
            features[0] += components['maintenance_cost']
            features[1] += components['downtime_cost']
            features[2] += components['reliability_improvement']
            features[3] += components['urgency_factor']
        
        return features

class MaxEntIRL:
    """
    Maximum Entropy Inverse Reinforcement Learning implementation for weight calibration.
    
    This algorithm learns the weights of the multi-objective heuristic function
    by maximizing the likelihood of expert demonstrations under the principle
    of maximum entropy.
    """
    
    def __init__(self, state_space, action_space, heuristic_func, learning_rate: float = 0.01):
        """
        Initialize MaxEnt IRL algorithm.
        
        Args:
            state_space: State space definition
            action_space: Available actions
            heuristic_func: Multi-objective heuristic function
            learning_rate: Learning rate for gradient optimization
        """
        self.state_space = state_space
        self.action_space = action_space
        self.heuristic_func = heuristic_func
        self.learning_rate = learning_rate
        
        # Initialize weights randomly
        self.weights = np.random.uniform(0.1, 0.5, 4)
        self.weights = self.weights / np.sum(self.weights)  # Normalize
        
        # Cache for efficiency
        self._policy_cache = {}
        self._feature_cache = {}
    
    def compute_policy(self, weights: np.ndarray, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Compute soft policy π(a|s) = exp(Q(s,a)/T) / Z(s) using current weights.
        
        Args:
            weights: Current weight vector
            temperature: Temperature parameter for softmax
            
        Returns:
            Policy dictionary mapping states to action probabilities
        """
        cache_key = (tuple(weights), temperature)
        if cache_key in self._policy_cache:
            return self._policy_cache[cache_key]
        
        policy = {}
        
        for state in self.state_space.states:
            action_values = []
            
            for action in self.state_space.actions:
                # Compute Q-value using weighted heuristic
                components = self.heuristic_func.get_components(state, action)
                q_value = (
                    weights[0] * components['maintenance_cost'] +
                    weights[1] * components['downtime_cost'] +
                    weights[2] * components['reliability_improvement'] +
                    weights[3] * components['urgency_factor']
                )
                action_values.append(-q_value / temperature)  # Negative because we minimize cost
            
            # Compute softmax policy
            action_probs = np.exp(action_values - logsumexp(action_values))
            policy[state] = action_probs
        
        self._policy_cache[cache_key] = policy
        return policy
    
    def compute_expected_features(self, policy: Dict[str, np.ndarray], 
                                demonstrations: List[ExpertDemonstration]) -> np.ndarray:
        """
        Compute expected feature counts under current policy.
        
        Args:
            policy: Current policy π(a|s)
            demonstrations: Expert demonstrations for context
            
        Returns:
            Expected feature vector
        """
        expected_features = np.zeros(4)
        total_steps = 0
        
        # Sample trajectories under current policy
        num_samples = 100
        max_trajectory_length = 50
        
        for demo in demonstrations:
            start_state = demo.trajectory[0]['state']
            
            for _ in range(num_samples):
                current_state = start_state
                trajectory_features = np.zeros(4)
                
                for step in range(max_trajectory_length):
                    if current_state not in policy:
                        break
                    
                    # Sample action according to policy
                    action_probs = policy[current_state]
                    action_idx = np.random.choice(len(self.state_space.actions), p=action_probs)
                    action = self.state_space.actions[action_idx]
                    
                    # Extract features for this step
                    components = self.heuristic_func.get_components(current_state, action)
                    step_features = np.array([
                        components['maintenance_cost'],
                        components['downtime_cost'],
                        components['reliability_improvement'],
                        components['urgency_factor']
                    ])
                    
                    trajectory_features += step_features
                    total_steps += 1
                    
                    # Transition to next state
                    next_state = self.state_space.transition(current_state, action)
                    if next_state is None or next_state == current_state:
                        break
                    current_state = next_state
                
                expected_features += trajectory_features
        
        return expected_features / max(total_steps, 1)
    
    def compute_expert_features(self, demonstrations: List[ExpertDemonstration]) -> np.ndarray:
        """
        Compute empirical feature counts from expert demonstrations.
        
        Args:
            demonstrations: List of expert demonstrations
            
        Returns:
            Empirical feature vector
        """
        expert_features = np.zeros(4)
        total_steps = 0
        
        for demo in demonstrations:
            demo_features = demo.get_feature_vector(self.heuristic_func)
            expert_features += demo_features
            total_steps += demo.length
        
        return expert_features / max(total_steps, 1)
    
    def compute_gradient(self, weights: np.ndarray, 
                        demonstrations: List[ExpertDemonstration]) -> np.ndarray:
        """
        Compute gradient of the log-likelihood with respect to weights.
        
        Args:
            weights: Current weight vector
            demonstrations: Expert demonstrations
            
        Returns:
            Gradient vector
        """
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Compute policy under current weights
        policy = self.compute_policy(weights)
        
        # Compute feature expectations
        expert_features = self.compute_expert_features(demonstrations)
        expected_features = self.compute_expected_features(policy, demonstrations)
        
        # Gradient is difference between expert and expected features
        gradient = expert_features - expected_features
        
        return gradient
    
    def compute_log_likelihood(self, weights: np.ndarray, 
                             demonstrations: List[ExpertDemonstration]) -> float:
        """
        Compute log-likelihood of demonstrations under current weights.
        
        Args:
            weights: Current weight vector
            demonstrations: Expert demonstrations
            
        Returns:
            Log-likelihood value
        """
        weights = weights / np.sum(weights)  # Normalize
        policy = self.compute_policy(weights)
        
        log_likelihood = 0.0
        
        for demo in demonstrations:
            for step in demo.trajectory:
                state = step['state']
                action = step['action']
                
                if state in policy:
                    action_idx = self.state_space.actions.index(action)
                    action_prob = policy[state][action_idx]
                    log_likelihood += np.log(max(action_prob, 1e-10))  # Avoid log(0)
        
        return log_likelihood
    
    def learn_weights(self, demonstrations: List[ExpertDemonstration], 
                     max_iterations: int = 100, 
                     convergence_threshold: float = 1e-4) -> Dict[str, Any]:
        """
        Learn optimal weights from expert demonstrations using gradient ascent.
        
        Args:
            demonstrations: List of expert demonstrations
            max_iterations: Maximum number of optimization iterations
            convergence_threshold: Convergence threshold for gradient norm
            
        Returns:
            Dictionary containing learned weights and optimization history
        """
        if not demonstrations:
            raise ValueError("No expert demonstrations provided")
        
        logger.info(f"Starting IRL weight learning with {len(demonstrations)} demonstrations")
        
        # Initialize optimization history
        history = {
            'weights': [],
            'log_likelihood': [],
            'gradient_norm': [],
            'iterations': 0
        }
        
        # Optimization objective (negative log-likelihood for minimization)
        def objective(weights):
            return -self.compute_log_likelihood(weights, demonstrations)
        
        # Gradient function
        def gradient(weights):
            return -self.compute_gradient(weights, demonstrations)
        
        # Constraints: weights must be positive and sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        ]
        bounds = [(0.01, 0.98) for _ in range(4)]  # Prevent weights from being exactly 0 or 1
        
        # Optimize using L-BFGS-B
        result = minimize(
            objective,
            self.weights,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iterations, 'ftol': convergence_threshold}
        )
        
        # Update learned weights
        if result.success:
            self.weights = result.x / np.sum(result.x)  # Ensure normalization
            logger.info(f"IRL converged after {result.nit} iterations")
        else:
            logger.warning(f"IRL optimization did not converge: {result.message}")
        
        # Record final results
        history['weights'] = self.weights.copy()
        history['log_likelihood'] = -result.fun
        history['gradient_norm'] = np.linalg.norm(result.jac)
        history['iterations'] = result.nit
        history['success'] = result.success
        
        return history

class IRLCalibrator:
    """
    High-level interface for IRL-based weight calibration in the CBR+STM framework.
    """
    
    def __init__(self, framework):
        """
        Initialize IRL calibrator with CBR+STM framework.
        
        Args:
            framework: CBR+STM framework instance
        """
        self.framework = framework
        self.irl = MaxEntIRL(
            framework.state_space,
            framework.state_space.actions,
            framework.heuristic
        )
        self.calibration_history = []
    
    def parse_expert_data(self, data: pd.DataFrame) -> List[ExpertDemonstration]:
        """
        Parse expert demonstration data from DataFrame.
        
        Expected columns:
        - trajectory_id: Identifier for each demonstration
        - step: Step number within trajectory
        - state: Current state
        - action: Action taken
        - next_state: Resulting state
        - maintenance_cost: Cost of maintenance action
        - downtime_cost: Cost of downtime
        - reliability_improvement: Reliability improvement achieved
        - urgency_factor: Urgency of the situation
        
        Args:
            data: DataFrame containing expert demonstration data
            
        Returns:
            List of ExpertDemonstration objects
        """
        demonstrations = []
        
        # Group by trajectory
        for traj_id, traj_group in data.groupby('trajectory_id'):
            trajectory = []
            
            for _, row in traj_group.sort_values('step').iterrows():
                step = {
                    'state': row['state'],
                    'action': row['action'],
                    'next_state': row['next_state'],
                    'cost': row.get('total_cost', 0)
                }
                trajectory.append(step)
            
            context = {
                'trajectory_id': traj_id,
                'equipment_type': traj_group.iloc[0].get('equipment_type', 'unknown'),
                'expert_id': traj_group.iloc[0].get('expert_id', 'unknown')
            }
            
            demonstrations.append(ExpertDemonstration(trajectory, context))
        
        return demonstrations
    
    def calibrate_from_demonstrations(self, demonstrations: List[ExpertDemonstration], 
                                    max_iterations: int = 100) -> Dict[str, Any]:
        """
        Calibrate weights using expert demonstrations.
        
        Args:
            demonstrations: List of expert demonstrations
            max_iterations: Maximum optimization iterations
            
        Returns:
            Calibration results including learned weights and metrics
        """
        logger.info("Starting IRL-based weight calibration")
        
        # Learn weights using IRL
        optimization_result = self.irl.learn_weights(demonstrations, max_iterations)
        
        # Update framework weights
        learned_weights = self.irl.weights.copy()
        self.framework.heuristic.weights = learned_weights
        
        # Compute calibration metrics
        metrics = self._compute_calibration_metrics(demonstrations, learned_weights)
        
        # Store calibration history
        calibration_record = {
            'timestamp': pd.Timestamp.now(),
            'weights': learned_weights,
            'num_demonstrations': len(demonstrations),
            'optimization_result': optimization_result,
            'metrics': metrics
        }
        self.calibration_history.append(calibration_record)
        
        logger.info(f"IRL calibration completed. Learned weights: {learned_weights}")
        
        return calibration_record
    
    def _compute_calibration_metrics(self, demonstrations: List[ExpertDemonstration], 
                                   weights: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics to evaluate calibration quality.
        
        Args:
            demonstrations: Expert demonstrations
            weights: Learned weights
            
        Returns:
            Dictionary of calibration metrics
        """
        # Compute log-likelihood of demonstrations under learned weights
        log_likelihood = self.irl.compute_log_likelihood(weights, demonstrations)
        
        # Compute action prediction accuracy
        policy = self.irl.compute_policy(weights)
        correct_predictions = 0
        total_predictions = 0
        
        for demo in demonstrations:
            for step in demo.trajectory:
                state = step['state']
                expert_action = step['action']
                
                if state in policy:
                    action_probs = policy[state]
                    predicted_action_idx = np.argmax(action_probs)
                    predicted_action = self.framework.state_space.actions[predicted_action_idx]
                    
                    if predicted_action == expert_action:
                        correct_predictions += 1
                    total_predictions += 1
        
        accuracy = correct_predictions / max(total_predictions, 1)
        
        # Compute weight entropy (measure of uncertainty)
        weight_entropy = -np.sum(weights * np.log(weights + 1e-10))
        
        return {
            'log_likelihood': log_likelihood,
            'action_accuracy': accuracy,
            'weight_entropy': weight_entropy,
            'total_demonstrations': len(demonstrations),
            'total_steps': sum(demo.length for demo in demonstrations)
        }
    
    def generate_synthetic_demonstrations(self, num_demos: int = 20) -> List[ExpertDemonstration]:
        """
        Generate synthetic expert demonstrations for testing IRL calibration.
        
        Args:
            num_demos: Number of demonstrations to generate
            
        Returns:
            List of synthetic expert demonstrations
        """
        demonstrations = []
        
        # Use current framework weights as "expert" behavior
        expert_weights = self.framework.heuristic.weights.copy()
        
        for demo_id in range(num_demos):
            # Random starting state (typically degraded)
            start_state_idx = np.random.randint(len(self.framework.state_space.states) // 2,
                                               len(self.framework.state_space.states))
            start_state = self.framework.state_space.states[start_state_idx]
            
            # Generate trajectory using expert policy
            trajectory = []
            current_state = start_state
            max_steps = 10
            
            for step in range(max_steps):
                # Choose action based on expert heuristic
                best_action = None
                best_score = float('inf')
                
                for action in self.framework.state_space.actions:
                    score = self.framework.heuristic.evaluate(current_state, action)
                    if score < best_score:
                        best_score = score
                        best_action = action
                
                # Add some noise to make it more realistic
                if np.random.random() < 0.1:  # 10% chance of suboptimal action
                    best_action = np.random.choice(self.framework.state_space.actions)
                
                next_state = self.framework.state_space.transition(current_state, best_action)
                
                step_data = {
                    'state': current_state,
                    'action': best_action,
                    'next_state': next_state,
                    'cost': best_score
                }
                trajectory.append(step_data)
                
                if next_state is None or next_state == current_state:
                    break
                current_state = next_state
            
            context = {
                'trajectory_id': f'synthetic_{demo_id}',
                'equipment_type': 'turbofan_engine',
                'expert_id': 'synthetic_expert'
            }
            
            demonstrations.append(ExpertDemonstration(trajectory, context))
        
        return demonstrations