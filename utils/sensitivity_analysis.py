"""
Sensitivity Analysis for CBR+STM Framework

This module implements sensitivity analysis for weight parameters as described
in the paper's practical calibration methods section.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
import itertools
from core.cbr_stm_framework import CBRSTMFramework

class SensitivityAnalyzer:
    """
    Sensitivity analysis for heuristic weight parameters.
    
    Implements parametric studies, Monte Carlo analysis, and threshold identification
    as described in the paper's sensitivity analysis methodology.
    """
    
    def __init__(self, framework: CBRSTMFramework):
        """
        Initialize sensitivity analyzer.
        
        Args:
            framework: CBR+STM framework instance
        """
        self.framework = framework
        self.base_weights = framework.heuristic.w1, framework.heuristic.w2, framework.heuristic.w3, framework.heuristic.w4
        
    def analyze_weight_sensitivity(self, base_weights: List[float], 
                                 perturbation_range: float = 0.2,
                                 num_samples: int = 500,
                                 start_state: str = None,
                                 goal_state: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive sensitivity analysis on weight parameters.
        
        Args:
            base_weights: Base weight configuration [w1, w2, w3, w4]
            perturbation_range: Perturbation range as fraction of base values
            num_samples: Number of Monte Carlo samples
            start_state: Start state for path analysis
            goal_state: Goal state for path analysis
            
        Returns:
            Comprehensive sensitivity analysis results
        """
        results = {
            'base_weights': base_weights,
            'perturbation_range': perturbation_range,
            'num_samples': num_samples
        }
        
        # 1. Parametric sensitivity study
        parametric_results = self._parametric_sensitivity_study(
            base_weights, perturbation_range, start_state, goal_state
        )
        results['parametric_analysis'] = parametric_results
        
        # 2. Monte Carlo sensitivity analysis
        monte_carlo_results = self._monte_carlo_sensitivity_analysis(
            base_weights, perturbation_range, num_samples, start_state, goal_state
        )
        results['monte_carlo_analysis'] = monte_carlo_results
        
        # 3. Critical threshold identification
        threshold_results = self._identify_critical_thresholds(
            base_weights, start_state, goal_state
        )
        results['threshold_analysis'] = threshold_results
        
        # 4. Calculate robustness metrics
        robustness_metrics = self._calculate_robustness_metrics(
            parametric_results, monte_carlo_results
        )
        results['robustness_metrics'] = robustness_metrics
        
        # 5. Generate recommendations
        recommendations = self._generate_deployment_recommendations(results)
        results['recommendations'] = recommendations
        
        # 6. Prepare data for visualization
        visualization_data = self._prepare_visualization_data(results)
        results.update(visualization_data)
        
        return results
    
    def _parametric_sensitivity_study(self, base_weights: List[float],
                                    perturbation_range: float,
                                    start_state: str = None,
                                    goal_state: str = None) -> Dict[str, Any]:
        """
        Perform parametric sensitivity study varying each weight individually.
        
        Args:
            base_weights: Base weight configuration
            perturbation_range: Perturbation range
            start_state: Start state
            goal_state: Goal state
            
        Returns:
            Parametric sensitivity results
        """
        weight_names = ['w1', 'w2', 'w3', 'w4']
        perturbation_steps = 21  # -100% to +100% in 10% increments
        
        results = {
            'weight_variations': {name: [] for name in weight_names},
            'cost_variations': {name: [] for name in weight_names},
            'path_variations': {name: [] for name in weight_names},
            'sensitivity_coefficients': {}
        }
        
        # Use default states if not provided
        if start_state is None:
            start_state = self.framework.state_space.states[0]
        if goal_state is None:
            goal_state = self.framework.state_space.states[-1]
        
        for i, weight_name in enumerate(weight_names):
            base_weight = base_weights[i]
            
            # Generate perturbation range
            min_perturbation = max(0.001, base_weight * (1 - perturbation_range))
            max_perturbation = base_weight * (1 + perturbation_range)
            
            weight_values = np.linspace(min_perturbation, max_perturbation, perturbation_steps)
            
            costs = []
            paths = []
            
            for weight_value in weight_values:
                # Create perturbed weights
                perturbed_weights = base_weights.copy()
                perturbed_weights[i] = weight_value
                
                # Update framework weights
                self.framework.heuristic.update_weights(perturbed_weights)
                
                # Execute algorithm
                try:
                    result = self.framework.execute_algorithm(
                        start_state=start_state,
                        goal_state=goal_state,
                        max_iterations=100
                    )
                    
                    if result['success']:
                        costs.append(result['total_cost'])
                        paths.append(len(result['path']))
                    else:
                        costs.append(float('inf'))  # Failed solution
                        paths.append(0)
                        
                except Exception:
                    costs.append(float('inf'))
                    paths.append(0)
            
            # Store results
            results['weight_variations'][weight_name] = weight_values.tolist()
            results['cost_variations'][weight_name] = costs
            results['path_variations'][weight_name] = paths
            
            # Calculate sensitivity coefficient (derivative approximation)
            finite_costs = [c for c in costs if c != float('inf')]
            if len(finite_costs) > 2:
                # Use central difference for sensitivity
                mid_idx = len(weight_values) // 2
                if mid_idx > 0 and mid_idx < len(costs) - 1:
                    if costs[mid_idx - 1] != float('inf') and costs[mid_idx + 1] != float('inf'):
                        sensitivity = (costs[mid_idx + 1] - costs[mid_idx - 1]) / (
                            weight_values[mid_idx + 1] - weight_values[mid_idx - 1]
                        )
                        results['sensitivity_coefficients'][weight_name] = sensitivity
                    else:
                        results['sensitivity_coefficients'][weight_name] = 0.0
                else:
                    results['sensitivity_coefficients'][weight_name] = 0.0
            else:
                results['sensitivity_coefficients'][weight_name] = 0.0
        
        # Restore original weights
        self.framework.heuristic.update_weights(base_weights)
        
        return results
    
    def _monte_carlo_sensitivity_analysis(self, base_weights: List[float],
                                        perturbation_range: float,
                                        num_samples: int,
                                        start_state: str = None,
                                        goal_state: str = None) -> Dict[str, Any]:
        """
        Perform Monte Carlo sensitivity analysis with weight uncertainty distributions.
        
        Args:
            base_weights: Base weight configuration
            perturbation_range: Perturbation range
            num_samples: Number of Monte Carlo samples
            start_state: Start state
            goal_state: Goal state
            
        Returns:
            Monte Carlo sensitivity results
        """
        # Use default states if not provided
        if start_state is None:
            start_state = self.framework.state_space.states[0]
        if goal_state is None:
            goal_state = self.framework.state_space.states[-1]
        
        # Generate random weight samples
        weight_samples = []
        cost_samples = []
        path_samples = []
        success_samples = []
        
        for _ in range(num_samples):
            # Generate perturbed weights using normal distribution
            perturbed_weights = []
            for base_weight in base_weights:
                # Normal distribution with std = perturbation_range * base_weight / 3
                # (99.7% of samples within ±perturbation_range)
                std_dev = base_weight * perturbation_range / 3
                perturbed_weight = max(0.001, np.random.normal(base_weight, std_dev))
                perturbed_weights.append(perturbed_weight)
            
            weight_samples.append(perturbed_weights)
            
            # Update framework and execute
            self.framework.heuristic.update_weights(perturbed_weights)
            
            try:
                result = self.framework.execute_algorithm(
                    start_state=start_state,
                    goal_state=goal_state,
                    max_iterations=100
                )
                
                if result['success']:
                    cost_samples.append(result['total_cost'])
                    path_samples.append(len(result['path']))
                    success_samples.append(1)
                else:
                    cost_samples.append(float('inf'))
                    path_samples.append(0)
                    success_samples.append(0)
                    
            except Exception:
                cost_samples.append(float('inf'))
                path_samples.append(0)
                success_samples.append(0)
        
        # Restore original weights
        self.framework.heuristic.update_weights(base_weights)
        
        # Analyze results
        finite_costs = [c for c in cost_samples if c != float('inf')]
        success_rate = np.mean(success_samples)
        
        results = {
            'weight_samples': weight_samples,
            'cost_samples': cost_samples,
            'path_samples': path_samples,
            'success_samples': success_samples,
            'success_rate': success_rate,
            'cost_statistics': {},
            'correlation_analysis': {}
        }
        
        if finite_costs:
            results['cost_statistics'] = {
                'mean': np.mean(finite_costs),
                'std': np.std(finite_costs),
                'min': np.min(finite_costs),
                'max': np.max(finite_costs),
                'percentiles': {
                    '5th': np.percentile(finite_costs, 5),
                    '25th': np.percentile(finite_costs, 25),
                    '50th': np.percentile(finite_costs, 50),
                    '75th': np.percentile(finite_costs, 75),
                    '95th': np.percentile(finite_costs, 95)
                }
            }
            
            # Correlation analysis between weights and costs
            weight_array = np.array(weight_samples)
            finite_cost_indices = [i for i, c in enumerate(cost_samples) if c != float('inf')]
            
            if len(finite_cost_indices) > 10:  # Need sufficient samples for correlation
                finite_weights = weight_array[finite_cost_indices]
                finite_costs_array = np.array([cost_samples[i] for i in finite_cost_indices])
                
                weight_names = ['w1', 'w2', 'w3', 'w4']
                for i, weight_name in enumerate(weight_names):
                    correlation, p_value = stats.pearsonr(finite_weights[:, i], finite_costs_array)
                    results['correlation_analysis'][weight_name] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
        
        return results
    
    def _identify_critical_thresholds(self, base_weights: List[float],
                                    start_state: str = None,
                                    goal_state: str = None) -> Dict[str, Any]:
        """
        Identify critical threshold values where optimal actions change.
        
        Args:
            base_weights: Base weight configuration
            start_state: Start state
            goal_state: Goal state
            
        Returns:
            Critical threshold analysis results
        """
        # Use default states if not provided
        if start_state is None:
            start_state = self.framework.state_space.states[0]
        if goal_state is None:
            goal_state = self.framework.state_space.states[-1]
        
        weight_names = ['w1', 'w2', 'w3', 'w4']
        thresholds = {}
        
        # Get baseline solution
        self.framework.heuristic.update_weights(base_weights)
        baseline_result = self.framework.execute_algorithm(
            start_state=start_state,
            goal_state=goal_state,
            max_iterations=100
        )
        
        if not baseline_result['success']:
            return {'error': 'Could not find baseline solution'}
        
        baseline_path = [step['action'] for step in baseline_result['path'] if 'action' in step]
        
        for i, weight_name in enumerate(weight_names):
            base_weight = base_weights[i]
            
            # Binary search for threshold where path changes
            threshold_found = False
            lower_bound = 0.001
            upper_bound = base_weight * 5  # Search up to 5x base weight
            
            threshold_value = base_weight
            
            for _ in range(20):  # Maximum 20 iterations for binary search
                test_weight = (lower_bound + upper_bound) / 2
                
                # Test this weight
                test_weights = base_weights.copy()
                test_weights[i] = test_weight
                
                self.framework.heuristic.update_weights(test_weights)
                test_result = self.framework.execute_algorithm(
                    start_state=start_state,
                    goal_state=goal_state,
                    max_iterations=100
                )
                
                if test_result['success']:
                    test_path = [step['action'] for step in test_result['path'] if 'action' in step]
                    
                    # Check if path changed significantly
                    if self._paths_significantly_different(baseline_path, test_path):
                        threshold_value = test_weight
                        threshold_found = True
                        upper_bound = test_weight
                    else:
                        lower_bound = test_weight
                else:
                    upper_bound = test_weight
                
                # Convergence check
                if abs(upper_bound - lower_bound) < base_weight * 0.01:
                    break
            
            thresholds[weight_name] = {
                'threshold_value': threshold_value,
                'base_value': base_weight,
                'threshold_ratio': threshold_value / base_weight,
                'threshold_found': threshold_found
            }
        
        # Restore original weights
        self.framework.heuristic.update_weights(base_weights)
        
        return {
            'thresholds': thresholds,
            'baseline_path': baseline_path,
            'analysis_bounds': {
                'lower_bound': 0.001,
                'upper_bound_multiplier': 5
            }
        }
    
    def _paths_significantly_different(self, path1: List[str], path2: List[str]) -> bool:
        """
        Check if two solution paths are significantly different.
        
        Args:
            path1: First path (list of actions)
            path2: Second path (list of actions)
            
        Returns:
            True if paths are significantly different
        """
        if len(path1) != len(path2):
            return True
        
        # Count different actions
        differences = sum(1 for a1, a2 in zip(path1, path2) if a1 != a2)
        
        # Consider significant if >30% of actions are different
        if len(path1) > 0:
            difference_ratio = differences / len(path1)
            return difference_ratio > 0.3
        
        return False
    
    def _calculate_robustness_metrics(self, parametric_results: Dict,
                                    monte_carlo_results: Dict) -> Dict[str, float]:
        """
        Calculate robustness metrics for deployment.
        
        Args:
            parametric_results: Parametric analysis results
            monte_carlo_results: Monte Carlo analysis results
            
        Returns:
            Dictionary of robustness metrics
        """
        metrics = {}
        
        # Cost stability (coefficient of variation)
        if 'cost_statistics' in monte_carlo_results and monte_carlo_results['cost_statistics']:
            cost_stats = monte_carlo_results['cost_statistics']
            if cost_stats['mean'] > 0:
                metrics['cost_stability'] = 1.0 - (cost_stats['std'] / cost_stats['mean'])
            else:
                metrics['cost_stability'] = 0.0
        else:
            metrics['cost_stability'] = 0.0
        
        # Path consistency (success rate)
        metrics['path_consistency'] = monte_carlo_results.get('success_rate', 0.0)
        
        # Sensitivity magnitude (average absolute sensitivity)
        sensitivity_coeffs = parametric_results.get('sensitivity_coefficients', {})
        if sensitivity_coeffs:
            avg_sensitivity = np.mean([abs(coeff) for coeff in sensitivity_coeffs.values()])
            metrics['sensitivity_magnitude'] = avg_sensitivity
        else:
            metrics['sensitivity_magnitude'] = 0.0
        
        # Critical threshold proximity
        weight_names = ['w1', 'w2', 'w3', 'w4']
        threshold_proximities = []
        
        for weight_name in weight_names:
            base_variations = parametric_results['weight_variations'].get(weight_name, [])
            if base_variations:
                base_value = base_variations[len(base_variations) // 2]  # Middle value
                # Simple threshold estimate based on cost variation
                cost_variations = parametric_results['cost_variations'].get(weight_name, [])
                if cost_variations:
                    finite_costs = [c for c in cost_variations if c != float('inf')]
                    if len(finite_costs) > 1:
                        cost_range = max(finite_costs) - min(finite_costs)
                        if cost_range > 0:
                            # Estimate threshold as weight value where cost changes significantly
                            threshold_proximity = min(2.0, cost_range / base_value)
                            threshold_proximities.append(threshold_proximity)
        
        if threshold_proximities:
            metrics['critical_threshold'] = np.mean(threshold_proximities)
        else:
            metrics['critical_threshold'] = 1.0
        
        # Overall robustness score (weighted combination)
        metrics['overall_robustness'] = (
            0.4 * metrics['cost_stability'] +
            0.3 * metrics['path_consistency'] +
            0.2 * (1.0 - min(1.0, metrics['sensitivity_magnitude'] / 10.0)) +
            0.1 * min(1.0, metrics['critical_threshold'])
        )
        
        return metrics
    
    def _generate_deployment_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate deployment recommendations based on sensitivity analysis.
        
        Args:
            results: Complete sensitivity analysis results
            
        Returns:
            List of deployment recommendations
        """
        recommendations = []
        
        robustness = results.get('robustness_metrics', {})
        
        # Cost stability recommendations
        cost_stability = robustness.get('cost_stability', 0.0)
        if cost_stability < 0.5:
            recommendations.append(
                "⚠️ Low cost stability detected. Consider implementing cost bounds or "
                "additional validation checks before deployment."
            )
        elif cost_stability > 0.8:
            recommendations.append(
                "✅ Good cost stability. The system shows robust performance under weight variations."
            )
        
        # Path consistency recommendations
        path_consistency = robustness.get('path_consistency', 0.0)
        if path_consistency < 0.7:
            recommendations.append(
                "⚠️ Low path consistency. Consider tightening weight bounds or "
                "implementing fallback strategies for failed solutions."
            )
        elif path_consistency > 0.9:
            recommendations.append(
                "✅ High path consistency. The algorithm reliably finds solutions across weight variations."
            )
        
        # Sensitivity magnitude recommendations
        sensitivity_magnitude = robustness.get('sensitivity_magnitude', 0.0)
        if sensitivity_magnitude > 10.0:
            recommendations.append(
                "⚠️ High sensitivity to weight changes. Implement precise weight calibration "
                "and regular recalibration procedures."
            )
        elif sensitivity_magnitude < 1.0:
            recommendations.append(
                "✅ Low sensitivity to weight changes. The system is robust to weight uncertainties."
            )
        
        # Correlation-based recommendations
        if 'monte_carlo_analysis' in results and 'correlation_analysis' in results['monte_carlo_analysis']:
            correlations = results['monte_carlo_analysis']['correlation_analysis']
            
            high_correlation_weights = []
            for weight_name, corr_data in correlations.items():
                if corr_data.get('significant', False) and abs(corr_data.get('correlation', 0)) > 0.7:
                    high_correlation_weights.append(weight_name)
            
            if high_correlation_weights:
                recommendations.append(
                    f"📊 High correlation detected for weights: {', '.join(high_correlation_weights)}. "
                    "Monitor these weights closely during operation."
                )
        
        # Overall robustness recommendation
        overall_robustness = robustness.get('overall_robustness', 0.0)
        if overall_robustness > 0.8:
            recommendations.append(
                "🎯 Excellent overall robustness. The system is ready for production deployment "
                "with current weight configuration."
            )
        elif overall_robustness > 0.6:
            recommendations.append(
                "⚡ Good robustness with minor concerns. Consider addressing specific issues "
                "mentioned above before full deployment."
            )
        else:
            recommendations.append(
                "🔄 Robustness concerns detected. Recommend weight recalibration and "
                "additional testing before production deployment."
            )
        
        # Add specific threshold recommendations
        if 'threshold_analysis' in results and 'thresholds' in results['threshold_analysis']:
            thresholds = results['threshold_analysis']['thresholds']
            unstable_weights = []
            
            for weight_name, threshold_data in thresholds.items():
                if threshold_data.get('threshold_found', False):
                    ratio = threshold_data.get('threshold_ratio', 1.0)
                    if ratio < 1.5:  # Threshold too close to base value
                        unstable_weights.append(weight_name)
            
            if unstable_weights:
                recommendations.append(
                    f"⚠️ Critical thresholds detected for: {', '.join(unstable_weights)}. "
                    "Small changes in these weights may significantly affect behavior."
                )
        
        if not recommendations:
            recommendations.append(
                "✅ No specific concerns detected. System appears stable for deployment."
            )
        
        return recommendations
    
    def _prepare_visualization_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data structures for visualization components.
        
        Args:
            results: Complete sensitivity analysis results
            
        Returns:
            Visualization-ready data structures
        """
        viz_data = {}
        
        # Prepare parametric variation data for plotting
        if 'parametric_analysis' in results:
            parametric = results['parametric_analysis']
            
            viz_data['weight_variations'] = []
            viz_data['cost_variations'] = []
            
            weight_names = ['w1', 'w2', 'w3', 'w4']
            for weight_name in weight_names:
                if weight_name in parametric['weight_variations']:
                    viz_data['weight_variations'].append(parametric['weight_variations'][weight_name])
                    viz_data['cost_variations'].append(parametric['cost_variations'][weight_name])
        
        # Prepare Monte Carlo scatter plot data
        if 'monte_carlo_analysis' in results:
            monte_carlo = results['monte_carlo_analysis']
            
            viz_data['monte_carlo_weights'] = monte_carlo.get('weight_samples', [])
            viz_data['monte_carlo_costs'] = monte_carlo.get('cost_samples', [])
            viz_data['monte_carlo_success'] = monte_carlo.get('success_samples', [])
        
        return viz_data
    
    def export_sensitivity_report(self, results: Dict[str, Any], filepath: str):
        """
        Export comprehensive sensitivity analysis report.
        
        Args:
            results: Sensitivity analysis results
            filepath: Output file path
        """
        import json
        from datetime import datetime
        
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'CBR+STM Weight Sensitivity Analysis',
                'base_weights': results['base_weights'],
                'perturbation_range': results['perturbation_range'],
                'num_samples': results['num_samples']
            },
            'results': results,
            'summary': {
                'robustness_score': results['robustness_metrics']['overall_robustness'],
                'primary_concerns': [],
                'deployment_ready': results['robustness_metrics']['overall_robustness'] > 0.7
            }
        }
        
        # Extract primary concerns from recommendations
        recommendations = results.get('recommendations', [])
        concerns = [rec for rec in recommendations if '⚠️' in rec or '🔄' in rec]
        report['summary']['primary_concerns'] = concerns
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
