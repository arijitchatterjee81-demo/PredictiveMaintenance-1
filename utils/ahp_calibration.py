"""
Analytic Hierarchy Process (AHP) Weight Calibration

This module implements AHP-based weight calibration for the multi-objective
heuristic function as mentioned in the paper's practical calibration methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import streamlit as st

class AHPCalibrator:
    """
    AHP-based calibration for heuristic weight coefficients.
    
    Implements expert elicitation using pairwise comparisons to derive
    relative importance weights as described in the paper.
    """
    
    def __init__(self):
        """Initialize AHP calibrator with criteria definitions."""
        self.criteria = [
            'Maintenance Cost',
            'Downtime Cost', 
            'Reliability Improvement',
            'Urgency Factor'
        ]
        
        self.criteria_codes = ['w1', 'w2', 'w3', 'w4']
        
        # AHP scale definitions
        self.ahp_scale = {
            1: 'Equal importance',
            3: 'Moderate importance', 
            5: 'Strong importance',
            7: 'Very strong importance',
            9: 'Extreme importance',
            2: 'Between equal and moderate',
            4: 'Between moderate and strong',
            6: 'Between strong and very strong',
            8: 'Between very strong and extreme'
        }
        
        # Consistency thresholds
        self.consistency_thresholds = {
            2: 0.0,
            3: 0.05,
            4: 0.09,
            5: 0.12,
            6: 0.15,
            7: 0.18,
            8: 0.20,
            9: 0.22,
            10: 0.24
        }
    
    def create_pairwise_comparison_matrix(self, comparisons: Dict[Tuple[int, int], float]) -> np.ndarray:
        """
        Create pairwise comparison matrix from expert judgments.
        
        Args:
            comparisons: Dictionary mapping (i,j) pairs to comparison values
            
        Returns:
            Pairwise comparison matrix
        """
        n = len(self.criteria)
        matrix = np.ones((n, n))
        
        # Fill upper triangle with comparisons
        for (i, j), value in comparisons.items():
            if i < j:
                matrix[i, j] = value
                matrix[j, i] = 1.0 / value  # Reciprocal relationship
        
        return matrix
    
    def calculate_weights(self, comparison_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate priority weights from comparison matrix using eigenvector method.
        
        Args:
            comparison_matrix: Pairwise comparison matrix
            
        Returns:
            Tuple of (weights, consistency_ratio)
        """
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
        
        # Find principal eigenvalue and corresponding eigenvector
        max_eigenvalue_idx = np.argmax(eigenvalues.real)
        max_eigenvalue = eigenvalues[max_eigenvalue_idx].real
        principal_eigenvector = eigenvectors[:, max_eigenvalue_idx].real
        
        # Normalize weights (ensure positive and sum to 1)
        weights = np.abs(principal_eigenvector)
        weights = weights / np.sum(weights)
        
        # Calculate consistency ratio
        consistency_ratio = self._calculate_consistency_ratio(comparison_matrix, max_eigenvalue)
        
        return weights, consistency_ratio
    
    def _calculate_consistency_ratio(self, matrix: np.ndarray, max_eigenvalue: float) -> float:
        """
        Calculate consistency ratio for AHP matrix.
        
        Args:
            matrix: Pairwise comparison matrix
            max_eigenvalue: Maximum eigenvalue
            
        Returns:
            Consistency ratio
        """
        n = matrix.shape[0]
        
        if n <= 2:
            return 0.0  # Perfect consistency for n <= 2
        
        # Calculate Consistency Index (CI)
        ci = (max_eigenvalue - n) / (n - 1)
        
        # Random Index (RI) values for different matrix sizes
        ri_values = {
            3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35,
            8: 1.40, 9: 1.45, 10: 1.49, 11: 1.52, 12: 1.54
        }
        
        ri = ri_values.get(n, 1.49)  # Default RI for large matrices
        
        # Calculate Consistency Ratio (CR)
        cr = ci / ri if ri > 0 else 0.0
        
        return cr
    
    def interactive_ahp_session(self) -> Dict[str, Any]:
        """
        Create interactive AHP session using Streamlit interface.
        
        Returns:
            Dictionary with calibrated weights and session results
        """
        st.subheader("AHP Weight Calibration Session")
        
        st.markdown("""
        **Instructions:** Compare the relative importance of each pair of criteria.
        Use the AHP scale where:
        - 1 = Equal importance
        - 3 = Moderate importance of left over right
        - 5 = Strong importance of left over right  
        - 7 = Very strong importance of left over right
        - 9 = Extreme importance of left over right
        - 2, 4, 6, 8 = Intermediate values
        """)
        
        # Collect pairwise comparisons
        comparisons = {}
        n = len(self.criteria)
        
        st.markdown("### Pairwise Comparisons")
        
        comparison_cols = st.columns(2)
        col_idx = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                with comparison_cols[col_idx % 2]:
                    criterion_i = self.criteria[i]
                    criterion_j = self.criteria[j]
                    
                    comparison_key = f"{criterion_i} vs {criterion_j}"
                    
                    # Create comparison slider
                    comparison_value = st.select_slider(
                        f"**{criterion_i}** vs **{criterion_j}**",
                        options=[1/9, 1/7, 1/5, 1/3, 1, 3, 5, 7, 9],
                        value=1,
                        format_func=lambda x: f"{x:.2f}" if x < 1 else f"{int(x)}",
                        key=f"ahp_comparison_{i}_{j}"
                    )
                    
                    if comparison_value < 1:
                        st.caption(f"{criterion_j} is {1/comparison_value:.0f}x more important")
                    elif comparison_value > 1:
                        st.caption(f"{criterion_i} is {comparison_value:.0f}x more important")
                    else:
                        st.caption("Equal importance")
                    
                    comparisons[(i, j)] = comparison_value
                
                col_idx += 1
        
        # Calculate weights when button is pressed
        if st.button("Calculate AHP Weights", type="primary"):
            return self._process_ahp_results(comparisons)
        
        return None
    
    def _process_ahp_results(self, comparisons: Dict[Tuple[int, int], float]) -> Dict[str, Any]:
        """
        Process AHP comparison results and display results.
        
        Args:
            comparisons: Pairwise comparison values
            
        Returns:
            Results dictionary with weights and metrics
        """
        # Create comparison matrix
        matrix = self.create_pairwise_comparison_matrix(comparisons)
        
        # Calculate weights and consistency
        weights, consistency_ratio = self.calculate_weights(matrix)
        
        # Display results
        st.subheader("AHP Calibration Results")
        
        # Weights table
        weights_df = pd.DataFrame({
            'Criterion': self.criteria,
            'Weight Code': self.criteria_codes,
            'Calculated Weight': weights,
            'Percentage': weights * 100
        })
        
        st.dataframe(weights_df.style.format({
            'Calculated Weight': '{:.4f}',
            'Percentage': '{:.1f}%'
        }))
        
        # Consistency check
        st.subheader("Consistency Analysis")
        
        consistency_threshold = 0.1  # Standard AHP threshold
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Consistency Ratio", f"{consistency_ratio:.3f}")
        with col2:
            st.metric("Threshold", f"{consistency_threshold:.1f}")
        with col3:
            if consistency_ratio <= consistency_threshold:
                st.success("✅ Consistent")
            else:
                st.error("❌ Inconsistent")
        
        if consistency_ratio > consistency_threshold:
            st.warning(f"""
            **Warning:** The consistency ratio ({consistency_ratio:.3f}) exceeds the recommended 
            threshold ({consistency_threshold}). Consider revising your pairwise comparisons 
            to improve consistency.
            """)
            
            # Provide suggestions for improvement
            self._suggest_consistency_improvements(matrix, comparisons)
        
        # Visualization
        self._visualize_ahp_results(weights_df, matrix)
        
        return {
            'weights': weights.tolist(),
            'consistency_ratio': consistency_ratio,
            'is_consistent': consistency_ratio <= consistency_threshold,
            'comparison_matrix': matrix.tolist(),
            'weights_df': weights_df
        }
    
    def _suggest_consistency_improvements(self, matrix: np.ndarray, 
                                        comparisons: Dict[Tuple[int, int], float]):
        """
        Suggest improvements for consistency.
        
        Args:
            matrix: Comparison matrix
            comparisons: Original comparisons
        """
        st.subheader("Consistency Improvement Suggestions")
        
        # Find most inconsistent comparisons
        n = matrix.shape[0]
        inconsistencies = []
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(n):
                    if k != i and k != j:
                        # Check transitivity: if A > B and B > C, then A should > C
                        expected = matrix[i, k] * matrix[k, j]
                        actual = matrix[i, j]
                        error = abs(np.log(expected) - np.log(actual))
                        
                        if error > 0.5:  # Threshold for significant inconsistency
                            inconsistencies.append({
                                'comparison': f"{self.criteria[i]} vs {self.criteria[j]}",
                                'current_value': actual,
                                'suggested_value': expected,
                                'error': error
                            })
        
        if inconsistencies:
            # Sort by error magnitude
            inconsistencies.sort(key=lambda x: x['error'], reverse=True)
            
            st.markdown("**Most inconsistent comparisons:**")
            
            for inc in inconsistencies[:3]:  # Show top 3
                st.write(f"- **{inc['comparison']}**: "
                        f"Current = {inc['current_value']:.2f}, "
                        f"Suggested = {inc['suggested_value']:.2f}")
    
    def _visualize_ahp_results(self, weights_df: pd.DataFrame, matrix: np.ndarray):
        """
        Create visualizations for AHP results.
        
        Args:
            weights_df: DataFrame with calculated weights
            matrix: Comparison matrix
        """
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Weights bar chart
        fig_weights = px.bar(
            weights_df, 
            x='Criterion', 
            y='Calculated Weight',
            title='AHP-Calibrated Weights',
            color='Calculated Weight',
            color_continuous_scale='viridis'
        )
        fig_weights.update_layout(showlegend=False)
        st.plotly_chart(fig_weights, use_container_width=True)
        
        # Comparison matrix heatmap
        fig_matrix = go.Figure(data=go.Heatmap(
            z=matrix,
            x=self.criteria,
            y=self.criteria,
            colorscale='RdYlBu_r',
            text=np.round(matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Comparison Value")
        ))
        
        fig_matrix.update_layout(
            title='Pairwise Comparison Matrix',
            xaxis_title='Criteria',
            yaxis_title='Criteria'
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    def get_default_weights(self) -> List[float]:
        """
        Get default AHP-calibrated weights based on typical maintenance priorities.
        
        Returns:
            List of default weights [w1, w2, w3, w4]
        """
        # Default expert judgment matrix (moderate preferences)
        default_comparisons = {
            (0, 1): 1/2,    # Maintenance cost vs Downtime cost (downtime slightly more important)
            (0, 2): 3,      # Maintenance cost vs Reliability (maintenance more important) 
            (0, 3): 1/3,    # Maintenance cost vs Urgency (urgency more important)
            (1, 2): 5,      # Downtime vs Reliability (downtime much more important)
            (1, 3): 1/2,    # Downtime vs Urgency (urgency slightly more important)
            (2, 3): 1/7     # Reliability vs Urgency (urgency much more important)
        }
        
        matrix = self.create_pairwise_comparison_matrix(default_comparisons)
        weights, _ = self.calculate_weights(matrix)
        
        return weights.tolist()
    
    def save_calibration_session(self, results: Dict[str, Any], filename: str):
        """
        Save AHP calibration session results.
        
        Args:
            results: Calibration results
            filename: Output filename
        """
        import json
        from datetime import datetime
        
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'method': 'AHP',
            'criteria': self.criteria,
            'weights': results['weights'],
            'consistency_ratio': results['consistency_ratio'],
            'is_consistent': results['is_consistent'],
            'comparison_matrix': results['comparison_matrix']
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def load_calibration_session(self, filename: str) -> Dict[str, Any]:
        """
        Load saved AHP calibration session.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded calibration results
        """
        import json
        
        with open(filename, 'r') as f:
            return json.load(f)
