"""
Visualization Components for CBR+STM Framework

This module provides visualization utilities for displaying framework results,
sensor data trends, solution paths, and analysis outputs.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional
import streamlit as st

class VisualizationComponents:
    """
    Visualization components for CBR+STM framework results and analysis.
    """
    
    def __init__(self):
        """Initialize visualization components with default styling."""
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'neutral': '#8c564b'
        }
        
        self.layout_defaults = {
            'font_family': 'Arial, sans-serif',
            'font_size': 12,
            'margin': dict(l=50, r=50, t=80, b=50),
            'height': 400
        }
    
    def plot_sensor_trends(self, data: pd.DataFrame, max_sensors: int = 6) -> go.Figure:
        """
        Plot sensor data trends over time for multiple units.
        
        Args:
            data: DataFrame with sensor data
            max_sensors: Maximum number of sensors to display
            
        Returns:
            Plotly figure with sensor trends
        """
        # Get sensor columns
        sensor_cols = [col for col in data.columns if col.startswith('sensor_')][:max_sensors]
        
        if not sensor_cols:
            # Create empty plot if no sensor data
            fig = go.Figure()
            fig.add_annotation(
                text="No sensor data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Create subplots
        rows = min(3, len(sensor_cols))
        cols = min(2, (len(sensor_cols) + rows - 1) // rows)
        
        subplot_titles = [f"Sensor {col.split('_')[1]}" for col in sensor_cols]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Sample units to avoid overcrowding
        sample_units = data['unit'].unique()[:5] if 'unit' in data.columns else [1]
        
        for i, sensor_col in enumerate(sensor_cols):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            for j, unit_id in enumerate(sample_units):
                if 'unit' in data.columns:
                    unit_data = data[data['unit'] == unit_id].sort_values('cycle')
                    x_values = unit_data['cycle'] if 'cycle' in data.columns else range(len(unit_data))
                    y_values = unit_data[sensor_col]
                    name = f"Unit {unit_id}"
                else:
                    x_values = range(len(data))
                    y_values = data[sensor_col]
                    name = "Data"
                
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode='lines',
                        name=name,
                        legendgroup=f"unit_{unit_id}",
                        showlegend=(i == 0),  # Only show legend for first subplot
                        line=dict(color=self.color_palette['primary'] if j == 0 else None)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Sensor Data Trends",
            height=300 * rows,
            **self.layout_defaults
        )
        
        return fig
    
    def plot_solution_path(self, path: List[Dict[str, Any]]) -> go.Figure:
        """
        Visualize the solution path from CBR+STM algorithm.
        
        Args:
            path: Solution path with states, actions, and costs
            
        Returns:
            Plotly figure showing the solution path
        """
        if not path:
            fig = go.Figure()
            fig.add_annotation(
                text="No solution path available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Extract path data
        steps = list(range(len(path)))
        states = [step.get('state', f'Step_{i}') for i, step in enumerate(path)]
        actions = [step.get('action', 'N/A') for step in path]
        costs = [step.get('cost', 0) for step in path]
        heuristics = [step.get('heuristic', 0) for step in path]
        total_costs = [step.get('total_cost', 0) for step in path]
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Solution Path', 'Step Costs', 'Heuristic Values', 'Cumulative Cost'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Solution path as connected states
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=states,
                mode='lines+markers',
                name='Path',
                line=dict(color=self.color_palette['primary'], width=3),
                marker=dict(size=8, color=self.color_palette['primary']),
                text=[f"Action: {action}" for action in actions],
                hovertemplate="<b>Step %{x}</b><br>State: %{y}<br>%{text}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # 2. Step costs
        fig.add_trace(
            go.Bar(
                x=steps[:-1],  # Exclude last step (goal state)
                y=costs[:-1],
                name='Step Cost',
                marker_color=self.color_palette['secondary'],
                text=[f"{cost:.2f}" for cost in costs[:-1]],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Heuristic values
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=heuristics,
                mode='lines+markers',
                name='Heuristic',
                line=dict(color=self.color_palette['success'], width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # 4. Cumulative cost
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=total_costs,
                mode='lines+markers',
                name='Total Cost',
                line=dict(color=self.color_palette['warning'], width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(self.color_palette['warning'])) + [0.3])}"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="CBR+STM Solution Path Analysis",
            height=600,
            showlegend=False,
            **self.layout_defaults
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Step", row=1, col=1)
        fig.update_xaxes(title_text="Step", row=1, col=2)
        fig.update_xaxes(title_text="Step", row=2, col=1)
        fig.update_xaxes(title_text="Step", row=2, col=2)
        
        fig.update_yaxes(title_text="State", row=1, col=1)
        fig.update_yaxes(title_text="Cost", row=1, col=2)
        fig.update_yaxes(title_text="Heuristic Value", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Cost", row=2, col=2)
        
        return fig
    
    def plot_state_transition_graph(self, state_space_data: Dict[str, Any]) -> go.Figure:
        """
        Visualize state transition graph.
        
        Args:
            state_space_data: State space visualization data
            
        Returns:
            Plotly figure with state transition graph
        """
        nodes = state_space_data.get('nodes', [])
        edges = state_space_data.get('edges', [])
        
        if not nodes:
            fig = go.Figure()
            fig.add_annotation(
                text="No state space data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Create NetworkX graph for layout
        G = nx.DiGraph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node['id'], **node)
        
        # Add edges
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # Calculate layout
        try:
            pos = nx.spring_layout(G, k=3, iterations=50)
        except:
            # Fallback to circular layout
            pos = nx.circular_layout(G)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in edges:
            x0, y0 = pos[edge['source']]
            x1, y1 = pos[edge['target']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{edge['source']} → {edge['target']}: {edge.get('action', 'Unknown')}")
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        for node in nodes:
            x, y = pos[node['id']]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node['label'])
            
            # Color based on health score
            health_score = node.get('health_score', 0.5)
            if health_score > 0.7:
                node_colors.append(self.color_palette['success'])
            elif health_score > 0.4:
                node_colors.append(self.color_palette['secondary'])
            else:
                node_colors.append(self.color_palette['warning'])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color=self.color_palette['neutral']),
            hoverinfo='none',
            mode='lines',
            name='Transitions'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=node_text,
            textposition='middle center',
            textfont=dict(size=10, color='white'),
            hoverinfo='text',
            hovertext=[f"State: {node['label']}<br>Health: {node.get('health_score', 0):.2f}" 
                      for node in nodes],
            name='States'
        ))
        
        fig.update_layout(
            title="State Transition Graph",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            **self.layout_defaults
        )
        
        return fig
    
    def plot_case_similarity_matrix(self, similarity_data: Dict[str, Any]) -> go.Figure:
        """
        Plot case similarity matrix heatmap.
        
        Args:
            similarity_data: Similarity scores between cases
            
        Returns:
            Plotly figure with similarity heatmap
        """
        # Extract similarity matrix (this would come from case base analysis)
        if 'similarity_matrix' not in similarity_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No similarity data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        matrix = similarity_data['similarity_matrix']
        case_labels = similarity_data.get('case_labels', [f"Case {i+1}" for i in range(len(matrix))])
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=case_labels,
            y=case_labels,
            colorscale='viridis',
            colorbar=dict(title="Similarity Score"),
            hoverongaps=False,
            text=np.round(matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Case Similarity Matrix",
            xaxis_title="Cases",
            yaxis_title="Cases",
            **self.layout_defaults
        )
        
        return fig
    
    def plot_heuristic_breakdown(self, heuristic_data: Dict[str, float]) -> go.Figure:
        """
        Plot breakdown of heuristic components.
        
        Args:
            heuristic_data: Heuristic component values
            
        Returns:
            Plotly figure with heuristic breakdown
        """
        components = [
            'Maintenance Cost',
            'Downtime Cost', 
            'Reliability Improvement',
            'Urgency Factor'
        ]
        
        values = [
            heuristic_data.get('weighted_maintenance', 0),
            heuristic_data.get('weighted_downtime', 0),
            abs(heuristic_data.get('weighted_reliability', 0)),  # Take absolute for display
            heuristic_data.get('weighted_urgency', 0)
        ]
        
        colors = [
            self.color_palette['primary'],
            self.color_palette['secondary'],
            self.color_palette['success'],
            self.color_palette['warning']
        ]
        
        # Create side-by-side bar and pie charts
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=("Component Values", "Relative Contribution")
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=components,
                y=values,
                marker_color=colors,
                text=[f"{v:.2f}" for v in values],
                textposition='auto',
                name='Components'
            ),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=components,
                values=[abs(v) for v in values],  # Use absolute values for pie chart
                marker_colors=colors,
                textinfo='label+percent',
                name='Distribution'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Multi-Objective Heuristic Breakdown",
            showlegend=False,
            **self.layout_defaults
        )
        
        return fig
    
    def plot_rul_prediction(self, rul_data: Dict[str, Any]) -> go.Figure:
        """
        Plot Remaining Useful Life predictions and trends.
        
        Args:
            rul_data: RUL prediction data
            
        Returns:
            Plotly figure with RUL analysis
        """
        if 'cycles' not in rul_data or 'rul_values' not in rul_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No RUL data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        cycles = rul_data['cycles']
        rul_values = rul_data['rul_values']
        
        fig = go.Figure()
        
        # RUL trend line
        fig.add_trace(go.Scatter(
            x=cycles,
            y=rul_values,
            mode='lines+markers',
            name='RUL Prediction',
            line=dict(color=self.color_palette['primary'], width=3),
            marker=dict(size=6)
        ))
        
        # Add confidence intervals if available
        if 'rul_upper' in rul_data and 'rul_lower' in rul_data:
            fig.add_trace(go.Scatter(
                x=cycles + cycles[::-1],
                y=rul_data['rul_upper'] + rul_data['rul_lower'][::-1],
                fill='toself',
                fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(self.color_palette['primary'])) + [0.2])}",
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
        
        # Add maintenance recommendations
        if 'maintenance_threshold' in rul_data:
            threshold = rul_data['maintenance_threshold']
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color=self.color_palette['warning'],
                annotation_text=f"Maintenance Threshold: {threshold}"
            )
        
        fig.update_layout(
            title="Remaining Useful Life Prediction",
            xaxis_title="Cycle",
            yaxis_title="Remaining Useful Life",
            **self.layout_defaults
        )
        
        return fig
    
    def create_dashboard_summary(self, summary_data: Dict[str, Any]) -> go.Figure:
        """
        Create dashboard summary with key metrics.
        
        Args:
            summary_data: Summary metrics data
            
        Returns:
            Plotly figure with dashboard summary
        """
        # Create 2x2 subplot grid for key metrics
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=("System Health", "Prediction Accuracy", 
                          "Action Distribution", "Cost Breakdown")
        )
        
        # System health indicator
        health_score = summary_data.get('system_health', 0.75)
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=health_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "System Health (%)"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': self.color_palette['primary']},
                'steps': [
                    {'range': [0, 50], 'color': self.color_palette['warning']},
                    {'range': [50, 80], 'color': self.color_palette['secondary']},
                    {'range': [80, 100], 'color': self.color_palette['success']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=1, col=1)
        
        # Prediction accuracy indicator
        accuracy = summary_data.get('prediction_accuracy', 0.85)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=accuracy * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Prediction Accuracy (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': self.color_palette['success']},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "green"}
                ]
            }
        ), row=1, col=2)
        
        # Action distribution
        actions = summary_data.get('action_distribution', {
            'Preventive': 40, 'Corrective': 30, 'Replacement': 20, 'Monitoring': 10
        })
        
        fig.add_trace(go.Bar(
            x=list(actions.keys()),
            y=list(actions.values()),
            marker_color=[self.color_palette['primary'], self.color_palette['secondary'],
                         self.color_palette['warning'], self.color_palette['success']],
            text=[f"{v}%" for v in actions.values()],
            textposition='auto'
        ), row=2, col=1)
        
        # Cost breakdown
        costs = summary_data.get('cost_breakdown', {
            'Maintenance': 45, 'Downtime': 35, 'Replacement': 20
        })
        
        fig.add_trace(go.Pie(
            labels=list(costs.keys()),
            values=list(costs.values()),
            marker_colors=[self.color_palette['primary'], self.color_palette['secondary'],
                          self.color_palette['warning']],
            textinfo='label+percent'
        ), row=2, col=2)
        
        fig.update_layout(
            title="CBR+STM Framework Dashboard",
            height=600,
            showlegend=False,
            **self.layout_defaults
        )
        
        return fig
    
    def export_visualization_report(self, figures: Dict[str, go.Figure], filepath: str):
        """
        Export all visualizations to HTML report.
        
        Args:
            figures: Dictionary of figure names and Plotly figures
            filepath: Output HTML file path
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CBR+STM Framework Visualization Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .figure-container {{ margin: 30px 0; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 10px; }}
            </style>
        </head>
        <body>
            <h1>CBR+STM Framework Visualization Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        for name, fig in figures.items():
            html_content += f"""
            <div class="figure-container">
                <h2>{name}</h2>
                <div id="{name.lower().replace(' ', '_')}">{fig.to_html(include_plotlyjs=False, div_id=name.lower().replace(' ', '_'))}</div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
