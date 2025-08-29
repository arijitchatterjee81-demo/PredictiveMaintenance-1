import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from core.cbr_stm_framework import CBRSTMFramework
from core.state_space import StateSpace
from core.heuristic import MultiObjectiveHeuristic
from data.nasa_loader import NASACMAPSSLoader
from utils.ahp_calibration import AHPCalibrator
from utils.sensitivity_analysis import SensitivityAnalyzer
from visualization.components import VisualizationComponents

# Page configuration
st.set_page_config(
    page_title="CBR+STM Predictive Maintenance Framework",
    page_icon="🔧",
    layout="wide"
)

# Initialize session state
if 'framework' not in st.session_state:
    st.session_state.framework = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'weights' not in st.session_state:
    st.session_state.weights = [0.3, 0.4, 0.2, 0.1]
if 'solution_path' not in st.session_state:
    st.session_state.solution_path = None

# Main title and description
st.title("🔧 CBR+STM Predictive Maintenance Framework")
st.markdown("""
This application implements the theoretical framework for Case-Based Reasoning with State Transition Mechanisms 
for predictive maintenance as described in the research paper. The framework combines CBR's adaptability with 
STM's systematic reasoning capabilities while maintaining transparency and auditability.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Dataset selection and loading
st.sidebar.subheader("Dataset Configuration")
dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    ["NASA C-MAPSS", "Custom Upload"]
)

if dataset_option == "NASA C-MAPSS":
    subset_option = st.sidebar.selectbox(
        "C-MAPSS Subset",
        ["FD001", "FD002", "FD003", "FD004"]
    )
    
    if st.sidebar.button("Load NASA C-MAPSS Dataset"):
        with st.spinner("Loading NASA C-MAPSS dataset..."):
            loader = NASACMAPSSLoader()
            st.session_state.dataset = loader.load_dataset(subset_option)
            st.sidebar.success(f"Loaded {subset_option} dataset")

# Heuristic weight configuration
st.sidebar.subheader("Heuristic Weight Configuration")
st.sidebar.markdown("Configure weights for the multi-objective heuristic function:")
st.sidebar.latex(r"h(s,a) = w_1 \cdot C_{maint}(a) + w_2 \cdot C_{downtime}(s,a) - w_3 \cdot R_{improvement}(s,a) + w_4 \cdot T_{urgency}(s)")

w1 = st.sidebar.slider("w₁ (Maintenance Cost)", 0.0, 1.0, st.session_state.weights[0], 0.05)
w2 = st.sidebar.slider("w₂ (Downtime Cost)", 0.0, 1.0, st.session_state.weights[1], 0.05)
w3 = st.sidebar.slider("w₃ (Reliability Improvement)", 0.0, 1.0, st.session_state.weights[2], 0.05)
w4 = st.sidebar.slider("w₄ (Urgency Factor)", 0.0, 1.0, st.session_state.weights[3], 0.05)

st.session_state.weights = [w1, w2, w3, w4]

# AHP Calibration
if st.sidebar.button("Use AHP Calibration"):
    ahp_calibrator = AHPCalibrator()
    st.session_state.weights = ahp_calibrator.get_default_weights()
    st.sidebar.success("Applied AHP-calibrated weights")
    st.rerun()

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dataset Overview", 
    "Framework Configuration", 
    "Case Retrieval & Analysis", 
    "State Navigation", 
    "Sensitivity Analysis"
])

with tab1:
    st.header("Dataset Overview")
    
    if st.session_state.dataset is not None:
        dataset = st.session_state.dataset
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", len(dataset['train']))
        with col2:
            st.metric("Test Samples", len(dataset['test']))
        with col3:
            st.metric("Features", dataset['train'].shape[1] - 2)  # Excluding unit and cycle columns
        
        # Display sample data
        st.subheader("Training Data Sample")
        st.dataframe(dataset['train'].head())
        
        # Visualize sensor data
        st.subheader("Sensor Data Visualization")
        if 'train' in dataset:
            viz = VisualizationComponents()
            fig = viz.plot_sensor_trends(dataset['train'])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please load a dataset from the sidebar to begin analysis.")

with tab2:
    st.header("Framework Configuration")
    
    if st.session_state.dataset is not None:
        st.subheader("State Space Definition")
        
        # State space configuration
        num_states = st.slider("Number of Health States", 3, 10, 5)
        state_names = []
        for i in range(num_states):
            state_name = st.text_input(f"State {i+1} Name", 
                                     value=f"Health_Level_{i+1}" if i < num_states-1 else "Failure")
            state_names.append(state_name)
        
        # Action space configuration
        st.subheader("Action Space Definition")
        actions = st.multiselect(
            "Available Maintenance Actions",
            ["Preventive_Maintenance", "Corrective_Maintenance", "Component_Replacement", 
             "Condition_Monitoring", "Emergency_Repair", "Scheduled_Overhaul"],
            default=["Preventive_Maintenance", "Corrective_Maintenance", "Component_Replacement"]
        )
        
        # Initialize framework
        if st.button("Initialize CBR+STM Framework"):
            with st.spinner("Initializing framework..."):
                # Create state space
                state_space = StateSpace(state_names, actions)
                
                # Create heuristic function
                heuristic = MultiObjectiveHeuristic(st.session_state.weights)
                
                # Initialize framework
                st.session_state.framework = CBRSTMFramework(
                    state_space=state_space,
                    heuristic=heuristic,
                    dataset=st.session_state.dataset
                )
                
                st.success("Framework initialized successfully!")
        
        # Display current configuration
        if st.session_state.framework is not None:
            st.success("✅ Framework is initialized and ready")
            st.json({
                "States": len(st.session_state.framework.state_space.states),
                "Actions": len(st.session_state.framework.state_space.actions),
                "Heuristic Weights": st.session_state.weights,
                "Case Base Size": len(st.session_state.framework.case_base.cases)
            })
    else:
        st.warning("Please load a dataset first.")

with tab3:
    st.header("Case Retrieval & Analysis")
    
    if st.session_state.framework is not None:
        st.subheader("Phase 1: State-Based Case Retrieval")
        
        # Select test case for analysis
        test_units = st.session_state.dataset['test']['unit'].unique()
        selected_unit = st.selectbox("Select Test Unit for Analysis", test_units)
        
        # Similarity threshold
        similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.7, 0.1)
        
        if st.button("Perform Case Retrieval"):
            with st.spinner("Retrieving similar cases..."):
                # Get current problem state
                unit_data = st.session_state.dataset['test'][
                    st.session_state.dataset['test']['unit'] == selected_unit
                ]
                current_state = st.session_state.framework.map_to_state_space(unit_data.iloc[0])
                
                # Retrieve similar cases
                similar_cases = st.session_state.framework.retrieve_similar_cases(
                    current_state, similarity_threshold
                )
                
                st.subheader("Retrieved Cases")
                st.write(f"Found {len(similar_cases)} similar cases")
                
                if similar_cases:
                    # Display similarity scores
                    similarity_df = pd.DataFrame([
                        {
                            'Case ID': case['id'],
                            'Similarity Score': case['similarity'],
                            'Initial RUL': case['initial_rul'],
                            'Final Outcome': case['outcome']
                        }
                        for case in similar_cases
                    ])
                    st.dataframe(similarity_df)
                    
                    # Extract patterns
                    patterns = st.session_state.framework.extract_patterns(similar_cases)
                    
                    st.subheader("Extracted Patterns")
                    for pattern_type, pattern_data in patterns.items():
                        st.write(f"**{pattern_type}:**")
                        st.json(pattern_data)
    else:
        st.warning("Please initialize the framework first.")

with tab4:
    st.header("State Navigation & Solution Generation")
    
    if st.session_state.framework is not None:
        st.subheader("Phase 2: Heuristic-Guided State Navigation")
        
        # Problem definition
        st.info("💡 **Navigation Logic**: Select a degraded initial state (e.g., high Health_Level) and a better goal state (e.g., low Health_Level or Failure state for replacement)")
        
        col1, col2 = st.columns(2)
        with col1:
            # Default to a worse state (higher degradation)
            default_start_idx = len(st.session_state.framework.state_space.states) - 2 if len(st.session_state.framework.state_space.states) > 1 else 0
            start_state = st.selectbox(
                "Initial State", 
                st.session_state.framework.state_space.states,
                index=default_start_idx
            )
        with col2:
            # Default to a better state (lower degradation) 
            goal_state = st.selectbox(
                "Goal State", 
                st.session_state.framework.state_space.states,
                index=0
            )
        
        # Algorithm parameters
        max_iterations = st.slider("Maximum Search Iterations", 10, 1000, 100)
        
        if st.button("Execute State Navigation Algorithm"):
            with st.spinner("Executing Algorithm 1: CBR+STM Framework..."):
                # Execute the algorithm
                result = st.session_state.framework.execute_algorithm(
                    start_state=start_state,
                    goal_state=goal_state,
                    max_iterations=max_iterations
                )
                
                st.session_state.solution_path = result
                
                if result['success']:
                    st.success(f"Solution found in {result['iterations']} iterations!")
                    
                    # Display solution path
                    st.subheader("Solution Path")
                    path_df = pd.DataFrame([
                        {
                            'Step': i,
                            'State': step['state'],
                            'Action': step.get('action', 'N/A'),
                            'Cost': step['cost'],
                            'Heuristic': step['heuristic'],
                            'Total Cost': step['total_cost']
                        }
                        for i, step in enumerate(result['path'])
                    ])
                    st.dataframe(path_df)
                    
                    # Visualize solution path
                    viz = VisualizationComponents()
                    fig = viz.plot_solution_path(result['path'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Decision justification
                    st.subheader("Decision Justification & Audit Trail")
                    for i, justification in enumerate(result['justifications']):
                        with st.expander(f"Step {i+1} Justification"):
                            st.json(justification)
                    
                    # Cost breakdown
                    st.subheader("Cost Breakdown Analysis")
                    cost_breakdown = result['cost_breakdown']
                    
                    total_cost = sum(cost_breakdown.values())
                    breakdown_df = pd.DataFrame([
                        {
                            'Component': comp,
                            'Cost': cost,
                            'Percentage': f"{(cost/total_cost)*100:.1f}%" if total_cost > 0 else "0.0%"
                        }
                        for comp, cost in cost_breakdown.items()
                    ])
                    st.dataframe(breakdown_df)
                    
                    # Cost breakdown chart
                    fig = px.pie(
                        breakdown_df, 
                        values='Cost', 
                        names='Component',
                        title='Cost Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"No solution found. Reason: {result.get('reason', 'Unknown')}")
    else:
        st.warning("Please initialize the framework first.")

with tab5:
    st.header("Sensitivity Analysis")
    
    if st.session_state.solution_path is not None:
        st.subheader("Weight Sensitivity Analysis")
        
        # Sensitivity analysis parameters
        perturbation_range = st.slider("Perturbation Range (±%)", 5, 50, 20)
        num_samples = st.slider("Number of Monte Carlo Samples", 100, 1000, 500)
        
        if st.button("Perform Sensitivity Analysis"):
            with st.spinner("Performing sensitivity analysis..."):
                analyzer = SensitivityAnalyzer(st.session_state.framework)
                
                # Perform analysis
                sensitivity_results = analyzer.analyze_weight_sensitivity(
                    base_weights=st.session_state.weights,
                    perturbation_range=perturbation_range / 100,
                    num_samples=num_samples,
                    start_state=st.session_state.solution_path['start_state'],
                    goal_state=st.session_state.solution_path['goal_state']
                )
                
                # Display results
                st.subheader("Sensitivity Analysis Results")
                
                # Weight impact on total cost
                fig = go.Figure()
                for i, weight_name in enumerate(['w1', 'w2', 'w3', 'w4']):
                    fig.add_trace(go.Scatter(
                        x=sensitivity_results['weight_variations'][i],
                        y=sensitivity_results['cost_variations'][i],
                        mode='markers',
                        name=f'{weight_name} ({"Maintenance" if i==0 else "Downtime" if i==1 else "Reliability" if i==2 else "Urgency"})',
                        opacity=0.6
                    ))
                
                fig.update_layout(
                    title='Weight Sensitivity on Total Cost',
                    xaxis_title='Weight Value',
                    yaxis_title='Total Solution Cost'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Robustness metrics
                st.subheader("Robustness Metrics")
                robustness = sensitivity_results['robustness_metrics']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cost Stability", f"{robustness['cost_stability']:.3f}")
                with col2:
                    st.metric("Path Consistency", f"{robustness['path_consistency']:.1%}")
                with col3:
                    st.metric("Critical Threshold", f"{robustness['critical_threshold']:.2f}")
                
                # Recommendations
                st.subheader("Deployment Recommendations")
                recommendations = sensitivity_results['recommendations']
                for rec in recommendations:
                    st.info(rec)
    else:
        st.warning("Please execute the state navigation algorithm first to generate a solution path.")

# Footer
st.markdown("---")
st.markdown("""
**CBR+STM Predictive Maintenance Framework** - Implementation of the theoretical framework 
for Case-Based Reasoning with State Transition Mechanisms. This tool provides explainable 
AI capabilities with full audit trail generation for regulatory compliance.
""")
