# streamlit_app.py
import streamlit as st
import traceback
import os
os.environ["STREAMLIT_WATCHDOG_DISABLED"] = "true"

# Helper to safely import modules
def safe_import(module_name, attr=None):
    try:
        module = __import__(module_name, fromlist=['*'])
        if attr:
            return getattr(module, attr)
        return module
    except Exception as e:
        st.session_state['import_status'][f"{module_name}{'.'+attr if attr else ''}"] = f"❌ Failed: {e}"
        return None

# Initialize import status
if 'import_status' not in st.session_state:
    st.session_state['import_status'] = {}

# Import core modules safely
CBRSTMFramework = safe_import('core.cbr_stm_framework', 'CBRSTMFramework')
StateSpace = safe_import('core.state_space', 'StateSpace')
MultiObjectiveHeuristic = safe_import('core.heuristic', 'MultiObjectiveHeuristic')
NASACMAPSSLoader = safe_import('data.nasa_loader', 'NASACMAPSSLoader')
AHPCalibrator = safe_import('utils.ahp_calibration', 'AHPCalibrator')
SensitivityAnalyzer = safe_import('utils.sensitivity_analysis', 'SensitivityAnalyzer')
ModernMethodsComparator = safe_import('utils.modern_methods_comparison', 'ModernMethodsComparator')
VisualizationComponents = safe_import('visualization.components', 'VisualizationComponents')

# Streamlit page config
st.set_page_config(
    page_title="CBR+STM Predictive Maintenance Framework",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 CBR+STM Predictive Maintenance Framework")
st.markdown("This app demonstrates the CBR+STM predictive maintenance framework with multi-tab interface.")

# Show import results
st.subheader("Module Import Status")
for module, status in st.session_state['import_status'].items():
    st.write(f"{module}: {status}")

# Attempt minimal dataset load
dataset = None
if NASACMAPSSLoader:
    try:
        loader = NASACMAPSSLoader()
        dataset = loader.load_dataset("FD001")
        st.success(f"✅ NASA C-MAPSS FD001 dataset loaded with {len(dataset['train'])} training samples")
    except Exception as e:
        st.error("❌ Dataset loading failed")
        st.text(traceback.format_exc())

# Multi-tab interface
try:
    tabs = st.tabs([
        "Dataset Overview", 
        "Framework Configuration", 
        "Case Retrieval & Analysis", 
        "State Navigation", 
        "Sensitivity Analysis",
        "IRL Calibration Results",
        "Demo & Validation"
    ])

    # Dataset Overview
    with tabs[0]:
        st.header("Dataset Overview")
        if dataset:
            try:
                st.write("**Training Samples:**", len(dataset['train']))
                st.write("**Test Samples:**", len(dataset['test']))
            except Exception as e:
                st.error("❌ Failed to display dataset")
                st.text(traceback.format_exc())
        else:
            st.info("Dataset not loaded. Check import status above.")

    # Framework Configuration
    with tabs[1]:
        st.header("Framework Configuration")
        if CBRSTMFramework and StateSpace and MultiObjectiveHeuristic:
            st.info("Framework modules loaded. Configuration UI can be implemented here.")
        else:
            st.warning("Framework modules missing. Cannot configure.")

    # Case Retrieval & Analysis
    with tabs[2]:
        st.header("Case Retrieval & Analysis")
        if CBRSTMFramework:
            st.info("Case retrieval functionality placeholder.")
        else:
            st.warning("CBRSTMFramework not loaded.")

    # State Navigation
    with tabs[3]:
        st.header("State Navigation")
        if StateSpace:
            st.info("State navigation UI placeholder.")
        else:
            st.warning("StateSpace module not loaded.")

    # Sensitivity Analysis
    with tabs[4]:
        st.header("Sensitivity Analysis")
        if SensitivityAnalyzer:
            st.info("Sensitivity analysis functionality placeholder.")
        else:
            st.warning("SensitivityAnalyzer module not loaded.")

    # IRL Calibration Results
    with tabs[5]:
        st.header("IRL Calibration Results")
        if AHPCalibrator:
            st.info("AHP/IRL calibration results placeholder.")
        else:
            st.warning("AHPCalibrator module not loaded.")

    # Demo & Validation
    with tabs[6]:
        st.header("Demo & Validation")
        if ModernMethodsComparator and VisualizationComponents:
            st.info("Demo and validation placeholder.")
        else:
            st.warning("Modules for demo/validation not loaded.")

except Exception as e:
    st.error("❌ Tab rendering failed")
    st.text(traceback.format_exc())

# Global error handling
try:
    pass
except Exception as e:
    st.error("❌ Unexpected error occurred")
    st.text(traceback.format_exc())
