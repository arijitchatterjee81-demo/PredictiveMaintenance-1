# CBR+STM Predictive Maintenance Framework

## Overview

This project implements a theoretical framework that combines Case-Based Reasoning (CBR) with State Transition Mechanisms (STM) for predictive maintenance applications. The framework is designed to provide systematic maintenance decision-making while maintaining transparency and auditability. The system processes equipment sensor data, particularly from NASA C-MAPSS datasets, to recommend optimal maintenance actions based on historical cases and multi-objective heuristic evaluation.

The application features a Streamlit-based web interface that allows users to configure parameters, analyze sensitivity, and visualize results through interactive charts and graphs.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Single-page application built with Streamlit providing an interactive dashboard
- **Visualization Layer**: Plotly-based interactive charts and graphs for displaying sensor trends, solution paths, and analysis results
- **Session State Management**: Streamlit session state handling for maintaining framework configuration and results across user interactions

### Backend Architecture
- **Core Framework (CBR+STM)**: Main algorithm implementation following the research paper's theoretical framework
- **State Space Management**: Discrete state representation with defined transitions between equipment health conditions
- **Case Base System**: Storage and retrieval system for historical maintenance cases with similarity-based matching
- **Multi-Objective Heuristic**: Decision-making function combining maintenance costs, downtime, reliability improvement, and urgency factors

### Data Processing Pipeline
- **NASA C-MAPSS Loader**: Specialized data loader for NASA Commercial Modular Aero-Propulsion System Simulation datasets
- **Feature Extraction**: Sensor data preprocessing and feature engineering for state representation
- **Similarity Calculation**: Multiple distance metrics (Euclidean, Manhattan, Cosine) for case-based reasoning

### Decision Support Components
- **AHP Calibration**: Analytic Hierarchy Process implementation for expert-driven weight calibration
- **Sensitivity Analysis**: Monte Carlo and parametric analysis for weight parameter optimization
- **Multi-Objective Optimization**: Weighted scoring system balancing cost, downtime, reliability, and urgency

### Core Design Patterns
- **Algorithm 1 Implementation**: Direct implementation of the CBR+STM algorithm from the research paper
- **State-Action Mapping**: Comprehensive state transition matrix defining equipment health progressions
- **Case Retrieval Strategy**: Similarity-based case matching with configurable thresholds
- **Heuristic Evaluation**: Four-component weighted scoring system for maintenance action evaluation

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Plotly**: Interactive visualization and charting
- **SciPy**: Scientific computing utilities for statistical analysis
- **Scikit-learn**: Machine learning utilities for similarity calculations
- **NetworkX**: Graph processing for state transition visualization

### Data Sources
- **NASA C-MAPSS Dataset**: Primary dataset for turbofan engine degradation simulation
- **Custom Data Upload**: Support for user-provided maintenance datasets

### Algorithmic Dependencies
- **Analytic Hierarchy Process (AHP)**: Weight calibration methodology
- **Monte Carlo Simulation**: Sensitivity analysis implementation
- **Multi-Objective Decision Making**: Weighted scoring algorithms

### Development Tools
- **Python 3.x**: Primary programming language
- **Type Hints**: Comprehensive type annotation for code clarity
- **Modular Architecture**: Component-based design for maintainability