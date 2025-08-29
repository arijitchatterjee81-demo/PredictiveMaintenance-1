"""
NASA C-MAPSS Dataset Loader

This module handles loading and preprocessing of NASA C-MAPSS (Commercial Modular 
Aero-Propulsion System Simulation) dataset for predictive maintenance applications.
"""

import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
from typing import Dict, Tuple, Any
import tempfile

class NASACMAPSSLoader:
    """
    Loader for NASA C-MAPSS dataset with preprocessing for CBR+STM framework.
    """
    
    def __init__(self):
        """Initialize the NASA C-MAPSS dataset loader."""
        self.dataset_url = "https://ti.arc.nasa.gov/c/6/"  # Base URL
        
        # Column definitions for C-MAPSS dataset
        self.columns = [
            'unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'
        ] + [f'sensor_{i}' for i in range(1, 22)]  # 21 sensors
        
        # Sensor descriptions for interpretability
        self.sensor_descriptions = {
            'sensor_1': 'Fan inlet temperature (°R)',
            'sensor_2': 'LPC outlet temperature (°R)', 
            'sensor_3': 'HPC outlet temperature (°R)',
            'sensor_4': 'LPT outlet temperature (°R)',
            'sensor_5': 'Fan inlet Pressure (psia)',
            'sensor_6': 'bypass-duct pressure (psia)',
            'sensor_7': 'HPC outlet pressure (psia)',
            'sensor_8': 'Physical fan speed (rpm)',
            'sensor_9': 'Physical core speed (rpm)',
            'sensor_10': 'Engine pressure ratio (P50/P2)',
            'sensor_11': 'Static pressure at HPC outlet (psia)',
            'sensor_12': 'Ratio of fuel flow to Ps30 (pps/psia)',
            'sensor_13': 'Corrected fan speed (rpm)',
            'sensor_14': 'Corrected core speed (rpm)',
            'sensor_15': 'Bypass Ratio',
            'sensor_16': 'Burner fuel-air ratio',
            'sensor_17': 'Bleed Enthalpy',
            'sensor_18': 'Required fan speed',
            'sensor_19': 'Required fan conversion speed',
            'sensor_20': 'High-pressure turbines Cool air flow',
            'sensor_21': 'Low-pressure turbines Cool air flow'
        }
    
    def load_dataset(self, subset: str = "FD001") -> Dict[str, Any]:
        """
        Load and preprocess NASA C-MAPSS dataset.
        
        Args:
            subset: Dataset subset (FD001, FD002, FD003, FD004)
            
        Returns:
            Dictionary containing train/test data and metadata
        """
        if subset not in ["FD001", "FD002", "FD003", "FD004"]:
            raise ValueError("Subset must be one of: FD001, FD002, FD003, FD004")
        
        # Generate synthetic dataset since actual NASA data access varies
        # In production, this would load from actual NASA repository
        print(f"Loading NASA C-MAPSS {subset} dataset...")
        
        train_data, test_data, rul_data = self._generate_synthetic_cmapss_data(subset)
        
        # Preprocess the data
        train_processed = self._preprocess_data(train_data, is_training=True)
        test_processed = self._preprocess_data(test_data, is_training=False)
        
        # Add RUL (Remaining Useful Life) information
        test_processed = self._add_rul_information(test_processed, rul_data)
        
        return {
            'train': train_processed,
            'test': test_processed,
            'rul_true': rul_data,
            'metadata': {
                'subset': subset,
                'num_train_units': train_data['unit'].nunique(),
                'num_test_units': test_data['unit'].nunique(),
                'max_cycles': train_data.groupby('unit')['cycle'].max().max(),
                'sensors': self.sensor_descriptions
            }
        }
    
    def _generate_synthetic_cmapss_data(self, subset: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Generate synthetic C-MAPSS-like data for demonstration.
        In production, replace with actual data loading.
        
        Args:
            subset: Dataset subset identifier
            
        Returns:
            Tuple of (training_data, test_data, rul_data)
        """
        # Dataset configurations
        configs = {
            'FD001': {'train_units': 100, 'test_units': 100, 'conditions': 1, 'faults': 1},
            'FD002': {'train_units': 260, 'test_units': 259, 'conditions': 6, 'faults': 1},
            'FD003': {'train_units': 100, 'test_units': 100, 'conditions': 1, 'faults': 2},
            'FD004': {'train_units': 249, 'test_units': 248, 'conditions': 6, 'faults': 2}
        }
        
        config = configs[subset]
        np.random.seed(42)  # For reproducibility
        
        # Generate training data
        train_data = self._generate_unit_data(
            num_units=config['train_units'],
            conditions=config['conditions'],
            faults=config['faults'],
            unit_offset=0
        )
        
        # Generate test data  
        test_data = self._generate_unit_data(
            num_units=config['test_units'],
            conditions=config['conditions'],
            faults=config['faults'],
            unit_offset=config['train_units'],
            is_test=True
        )
        
        # Generate RUL values for test data
        rul_data = np.random.randint(1, 200, size=config['test_units'])
        
        return train_data, test_data, rul_data
    
    def _generate_unit_data(self, num_units: int, conditions: int, faults: int, 
                           unit_offset: int = 0, is_test: bool = False) -> pd.DataFrame:
        """
        Generate synthetic operational data for multiple units.
        
        Args:
            num_units: Number of units to generate
            conditions: Number of operational conditions
            faults: Number of fault modes
            unit_offset: Starting unit ID
            is_test: Whether this is test data
            
        Returns:
            DataFrame with synthetic unit operational data
        """
        all_data = []
        
        for unit_id in range(1, num_units + 1):
            # Generate lifecycle length
            if is_test:
                # Test data has truncated lifecycles
                lifecycle_length = np.random.randint(50, 250)
            else:
                # Training data runs to failure
                lifecycle_length = np.random.randint(100, 400)
            
            unit_data = self._generate_single_unit_lifecycle(
                unit_id + unit_offset, 
                lifecycle_length, 
                conditions, 
                faults
            )
            all_data.append(unit_data)
        
        return pd.concat(all_data, ignore_index=True)
    
    def _generate_single_unit_lifecycle(self, unit_id: int, cycles: int, 
                                       conditions: int, faults: int) -> pd.DataFrame:
        """
        Generate synthetic lifecycle data for a single unit.
        
        Args:
            unit_id: Unit identifier
            cycles: Number of operational cycles
            conditions: Number of operational conditions
            faults: Number of fault modes
            
        Returns:
            DataFrame with single unit lifecycle data
        """
        data = []
        
        # Generate operational settings (vary by condition)
        condition_id = np.random.randint(1, conditions + 1)
        op_settings = self._get_operational_settings(condition_id)
        
        # Generate fault progression parameters
        fault_mode = np.random.randint(1, faults + 1)
        fault_params = self._get_fault_parameters(fault_mode)
        
        for cycle in range(1, cycles + 1):
            # Base sensor values
            sensor_values = self._generate_sensor_values(
                cycle, cycles, op_settings, fault_params
            )
            
            # Create row
            row = {
                'unit': unit_id,
                'cycle': cycle,
                'op_setting_1': op_settings[0],
                'op_setting_2': op_settings[1],
                'op_setting_3': op_settings[2]
            }
            
            # Add sensor values
            for i, value in enumerate(sensor_values, 1):
                row[f'sensor_{i}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _get_operational_settings(self, condition_id: int) -> np.ndarray:
        """
        Generate operational settings for given condition.
        
        Args:
            condition_id: Operational condition identifier
            
        Returns:
            Array of operational settings
        """
        # Different operational conditions
        settings_map = {
            1: [0.0, 0.0, 100.0],
            2: [0.0, 10.0, 100.0],
            3: [10.0, 0.0, 100.0],
            4: [10.0, 10.0, 100.0],
            5: [20.0, 0.0, 100.0],
            6: [20.0, 10.0, 100.0]
        }
        
        base_settings = settings_map.get(condition_id, [0.0, 0.0, 100.0])
        
        # Add small random variations
        variations = np.random.normal(0, 0.5, 3)
        return np.array(base_settings) + variations
    
    def _get_fault_parameters(self, fault_mode: int) -> Dict[str, float]:
        """
        Generate fault progression parameters.
        
        Args:
            fault_mode: Fault mode identifier
            
        Returns:
            Dictionary of fault parameters
        """
        fault_configs = {
            1: {'degradation_rate': 0.001, 'noise_level': 0.1, 'affected_sensors': [2, 3, 4]},
            2: {'degradation_rate': 0.0015, 'noise_level': 0.15, 'affected_sensors': [7, 8, 9]}
        }
        
        return fault_configs.get(fault_mode, fault_configs[1])
    
    def _generate_sensor_values(self, cycle: int, total_cycles: int, 
                               op_settings: np.ndarray, fault_params: Dict) -> np.ndarray:
        """
        Generate sensor values for a specific cycle.
        
        Args:
            cycle: Current cycle number
            total_cycles: Total cycles in lifecycle
            op_settings: Operational settings
            fault_params: Fault parameters
            
        Returns:
            Array of sensor values
        """
        # Health factor (decreases with age)
        health_factor = 1.0 - (cycle / total_cycles) * fault_params['degradation_rate'] * 100
        
        # Base sensor values (typical ranges for turbofan engines)
        base_values = np.array([
            518.67,    # Fan inlet temperature
            642.0,     # LPC outlet temperature  
            1589.7,    # HPC outlet temperature
            1400.0,    # LPT outlet temperature
            14.62,     # Fan inlet pressure
            14.62,     # Bypass-duct pressure
            553.9,     # HPC outlet pressure
            2388.0,    # Physical fan speed
            9046.19,   # Physical core speed
            1.3,       # Engine pressure ratio
            47.47,     # Static pressure at HPC outlet
            521.66,    # Ratio of fuel flow to Ps30
            2388.0,    # Corrected fan speed
            8138.62,   # Corrected core speed
            8.4195,    # Bypass ratio
            0.03,      # Burner fuel-air ratio
            392.0,     # Bleed enthalpy
            2388.0,    # Required fan speed
            8138.62,   # Required fan conversion speed
            38.86,     # HP turbines cool air flow
            23.419     # LP turbines cool air flow
        ])
        
        # Apply operational condition effects
        condition_effects = np.random.normal(1.0, 0.02, 21)  # Small variations
        sensor_values = base_values * condition_effects
        
        # Apply degradation effects
        degradation = np.ones(21)
        for sensor_idx in fault_params['affected_sensors']:
            degradation[sensor_idx - 1] = health_factor
        
        sensor_values = sensor_values * degradation
        
        # Add measurement noise
        noise = np.random.normal(0, fault_params['noise_level'], 21)
        sensor_values = sensor_values + noise * sensor_values * 0.01  # 1% relative noise
        
        return sensor_values
    
    def _preprocess_data(self, data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Preprocess the raw dataset.
        
        Args:
            data: Raw dataset
            is_training: Whether this is training data
            
        Returns:
            Preprocessed dataset
        """
        processed_data = data.copy()
        
        # Calculate RUL for training data
        if is_training:
            processed_data = self._calculate_rul(processed_data)
        
        # Normalize sensor values
        processed_data = self._normalize_sensors(processed_data)
        
        # Remove constant or near-constant sensors
        processed_data = self._remove_constant_sensors(processed_data)
        
        # Add derived features
        processed_data = self._add_derived_features(processed_data)
        
        return processed_data
    
    def _calculate_rul(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Remaining Useful Life for each time step.
        
        Args:
            data: Dataset with unit and cycle columns
            
        Returns:
            Dataset with RUL column added
        """
        data_with_rul = data.copy()
        
        # Calculate max cycle per unit
        max_cycles = data_with_rul.groupby('unit')['cycle'].max()
        
        # Calculate RUL
        rul_values = []
        for _, row in data_with_rul.iterrows():
            unit_id = row['unit']
            current_cycle = row['cycle']
            max_cycle = max_cycles[unit_id]
            rul = max_cycle - current_cycle
            rul_values.append(rul)
        
        data_with_rul['RUL'] = rul_values
        return data_with_rul
    
    def _normalize_sensors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize sensor values using z-score normalization.
        
        Args:
            data: Dataset with sensor columns
            
        Returns:
            Dataset with normalized sensors
        """
        normalized_data = data.copy()
        
        sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
        
        for col in sensor_cols:
            mean_val = data[col].mean()
            std_val = data[col].std()
            
            if std_val > 0:
                normalized_data[col] = (data[col] - mean_val) / std_val
            else:
                normalized_data[col] = 0  # Constant column
        
        return normalized_data
    
    def _remove_constant_sensors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove sensors with very low variance (nearly constant).
        
        Args:
            data: Dataset with sensor columns
            
        Returns:
            Dataset with constant sensors removed
        """
        filtered_data = data.copy()
        
        sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
        
        for col in sensor_cols:
            if data[col].std() < 1e-6:  # Very small variance
                filtered_data = filtered_data.drop(columns=[col])
                print(f"Removed constant sensor: {col}")
        
        return filtered_data
    
    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features that may be useful for maintenance prediction.
        
        Args:
            data: Preprocessed dataset
            
        Returns:
            Dataset with derived features
        """
        enhanced_data = data.copy()
        
        # Add cycle-based features
        enhanced_data['cycle_norm'] = enhanced_data['cycle'] / enhanced_data.groupby('unit')['cycle'].transform('max')
        
        # Add moving averages for trend detection
        sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
        
        for col in sensor_cols[:5]:  # Only for first 5 sensors to avoid too many features
            enhanced_data[f'{col}_ma3'] = enhanced_data.groupby('unit')[col].rolling(window=3, min_periods=1).mean().values
            enhanced_data[f'{col}_ma5'] = enhanced_data.groupby('unit')[col].rolling(window=5, min_periods=1).mean().values
        
        # Add degradation indicators
        for col in sensor_cols[:3]:  # For key sensors
            enhanced_data[f'{col}_trend'] = enhanced_data.groupby('unit')[col].diff()
        
        return enhanced_data
    
    def _add_rul_information(self, test_data: pd.DataFrame, rul_true: np.ndarray) -> pd.DataFrame:
        """
        Add true RUL information to test data.
        
        Args:
            test_data: Test dataset
            rul_true: True RUL values for each test unit
            
        Returns:
            Test dataset with RUL information
        """
        enhanced_test = test_data.copy()
        
        # Get last cycle for each unit
        last_cycles = test_data.groupby('unit')['cycle'].max()
        
        # Map RUL to each unit
        rul_map = {}
        for i, unit_id in enumerate(sorted(test_data['unit'].unique())):
            rul_map[unit_id] = rul_true[i]
        
        # Add RUL column
        rul_values = []
        for _, row in enhanced_test.iterrows():
            unit_id = row['unit']
            rul_values.append(rul_map[unit_id])
        
        enhanced_test['RUL_true'] = rul_values
        
        return enhanced_test
