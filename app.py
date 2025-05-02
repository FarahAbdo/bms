import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

class BatteryAnalyzer:
    """A comprehensive system for battery data analysis, SOC and SOH prediction"""
    
    def __init__(self, file_path=None):
        """Initialize the BatteryAnalyzer with optional file path"""
        self.file_path = file_path
        self.df = None
        self.processed_df = None
        self.soc_model = None
        self.soh_model = None
        self.column_mapping = {}
        # Store model parameters for predictions
        self.min_voltage = 0
        self.max_voltage = 5  # Default values
        self.nominal_capacity = 1.0  # Default value
        
    def load_data(self, file_path=None):
        """Load battery data from file"""
        if file_path:
            self.file_path = file_path
            
        if not self.file_path:
            raise ValueError("No file path provided.")
            
        st.info(f"Loading data from {getattr(self.file_path, 'name', 'uploaded file')}...")
        
        try:
            # For Excel files
            if (hasattr(self.file_path, 'name') and (self.file_path.name.endswith('.xlsx') or self.file_path.name.endswith('.xls'))):
                self.df = pd.read_excel(self.file_path)
            # For CSV files
            elif hasattr(self.file_path, 'name') and self.file_path.name.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            else:
                # Try to read as Excel first, then CSV if that fails
                try:
                    self.df = pd.read_excel(self.file_path)
                except:
                    try:
                        self.df = pd.read_csv(self.file_path)
                    except Exception as e:
                        raise ValueError(f"Could not read file as Excel or CSV: {e}")
            

                
            st.success(f"\nLoaded {len(self.df)} records.")
            return self.df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            raise
    
    def standardize_column_names(self):
        """Map the actual column names to expected standard names"""
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
            
        st.info("\nStandardizing column names...")
        
        # Dictionary to map common variations of column names
        self.column_mapping = {}
        
        # Define common patterns for each type of data
        column_patterns = {
            'Cycle_Index': ['cycle', 'index', 'cycle_index', 'cycle index'],
            'Current(A)': ['current', 'curr', 'i(', 'ma'],
            'Voltage(V)': ['voltage', 'volt', 'v(', 'mv'],
            'Charge_Capacity(Ah)': ['charge_cap', 'charge cap', 'chg cap'],
            'Discharge_Capacity(Ah)': ['discharge_cap', 'discharge cap', 'dischg cap'],
            'Charge_Energy(Wh)': ['charge_energy', 'charge energy', 'chg energy'],
            'Discharge_Energy(Wh)': ['discharge_energy', 'discharge energy', 'dischg energy'],
            'Internal_Resistance(Ohm)': ['resist', 'internal', 'resistance'],
            'Date_Time': ['date', 'time', 'date_time'],
            'Temperature(C)': ['temp', 'temperature', 'celsius']
        }
        
        # Find matching columns
        for standard_name, patterns in column_patterns.items():
            for col in self.df.columns:
                if any(pattern in col.lower() for pattern in patterns):
                    self.column_mapping[col] = standard_name
                    break
        
        st.write("\nColumn Mapping:")
        for original, mapped in self.column_mapping.items():
            st.write(f"  - {original} -> {mapped}")
        
        # Create a copy with renamed columns
        self.df_renamed = self.df.copy()
        self.df_renamed = self.df_renamed.rename(columns=self.column_mapping)
        
        return self.df_renamed
    
    def preprocess_data(self):
        """Clean and preprocess the battery cycling data"""
        if not hasattr(self, 'df_renamed'):
            raise ValueError("Column standardization not performed. Run standardize_column_names() first.")
            
        st.info("\nPreprocessing data...")
        
        df = self.df_renamed.copy()
        
        # Check if we have the required columns after mapping
        required_columns = ['Cycle_Index', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"\nWarning: The following required columns are missing: {missing_columns}")
            st.warning("Attempting to use alternative columns or create derived columns...")
        
        # If Cycle_Index is missing, try to create it
        if 'Cycle_Index' not in df.columns:
            # Try to find a column with sequential numbers
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].is_monotonic_increasing and df[col].nunique() > len(df) * 0.8:
                    st.write(f"Using column '{col}' as Cycle_Index")
                    df['Cycle_Index'] = df[col]
                    break
            else:
                # If no suitable column found, create a sequence
                st.write("Creating Cycle_Index as a sequential number")
                df['Cycle_Index'] = range(1, len(df) + 1)
        
        # Convert date_time column to datetime if it exists
        if 'Date_Time' in df.columns:
            df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')
        
        # Handle units conversion if needed (e.g., mV to V, mA to A)
        if 'Current(A)' in df.columns and df['Current(A)'].abs().max() > 100:
            st.write("Converting current from mA to A")
            df['Current(A)'] = df['Current(A)'] / 1000
            
        if 'Voltage(V)' in df.columns and df['Voltage(V)'].abs().max() > 100:
            st.write("Converting voltage from mV to V")
            df['Voltage(V)'] = df['Voltage(V)'] / 1000
        
        # Drop rows with missing values in critical columns that exist
        existing_critical_cols = [col for col in ['Cycle_Index', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)'] 
                                if col in df.columns]
        if existing_critical_cols:
            initial_count = len(df)
            df = df.dropna(subset=existing_critical_cols)
            dropped_count = initial_count - len(df)
            if dropped_count > 0:
                st.write(f"Dropped {dropped_count} rows with missing values in critical columns.")
        
        # If we don't have capacity columns, we need to estimate them
        if 'Charge_Capacity(Ah)' not in df.columns or 'Discharge_Capacity(Ah)' not in df.columns:
            st.warning("Warning: Capacity columns are missing. Creating synthetic data for demonstration.")
            # Create synthetic capacity data with degradation over cycles
            max_cycle = df['Cycle_Index'].max()
            initial_capacity = 1.0  # Assumed initial capacity in Ah
            
            # Linear degradation model (simplified)
            df['Discharge_Capacity(Ah)'] = initial_capacity * (1 - 0.0002 * df['Cycle_Index'])
            df['Charge_Capacity(Ah)'] = df['Discharge_Capacity(Ah)'] * 1.05  # Charge is typically higher than discharge
        
        # Calculate the initial capacity (from first cycle) to use as reference for SOH
        min_cycle = df['Cycle_Index'].min()
        self.nominal_capacity = df.loc[df['Cycle_Index'] == min_cycle, 'Discharge_Capacity(Ah)'].values[0]
        
        # Calculate State of Health (SOH)
        # SOH is typically defined as the ratio of current capacity to nominal capacity
        df['SOH'] = (df['Discharge_Capacity(Ah)'] / self.nominal_capacity) * 100
        
        # If voltage column exists, use it for SOC estimation
        if 'Voltage(V)' in df.columns:
            # Store min and max voltage for normalization in future predictions
            self.min_voltage = df['Voltage(V)'].min()
            self.max_voltage = df['Voltage(V)'].max()
            
            # Normalize Voltage to range [0,1] for each cycle
            df['Voltage_Normalized'] = (df['Voltage(V)'] - self.min_voltage) / \
                                    (self.max_voltage - self.min_voltage)
            
            # Map voltage to SOC using a simplified model
            df['SOC'] = (df['Voltage(V)'] - self.min_voltage) / (self.max_voltage - self.min_voltage) * 100
            df['SOC'] = df['SOC'].clip(0, 100)  # Ensure SOC is between 0-100%
        else:
            st.warning("Warning: Voltage column is missing. Creating synthetic SOC data.")
            # Create synthetic SOC data
            df['SOC'] = 100 - (df['Cycle_Index'] % 10) * 10  # Simplified pattern
            df['Voltage_Normalized'] = df['SOC'] / 100
            # Set default voltage range for future normalization
            self.min_voltage = 2.5
            self.max_voltage = 4.2
        
        # Create additional features if possible
        if all(col in df.columns for col in ['Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)']):
            # Calculate Coulombic Efficiency
            df['Coulombic_Efficiency'] = (df['Discharge_Capacity(Ah)'] / df['Charge_Capacity(Ah)']) * 100
        
        if all(col in df.columns for col in ['Charge_Energy(Wh)', 'Discharge_Energy(Wh)']):
            # Calculate Energy Efficiency
            df['Energy_Efficiency'] = (df['Discharge_Energy(Wh)'] / df['Charge_Energy(Wh)']) * 100
        
        if all(col in df.columns for col in ['Discharge_Energy(Wh)', 'Discharge_Capacity(Ah)']):
            # Calculate Energy Density
            df['Energy_Density'] = df['Discharge_Energy(Wh)'] / df['Discharge_Capacity(Ah)']
        
        self.processed_df = df
        st.success(f"\nPreprocessing complete. Final dataset has {len(df)} records.")
        
        return df
    
    def visualize_battery_metrics(self):
        """Create visualizations of battery metrics including SOC and SOH"""
        if self.processed_df is None:
            raise ValueError("No processed data available. Run preprocess_data() first.")
            
        st.info("\nVisualizing battery metrics...")
        df = self.processed_df
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Discharge Capacity vs Cycle Index (if available)
        plt.subplot(3, 2, 1)
        if 'Discharge_Capacity(Ah)' in df.columns:
            plt.plot(df['Cycle_Index'], df['Discharge_Capacity(Ah)'], 'b-', linewidth=2)
            plt.xlabel('Cycle Index')
            plt.ylabel('Discharge Capacity (Ah)')
            plt.title('Capacity Fade with Cycling')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Discharge Capacity data not available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Discharge Capacity (Not Available)')
        
        # Plot 2: SOH vs Cycle Index
        plt.subplot(3, 2, 2)
        plt.plot(df['Cycle_Index'], df['SOH'], 'g-', linewidth=2)
        plt.xlabel('Cycle Index')
        plt.ylabel('State of Health (%)')
        plt.title('SOH Degradation with Cycling')
        plt.grid(True)
        
        # Plot 3: Voltage vs SOC (if voltage available)
        plt.subplot(3, 2, 3)
        if 'Voltage(V)' in df.columns:
            plt.scatter(df['SOC'], df['Voltage(V)'], alpha=0.5)
            plt.xlabel('State of Charge (%)')
            plt.ylabel('Voltage (V)')
            plt.title('Voltage vs. SOC Relationship')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Voltage data not available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Voltage vs. SOC (Not Available)')
        
        # Plot 4: Internal Resistance vs Cycle (if available)
        plt.subplot(3, 2, 4)
        if 'Internal_Resistance(Ohm)' in df.columns:
            plt.plot(df['Cycle_Index'], df['Internal_Resistance(Ohm)'], 'r-', linewidth=2)
            plt.xlabel('Cycle Index')
            plt.ylabel('Internal Resistance (Ohm)')
            plt.title('Internal Resistance vs Cycle')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Internal Resistance data not available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Internal Resistance (Not Available)')
        
        # Plot 5: Coulombic Efficiency (if available)
        plt.subplot(3, 2, 5)
        if 'Coulombic_Efficiency' in df.columns:
            plt.plot(df['Cycle_Index'], df['Coulombic_Efficiency'], 'm-', linewidth=2)
            plt.xlabel('Cycle Index')
            plt.ylabel('Coulombic Efficiency (%)')
            plt.title('Coulombic Efficiency vs Cycle')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Coulombic Efficiency data not available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Coulombic Efficiency (Not Available)')
        
        # Plot 6: Energy Efficiency (if available)
        plt.subplot(3, 2, 6)
        if 'Energy_Efficiency' in df.columns:
            plt.plot(df['Cycle_Index'], df['Energy_Efficiency'], 'c-', linewidth=2)
            plt.xlabel('Cycle Index')
            plt.ylabel('Energy Efficiency (%)')
            plt.title('Energy Efficiency vs Cycle')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Energy Efficiency data not available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Energy Efficiency (Not Available)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        return
    
    def prepare_soc_features(self):
        """Prepare features for SOC prediction model"""
        if self.processed_df is None:
            raise ValueError("No processed data available. Run preprocess_data() first.")
            
        st.info("\nPreparing features for SOC modeling...")
        df = self.processed_df
        
        # Define potential features for SOC prediction
        potential_features = ['Voltage(V)', 'Current(A)', 'Internal_Resistance(Ohm)', 
                             'Cycle_Index', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
                             'Energy_Density', 'Voltage_Normalized', 'Temperature(C)']
        
        # Use only the columns that exist in the dataframe
        soc_features = [f for f in potential_features if f in df.columns]
        
        if not soc_features:
            raise ValueError("No valid features found for SOC prediction")
        
        st.write(f"\nUsing these features for SOC prediction: {soc_features}")
        
        X_soc = df[soc_features]
        y_soc = df['SOC']
        
        return X_soc, y_soc
    
    def prepare_soh_features(self):
        """Prepare features for SOH prediction model"""
        if self.processed_df is None:
            raise ValueError("No processed data available. Run preprocess_data() first.")
            
        st.info("\nPreparing features for SOH modeling...")
        df = self.processed_df
        
        # Define potential features for SOH prediction
        potential_features = ['Cycle_Index', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)', 
                             'Charge_Energy(Wh)', 'Discharge_Energy(Wh)', 'Internal_Resistance(Ohm)',
                             'Coulombic_Efficiency', 'Energy_Efficiency', 'Voltage(V)', 'Temperature(C)']
        
        # Use only the columns that exist in the dataframe
        soh_features = [f for f in potential_features if f in df.columns]
        
        if not soh_features:
            raise ValueError("No valid features found for SOH prediction")
        
        st.write(f"\nUsing these features for SOH prediction: {soh_features}")
        
        X_soh = df[soh_features]
        y_soh = df['SOH']
        
        return X_soh, y_soh
    
    def train_models(self, model_type='randomforest', test_size=0.2, random_state=42):
        """Train SOC and SOH prediction models using the specified algorithm"""
        st.info(f"\nTraining models using {model_type}...")
        
        # Select the model class based on user input
        if model_type.lower() == 'randomforest':
            model_class = RandomForestRegressor
        elif model_type.lower() == 'gradientboosting':
            model_class = GradientBoostingRegressor
        elif model_type.lower() == 'xgboost':
            model_class = xgb.XGBRegressor
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Choose 'randomforest', 'gradientboosting', or 'xgboost'.")
        
        # Train SOC model
        try:
            X_soc, y_soc = self.prepare_soc_features()
            
            # Save feature names for prediction
            self.soc_feature_names = X_soc.columns.tolist()
            
            st.write("\nTraining the SOC model...")
            self.soc_model = self._train_single_model(X_soc, y_soc, model_class, "SOC", test_size, random_state)
        except Exception as e:
            st.error(f"Error in SOC model training: {e}")
            self.soc_model = None
        
        # Train SOH model  
        try:
            X_soh, y_soh = self.prepare_soh_features()
            
            # Save feature names for prediction
            self.soh_feature_names = X_soh.columns.tolist()
            
            st.write("\nTraining the SOH model...")
            self.soh_model = self._train_single_model(X_soh, y_soh, model_class, "SOH", test_size, random_state)
        except Exception as e:
            st.error(f"Error in SOH model training: {e}")
            self.soh_model = None
            
        return self.soc_model, self.soh_model
    
    def _train_single_model(self, X, y, model_class, model_name, test_size=0.2, random_state=42):
        """Train a single model (helper method)"""
        # Handle small datasets
        if len(X) < 10:
            st.warning(f"Warning: Very small dataset ({len(X)} records). Using simple model without validation.")
            model = model_class(random_state=random_state)
            model.fit(X, y)
            return model
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model_class(random_state=random_state))
        ])
        
        # Define hyperparameters to tune
        if isinstance(model_class(), RandomForestRegressor):
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [None, 10, 20]
            }
        elif isinstance(model_class(), GradientBoostingRegressor):
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5]
            }
        elif isinstance(model_class(), xgb.XGBRegressor):
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5]
            }
        
        # Choose cross-validation based on dataset size
        cv = min(3, len(X_train) // 3) if len(X_train) >= 9 else 2
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"\n{model_name} Model Performance:")
        st.write(f"Best Parameters: {grid_search.best_params_}")
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"R² Score: {r2:.4f}")
        
        # Feature importance
        feature_importances = best_model.named_steps['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        st.write(f"\n{model_name} Feature Importance:")
        st.dataframe(feature_importance_df)
        
        # Plot feature importance
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(f'Feature Importance for {model_name} Prediction')
        plt.tight_layout()
        st.pyplot(fig)
        
        return best_model
    
    def prepare_input_data(self, input_data):
        """Prepare input data for prediction by adding derived features"""
        # Make a copy to avoid modifying the original
        df = input_data.copy()
        
        # Create Voltage_Normalized if Voltage(V) exists but Voltage_Normalized doesn't
        if 'Voltage(V)' in df.columns and 'Voltage_Normalized' not in df.columns:
            df['Voltage_Normalized'] = (df['Voltage(V)'] - self.min_voltage) / (self.max_voltage - self.min_voltage)
        
        # Calculate Coulombic Efficiency if needed
        if 'Coulombic_Efficiency' not in df.columns and all(col in df.columns for col in ['Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)']):
            df['Coulombic_Efficiency'] = (df['Discharge_Capacity(Ah)'] / df['Charge_Capacity(Ah)']) * 100
        
        # Calculate Energy Efficiency if needed
        if 'Energy_Efficiency' not in df.columns and all(col in df.columns for col in ['Charge_Energy(Wh)', 'Discharge_Energy(Wh)']):
            df['Energy_Efficiency'] = (df['Discharge_Energy(Wh)'] / df['Charge_Energy(Wh)']) * 100
        
        # Calculate Energy Density if needed
        if 'Energy_Density' not in df.columns and all(col in df.columns for col in ['Discharge_Energy(Wh)', 'Discharge_Capacity(Ah)']):
            df['Energy_Density'] = df['Discharge_Energy(Wh)'] / df['Discharge_Capacity(Ah)']
        
        return df
    
    def predict_battery_states(self, new_data=None):
        """Predict SOC and SOH for new battery data"""
        if self.soc_model is None or self.soh_model is None:
            raise ValueError("Models not trained. Run train_models() first.")
            
        if new_data is None:
            # Use the last cycle as an example of new data
            st.write("\nUsing the last cycle as sample data for prediction...")
            new_data = self.processed_df[self.processed_df['Cycle_Index'] == 
                                       self.processed_df['Cycle_Index'].max()].copy()
        
        try:
            # Prepare input data with derived features
            prepared_data = self.prepare_input_data(new_data)
            
            # Prepare features for SOC prediction
            X_soc_cols = self.soc_feature_names
            missing_soc_cols = [col for col in X_soc_cols if col not in prepared_data.columns]
            
            if missing_soc_cols:
                st.warning(f"Missing features for SOC prediction: {missing_soc_cols}")
                for col in missing_soc_cols:
                    prepared_data[col] = 0  # Placeholder value
                    
            X_soc = prepared_data[X_soc_cols]
            
            # Prepare features for SOH prediction
            X_soh_cols = self.soh_feature_names
            missing_soh_cols = [col for col in X_soh_cols if col not in prepared_data.columns]
            
            if missing_soh_cols:
                st.warning(f"Missing features for SOH prediction: {missing_soh_cols}")
                for col in missing_soh_cols:
                    prepared_data[col] = 0  # Placeholder value
                    
            X_soh = prepared_data[X_soh_cols]
            
            # Predict SOC
            soc_predictions = self.soc_model.predict(X_soc)
            
            # Predict SOH
            soh_predictions = self.soh_model.predict(X_soh)
            
            # Add predictions to the dataframe
            result_data = new_data.copy()
            result_data['Predicted_SOC'] = soc_predictions
            result_data['Predicted_SOH'] = soh_predictions
            
            # Print prediction for the latest cycle
            st.write("\nPrediction Results:")
            st.write(f"Cycle Index: {result_data['Cycle_Index'].values[0]}")
            st.write(f"Predicted SOC: {result_data['Predicted_SOC'].values[0]:.2f}%")
            st.write(f"Predicted SOH: {result_data['Predicted_SOH'].values[0]:.2f}%")
            
            return result_data
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            raise
    
    def evaluate_model_results(self, df_with_predictions=None):
        """Evaluate model predictions with visualizations"""
        if self.processed_df is None:
            raise ValueError("No processed data available. Run preprocess_data() first.")

        if df_with_predictions is None:
            # If no predictions provided, make predictions on all data
            prepared_data = self.prepare_input_data(self.processed_df)
            
            X_soc = prepared_data[self.soc_feature_names]
            X_soh = prepared_data[self.soh_feature_names]

            df_with_predictions = self.processed_df.copy()
            df_with_predictions['Predicted_SOC'] = self.soc_model.predict(X_soc)
            df_with_predictions['Predicted_SOH'] = self.soh_model.predict(X_soh)
        else:
            # If predictions are provided for a subset (e.g., last cycle), 
            # filter the processed_df to match the cycle index
            df_with_predictions = df_with_predictions.reset_index(drop=True)  # Reset index of predictions
            cycle_indices_to_keep = df_with_predictions['Cycle_Index'].unique()
            filtered_df = self.processed_df[self.processed_df['Cycle_Index'].isin(cycle_indices_to_keep)]
            df_with_predictions = df_with_predictions[df_with_predictions['Cycle_Index'].isin(cycle_indices_to_keep)]

        # Ensure both DataFrames have the same index for alignment
        df_with_predictions = df_with_predictions.set_index(self.processed_df.index)
        
        fig = plt.figure(figsize=(14, 10))

        # Plot 1: Actual vs Predicted SOC
        plt.subplot(2, 2, 1)
        plt.scatter(self.processed_df['SOC'], df_with_predictions['Predicted_SOC'], alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')  # Diagonal line
        plt.xlabel('Actual SOC (%)')
        plt.ylabel('Predicted SOC (%)')
        plt.title('Actual vs Predicted SOC')
        plt.grid(True)

        
        # Plot 2: Actual vs Predicted SOH
        plt.subplot(2, 2, 2)
        plt.scatter(self.processed_df['SOH'], df_with_predictions['Predicted_SOH'], alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')  # Diagonal line
        plt.xlabel('Actual SOH (%)')
        plt.ylabel('Predicted SOH (%)')
        plt.title('Actual vs Predicted SOH')
        plt.grid(True)
        
        # Plot 3: SOC Prediction Error
        plt.subplot(2, 2, 3)
        soc_error = self.processed_df['SOC'] - df_with_predictions['Predicted_SOC']
        plt.hist(soc_error, bins=30)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('SOC Prediction Error Distribution')
        plt.grid(True)
        
        # Plot 4: SOH Prediction Error
        plt.subplot(2, 2, 4)
        soh_error = self.processed_df['SOH'] - df_with_predictions['Predicted_SOH']
        plt.hist(soh_error, bins=30)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('SOH Prediction Error Distribution')
        plt.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        return

    def run_analysis(self, file_path=None, model_type='randomforest'):
        """Run the complete battery analysis workflow"""
        if file_path:
            self.load_data(file_path)
        elif self.df is None:
            raise ValueError("No data loaded and no file path provided.")
            
        self.standardize_column_names()
        self.preprocess_data()
        self.visualize_battery_metrics()
        
        # Train models
        self.train_models(model_type=model_type)
        
        # Make predictions on test data
        if self.soc_model and self.soh_model:
            df_with_predictions = self.predict_battery_states()
            self.evaluate_model_results(df_with_predictions)
            
        return self.soc_model, self.soh_model, self.processed_df

# Streamlit App
def main():
    st.title("🔋 Battery State Analyzer")
    st.markdown("""
    This app analyzes battery cycling data to predict State of Charge (SOC) and 
    State of Health (SOH) using machine learning models.
    """)
    
    # File upload
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a battery data file (CSV or Excel)",
        type=["csv", "xlsx", "xls"]
    )
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Random Forest", "Gradient Boosting", "XGBoost"],
        index=0
    ).lower().replace(" ", "")
    
    # Initialize analyzer
    analyzer = BatteryAnalyzer()
    
    if uploaded_file is not None:
        try:
            # Run analysis
            with st.spinner("Analyzing battery data..."):
                analyzer.run_analysis(uploaded_file, model_type)
            
            st.success("Analysis complete!")
            
                
            # Manual prediction section
            st.header("Manual Prediction")
            st.write("Enter values for prediction:")
            
            # Create input form
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    cycle_index = st.number_input("Cycle Index", min_value=0)
                    voltage = st.number_input("Voltage (V)", min_value=0.0)
                    current = st.number_input("Current (A)", min_value=0.0)
                    
                with col2:
                    discharge_cap = st.number_input("Discharge Capacity (Ah)", min_value=0.0)
                    charge_cap = st.number_input("Charge Capacity (Ah)", min_value=0.0)
                    temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=100.0)
                
                submitted = st.form_submit_button("Predict")
                
                if submitted and analyzer.soc_model and analyzer.soh_model:
                    try:
                        # Create input dataframe
                        input_data = pd.DataFrame({
                            'Cycle_Index': [cycle_index],
                            'Voltage(V)': [voltage],
                            'Current(A)': [current],
                            'Discharge_Capacity(Ah)': [discharge_cap],
                            'Charge_Capacity(Ah)': [charge_cap],
                            'Temperature(C)': [temp]
                        })
                        
                        # Make prediction
                        result = analyzer.predict_battery_states(input_data)
                        
                        # Display results
                        st.success(f"Predicted SOC: {result['Predicted_SOC'].values[0]:.2f}%")
                        st.success(f"Predicted SOH: {result['Predicted_SOH'].values[0]:.2f}%")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
