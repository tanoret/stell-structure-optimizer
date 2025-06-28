# profile_manager.py

"""
This module serves as the production-ready processing layer between raw profile
data (stored in pandas DataFrames) and the structural analysis code.

It takes DataFrames for different profile types, robustly cleans and standardizes
the data, and populates a unified database. It provides a simple interface to
retrieve all necessary properties for a given profile name.
"""
import pandas as pd
import numpy as np
import re # Import regular expressions for better string cleaning

# This dictionary will hold the final, clean data for all profiles.
PROFILES_DATABASE = {}

def _clean_column_names(df):
    """
    Robustly standardizes column names to a consistent 'lowercase_with_underscores' format.
    Handles different capitalization, special characters, and spacing.
    """
    # This function now uses your method of directly renaming the first column.
    df.rename(columns={df.columns[0]: 'profile'}, inplace=True)

    new_cols = {}
    for col in df.columns:
        if col == 'profile':
            new_cols[col] = col
            continue

        # For all other columns: make lowercase, remove units in parentheses
        clean = col.lower().split('(')[0].strip()
        # Replace common special characters with spaces first
        clean = re.sub(r'[=/]', ' ', clean)
        # Replace any sequence of non-alphanumeric chars with a single underscore
        clean = re.sub(r'[^a-z0-9]+', '_', clean)
        clean = clean.strip('_')
        new_cols[col] = clean

    df = df.rename(columns=new_cols)
    return df

def _process_i_channel_df(df):
    """Processes I-beams and Channels (IPN, IPE, UPN, W)."""
    df = _clean_column_names(df)
    processed_profiles = {}

    prop_map = {'A': 'ag', 'Iz': 'x_x_ix', 'Iy': 'y_y_iy', 'J': 'j', 'd': 'dimensiones_d', 'bf': 'bf'}

    for _, row in df.iterrows():
        # --- FIX: Standardize profile name to uppercase and remove whitespace ---
        name = re.sub(r'\s+', '', str(row['profile'])).upper()
        props = {}
        try:
            props['A'] = float(row[prop_map['A']])
            props['Iz'] = float(row[prop_map['Iz']])
            props['Iy'] = float(row[prop_map['Iy']])
            props['J'] = float(row.get(prop_map['J'], 0.0))
            props['cy'] = float(row[prop_map['d']]) / 2.0
            props['cz'] = float(row[prop_map['bf']]) / 2.0
            processed_profiles[name] = {k: v for k, v in props.items() if pd.notna(v)}
        except (KeyError, ValueError) as e:
            print(f"Skipping I-beam/Channel profile '{name}' due to missing/invalid data: {e}")
            continue
    return processed_profiles

def _process_hss_square_df(df):
    """Processes square hollow sections."""
    df = _clean_column_names(df)
    processed_profiles = {}

    prop_map = {'A': 'ag', 'Iz': 'ix_iy', 'Iy': 'ix_iy', 'J': 'j', 'B': 'b'}

    for _, row in df.iterrows():
        # --- FIX: Standardize profile name to uppercase and remove whitespace ---
        name = re.sub(r'\s+', '', str(row['profile'])).upper()
        props = {}
        try:
            props['A'] = float(row[prop_map['A']])
            props['Iz'] = float(row[prop_map['Iz']])
            props['Iy'] = float(row[prop_map['Iy']])
            props['J'] = float(row.get(prop_map['J'], 0.0))
            B = float(row[prop_map['B']])
            props['cy'] = B / 2.0
            props['cz'] = B / 2.0
            processed_profiles[name] = {k: v for k, v in props.items() if pd.notna(v)}
        except (KeyError, ValueError) as e:
            print(f"Skipping Square HSS profile '{name}' due to missing/invalid data: {e}")
            continue
    return processed_profiles

def _process_hss_rectangular_df(df):
    """Processes rectangular hollow sections."""
    df = _clean_column_names(df)
    processed_profiles = {}

    prop_map = {'A': 'ag', 'Iz': 'ix', 'Iy': 'iy', 'J': 'j', 'H': 'h', 'B': 'b'}

    for _, row in df.iterrows():
        # --- FIX: Standardize profile name to uppercase and remove whitespace ---
        name = re.sub(r'\s+', '', str(row['profile'])).upper()
        props = {}
        try:
            props['A'] = float(row[prop_map['A']])
            props['Iz'] = float(row[prop_map['Iz']])
            props['Iy'] = float(row[prop_map['Iy']])
            props['J'] = float(row.get(prop_map['J'], 0.0))
            props['cy'] = float(row[prop_map['H']]) / 2.0
            props['cz'] = float(row[prop_map['B']]) / 2.0
            processed_profiles[name] = {k: v for k, v in props.items() if pd.notna(v)}
        except (KeyError, ValueError) as e:
            print(f"Skipping Rectangular HSS profile '{name}' due to missing/invalid data: {e}")
            continue
    return processed_profiles

def _process_hss_circular_df(df):
    """Processes circular hollow sections."""
    df = _clean_column_names(df)
    processed_profiles = {}

    prop_map = {'A': 'ag', 'I': 'i', 'J': 'j', 'D': 'd'}

    for _, row in df.iterrows():
        # --- FIX: Standardize profile name to uppercase and remove whitespace ---
        name = re.sub(r'\s+', '', str(row['profile'])).upper()
        props = {}
        try:
            props['A'] = float(row[prop_map['A']])
            I_val = float(row[prop_map['I']])
            props['Iz'] = I_val
            props['Iy'] = I_val
            props['J'] = float(row.get(prop_map['J'], 0.0))
            D = float(row[prop_map['D']])
            props['cy'] = D / 2.0
            props['cz'] = D / 2.0
            processed_profiles[name] = {k: v for k, v in props.items() if pd.notna(v)}
        except (KeyError, ValueError) as e:
            print(f"Skipping Circular HSS profile '{name}' due to missing/invalid data: {e}")
            continue
    return processed_profiles

def _process_l_df(df):
    """Processes L-profiles (angles)."""
    df = _clean_column_names(df)
    processed_profiles = {}

    prop_map = {'A': 'ag', 'Iz': 'x_x_y_y_ix_iy', 'Iy': 'x_x_y_y_ix_iy', 'J': 'j', 'b': 'dimensiones_b'}

    for _, row in df.iterrows():
        # --- FIX: Standardize profile name to uppercase and remove whitespace ---
        name = re.sub(r'\s+', '', str(row['profile'])).upper()
        props = {}
        try:
            props['A'] = float(row[prop_map['A']])
            props['Iz'] = float(row[prop_map['Iz']])
            props['Iy'] = float(row[prop_map['Iy']])
            props['J'] = float(row.get(prop_map['J'], 0.0))
            b = float(row[prop_map['b']])
            props['cy'] = b
            props['cz'] = b
            processed_profiles[name] = {k: v for k, v in props.items() if pd.notna(v)}
        except (KeyError, ValueError) as e:
            print(f"Skipping L-profile '{name}' due to missing/invalid data: {e}")
            continue
    return processed_profiles


def populate_database(df_dict: dict):
    """Populates the master database from a dictionary of DataFrames."""
    global PROFILES_DATABASE

    processing_map = {
        'ipn': _process_i_channel_df,
        'upn': _process_i_channel_df,
        'square': _process_hss_square_df,
        'rectangular': _process_hss_rectangular_df,
        'circular': _process_hss_circular_df,
        'l': _process_l_df
    }

    for profile_type, df in df_dict.items():
        profile_type_key = profile_type.lower()
        if profile_type_key in processing_map:
            print(f"Processing {profile_type} profiles...")
            # Create a copy to avoid modifying the original DataFrame in memory
            processed_data = processing_map[profile_type_key](df.copy())
            PROFILES_DATABASE.update(processed_data)
            print(f"-> Added {len(processed_data)} profiles.")
        else:
            print(f"Warning: No processing function for type '{profile_type}'.")

    print(f"\nDatabase populated. Total profiles: {len(PROFILES_DATABASE)}")


def get_beam_properties(profile_name: str, E: float, nu: float, rho: float) -> dict:
    """Retrieves section properties from the database and combines them with material properties."""
    try:
        # --- FIX: Standardize lookup name to uppercase and remove whitespace ---
        profile_name = re.sub(r'\s+', '', profile_name).upper()
        section_props = PROFILES_DATABASE[profile_name]
        full_props = {'E': E, 'nu': nu, 'rho': rho, **section_props}
        return full_props
    except KeyError:
        print(f"ERROR: Profile '{profile_name}' not found in the database.")
        raise
