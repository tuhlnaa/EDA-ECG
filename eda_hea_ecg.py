import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
from collections import defaultdict

def create_directories(base_path, dataset_names):
    """
    Create recursive directory structure for each dataset
    
    Args:
        base_path (str): Base directory path
        dataset_names (list): List of dataset names
    """
    base_path = Path(base_path)
    
    for dataset in dataset_names:
        dataset_dir = base_path / dataset
        # Create subdirectories
        (dataset_dir / 'processed').mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels').mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'metadata').mkdir(parents=True, exist_ok=True)
        print(f"Created directories for dataset: {dataset}")

def load_dx_mapping(dx_map_path):
    """
    Load the Dx mapping CSV file
    
    Args:
        dx_map_path (str): Path to Dx_map.csv
        
    Returns:
        pd.DataFrame: DataFrame containing diagnosis mappings
    """
    try:
        df = pd.read_csv(dx_map_path)
        print(f"Loaded Dx mapping with {len(df)} entries")
        return df
    except Exception as e:
        print(f"Error loading Dx mapping: {e}")
        return None

def parse_dx_codes(dx_string):
    """
    Parse diagnosis codes from comment string
    
    Args:
        dx_string (str): Comma-separated diagnosis codes
        
    Returns:
        list: List of diagnosis codes
    """
    if not dx_string or dx_string == 'Unknown':
        return []
    
    # Split by comma and clean up
    codes = [code.strip() for code in dx_string.split(',')]
    return [code for code in codes if code]

def extract_patient_info_from_comments(comments):
    """
    Extract patient information from ECG comments
    
    Args:
        comments (list): List of comment strings
        
    Returns:
        dict: Dictionary with extracted patient info
    """
    patient_info = {
        'age': 'Unknown',
        'sex': 'Unknown', 
        'dx': 'Unknown',
        'rx': 'Unknown',
        'hx': 'Unknown',
        'sx': 'Unknown'
    }
    
    for comment in comments:
        comment = comment.strip()
        
        # Extract Age
        age_match = re.search(r'Age:\s*(\d+)', comment)
        if age_match:
            patient_info['age'] = age_match.group(1)
        
        # Extract Sex
        sex_match = re.search(r'Sex:\s*(\w+)', comment)
        if sex_match:
            patient_info['sex'] = sex_match.group(1)
        
        # Extract Dx (Diagnosis codes)
        dx_match = re.search(r'Dx:\s*([0-9,\s]+)', comment)
        if dx_match:
            patient_info['dx'] = dx_match.group(1).strip()
        
        # Extract Rx (Prescriptions)
        rx_match = re.search(r'Rx:\s*(\w+)', comment)
        if rx_match:
            patient_info['rx'] = rx_match.group(1)
        
        # Extract Hx (History)
        hx_match = re.search(r'Hx:\s*(\w+)', comment)
        if hx_match:
            patient_info['hx'] = hx_match.group(1)
        
        # Extract Sx (Symptoms)
        sx_match = re.search(r'Sx:\s*(\w+)', comment)
        if sx_match:
            patient_info['sx'] = sx_match.group(1)
    
    return patient_info

def read_ecg_with_labels(file_path, dx_mapping_df):
    """
    Read ECG file and extract labels with mapping
    
    Args:
        file_path (str): Path to ECG file without extension
        dx_mapping_df (pd.DataFrame): DataFrame with diagnosis mappings
        
    Returns:
        dict: ECG metadata with mapped labels
    """
    try:
        # Read the header file
        record = wfdb.rdheader(file_path)
        
        # Extract patient info from comments
        patient_info = extract_patient_info_from_comments(record.comments)
        
        # Parse diagnosis codes
        dx_codes = parse_dx_codes(patient_info['dx'])
        
        # Map diagnosis codes to descriptions
        mapped_diagnoses = []
        for code in dx_codes:
            # Look up in mapping DataFrame
            mapping = dx_mapping_df[dx_mapping_df['SNOMED CT Code'].astype(str) == str(code)]
            if not mapping.empty:
                mapped_diagnoses.append({
                    'code': code,
                    'diagnosis': mapping.iloc[0]['Dx'],
                    'abbreviation': mapping.iloc[0]['Abbreviation']
                })
            else:
                mapped_diagnoses.append({
                    'code': code,
                    'diagnosis': 'Unknown',
                    'abbreviation': 'Unknown'
                })
        
        # Compile metadata
        metadata = {
            'record_name': record.record_name,
            'sampling_frequency': record.fs,
            'signal_length': record.sig_len,
            'duration_seconds': record.sig_len / record.fs if record.fs > 0 else 0,
            'number_of_signals': record.n_sig,
            'patient_info': patient_info,
            'dx_codes': dx_codes,
            'mapped_diagnoses': mapped_diagnoses,
            'raw_comments': record.comments
        }
        
        return metadata
        
    except Exception as e:
        print(f"Error reading ECG file {file_path}: {e}")
        return None

def process_dataset(dataset_path, dx_mapping_df, dataset_name, output_base_path):
    """
    Process all ECG files in a dataset directory
    
    Args:
        dataset_path (str): Path to dataset directory
        dx_mapping_df (pd.DataFrame): Diagnosis mapping DataFrame
        dataset_name (str): Name of the dataset
        output_base_path (str): Base output path
        
    Returns:
        dict: Processing results and statistics
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_base_path) / dataset_name
    
    # Find all .hea files (header files)
    hea_files = list(dataset_path.rglob('*.hea'))
    
    all_records = []
    unique_diagnoses = set()
    sample_count = 0
    
    # Lists to collect sampling frequencies and durations for statistics
    sampling_frequencies = []
    durations_seconds = []
    
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Found {len(hea_files)} ECG files")
    
    for hea_file in hea_files:
        # Get file path without extension
        file_stem = str(hea_file.with_suffix(''))
        
        # Read ECG with labels
        metadata = read_ecg_with_labels(file_stem, dx_mapping_df)
        
        if metadata:
            all_records.append(metadata)
            sample_count += 1
            
            # Collect sampling frequency and duration data for statistics
            sampling_frequencies.append(metadata['sampling_frequency'])
            durations_seconds.append(metadata['duration_seconds'])
            
            # Collect unique diagnoses
            for dx in metadata['mapped_diagnoses']:
                if dx['diagnosis'] != 'Unknown':
                    unique_diagnoses.add(dx['diagnosis'])
    
    # Create labels DataFrame for this dataset
    labels_data = []
    for record in all_records:
        for dx in record['mapped_diagnoses']:
            labels_data.append({
                'record_name': record['record_name'],
                'age': record['patient_info']['age'],
                'sex': record['patient_info']['sex'],
                'dx_code': dx['code'],
                'diagnosis': dx['diagnosis'],
                'abbreviation': dx['abbreviation']
            })
    
    if labels_data:
        labels_df = pd.DataFrame(labels_data)
        
        # Save complete labels file
        labels_file = output_path / 'labels' / f'{dataset_name}_labels.csv'
        labels_df.to_csv(labels_file, index=False)
        print(f"Saved labels file: {labels_file}")
        
        # Create unique diagnoses file
        unique_dx_df = labels_df[['diagnosis', 'dx_code', 'abbreviation']].drop_duplicates()
        unique_dx_file = output_path / 'labels' / f'{dataset_name}_unique_diagnoses.csv'
        unique_dx_df.to_csv(unique_dx_file, index=False)
        print(f"Saved unique diagnoses file: {unique_dx_file}")
    
    # Calculate statistics for sampling frequencies and durations
    sf_stats = None
    duration_stats = None
    
    if sampling_frequencies:
        sf_stats = {
            'min': min(sampling_frequencies),
            'max': max(sampling_frequencies),
            'average': sum(sampling_frequencies) / len(sampling_frequencies)
        }
    
    if durations_seconds:
        duration_stats = {
            'min': min(durations_seconds),
            'max': max(durations_seconds),
            'average': sum(durations_seconds) / len(durations_seconds)
        }
    
    # Create summary statistics
    stats = {
        'dataset_name': dataset_name,
        'total_samples': sample_count,
        'unique_diagnoses_count': len(unique_diagnoses),
        'unique_diagnoses': list(unique_diagnoses),
        'sampling_frequency_stats': sf_stats,
        'duration_seconds_stats': duration_stats
    }
    
    # Save statistics
    stats_file = output_path / 'metadata' / f'{dataset_name}_stats.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total Samples: {sample_count}\n")
        f.write(f"Unique Diagnoses: {len(unique_diagnoses)}\n")
        
        if sf_stats:
            f.write(f"\nSampling Frequency Statistics (Hz):\n")
            f.write(f"  Minimum: {sf_stats['min']:.2f}\n")
            f.write(f"  Maximum: {sf_stats['max']:.2f}\n")
            f.write(f"  Average: {sf_stats['average']:.2f}\n")
        
        if duration_stats:
            f.write(f"\nDuration Statistics (seconds):\n")
            f.write(f"  Minimum: {duration_stats['min']:.2f}\n")
            f.write(f"  Maximum: {duration_stats['max']:.2f}\n")
            f.write(f"  Average: {duration_stats['average']:.2f}\n")
        
        f.write(f"\nDiagnoses List:\n")
        for dx in sorted(unique_diagnoses):
            f.write(f"  - {dx}\n")
    
    print(f"Dataset {dataset_name} processed: {sample_count} samples, {len(unique_diagnoses)} unique diagnoses")
    
    return stats

def filter_dx_mapping(dx_mapping_df, filter_criteria=None):
    """
    Filter the Dx mapping DataFrame based on criteria
    
    Args:
        dx_mapping_df (pd.DataFrame): Original mapping DataFrame
        filter_criteria (dict): Dictionary with filter criteria
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if filter_criteria is None:
        return dx_mapping_df
    
    filtered_df = dx_mapping_df.copy()
    
    # Example filters - modify as needed
    if 'exclude_unknown' in filter_criteria:
        filtered_df = filtered_df[~filtered_df['Dx'].str.contains('Unknown', case=False, na=False)]
    
    if 'include_only' in filter_criteria:
        include_terms = filter_criteria['include_only']
        mask = False
        for term in include_terms:
            mask |= filtered_df['Dx'].str.contains(term, case=False, na=False)
        filtered_df = filtered_df[mask]
    
    return filtered_df

def main():
    """
    Main processing function
    """
    # Configuration (Path to your Dx mapping file)
    dx_map_path = 'G12EC/Dx_map.csv'
    datasets_config = {
        'dataset1': './G12EC/cpsc_2018',
        #'dataset2': '/path/to/dataset2',
        # Add more datasets as needed
    }
    output_base_path = 'processed_ecg_data'
    
    # Load Dx mapping
    print("Loading diagnosis mapping...")
    dx_mapping_df = load_dx_mapping(dx_map_path)
    if dx_mapping_df is None:
        print("Failed to load Dx mapping. Exiting.")
        return
    
    # Optional: Filter Dx mapping
    # filter_criteria = {'exclude_unknown': True}
    # dx_mapping_df = filter_dx_mapping(dx_mapping_df, filter_criteria)
    
    # Create output directories
    dataset_names = list(datasets_config.keys())
    create_directories(output_base_path, dataset_names)
    
    # Process each dataset
    all_stats = []
    total_samples = 0
    
    # Lists to collect overall statistics across all datasets
    all_sampling_frequencies = []
    all_durations_seconds = []
    
    for dataset_name, dataset_path in datasets_config.items():
        if os.path.exists(dataset_path):
            stats = process_dataset(dataset_path, dx_mapping_df, dataset_name, output_base_path)
            all_stats.append(stats)
            total_samples += stats['total_samples']
            
            # Collect data for overall statistics
            if stats['sampling_frequency_stats']:
                # Note: We would need the raw data for accurate overall stats
                # For now, we'll approximate using the dataset averages
                pass
            
        else:
            print(f"Warning: Dataset path does not exist: {dataset_path}")
    
    # Print final summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total datasets processed: {len(all_stats)}")
    print(f"Total samples across all datasets: {total_samples}")
    
    for stats in all_stats:
        print(f"\n{stats['dataset_name']}:")
        print(f"  Samples: {stats['total_samples']}")
        print(f"  Unique diagnoses: {stats['unique_diagnoses_count']}")
        
        if stats['sampling_frequency_stats']:
            sf_stats = stats['sampling_frequency_stats']
            print(f"  Sampling Frequency (Hz) - Min: {sf_stats['min']:.2f}, "
                  f"Max: {sf_stats['max']:.2f}, Avg: {sf_stats['average']:.2f}")
        
        if stats['duration_seconds_stats']:
            dur_stats = stats['duration_seconds_stats']
            print(f"  Duration (seconds) - Min: {dur_stats['min']:.2f}, "
                  f"Max: {dur_stats['max']:.2f}, Avg: {dur_stats['average']:.2f}")
    
    # Save combined summary
    summary_file = Path(output_base_path) / 'combined_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("ECG Datasets Processing Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total datasets: {len(all_stats)}\n")
        f.write(f"Total samples: {total_samples}\n\n")
        
        for stats in all_stats:
            f.write(f"Dataset: {stats['dataset_name']}\n")
            f.write(f"  Samples: {stats['total_samples']}\n")
            f.write(f"  Unique diagnoses: {stats['unique_diagnoses_count']}\n")
            
            if stats['sampling_frequency_stats']:
                sf = stats['sampling_frequency_stats']
                f.write(f"  Sampling Frequency (Hz) - Min: {sf['min']:.2f}, "
                       f"Max: {sf['max']:.2f}, Avg: {sf['average']:.2f}\n")
            
            if stats['duration_seconds_stats']:
                dur = stats['duration_seconds_stats']
                f.write(f"  Duration (seconds) - Min: {dur['min']:.2f}, "
                       f"Max: {dur['max']:.2f}, Avg: {dur['average']:.2f}\n")
            
            f.write(f"  Diagnoses: {', '.join(stats['unique_diagnoses'][:5])}...\n\n")

if __name__ == "__main__":
    main()

    
"""
Output Structure:

processed_ecg_data/
├── dataset1/
│   ├── labels/
│   │   ├── dataset1_labels.csv
│   │   └── dataset1_unique_diagnoses.csv
│   ├── metadata/
│   │   └── dataset1_stats.txt
│   └── processed/
├── dataset2/
│   └── ...
└── combined_summary.txt
"""