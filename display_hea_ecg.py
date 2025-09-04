import wfdb
import numpy as np
from pathlib import Path

def read_ecg_metadata(file_path):
    """
    Read ECG file metadata from WFDB format files (.hea and .mat)
    
    Args:
        file_path (str): Path to the ECG file WITHOUT extension (e.g., 'JS03244')
    
    Returns:
        dict: Dictionary containing metadata information
    """
    try:
        # Read the header file which contains metadata
        record = wfdb.rdheader(file_path)
        
        # Extract basic signal information
        metadata = {
            'record_name': record.record_name,
            'sampling_frequency': record.fs,  # Hz
            'signal_length': record.sig_len,  # number of samples
            'duration_seconds': record.sig_len / record.fs if record.fs > 0 else 0,
            'duration_minutes': (record.sig_len / record.fs) / 60 if record.fs > 0 else 0,
            'number_of_signals': record.n_sig,
            'signal_names': record.sig_name,
            'units': record.units,
            'gain': record.adc_gain,
            'baseline': record.baseline,
            'comments': record.comments if hasattr(record, 'comments') else []
        }
        
        # Extract patient information from comments (if available)
        patient_info = {}
        if hasattr(record, 'comments') and record.comments:
            for comment in record.comments:
                comment = comment.strip()
                # Common patterns in ECG metadata
                if 'Age:' in comment:
                    patient_info['age'] = comment.split('Age:')[1].strip()
                elif 'Sex:' in comment or 'Gender:' in comment:
                    patient_info['gender'] = comment.split(':')[1].strip()
                elif 'Diagnosis:' in comment:
                    patient_info['diagnosis'] = comment.split('Diagnosis:')[1].strip()
                elif 'Medications:' in comment:
                    patient_info['medications'] = comment.split('Medications:')[1].strip()
                elif comment.startswith('#'):
                    # Some files use # for patient info
                    patient_info['additional_info'] = comment[1:].strip()
        
        metadata['patient_info'] = patient_info
        
        return metadata
    
    except Exception as e:
        print(f"Error reading ECG file: {e}")
        return None


def display_ecg_info(file_path):
    """
    Display ECG metadata in a readable format
    
    Args:
        file_path (str): Path to the ECG file WITHOUT extension
    """
    metadata = read_ecg_metadata(file_path)
    
    if metadata is None:
        print("Failed to read ECG metadata")
        return
    
    print("=" * 50)
    print(f"ECG FILE METADATA: {metadata['record_name']}")
    print("=" * 50)
    
    # Basic signal information
    print("\n📊 SIGNAL INFORMATION:")
    print(f"Sampling Frequency: {metadata['sampling_frequency']} Hz")
    print(f"Duration: {metadata['duration_seconds']:.2f} seconds ({metadata['duration_minutes']:.2f} minutes)")
    print(f"Total Samples: {metadata['signal_length']}")
    print(f"Number of Channels: {metadata['number_of_signals']}")
    
    # Signal details
    if metadata['signal_names']:
        print(f"\n📈 SIGNAL CHANNELS:")
        for i, (name, unit, gain) in enumerate(zip(
            metadata['signal_names'], 
            metadata['units'], 
            metadata['gain']
        )):
            print(f"  Channel {i+1}: {name} (Unit: {unit}, Gain: {gain})")
    
    # Patient information
    if metadata['patient_info']:
        print(f"\n👤 PATIENT INFORMATION:")
        for key, value in metadata['patient_info'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Additional comments
    if metadata['comments']:
        print(f"\n💬 ADDITIONAL COMMENTS:")
        for comment in metadata['comments']:
            print(f"  • {comment}")


# Alternative method using scipy for .mat files
def read_mat_file_alternative(mat_file_path):
    """
    Alternative method to read .mat files using scipy
    Useful if WFDB format is non-standard
    """
    try:
        from scipy.io import loadmat
        
        data = loadmat(mat_file_path)
        print("MAT file contents:")
        for key, value in data.items():
            if not key.startswith('__'):  # Skip metadata keys
                print(f"  {key}: {type(value)} - Shape: {getattr(value, 'shape', 'N/A')}")
        
        return data
    except Exception as e:
        print(f"Error reading MAT file: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Replace 'JS03244' with your actual file path (without extension)
    ecg_file = r"D:\ECG-dataset\G12EC\ptb-xl\g1\HR00001"  # This will look for JS03244.hea and JS03244.mat
    
    print("Method 1: Using WFDB library")
    display_ecg_info(ecg_file)
    
    print("\n" + "="*50)
    print("Method 2: Direct MAT file reading (if needed)")
    print("="*50)
    # mat_data = read_mat_file_alternative('JS03244.mat')

# Installation note:
# pip install wfdb scipy