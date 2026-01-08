import os
import sys
import numpy as np
from CellTOSG_Loader import CellTOSGDataLoader

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.path_config import get_path

# Module-level constants (if these values don't change)
DATA_ROOT = get_path('external.omnicell_data_root', absolute=True)
DOWNSTREAM_TASK = 'gender' # gender or disease
LABEL_COLUMN = 'gender'    # gender or disease
SAMPLE_RATIO = 1.0
SAMPLE_SIZE = None
BALANCED = True
SHUFFLE = True
RANDOM_STATE = 2025
TRAIN_TEXT = True
TRAIN_BIO = False
# Use absolute path to avoid permission errors when running as microservice
OUTPUT_DIR = get_path('data.dataset_outputs', absolute=True, create=True)

def omic_fetch(fetch_dict: dict) -> object:
    """
    Fetch data from CellTOSG based on the provided dictionary.

    Args:
        fetch_dict (dict): A dictionary containing keys and values to fetch data.

    Returns:
        object: The fetched data from CellTOSG.
    """
    tissue_general = fetch_dict.get("organ", None)
    tissue = fetch_dict.get("tissue", None)
    suspension_type = fetch_dict.get("suspension_type", None)
    cell_type = fetch_dict.get("cell type", None)
    disease_name = fetch_dict.get("disease", None)
    gender = fetch_dict.get("gender", None)

    # Load dataset with conditions
    dataset = CellTOSGDataLoader(
        root=DATA_ROOT,
        conditions={
            "tissue_general": tissue_general,
            "tissue": tissue,
            "suspension_type": suspension_type,
            "cell_type": cell_type,
            "disease": disease_name,
            "gender": gender,
        },
        downstream_task=DOWNSTREAM_TASK,
        label_column=LABEL_COLUMN,
        sample_ratio=SAMPLE_RATIO,
        sample_size=SAMPLE_SIZE,
        balanced=BALANCED,
        shuffle=SHUFFLE,
        random_state=RANDOM_STATE,
        train_text=TRAIN_TEXT,
        train_bio=TRAIN_BIO,
        output_dir=OUTPUT_DIR
    )
    
    data_dict, file_path_dict = save_fetch_data(dataset, OUTPUT_DIR)
    return data_dict, file_path_dict, disease_name


def save_fetch_data(dataset: object, output_dir: str) -> tuple:
    """
    Save the fetched dataset to files and return data dictionary and file paths.

    Args:
        dataset (object): The dataset to save.
        output_dir (str): The output directory path.

    Returns:
        tuple: (data_dict, file_paths) containing the data arrays and their file paths.
    """
    omic_feature = dataset.data
    omic_label = dataset.labels
    normal_omic_feature = omic_feature[omic_label == 0]
    disease_omic_feature = omic_feature[omic_label == 1]
    
    # Define data and file structure
    data_dict = {
        "normal_omic_feature": normal_omic_feature,
        "disease_omic_feature": disease_omic_feature,
        "omic_label": omic_label
    }
    
    # Save arrays and build file paths
    file_path_dict = {}
    for key, data in data_dict.items():
        file_path = os.path.join(output_dir, f"{key}.npy")
        np.save(file_path, data)
        file_path_dict[key] = file_path
    
    return data_dict, file_path_dict