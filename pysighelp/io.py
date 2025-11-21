import pickle
import os
import numpy as np
import time

def deep_compare(a, b):
    """Recursively compare nested structures with NumPy array support."""
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(deep_compare(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(deep_compare(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b

def save_pickle(path, name, data, timestamp=True, check_integrity=True):
    """
    Save data to a pickle file with optional timestamp and integrity check.
    Parameters
    ----------
    path : str
        Directory path where the file will be saved.
    name : str
        Name of the file (without extension).
    data : dict, list, or np.ndarray
        Data to be saved.
    timestamp : bool, optional
        If True, adds a timestamp to the filename. Default is True.
    check_integrity : bool, optional
        If True, performs an integrity check after saving. Default is True.


    Example
    -------
    >>> save_pickle('/path/to/dir', 'my_data', {'key': 'value'})
    >>> save_pickle('/path/to/dir', 'my_data', np.array([1, 2, 3]), timestamp=False)
    >>> save_pickle('/path/to/dir', 'my_data', [1, 2, 3], check_integrity=False)
    """
    try:
        #Add timestamp after filename
        timestamp_value = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(path):
            os.makedirs(path)
        if not isinstance(name, str):
            raise ValueError("Name must be a string.")
        if not isinstance(data, (dict, list, np.ndarray)):
            raise ValueError("Data must be a dictionary, list, or NumPy array.")
        if not os.path.isdir(path):
            raise ValueError(f"Path '{path}' is not a directory.")
        if not os.access(path, os.W_OK):
            raise PermissionError(f"Path '{path}' is not writable.")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Path '{path}' is not readable.")

        if not timestamp:
            filename = f"{path}/{name}.pickle"
        else:
            # Add timestamp to filename
            filename = f"{path}/{timestamp_value}_{name}.pickle"


        if check_integrity:
            with open(filename, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Data saved to {filename}")

            # Verify file size is nonzero
            if os.path.getsize(filename) == 0:
                raise Exception("File is empty after saving.")

            with open(filename, 'rb') as f:
                loaded_data = pickle.load(f)
                if deep_compare(loaded_data, data):
                    print("Data integrity check passed.")
                else:
                    raise Exception("Data integrity check failed.")
    except EOFError:
        raise Exception("EOFError: File may be empty or corrupted.")
    except Exception as e:
        raise Exception(f"Error during save or verification: {e}")
    
def parse_ni_results(results):
    """
    Parse results from a National Instruments data acquisition system.

    Parameters
    ----------
    results : dict
        Dictionary containing the results from the NI system from get_measurement_dict() method.

    Returns
    -------
    parsed_results : dict
        Dictionary with parsed results.
    """
    parsed_results = {}

    for key in results.keys():
        for i, name in enumerate(results[key]['channel_names']):
            parsed_results[name] = results[key]['data'][:,i]
        parsed_results['t'] = results[key]['time']
    
    return parsed_results
