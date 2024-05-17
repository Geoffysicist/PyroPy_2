from typing import Optional
from pathlib import Path
import yaml
import base64
from io import StringIO
import pandas as pd

def parse_weather(contents, column_names):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string).decode('utf-8')
    try:
        # Find the header row containing 'date'
        header_row = next(i for i, line in enumerate(decoded.split('\n')) if 'date' in line.lower())
        df = pd.read_csv(StringIO(decoded), header=header_row, na_values='--')
        df.dropna(inplace=True)
        df['DateTime'] = pd.to_datetime(df['Local Date'] + ' ' + df['Local Time'])
        df = df.rename(columns={value: key for key, value in column_names.items()})
        return df

    except Exception as e:
        print(e)
        return pd.DataFrame()

def find_header_row(csv_file: str, target_string: str) -> Optional[int]:
    """
    Find the row number of the header in a CSV file containing a specific target string.

    Args:
        csv_file (str): Path to the CSV file.
        target_string (str): The target string to search for in the header.

    Returns:
        int or None: The row number of the header containing the target string,
                     or None if the header is not found.
    """
    # Open the CSV file
    with open(csv_file, 'r') as file:
        # Iterate through each line
        for line_num, line in enumerate(file):
            # Check if the line contains the target string
            if target_string.lower() in line.lower():
                return line_num
    return None  # If header row not found

def get_settings(pth: Path) -> dict:
    '''loads the settings from a yaml file
    
    args:
        pth: path to the yaml file
    raises:
        FileNotFoundError if path doesn't exist
    '''
    pth = Path(pth)
    try:
        with open(pth) as f:
            settings = yaml.safe_load(f)
            if not isinstance(settings, dict):
                return TypeError(f'{pth} is not a validly formatted settings yaml')
                
    except FileNotFoundError as e:
        # msg = "Error: settings.yaml file not found - see README.md"
        # logger.info(msg)
        # sys.exit(f'\n{msg}\n')
        return e
    
    return settings