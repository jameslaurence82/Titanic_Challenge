"""
Titanic - Machine Learning from Disaster.

@Author James Laurence
@Date August 25th, 2024
"""
# %% Imports, Filepaths, Variables
import os
import logging
from kaggle.api.kaggle_api_extended import KaggleApi


# %% Extract data using the Kaggle API function
def extract_data(competition, file_name, file_path, force=False):  # Added force as a parameter
    api = KaggleApi()
    api.authenticate()  # Authenticate using credentials stored in ~/.kaggle/kaggle.json

    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)

    # Download the specific file from the competition dataset
    api.competition_download_file(competition, file_name, path=file_path, force=force)

    # Unzip the file if it's zipped
    file_zip_path = os.path.join(file_path, file_name + '.zip')
    if os.path.exists(file_zip_path):
        with ZipFile(file_zip_path, 'r') as zip_ref:
            zip_ref.extractall(file_path)
        os.remove(file_zip_path)


# %% Main Function
def main(project_dir):
    # Get logger
    logger = logging.getLogger(__name__)
    logger.info('Getting raw data')

    # Competition name
    competition = 'titanic'

    # File names
    train_file = 'train.csv'
    test_file = 'test.csv'

    # File paths
    raw_data_path = os.path.join(project_dir, 'data', 'raw')

    # Extract data with force option enabled
    extract_data(competition, train_file, raw_data_path, force=True)
    extract_data(competition, test_file, raw_data_path, force=True)

    logger.info('Downloaded raw training and test data')


if __name__ == '__main__':
    # Getting root directory
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    
    # Setup logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Call the main function
    main(project_dir)
