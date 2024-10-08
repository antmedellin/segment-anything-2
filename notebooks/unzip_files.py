import os
import zipfile

def unzip_files(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir():
                continue

            print(f'Extracting {file_info.filename} to {extract_to}')

            # Extract the file to the specified directory
            zip_ref.extract(file_info, extract_to)

            # Check if the file is a zip file
            if file_info.filename.endswith('.zip'):
                # Create a folder to extract the nested zip file
                nested_folder = os.path.join(extract_to, os.path.splitext(file_info.filename)[0])
                os.makedirs(nested_folder, exist_ok=True)

                # Extract the nested zip file to the created folder
                unzip_files(os.path.join(extract_to, file_info.filename), nested_folder)

# Example usage
# zip_file_path = '/home/anthony/HyperWorkspace/hsi_tracking/training.zip'
zip_file_path = '/workspaces/hsi_tracking/datasets/ranking.zip'
extract_to = '/workspaces/hsi_tracking/datasets/'

unzip_files(zip_file_path, extract_to)
print('Files extracted successfully.')