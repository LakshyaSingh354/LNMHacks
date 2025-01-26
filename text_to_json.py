import os

def process_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            # Display the file being processed
            print(f"Processing file: {filename}")
            
            # Read the contents of the file
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Remove the first and last lines
            if len(lines) > 2:
                lines = lines[1:-1]
                print(f"Removed first and last lines from {filename}")
            else:
                print(f"File {filename} has too few lines to remove.")
            
            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)
            
            # Rename the file with a .json extension
            new_filename = os.path.splitext(filename)[0] + ".json"
            new_file_path = os.path.join(directory, new_filename)
            os.rename(file_path, new_file_path)
            print(f"Renamed {filename} to {new_filename}\n")

# Specify the directory containing the txt files
directory = "data_json"

# Process the files
process_files(directory)
