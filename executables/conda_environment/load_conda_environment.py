import subprocess

# Specify the path to the YAML file containing environment information
yaml_file = '../../conda_environment.yml'

# Execute the "conda env create" command to create the Conda environment
command = f"conda env create -f {yaml_file}"
process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
output, _ = process.communicate()

# Check if the environment was created successfully
if process.returncode == 0:
    print("Conda environment created successfully.")
else:
    print("Failed to create Conda environment.")