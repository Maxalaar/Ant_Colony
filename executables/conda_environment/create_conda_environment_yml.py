import os
import subprocess
import yaml

if __name__ == '__main__':
    # Get the current Conda environment
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    if current_env is None:
        print("You are not in a Conda environment. Please activate a Conda environment before running this script.")
        exit(1)

    # Run the "conda env export" command to obtain dependencies from the current environment
    command = f"conda env export -n {current_env}"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, _ = process.communicate()

    # Parse the output to extract environment information
    env_info = yaml.safe_load(output.decode())

    # Write the information to a YAML file
    with open('../conda_environment.yml', 'w') as yaml_file:
        yaml.dump(env_info, yaml_file, default_flow_style=False)

    print(f"conda_environment.yml file created from the current Conda environment: {current_env}.")