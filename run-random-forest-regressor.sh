#!/bin/bash

# Name of the environment variable
ENV_NAME=cse6242team114-rf

# Path of python script file
PYTHON_SCRIPT_PATH="python_scripts/RandomForestRegression.py"

# Check if environment exists
conda env list | awk '{print $1}' | grep -x $ENV_NAME
if [ $? -eq 0 ]; then
    echo "Environment exists"
else
    # Create a new environment, if it doesn't exist already
    conda create -y -n $ENV_NAME python=3.8
    echo "Environment $ENV_NAME created."
fi

# Activate the environment
conda activate $ENV_NAME

# Install requirements from requirements.txt in the environment
pip install -r requirements.txt

# Run the python script
python $PYTHON_SCRIPT_PATH

echo "Script executed successfully."