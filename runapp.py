import streamlit as st
import os
import subprocess


def run_streamlit_app():
    # Define the path to your Streamlit app Python file
    
    name_of_file = "app3.py"
    app_file_path = os.path.join(os.getcwd(), name_of_file)


    # Construct the command to run the Streamlit app
    command = ["streamlit", "run", app_file_path]
    
    # Execute the command
    subprocess.Popen(command, shell=True)
# Call the function to run the Streamlit app
run_streamlit_app()