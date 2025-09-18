import streamlit as st
import subprocess
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="Model Training Hub", page_icon="🚀", layout="wide")

# --- Page Title and Introduction ---
st.title("🚀 Model Training Hub")
st.markdown("""
This page serves as the control center for the machine learning pipeline. You can trigger a new model training run and access key project resources like the MLflow UI and data quality profiles.
""")

# --- Section 1: Run Training Pipeline ---
st.header("1. Trigger a New Model Training Run")
st.warning("This process can take a few minutes. Please ensure the MLflow tracking server is running in a separate terminal before you begin.", icon="⚠️")

script_path = "run_pipeline.sh"

if 'pipeline_output' not in st.session_state:
    st.session_state.pipeline_output = ""
if 'pipeline_error' not in st.session_state:
    st.session_state.pipeline_error = ""

if st.button("🚀 Run Training Pipeline"):
    st.session_state.pipeline_output = ""
    st.session_state.pipeline_error = ""
    
    if not os.path.exists(script_path):
        st.error(f"Error: The script '{script_path}' was not found in the project root directory.")
    else:
        with st.spinner("Executing training pipeline... This may take a moment."):
            try:
                # We execute the script from the project root. Streamlit runs from the root.
                process = subprocess.Popen(
                    [f"./{script_path}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd="." # Ensure it runs from the project root
                )

                # Stream output
                stdout_placeholder = st.empty()
                stderr_placeholder = st.empty()
                full_stdout = ""
                full_stderr = ""
                
                for line in iter(process.stdout.readline, ''):
                    full_stdout += line
                    stdout_placeholder.code(full_stdout, language="bash")
                
                process.stdout.close()
                process.wait()

                full_stderr = process.stderr.read()
                if full_stderr:
                     stderr_placeholder.error(full_stderr)

                st.session_state.pipeline_output = full_stdout
                st.session_state.pipeline_error = full_stderr

                if process.returncode == 0:
                    st.success("Pipeline executed successfully!")
                else:
                    st.error("Pipeline execution failed. See error details above.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


# --- Section 2: Project Resources ---
st.markdown("---")
st.header("2. Access Project Resources")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("**MLflow Tracking UI**")
    st.markdown("Track experiments, compare runs, and manage models.")
    st.link_button("Go to MLflow", "http://127.0.0.1:5000")

with col2:
    st.info("**Raw Data Profile**")
    st.markdown("An in-depth report on the initial, unprocessed training data.")
    if st.button("View Raw Profile"):
        st.session_state.show_profile = 'raw'


with col3:
    st.info("**Processed Data Profile**")
    st.markdown("An in-depth report on the data after feature engineering.")
    if st.button("View Processed Profile"):
        st.session_state.show_profile = 'processed'


# Display the selected profile in a modal dialog
if 'show_profile' in st.session_state and st.session_state.show_profile:
    profile_type = st.session_state.show_profile
    file_path = f"model/{'Raw' if profile_type == 'raw' else 'Processed'}_training_data_profile.html"
    
    @st.dialog(f"{profile_type.capitalize()} Data Profile")
    def show_profile_modal():
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=600, scrolling=True)
        else:
            st.error(f"Could not find the profile report at '{file_path}'. Please run the pipeline to generate it.")
    
    show_profile_modal()
    # Clear the state so the dialog doesn't reappear on every rerun
    st.session_state.show_profile = None
