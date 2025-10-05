import streamlit as st

st.set_page_config(
    page_title="Prompt Evolution Engine", 
    layout="wide"
)

st.title("🧠 Prompt Evolution Engine - Welcome")
st.markdown("---")

st.markdown("""
Welcome to the Prompt Evolution Engine!

This tool automatically mutates and tests your base prompt to find the optimal version. 
Navigate using the sidebar to the **Engine** page to begin the evolution process.
""")