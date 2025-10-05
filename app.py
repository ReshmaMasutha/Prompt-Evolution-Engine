
import streamlit as st
import pandas as pd
import os
import plotly.express as px # Import Plotly for the visualization

# Import all necessary components from your utility file
from core_utils import generate_variations_with_meta, get_model_response, calculate_metrics 

# --- SESSION STATE INITIALIZATION (MUST be at the very top) ---
# Initialize session state for the prompt text and the default prompt
DEFAULT_PROMPT = "Write a blog post about the importance of AI in education."

if 'base_prompt_text' not in st.session_state:
    st.session_state['base_prompt_text'] = DEFAULT_PROMPT
    
# --- REPORT GENERATION FUNCTION ---

def generate_report_content(df, base_prompt):
    """Formats the DataFrame and results into a human-readable string for download."""
    report = f"--- Prompt Evolution Report ---\n"
    report += f"Base Prompt: {base_prompt}\n"
    report += f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    best_row = df.iloc[0]
    report += f"🏆 BEST PROMPT:\n{best_row['Prompt Version']}\n"
    report += f"Final Score: {best_row['Final Score']}%\n\n"
    
    report += "--- Full Leaderboard ---\n"
    # Select columns to display in the report text file
    report += df[['Rank', 'ID', 'Final Score', 'Prompt Version']].to_string(index=False)
    report += "\n\n"
    
    report += "--- AI Outputs and Metrics ---\n"
    for index, row in df.iterrows():
        report += f"== Prompt #{row['ID']} (Rank {row['Rank']}) ==\n"
        report += f"Prompt: {row['Prompt Version']}\n"
        report += f"Metrics: Relevance={row['Relevance (%)']}%, Readability={row['Readability (%)']}%, Length={row['Length Score (%)']}%\n"
        report += f"AI Response:\n{row['AI Output']}\n\n"
        
    return report

# --- CONFIGURATION AND API CHECK ---
import os
openai.api_key = os.getenv("OPENAI_API_KEY") # Reads key from environment
# Check if the API key is set before proceeding
if not os.getenv('OPENAI_API_KEY'):
    st.error("🚨 OpenAI API Key not found!")
    st.info("Please set the environment variable in your terminal: $env:OPENAI_API_KEY='Your_API_Key'")
    st.stop() 

st.set_page_config(layout="wide", page_title="Prompt Evolution Engine")
st.title("🧠 Prompt Evolution Engine")
st.markdown("Let AI evolve and test your prompt to find the optimal version.")

# --- 1. Input Section ---

st.header("📝 1. Enter Your Base Prompt")
# Use the session state value for the text area
base_prompt = st.text_area(
    "Base Prompt:", 
    value=st.session_state['base_prompt_text'],
    height=150
)

col1, col2 = st.columns(2)
num_variations = col1.slider('Number of Variations to Generate', 3, 10, 5)
temperature = col2.slider('Model Temperature (Creativity)', 0.0, 1.0, 0.7)

# --- 2. Generation and Testing Button ---

if st.button("Generate & Test Prompts 🔁", type="primary", use_container_width=True):
    
    if not base_prompt.strip():
        st.warning("Please enter a base prompt to begin evolution.")
        st.stop()

    # Use Streamlit's spinner for a professional look during long tasks
    with st.spinner("Step 1/2: Generating and Mutating Prompts..."):
        # Get the list of evolved prompts from the LLM
        variations = generate_variations_with_meta(
            base_prompt=base_prompt, 
            n=num_variations,
        )
        
    st.subheader("✨ Generated Prompt Variations")
    if not variations:
        st.error("The LLM failed to generate valid prompt variations. Check your API key or try again.")
        st.stop()

    for i, v in enumerate(variations):
        prompt_id = v.get('id', i+1)
        st.code(f"Prompt #{prompt_id}: {v['prompt']}")
        
    # --- 3. Evaluation Loop ---
    
    st.subheader("📊 Evaluation & Testing")
    results = []
    
    progress_bar = st.progress(0, text="Starting AI response and evaluation...")

    for i, v in enumerate(variations):
        prompt = v['prompt']
        prompt_id = v.get('id', i+1)

        # Update progress bar
        progress_value = (i + 1) / len(variations)
        progress_bar.progress(progress_value, text=f"Step 2/2: Testing Prompt {i+1} of {len(variations)}...")

        # Get the AI response
        model_output = get_model_response(prompt, temperature=temperature)
        
        # Calculate scores using the function from core_utils
        scores = calculate_metrics(base_prompt, model_output)
        
        # Compile all results
        result = {
            'ID': prompt_id,
            'Prompt Version': prompt,
            'Final Score': round(scores['Final_Score'] * 100, 1), 
            'Relevance (%)': round(scores['Semantic_Relevance'] * 100, 1),
            'Readability (%)': round(scores['Readability'] * 100, 1),
            'Length Score (%)': round(scores['Length_Score'] * 100, 1),
            'AI Output': model_output
        }
        results.append(result)

    progress_bar.empty() 
    st.success("Testing and Evaluation Complete!")

    # --- 4. Display Leaderboard ---

    df = pd.DataFrame(results).sort_values('Final Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    
    st.header("🧾 Prompt Leaderboard")
    
    best_prompt_row = df.iloc[0]
    
    st.dataframe(
        df[['Rank', 'ID', 'Prompt Version', 'Final Score', 'Relevance (%)', 'Readability (%)']],
        hide_index=True,
        column_config={
            "Final Score": st.column_config.ProgressColumn(
                "Final Score",
                help="Weighted average of all metrics (0-100)",
                format="%f",
                min_value=0,
                max_value=100,
            )
        }
    )

    # --- 4.5 Visualization ---
    st.subheader("📈 Prompt Performance Chart")
    
    df['Label'] = 'ID ' + df['ID'].astype(str) + ' (Rank ' + df['Rank'].astype(str) + ')'
    
    fig = px.bar(
        df.sort_values('Final Score', ascending=True),
        x='Final Score', 
        y='Label', 
        orientation='h',
        title='Prompt Final Score Comparison (0-100%)',
        color='Final Score', 
        color_continuous_scale=px.colors.sequential.Plotly3 
    )
    
    fig.update_layout(
        xaxis_title="Final Score (%)", 
        yaxis_title="Prompt Version",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


    # --- 5. Best Prompt Display ---
    st.subheader("🏆 Best Performing Prompt")
    
    st.markdown(f"**Score: {best_prompt_row['Final Score']}%**")
    st.success(f"**{best_prompt_row['Prompt Version']}**")

    # Save the winning prompt to Streamlit's session state
    st.session_state['winning_prompt'] = best_prompt_row['Prompt Version']
    
    # --- Action Buttons (Evolve & Download) ---
    button_col1, button_col2 = st.columns(2)

    # Column 1: Evolve Again Button
    with button_col1:
        if st.button("🔁 Evolve Again (Use Winning Prompt as New Base)", type="secondary", use_container_width=True):
            st.session_state['base_prompt_text'] = st.session_state['winning_prompt']
            st.experimental_rerun()

    # Column 2: Download Report Button
    with button_col2:
        report_content = generate_report_content(df, base_prompt) 
        st.download_button(
            label="💾 Download Full Report (TXT)",
            data=report_content,
            file_name="prompt_evolution_report.txt",
            mime="text/plain",
            use_container_width=True
        )
        
    st.markdown("---")
    
    # --- 6. AI Output Display (Expanders) ---
    st.header("📝 Full Results and AI Outputs")
    for index, row in df.iterrows():
        with st.expander(f"Rank {row['Rank']} (ID {row['ID']}) - Score: {row['Final Score']}%"):
            st.markdown(f"**Prompt Used:**")
            st.code(row['Prompt Version'])
            st.markdown(f"**Metrics:** Relevance: {row['Relevance (%)']}%, Readability: {row['Readability (%)']}%, Length: {row['Length Score (%)']}%")
            st.markdown(f"**AI Response:**")
            st.write(row['AI Output'])
