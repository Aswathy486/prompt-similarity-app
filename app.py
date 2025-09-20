import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")  # Good for long prompts

model = load_model()

# Streamlit interface
st.title("🔎 Prompt Similarity Scorer")
st.write("Compare two prompts and get a similarity score with points awarded.")

prompt1 = st.text_area("Enter first prompt (Host):")
prompt2 = st.text_area("Enter second prompt (Team):")

if st.button("Check Similarity"):
    if prompt1 and prompt2:
        emb1 = model.encode(prompt1, convert_to_tensor=True)
        emb2 = model.encode(prompt2, convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2).item()

        # Points logic
        if score >= 0.9:
            points = 10
        elif score >= 0.8:
            points = 8
        elif score >= 0.6:
            points = 6
        elif score >= 0.8:
            points = 10
        elif score >= 0.4:
            points = 4
        elif score >= 0.2:
            points = 2
        else:
            points = 0

        # Show results
        st.subheader("📊 Results")
        st.metric("Similarity Score", f"{score:.4f}")
        st.metric("Points Awarded", points)
    else:
        st.warning("⚠️ Please enter both prompts before checking similarity.")
