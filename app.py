# app_pretty.py
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("track_model.pkl")
le = joblib.load("label_encoder.pkl")
X_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="توقع التراك للطالب", layout="centered")
st.title("🎯 توقع التراك المناسب للطالب")

with st.form("student_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    logical_reasoning = st.number_input("Logical Reasoning", min_value=0, max_value=20, value=10)
    memory_recall = st.number_input("Memory & Recall", min_value=0, max_value=10, value=5)
    problem_solving = st.number_input("Problem Solving", min_value=0, max_value=20, value=10)
    analytical_thinking = st.number_input("Analytical Thinking", min_value=0, max_value=20, value=10)
    abstract_thinking = st.number_input("Abstract Thinking", min_value=0, max_value=10, value=5)
    critical_evaluation = st.number_input("Critical Evaluation", min_value=0, max_value=20, value=10)
    mathematical_reasoning = st.number_input("Mathematical Reasoning", min_value=0, max_value=20, value=10)
    decision_making = st.number_input("Decision Making", min_value=0, max_value=10, value=5)
    comprehension = st.number_input("Comprehension", min_value=0, max_value=20, value=10)
    spatial_intelligence = st.number_input("Spatial Intelligence", min_value=0, max_value=10, value=5)
    coding_score = st.number_input("Coding Score", min_value=0, max_value=20, value=10)
    
    submitted = st.form_submit_button("توقع التراك")

if submitted:
    thinking_score = (analytical_thinking + abstract_thinking + critical_evaluation)/3
    logic_math_score = (logical_reasoning + problem_solving + mathematical_reasoning)/3

    sample_dict = {
        'age': age,
        'gender': 0 if gender=="Male" else 1,
        'logical_reasoning': logical_reasoning,
        'memory_recall': memory_recall,
        'problem_solving': problem_solving,
        'analytical_thinking': analytical_thinking,
        'abstract_thinking': abstract_thinking,
        'critical_evaluation': critical_evaluation,
        'mathematical_reasoning': mathematical_reasoning,
        'decision_making': decision_making,
        'comprehension': comprehension,
        'spatial_intelligence': spatial_intelligence,
        'coding_score': coding_score,
        'thinking_score': thinking_score,
        'logic_math_score': logic_math_score
    }

    for col in [c for c in X_columns if 'university_major_' in c]:
        sample_dict[col] = 0

    sample_df = pd.DataFrame([sample_dict])
    sample_df = sample_df[X_columns]

    probs = model.predict_proba(sample_df.values)[0]
    pred = model.predict(sample_df.values)[0]

    track_name = le.inverse_transform([pred])[0]

    st.subheader(f"✅ التراك المقترح: {track_name}")
    st.subheader("📊 احتمالية كل التراكات:")

    for cls, prob in zip(le.classes_, probs):
        st.write(f"{cls}: {prob*100:.2f}%")
        st.progress(float(prob))

