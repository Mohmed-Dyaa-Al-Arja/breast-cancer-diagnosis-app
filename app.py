import streamlit as st
import pandas as pd
import joblib
import altair as alt

st.set_page_config(page_title="Breast Cancer Diagnosis", layout="wide")
st.title("🔬 Breast Cancer Diagnosis - Decision Tree Model")


model = joblib.load("tree_model.pkl")

features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

mode = st.radio("Choose input method:", ["🧍 Manual Input", "📁 Upload CSV File"])

if mode == "🧍 Manual Input":
    st.subheader("📝 Enter tumor feature values:")
    col1, col2 = st.columns(2)
    input_vals = []


    with col1:
        for feat in features[:len(features)//2]:
            val = st.number_input(feat, format="%.5f", value=0.0)
            input_vals.append(val)
    
    with col2:
        for feat in features[len(features)//2:]:
            val = st.number_input(feat, format="%.5f", value=0.0)
            input_vals.append(val)

    if st.button("Diagnosis"):
        input_df = pd.DataFrame([input_vals], columns=features)
        try:
            prediction = model.predict(input_df)[0]
            label = "Malignant" if prediction == 1 else "Benign"
            st.subheader("Diagnosis Result:")
            st.success(f"Diagnosis: {label}")
            st.dataframe(input_df)
        except Exception as e:
            st.error(f"Error during diagnosis: {e}")

else:
    st.subheader("📤 Upload CSV file for bulk diagnosis")
    uploaded_file = st.file_uploader("Upload file here", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        original_data = data.copy()

       
        process_data = data[[col for col in data.columns if col in features]]


        categorical_cols = []  
        numerical_cols = features

        try:
            processed = process_data.copy()
            predictions = model.predict(processed)

            original_data = original_data.loc[processed.index]  
            original_data.insert(0, "person_id", range(1, len(original_data) + 1))
            original_data["diagnosis_prediction"] = predictions

            original_data["diagnosis_label"] = original_data["diagnosis_prediction"].replace({0: "حميد (Benign)", 1: "خبيث (Malignant)"})

            st.success("✅ Diagnosis successfully completed!")

            st.subheader("📋 Comprehensive table of all cases with diagnosis")
            st.dataframe(original_data)

            st.subheader("👩‍⚕️ Show malignant cases only")
            st.dataframe(original_data[original_data["diagnosis_prediction"] == 1])

            st.subheader("🧘 Show benign cases only")
            st.dataframe(original_data[original_data["diagnosis_prediction"] == 0])

            chart_type = st.selectbox("Choose chart type:", ["Bar Chart", "Area Chart", "Line Chart", "Circle Pack Chart"])

            def plot_chart(data, col, chart_type):
                if chart_type == "Bar Chart":
                    chart = alt.Chart(data).mark_bar().encode(
                        x=alt.X(f"{col}:N" if data[col].dtype == 'object' else f"{col}:Q", title=col),
                        y='count()',
                        color='diagnosis_label:N',
                        tooltip=['count()']
                    ).interactive().properties(width=600)
                elif chart_type == "Area Chart":
                    chart = alt.Chart(data).mark_area(
                        opacity=0.4, interpolate='step'
                    ).encode(
                        x=alt.X(f"{col}:Q" if data[col].dtype != 'object' else f"{col}:N",
                                bin=alt.Bin(maxbins=40) if data[col].dtype != 'object' else None, title=col),
                        y='count()',
                        color='diagnosis_label:N',
                        tooltip=['count()']
                    ).interactive().properties(width=600)
                elif chart_type == "Line Chart":
                    chart = alt.Chart(data).mark_line(point=True).encode(
                        x=alt.X(f"{col}:Q" if data[col].dtype != 'object' else f"{col}:N",
                                bin=alt.Bin(maxbins=40) if data[col].dtype != 'object' else None, title=col),
                        y='count()',
                        color='diagnosis_label:N',
                        tooltip=['count()']
                    ).interactive().properties(width=600)
                elif chart_type == "Circle Pack Chart":
                    chart = alt.Chart(data).mark_circle().encode(
                        x=alt.X(f"{col}:N" if data[col].dtype == 'object' else f"{col}:Q", title=col),
                        y='count()',
                        size='count()',
                        color='diagnosis_label:N',
                        tooltip=['count()']
                    ).interactive().properties(width=600)
                else:
                    chart = None
                return chart

            plot_features = ['diagnosis_label', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean']

            for col in plot_features:
                if col in original_data.columns:
                    st.markdown(f"### 📌 Distribution of `{col}` by diagnosis")
                    chart = plot_chart(original_data, col, chart_type)
                    if chart:
                        st.altair_chart(chart)

        except Exception as e:
            st.error(f"❌ Error in diagnosis: {e}")


st.markdown("""
---
<div style='text-align:center; color:gray; font-size:13px;'>
    Developed by <b>Mohamed Dyaa</b><br>
</div>
""", unsafe_allow_html=True)


# streamlit run "D:\mohamed dyaa\1. Breast Cancer\app.py"

