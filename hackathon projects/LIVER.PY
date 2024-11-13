import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the dataset and rename columns for consistency
file_path = 'indian_liver_patient.csv'
data = pd.read_csv(file_path)
data = data.rename(columns={
    'Alkaline_Phosphotase': 'Alkaline_Phosphatase',
    'Alamine_Aminotransferase': 'Alanine_Aminotransferase',
    'Total_Protiens': 'Total_Proteins',
    'Albumin_and_Globulin_Ratio': 'Alkaline_Phosphatase_Ratio',
    'Dataset': 'Liver_disease'
})

# Map 'Gender' to numeric values
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Convert target column to binary format (1 for disease, 0 for no disease)
data['Liver_disease'] = data['Liver_disease'].map({1: 1, 2: 0})

# Prepare features and target variable
X = data.drop('Liver_disease', axis=1)
y = data['Liver_disease']

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Define the prediction function
def predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphatase, alanine_aminotransferase, aspartate_aminotransferase, total_proteins, albumin, alkaline_phosphatase_ratio):
    # Convert gender to numeric
    gender = 1 if gender == "Male" else 0

    # Create a dataframe with input features
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Total_Bilirubin': [total_bilirubin],
        'Direct_Bilirubin': [direct_bilirubin],
        'Alkaline_Phosphatase': [alkaline_phosphatase],
        'Alanine_Aminotransferase': [alanine_aminotransferase],
        'Aspartate_Aminotransferase': [aspartate_aminotransferase],
        'Total_Proteins': [total_proteins],
        'Albumin': [albumin],
        'Alkaline_Phosphatase_Ratio': [alkaline_phosphatase_ratio]
    })
    
    # Make the prediction
    try:
        prediction = model.predict(input_data)
        return "Liver Disease Detected" if prediction == 1 else "No Liver Disease"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_liver_disease,
    inputs=[
        gr.Slider(0, 100, step=1, label="Age"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Slider(0, 20, step=0.1, label="Total Bilirubin"),
        gr.Slider(0, 10, step=0.1, label="Direct Bilirubin"),
        gr.Slider(0, 500, step=1, label="Alkaline Phosphatase"),
        gr.Slider(0, 200, step=1, label="Alanine Aminotransferase"),
        gr.Slider(0, 200, step=1, label="Aspartate Aminotransferase"),
        gr.Slider(0, 10, step=0.1, label="Total Proteins"),
        gr.Slider(0, 5, step=0.1, label="Albumin"),
        gr.Slider(0, 10, step=0.1, label="Alkaline Phosphatase Ratio")
    ],
    outputs="text",
    live=False,
    title="Liver Disease Prediction",
    description="This tool predicts whether you have liver disease based on various medical parameters."
)

# Launch the interface
iface.launch()
