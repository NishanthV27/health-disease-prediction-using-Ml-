import gradio as gr
import numpy as np
import joblib  # For loading the saved model

# Load your pre-trained model (replace 'model.pkl' with your actual model file path)
model = joblib.load("model1.pkl")

# Define a function to make predictions based on user input
def predict_diabetes(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age):
    try:
        # Convert inputs into a numpy array in the format expected by the model
        input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])
        
        # Make a prediction
        prediction = model.predict(input_data)
        
        # Return result
        return "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"
    except Exception as e:
        return f"An error occurred: {e}"

# Set up Gradio inputs and outputs using the updated components
inputs = [
    gr.Number(label="Number of Pregnancies"),
    gr.Number(label="Glucose Level (mg/dL)"),
    gr.Number(label="Blood Pressure (mmHg)"),
    gr.Number(label="Skin Thickness (mm)"),
    gr.Number(label="Insulin Level (IU/mL)"),
    gr.Number(label="Body Mass Index (kg/m²)"),
    gr.Number(label="Diabetes Pedigree Function"),
    gr.Number(label="Patient Age in Years")
]

# Set up Gradio interface
outputs = gr.Textbox(label="Diabetes Prediction Result")
interface = gr.Interface(fn=predict_diabetes, inputs=inputs, outputs=outputs, title="Diabetes Prediction")

# Launch the app
interface.launch()
