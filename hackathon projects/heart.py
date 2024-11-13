import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import gradio as gr

# Load and preprocess the dataset
data = pd.read_csv("heart.csv")

# Splitting features and target variable
X = data.drop(columns="target")
y = data["target"]

# Scale the features for better convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a logistic regression model with a higher max_iter
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# Check model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Prediction function for Gradio
def predict_heart_attack(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Arrange inputs into a DataFrame and scale using the fitted scaler
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=X.columns)
    input_data_scaled = scaler.transform(input_data)
    
    # Predict and interpret result
    prediction = model.predict(input_data_scaled)[0]
    return "Likely Heart Attack" if prediction == 1 else "Unlikely Heart Attack"

# Define Gradio Interface
interface = gr.Interface(
    fn=predict_heart_attack,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio([0, 1], label="Sex (0 = Female, 1 = Male)"),
        gr.Radio([0, 1, 2, 3], label="Chest Pain Type (0-3)"),
        gr.Number(label="Resting Blood Pressure"),
        gr.Number(label="Cholesterol"),
        gr.Radio([0, 1], label="Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)"),
        gr.Radio([0, 1, 2], label="Resting ECG Results (0-2)"),
        gr.Number(label="Max Heart Rate Achieved"),
        gr.Radio([0, 1], label="Exercise Induced Angina (1 = Yes, 0 = No)"),
        gr.Number(label="Oldpeak (ST depression induced by exercise)"),
        gr.Radio([0, 1, 2], label="Slope of Peak Exercise ST Segment (0-2)"),
        gr.Number(label="Number of Major Vessels (0-3)"),
        gr.Radio([0, 1, 2, 3], label="Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)")
    ],
    outputs="text",
    title="Heart Attack Prediction Model",
    description="Enter the patient information to predict the likelihood of a heart attack."
)

# Launch Gradio interface
interface.launch()
