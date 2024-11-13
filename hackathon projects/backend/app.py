import pickle

# Load the saved model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

# Example prediction
sample_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]  # Sample input matching the feature columns
prediction = model.predict(sample_data)
print("Prediction:", "Diabetes" if prediction[0] == 1 else "No Diabetes")
