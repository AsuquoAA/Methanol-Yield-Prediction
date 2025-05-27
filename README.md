# Methanol Synthesis Yield Predictor (Backend)
This repository contains the backend code for the Methanol Synthesis Yield Predictor, a FastAPI-based API that serves predictions from a machine learning model. The model, a Random Forest Regressor, predicts methanol yield based on reaction conditions such as temperature, pressure, and residence times. The API is designed to be called by the Streamlit frontend, providing real-time predictions for user-inputted conditions.

---

## Project Overview
This project involves the development of a machine learning model to predict methanol yield based on reaction conditions. The process began with the generation of a synthetic dataset using the Langmuir-Hinshelwood (L-H) kinetic model, simulating a double-pass Plug Flow Reactor (PFR) with 90% methanol removal after the first pass. This dataset was then used to train a Random Forest Regressor model, which was subsequently integrated into a FastAPI backend for real-time predictions.

---

## Features
- **FastAPI-powered API** for efficient request handling.
- **Machine Learning Model** (Random Forest Regressor) for methanol yield prediction.
- **Containerized with Docker** for easy deployment.
- **Supports POST requests** to the /predict endpoint for yield predictions.

---

## Technologies Used
- **FastAPI**: For building the API.
- **scikit-learn**: For the Random Forest model.
- **joblib**: For model serialization.
- **Docker**: For containerization.
- **Render**: For deployment.

---

## Setup Instructions
### Prerequisites
- Python 3.10+
- Docker (for local testing and deployment)
- Git

### Local Setup
Clone the Repository:
- git clone https://github.com/AsuquoAA/Methanol-Yield-Prediction_Backend.git
- cd Methanol-Yield-Prediction


### Install Dependencies:
pip install -r requirements.txt


### Load the Model:
Ensure model.joblib and yield_boxcox_transformer.pkl are in the objects/ directory (or adjust paths in the code).


### Run the FastAPI Server:
uvicorn app.main:app --host 0.0.0.0 --port 8000


### Docker Setup
#### Build the Docker Image:
docker build -t methanol-yield-api .

#### Run the Docker Container:
docker run -p 8000:8000 methanol-yield-api


### API Usage
**Endpoint:** `POST /predict`  
**Description:** Predicts methanol yield based on input conditions.

**Input Format (JSON):**
```json
{
  "Temperature (K)": float,
  "Pressure (bar)": float,
  "Residence Time (s)_1": float,
  "Residence Time (s)_2": float
}
```

Output Format (JSON):
```json
{
  "Predicted Percentage yield": "string"
}
```

Example Request:
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"Temperature (K)": 523.0, "Pressure (bar)": 100.0, "Residence Time (s)_1": 10.0, "Residence Time (s)_2": 8.0}'


### Deployment
Render Deployment

### Push to GitHub
Ensure Dockerfile and requirements.txt are included in the repository.

---

## Deploy on Render
- Connect your GitHub repository to Render.
- Select "Web Service" with the Docker environment.
- Set the instance type (e.g., Free or Starter).


### Model Files
Upload model.joblib and yield_boxcox_transformer.pkl to Render’s disk or configure the container to download them during startup.



## Model Information
Training Data: 5,000 synthetic samples simulating a double-pass Plug Flow Reactor (PFR) with 90% methanol removal, based on Langmuir-Hinshelwood kinetics.
Model: Random Forest Regressor (R² = 0.9877 on test set).
Standard Input Ranges:
Temperature: 473–573 K
Pressure: 50–150 bar
Residence Times: 1–30 s (for each pass)



## License
This project is licensed under the MIT License. The MIT License is a permissive open-source license that allows others to use, modify, and share your code freely, as long as they include the original license and copyright notice.

### Contact
For questions or contributions, please contact Anthony Asuquo at asuquoanthony2@gmail.com.
