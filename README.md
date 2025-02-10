# Wine Prediction API

A FastAPI application that manages a wine dataset, trains a machine learning model using scikit-learn, and provides endpoints to perform CRUD operations, retraining, and predictions.

## Features

- **CRUD Endpoints:**
  - **Create a new entry:** `POST /entry/`
  - **Get all entries:** `GET /entries/`
  - **Get a single entry:** `GET /entry/{entry_id}`
  - **Update an entry:** `PUT /entry/{entry_id}`
  - **Delete an entry:** `DELETE /entry/{entry_id}`
- **ML Endpoints:**
  - **Retrain the model:** `POST /train/`
  - **Make a prediction:** `POST /predict/`
- **Technology Stack:**
  - Python 3.8, FastAPI, MongoDB, pandas, scikit-learn, Docker
- **CI/CD:**
  - GitHub Actions is used to run tests on each push or pull request.

## Setup

### Prerequisites

- **Docker & Docker Compose:** Ensure Docker is installed and running.
- **Git:** For version control.

### Local Development

1. Clone the Repository:
   ```bash
   git clone <YOUR_REMOTE_REPOSITORY_URL>
   cd wine_api_project

2. Build and Run the Application:
    docker-compose up --build
   

The FastAPI app will be available at http://localhost:8000.
Swagger UI documentation can be accessed at http://localhost:8000/docs.

Running Tests:
    To run tests inside the Docker container:
    docker exec -it wine_api_project-app-1 pytest test_main.py

Continuous Integration (CI)
This project uses GitHub Actions for CI. The workflow file is located at .github/workflows/tests.yml. It automatically runs tests on every push or pull request.

Usage
CRUD operations: Use the endpoints provided by the API to add, retrieve, update, or delete wine entries.
Prediction: Post the required features to /predict/ to get a wine classification.
Retraining: Use /train/ to retrain the ML model on the current dataset in the database.

For any questions or issues, please contact Hanganee Hamunyela at madalahamunyela@gmail.com.
