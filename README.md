# Wine Prediction API

A FastAPI application that manages a wine dataset, trains a machine learning model using scikit-learn, and provides endpoints to perform CRUD operations, retraining, and predictions. The API uses Pydantic models for input validation and automatic documentation generation.

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
  - **Get prediction example:** `GET /predict/example`
  - **Get prediction schema:** `GET /predict/schema`
- **Technology Stack:**
  - Python 3.8, FastAPI, MongoDB, pandas, scikit-learn, Docker
  - Pydantic for data validation and API documentation
- **CI/CD:**
  - GitHub Actions is used to run tests on each push or pull request.

## Setup

### Prerequisites

- **Docker & Docker Compose:** Ensure Docker is installed and running
- **Git:** For version control
- **Python 3.8+:** For local development (optional)

### Local Development

1. Clone the Repository:
   ```bash
   git clone https://github.com/UchihaMadala/wine_api_project
   cd wine_api_project
   ```

2. Build and Run the Application:
   ```bash
   docker-compose up --build
   ```

The FastAPI app will be available at http://localhost:8000.
Interactive API documentation (Swagger UI) can be accessed at http://localhost:8000/docs.

### Running Tests
To run tests inside the Docker container:
```bash
docker exec -it wine_api_project-app-1 pytest test_main.py
```

## API Usage

### Making Predictions

1. **Get Example Input:**
   ```bash
   curl http://localhost:8000/predict/example
   ```

2. **Make a Prediction:**
   ```bash
   curl -X POST "http://localhost:8000/predict/" \
        -H "Content-Type: application/json" \
        -d '{
            "alcohol": 14.23,
            "malic_acid": 1.71,
            "ash": 2.43,
            "alcalinity_of_ash": 15.6,
            "magnesium": 127.0,
            "total_phenols": 2.80,
            "flavanoids": 3.06,
            "nonflavanoid_phenols": 0.28,
            "proanthocyanins": 2.29,
            "color_intensity": 5.64,
            "hue": 1.04,
            "od280_od315_of_diluted_wines": 3.92,
            "proline": 1065.0
        }'
   ```

### Required Features for Prediction

All measurements must be positive numbers and provided in the following units:
- `alcohol`: Percentage (%)
- `malic_acid`: Grams per liter (g/L)
- `ash`: Grams per liter (g/L)
- `alcalinity_of_ash`: Grams per liter (g/L)
- `magnesium`: Milligrams per liter (mg/L)
- `total_phenols`: Grams per liter (g/L)
- `flavanoids`: Grams per liter (g/L)
- `nonflavanoid_phenols`: Grams per liter (g/L)
- `proanthocyanins`: Grams per liter (g/L)
- `color_intensity`: Absorbance value
- `hue`: Ratio value
- `od280_od315_of_diluted_wines`: Ratio value
- `proline`: Milligrams per liter (mg/L)

### Response Format

The prediction endpoint returns:
```json
{
    "prediction": 1,
    "confidence_scores": {
        "1": 0.8,
        "2": 0.15,
        "3": 0.05
    }
}
```

### Other Operations

- **Add New Data:**
  ```bash
  curl -X POST "http://localhost:8000/entry/" -H "Content-Type: application/json" -d '{"wine": 1, ...features...}'
  ```

- **Retrain Model:**
  ```bash
  curl -X POST "http://localhost:8000/train/"
  ```

## Continuous Integration (CI)

This project uses GitHub Actions for CI. The workflow file is located at `.github/workflows/tests.yml`. It automatically runs tests on every push or pull request.

## API Documentation

- **Swagger UI:** Available at `/docs` - Interactive API documentation
- **ReDoc:** Available at `/redoc` - Alternative API documentation
- **OpenAPI Schema:** Available at `/openapi.json` - Raw API schema

## Error Handling

The API includes comprehensive error handling:
- Invalid input validation
- Missing required fields
- Out-of-range values
- Database operation failures
- Model prediction errors

## Contact

For any questions or issues, please contact Hanganee Hamunyela at madalahamunyela@gmail.com.
