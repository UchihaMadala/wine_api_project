from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Wine Prediction API is running!"}

def test_create_entry():
    entry = {"wine": 2, "alcohol": 13.0, "malic_acid": 2.0}
    response = client.post("/entry/", json=entry)
    assert response.status_code == 200
    assert response.json() == {"message": "Entry added successfully"}

def test_get_entries():
    response = client.get("/entries/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict():
    features = {"alcohol": 13.0, "malic_acid": 2.0}
    response = client.post("/predict/", json=features)
    assert response.status_code == 200
    assert "prediction" in response.json()
