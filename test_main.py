from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)

# Mock MongoDB connection and data
mock_data = [
    {
        "id": 1,
        "wine": 1,
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
    }
]

# Mock MongoDB collection methods
class MockCollection:
    def find(self, *args, **kwargs):
        return mock_data
        
    def find_one(self, *args, **kwargs):
        return mock_data[0]
        
    def insert_one(self, *args, **kwargs):
        return True

# Mock MongoDB database and client
mock_db = MagicMock()
mock_db.wine_data = MockCollection()

@patch('main.db', mock_db)
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Wine Prediction API is running!"}

@patch('main.db', mock_db)
def test_create_entry():
    entry = {
        "wine": 2,
        "alcohol": 13.0,
        "malic_acid": 2.0,
        "ash": 2.3,
        "alcalinity_of_ash": 15.0,
        "magnesium": 127.0,
        "total_phenols": 2.8,
        "flavanoids": 3.0,
        "nonflavanoid_phenols": 0.3,
        "proanthocyanins": 2.3,
        "color_intensity": 5.6,
        "hue": 1.0,
        "od280_od315_of_diluted_wines": 3.9,
        "proline": 1065.0
    }
    response = client.post("/entry/", json=entry)
    assert response.status_code == 200
    assert response.json() == {"message": "Entry added successfully"}

@patch('main.db', mock_db)
def test_get_entries():
    response = client.get("/entries/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

@patch('main.db', mock_db)
def test_predict():
    features = {
        "alcohol": 13.0,
        "malic_acid": 2.0,
        "ash": 2.3,
        "alcalinity_of_ash": 15.0,
        "magnesium": 127.0,
        "total_phenols": 2.8,
        "flavanoids": 3.0,
        "nonflavanoid_phenols": 0.3,
        "proanthocyanins": 2.3,
        "color_intensity": 5.6,
        "hue": 1.0,
        "od280_od315_of_diluted_wines": 3.9,
        "proline": 1065.0
    }
    response = client.post("/predict/", json=features)
    assert response.status_code == 200
    assert "prediction" in response.json()
