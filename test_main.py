from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running!"}

client = TestClient(app)

def test_root():
    """Test if FastAPI works"""
    response = client.get("/")
    assert response.status_code == 200
