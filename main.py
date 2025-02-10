import os
import time
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# If running in GitHub Actions, wait a few seconds for the MongoDB service to be ready.
if os.getenv("GITHUB_ACTIONS"):
    print("Detected GitHub Actions environment. Waiting 15 seconds for MongoDB service to be ready...")
    time.sleep(15)

# --- MongoDB Setup ---
MONGO_URI = "mongodb://mongo:27017/"
client = MongoClient(MONGO_URI)
db = client["wine_database"]
collection = db["wine_data"]

# --- Check if MongoDB is Empty and Load CSV Data ---
if collection.count_documents({}) == 0:
    print("⚠️ No data found in MongoDB! Loading data from wine.csv...")
    df = pd.read_csv("wine.csv")

    # Print the CSV column names for debugging
    print("📊 CSV Columns:", df.columns.tolist())

    # Insert data into MongoDB as-is (no id column added)
    collection.insert_many(df.to_dict(orient="records"))
    print("✅ Data loaded into MongoDB from wine.csv.")

# --- Fetch Data from MongoDB ---
data = list(collection.find({}, {"_id": 0}))
df = pd.DataFrame(data)

# Print MongoDB column names for debugging
print("📝 MongoDB Columns:", df.columns.tolist())

# Ensure 'wine' exists as the target column
if "wine" not in df.columns:
    raise ValueError("❌ 'wine' column is missing! Check MongoDB data.")

# Train Model with hyperparameter tuning
def train_model():
    global model, X_train, X_test, y_train, y_test, df

    y = df["wine"]
    X = df.drop("wine", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning: find best n_neighbors from 1 to 29
    param_grid = {"n_neighbors": range(1, 30)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    print(f"✅ Best n_neighbors: {grid_search.best_params_['n_neighbors']}")
    print(f"📊 Training Accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"📊 Test Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

# Initial training
train_model()

# --- FastAPI Setup ---
app = FastAPI(
    title="Wine Prediction API",
    description="An API for managing wine data, retraining an ML model, and making predictions.",
    version="1.0.0",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "none"
    }
)

@app.get("/")
def root():
    return {"message": "Wine Prediction API is running!"}

@app.post("/entry/")
def create_entry(entry: dict):
    """Create a new entry in the database."""
    if "wine" not in entry:
        raise HTTPException(status_code=400, detail="Missing 'wine' field in entry.")
    collection.insert_one(entry)
    return {"message": "Entry added successfully"}

@app.get("/entries/")
def get_entries():
    """Retrieve all wine data entries."""
    return list(collection.find({}, {"_id": 0}))

@app.get("/entry/{entry_id}")
def get_entry(entry_id: int):
    """Retrieve a single entry."""
    entry = collection.find_one({"id": entry_id}, {"_id": 0})
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry

@app.put("/entry/{entry_id}")
def update_entry(entry_id: int, update: dict):
    """Update an entry."""
    result = collection.update_one({"id": entry_id}, {"$set": update})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Entry not found or no changes made")
    return {"message": "Entry updated successfully"}

@app.delete("/entry/{entry_id}")
def delete_entry(entry_id: int):
    """Delete an entry."""
    result = collection.delete_one({"id": entry_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"message": "Entry deleted successfully"}

@app.post("/train/")
def retrain_model():
    """Retrain the model using all data in the database."""
    global df
    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)
    
    if df.empty:
        raise HTTPException(status_code=400, detail="No data available for training.")
    
    train_model()
    return {"message": "Model retrained successfully"}

@app.post("/predict/")
def predict_wine(features: dict):
    """Make a prediction."""
    feature_df = pd.DataFrame([features])
    try:
        prediction = model.predict(feature_df)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {e}")
    return {"prediction": int(prediction)}
