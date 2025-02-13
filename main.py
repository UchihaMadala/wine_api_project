from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# --- Pydantic Models ---
class WineFeatures(BaseModel):
    alcohol: float = Field(
        ...,
        description="Alcohol content in percentage",
        example=14.23,
        ge=0,  # greater than or equal to 0
        le=100  # less than or equal to 100
    )
    malic_acid: float = Field(
        ...,
        description="Malic acid content in g/L",
        example=1.71,
        ge=0
    )
    ash: float = Field(
        ...,
        description="Ash content in g/L",
        example=2.43,
        ge=0
    )
    alcalinity_of_ash: float = Field(
        ...,
        description="Alcalinity of ash in g/L",
        example=15.6,
        ge=0
    )
    magnesium: float = Field(
        ...,
        description="Magnesium content in mg/L",
        example=127.0,
        ge=0
    )
    total_phenols: float = Field(
        ...,
        description="Total phenols in g/L",
        example=2.80,
        ge=0
    )
    flavanoids: float = Field(
        ...,
        description="Flavanoids content in g/L",
        example=3.06,
        ge=0
    )
    nonflavanoid_phenols: float = Field(
        ...,
        description="Nonflavanoid phenols in g/L",
        example=0.28,
        ge=0
    )
    proanthocyanins: float = Field(
        ...,
        description="Proanthocyanins content in g/L",
        example=2.29,
        ge=0
    )
    color_intensity: float = Field(
        ...,
        description="Color intensity (absorbance at specific wavelength)",
        example=5.64,
        ge=0
    )
    hue: float = Field(
        ...,
        description="Hue (ratio of absorbances)",
        example=1.04,
        ge=0
    )
    od280_od315_of_diluted_wines: float = Field(
        ...,
        description="OD280/OD315 of diluted wines (ratio of absorbances)",
        example=3.92,
        ge=0
    )
    proline: float = Field(
        ...,
        description="Proline content in mg/L",
        example=1065.0,
        ge=0
    )

    class Config:
        schema_extra = {
            "example": {
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
        }

class WineEntry(WineFeatures):
    id: Optional[int] = Field(None, description="Entry ID")
    wine: int = Field(..., description="Wine class (1, 2, or 3)", example=1)

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted wine class (1, 2, or 3)", example=1)
    confidence_scores: Dict[str, float] = Field(
        ..., 
        description="Confidence scores for each class",
        example={"1": 0.8, "2": 0.15, "3": 0.05}
    )

class MessageResponse(BaseModel):
    message: str = Field(..., description="Response message")

# --- MongoDB Setup ---
MONGO_URI = "mongodb://mongo:27017/"
client = MongoClient(MONGO_URI)
db = client["wine_database"]
collection = db["wine_data"]

# Optional: Drop the collection if needed (uncomment the next line)
collection.drop()

# --- Check if MongoDB is Empty and Load CSV Data ---
if collection.count_documents({}) == 0:
    print("⚠️ No data found in MongoDB! Loading data from wine.csv...")
    df = pd.read_csv("wine.csv")
    
    # Ensure 'id' column exists
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    
    # Insert data into MongoDB
    collection.insert_many(df.to_dict(orient="records"))
    print("✅ Data loaded into MongoDB from wine.csv.")

# --- Fetch Data from MongoDB ---
data = list(collection.find({}, {"_id": 0}))
df = pd.DataFrame(data)

# Debug: Check MongoDB Columns
print("📝 MongoDB Columns:", df.columns.tolist())

# Ensure 'wine' exists as the target column
if "wine" not in df.columns:
    raise ValueError("❌ 'wine' column is missing! Check MongoDB data.")

# Train Model
def train_model():
    global model, X_train, X_test, y_train, y_test
    y = df["wine"]
    X = df.drop(columns=["wine", "id"], errors="ignore")  # Drop 'id' if it exists
    X = X.fillna(X.mean())  # Handle missing values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
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
    title="Wine Classification API",
    description="API for wine classification using machine learning",
    version="1.0.0"
)

@app.get("/", response_model=MessageResponse)
def root():
    """Root endpoint to check if the API is running."""
    return MessageResponse(message="Wine Prediction API is running!")

@app.post("/entry/", response_model=MessageResponse)
def create_entry(entry: WineEntry):
    """
    Create a new wine entry in the database.
    
    Args:
        entry: Wine entry data including features and class
    
    Returns:
        Message confirming entry creation
    """
    entry_dict = entry.dict()
    
    # Ensure 'id' is set correctly
    last_entry = collection.find_one(sort=[("id", -1)])
    new_id = (last_entry["id"] + 1) if last_entry else 1
    entry_dict["id"] = new_id
    
    collection.insert_one(entry_dict)
    return MessageResponse(message="Entry added successfully")

@app.get("/entries/", response_model=List[WineEntry])
def get_entries():
    """Retrieve all wine data entries."""
    return list(collection.find({}, {"_id": 0}))

@app.get("/entry/{entry_id}", response_model=WineEntry)
def get_entry(entry_id: int):
    """
    Retrieve a single wine entry.
    
    Args:
        entry_id: ID of the entry to retrieve
    
    Returns:
        Wine entry data
    """
    entry = collection.find_one({"id": entry_id}, {"_id": 0})
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry

@app.put("/entry/{entry_id}", response_model=MessageResponse)
def update_entry(entry_id: int, update: WineEntry):
    """
    Update a wine entry.
    
    Args:
        entry_id: ID of the entry to update
        update: Updated wine entry data
    
    Returns:
        Message confirming update
    """
    result = collection.update_one({"id": entry_id}, {"$set": update.dict(exclude_unset=True)})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Entry not found or no changes made")
    return MessageResponse(message="Entry updated successfully")

@app.delete("/entry/{entry_id}", response_model=MessageResponse)
def delete_entry(entry_id: int):
    """Delete a wine entry."""
    result = collection.delete_one({"id": entry_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Entry not found")
    return MessageResponse(message="Entry deleted successfully")

@app.post("/train/", response_model=MessageResponse)
def retrain_model():
    """Retrain the model using all data in the database."""
    global df
    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)
    
    if df.empty:
        raise HTTPException(status_code=400, detail="No data available for training.")
    
    train_model()
    return MessageResponse(message="Model retrained successfully")

@app.get("/predict/example", response_model=Dict[str, dict])
def get_prediction_example():
    """
    Get an example of the prediction input format and expected response.
    
    Returns:
        Dictionary containing example request and response formats
    """
    example_input = WineFeatures.Config.schema_extra["example"]
    example_response = {
        "prediction": 1,
        "confidence_scores": {"1": 0.8, "2": 0.15, "3": 0.05}
    }
    
    return {
        "example_request": example_input,
        "example_response": example_response
    }

@app.get("/predict/schema")
def get_prediction_schema():
    """
    Get the complete schema for prediction input, including field descriptions and constraints.
    """
    return WineFeatures.schema()

@app.post("/predict/", response_model=PredictionResponse)
def predict_wine(features: WineFeatures):
    """
    Make a wine classification prediction.
    
    The model expects chemical analysis measurements of wine samples.
    All measurements should be positive numbers and in their respective units.
    
    Args:
        features: Wine chemical analysis measurements including:
            - alcohol: Alcohol content (%)
            - malic_acid: Malic acid content (g/L)
            - ash: Ash content (g/L)
            - alcalinity_of_ash: Alcalinity of ash (g/L)
            - magnesium: Magnesium content (mg/L)
            - total_phenols: Total phenols (g/L)
            - flavanoids: Flavanoids content (g/L)
            - nonflavanoid_phenols: Nonflavanoid phenols (g/L)
            - proanthocyanins: Proanthocyanins (g/L)
            - color_intensity: Color intensity (absorbance)
            - hue: Hue (ratio)
            - od280_od315_of_diluted_wines: OD280/OD315 ratio
            - proline: Proline (mg/L)
    
    Returns:
        PredictionResponse: Object containing:
            - prediction: Predicted wine class (1, 2, or 3)
            - confidence_scores: Probability scores for each class
    
    Example:
        ```
        curl -X POST "http://localhost:8000/predict/" -H "Content-Type: application/json" -d '{
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
    """
    feature_df = pd.DataFrame([features.dict()])
    try:
        prediction = model.predict(feature_df)[0]
        proba = model.predict_proba(feature_df)[0]
        confidence_scores = {str(i+1): float(score) for i, score in enumerate(proba)}
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence_scores=confidence_scores
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {e}")