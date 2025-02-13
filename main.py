import time
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import os

# --- Pydantic Models ---
class WineFeatures(BaseModel):
    alcohol: float = Field(
        ...,
        description="Alcohol content in percentage",
        example=14.23,
        ge=0,
        le=100
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

# --- MongoDB Setup with Connection Retry Logic ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017/")

def get_database():
    retries = 5
    while retries > 0:
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # 5 second timeout
            # Verify connection
            client.server_info()  # Will throw an error if MongoDB is not available
            db = client["wine_database"]
            print("Connected to MongoDB")
            return db
        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            retries -= 1
            if retries == 0:
                print(f"Failed to connect to MongoDB after multiple attempts: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to connect to MongoDB. Please check if MongoDB is running."
                )
            else:
                print(f"Retrying to connect to MongoDB... Attempts remaining: {retries}")
                time.sleep(2)  # Wait for 2 seconds before retrying

# Initialize database connection
try:
    db = get_database()
    collection = db["wine_data"]
except Exception as e:
    print(f"Failed to initialize database: {e}")
    raise

# --- Check if MongoDB is Empty and Load CSV Data ---
def initialize_data():
    try:
        if collection.count_documents({}) == 0:
            print("⚠️ No data found in MongoDB! Loading data from wine.csv...")
            
            # Check if wine.csv exists
            if not os.path.exists("wine.csv"):
                print("❌ wine.csv not found!")
                # Create some dummy data for testing
                dummy_data = {
                    "id": [1],
                    "wine": [1],
                    "alcohol": [14.23],
                    "malic_acid": [1.71],
                    "ash": [2.43],
                    "alcalinity_of_ash": [15.6],
                    "magnesium": [127.0],
                    "total_phenols": [2.80],
                    "flavanoids": [3.06],
                    "nonflavanoid_phenols": [0.28],
                    "proanthocyanins": [2.29],
                    "color_intensity": [5.64],
                    "hue": [1.04],
                    "od280_od315_of_diluted_wines": [3.92],
                    "proline": [1065.0]
                }
                df = pd.DataFrame(dummy_data)
            else:
                df = pd.read_csv("wine.csv")
                
            # Normalize CSV column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Ensure 'id' column exists
            if "id" not in df.columns:
                df.insert(0, "id", range(1, len(df) + 1))
            
            # Insert data into MongoDB
            collection.insert_many(df.to_dict(orient="records"))
            print("✅ Data loaded into MongoDB successfully.")
    except Exception as e:
        print(f"Error initializing data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize database data: {str(e)}"
        )

# Initialize data
initialize_data()

# --- FastAPI Setup ---
app = FastAPI(
    title="Wine Classification API",
    description="API for wine classification using machine learning",
    version="1.0.0"
)

# --- ML Model Setup ---
def train_model():
    global model, X_train, X_test, y_train, y_test
    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)
    
    y = df["wine"]
    X = df.drop(columns=["wine", "id"], errors="ignore")
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

@app.get("/", response_model=MessageResponse)
async def root():
    """Root endpoint to check if the API is running."""
    return MessageResponse(message="Wine Prediction API is running!")

@app.post("/entry/", response_model=MessageResponse)
async def create_entry(entry: WineEntry):
    """Create a new wine entry in the database."""
    try:
        entry_dict = entry.dict()
        
        # Ensure 'id' is set correctly
        last_entry = collection.find_one(sort=[("id", -1)])
        new_id = (last_entry["id"] + 1) if last_entry else 1
        entry_dict["id"] = new_id
        
        collection.insert_one(entry_dict)
        return MessageResponse(message="Entry added successfully")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create entry: {str(e)}"
        )
    
@app.get("/entries/", response_model=List[WineEntry])
async def get_entries():
    """Retrieve all wine data entries."""
    try:
        entries = list(collection.find({}, {"_id": 0}))
        print(f"Found {len(entries)} entries")  # Debug log
        
        processed_entries = []
        for entry in entries:
            try:
                processed_entry = {
                    "id": int(float(entry.get("id", 0))),
                    "wine": int(float(entry.get("wine", 0))),
                    "alcohol": float(entry.get("alcohol", 0.0)),
                    "malic_acid": float(entry.get("malic_acid", 0.0)),
                    "ash": float(entry.get("ash", 0.0)),
                    "alcalinity_of_ash": float(entry.get("alcalinity_of_ash", 0.0)),
                    "magnesium": float(entry.get("magnesium", 0.0)),
                    "total_phenols": float(entry.get("total_phenols", 0.0)),
                    "flavanoids": float(entry.get("flavanoids", 0.0)),
                    "nonflavanoid_phenols": float(entry.get("nonflavanoid_phenols", 0.0)),
                    "proanthocyanins": float(entry.get("proanthocyanins", 0.0)),
                    "color_intensity": float(entry.get("color_intensity", 0.0)),
                    "hue": float(entry.get("hue", 0.0)),
                    "od280_od315_of_diluted_wines": float(entry.get("od280_od315_of_diluted_wines", 0.0)),
                    "proline": float(entry.get("proline", 0.0))
                }
                processed_entries.append(processed_entry)
            except Exception as e:
                print(f"Error processing entry: {entry}")  # Debug log
                print(f"Error details: {str(e)}")  # Debug log
                continue
        
        print(f"Successfully processed {len(processed_entries)} entries")  # Debug log
        return processed_entries
    except Exception as e:
        print(f"Error in get_entries: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/entry/{entry_id}", response_model=WineEntry)
async def get_entry(entry_id: int):
    """Retrieve a single wine entry."""
    try:
        print(f"Looking for entry_id: {entry_id}")  # Debug log
        entry = collection.find_one({"id": entry_id}, {"_id": 0})
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        
        print(f"Found entry: {entry}")  # Debug log
        
        processed_entry = {
            "id": int(float(entry.get("id", 0))),
            "wine": int(float(entry.get("wine", 0))),
            "alcohol": float(entry.get("alcohol", 0.0)),
            "malic_acid": float(entry.get("malic_acid", 0.0)),
            "ash": float(entry.get("ash", 0.0)),
            "alcalinity_of_ash": float(entry.get("alcalinity_of_ash", 0.0)),
            "magnesium": float(entry.get("magnesium", 0.0)),
            "total_phenols": float(entry.get("total_phenols", 0.0)),
            "flavanoids": float(entry.get("flavanoids", 0.0)),
            "nonflavanoid_phenols": float(entry.get("nonflavanoid_phenols", 0.0)),
            "proanthocyanins": float(entry.get("proanthocyanins", 0.0)),
            "color_intensity": float(entry.get("color_intensity", 0.0)),
            "hue": float(entry.get("hue", 0.0)),
            "od280_od315_of_diluted_wines": float(entry.get("od280_od315_of_diluted_wines", 0.0)),
            "proline": float(entry.get("proline", 0.0))
        }
        
        print(f"Processed entry: {processed_entry}")  # Debug log
        return processed_entry
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_entry: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.put("/entry/{entry_id}", response_model=MessageResponse)
async def update_entry(entry_id: int, update: WineEntry):
    """Update a wine entry."""
    try:
        result = collection.update_one(
            {"id": entry_id}, 
            {"$set": update.dict(exclude_unset=True)}
        )
        if result.modified_count == 0:
            raise HTTPException(
                status_code=404, 
                detail="Entry not found or no changes made"
            )
        return MessageResponse(message="Entry updated successfully")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update entry: {str(e)}"
        )

@app.delete("/entry/{entry_id}", response_model=MessageResponse)
async def delete_entry(entry_id: int):
    """Delete a wine entry."""
    try:
        result = collection.delete_one({"id": entry_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Entry not found")
        return MessageResponse(message="Entry deleted successfully")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete entry: {str(e)}"
        )

@app.post("/train/", response_model=MessageResponse)
async def retrain_model():
    """Retrain the model using all data in the database."""
    try:
        train_model()
        return MessageResponse(message="Model retrained successfully")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrain model: {str(e)}"
        )

@app.get("/predict/example", response_model=Dict[str, dict])
async def get_prediction_example():
    """Get an example of the prediction input format and expected response."""
    example_input = WineFeatures.Config.schema_extra["example"]
    example_response = {
        "prediction": 1,
        "confidence_scores": {"1": 0.8, "2": 0.15, "3": 0.05}
    }
    return {"example_request": example_input, "example_response": example_response}

@app.get("/predict/schema")
async def get_prediction_schema():
    """Get the complete schema for prediction input."""
    return WineFeatures.schema()

@app.post("/predict/", response_model=PredictionResponse)
async def predict_wine(features: WineFeatures):
    """Make a wine classification prediction."""
    try:
        feature_df = pd.DataFrame([features.dict()])
        prediction = model.predict(feature_df)[0]
        proba = model.predict_proba(feature_df)[0]
        confidence_scores = {str(i+1): float(score) for i, score in enumerate(proba)}
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence_scores=confidence_scores
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in prediction: {str(e)}"
        )
