from fastapi import FastAPI, HTTPException, BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import List
from model import get_image_for_prediction  # Import the prediction logic
import tensorflow as tf
from contextlib import asynccontextmanager
from datetime import datetime

app = FastAPI()

# Global variable to hold the model
model = None

# MongoDB connection
client = AsyncIOMotorClient("mongodb+srv://satishbisa:HiThere!123@cluster0.vji5t.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['runwai_db']
collection = db['clothing_catalog']
feedback_collection = db['user_feedback']  # Feedback collection

# Clothing item schema
class ClothingItem(BaseModel):
    clothing_id: str
    item_name: str
    category: str
    image_url: str
    price: float
    colors_available: List[str]
    brand: str
    size: List[str]

# Lifespan event handler for model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading model...")
    model = tf.keras.models.load_model("clothing_model.keras")  # Load the model once
    print("Model loaded successfully.")
    yield
    print("Shutting down...")

# Use the lifespan context in the FastAPI app
app = FastAPI(lifespan=lifespan)

# Route to add a new clothing item
@app.post("/add-item")
async def add_clothing_item(item: ClothingItem):
    result = await collection.insert_one(item.dict())
    if result.acknowledged:
        return {"message": "Item added successfully", "id": str(result.inserted_id)}
    raise HTTPException(status_code=500, detail="Failed to add item")

# Route to get all clothing items
@app.get("/items", response_model=List[ClothingItem])
async def get_all_clothing_items():
    items = await collection.find().to_list(100)  # Limits to 100 items
    return items

# Route to get a clothing item by ID
@app.get("/item/{item_id}")
async def get_clothing_item(item_id: str):
    item = await collection.find_one({"clothing_id": item_id})
    if item:
        return item
    raise HTTPException(status_code=404, detail="Item not found")

# Route to predict if a user will like a specific clothing item
@app.get("/predict-like/{item_id}")
async def predict_user_like(item_id: str, background_tasks: BackgroundTasks):
    try:
        # Add image fetching and prediction to background tasks
        background_tasks.add_task(async_predict_user_like, item_id)
        return {"message": "Prediction in progress. Check back later!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background task for prediction
async def async_predict_user_like(item_id: str):
    try:
        prediction, item = await get_image_for_prediction(item_id, model)
        liked = "like" if prediction >= 0.5 else "dislike"
        print(f"The user would {liked} the clothing item {item['item_name']}.")
    except Exception as e:
        print(f"Error: {str(e)}")

# POST route for submitting feedback
@app.post("/feedback")
async def user_feedback(item_id: str, liked: bool):
    # Store the feedback in MongoDB
    feedback_data = {
        "user_id": "user123",  # Replace with actual user id if applicable
        "item_id": item_id,
        "liked": liked,
        "timestamp": datetime.utcnow()
    }
    result = await feedback_collection.insert_one(feedback_data)
    if result.acknowledged:
        return {"message": "Feedback recorded successfully"}
    raise HTTPException(status_code=500, detail="Failed to record feedback")
