from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Request
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import List
from model import get_image_for_prediction  # Import the prediction logic
import tensorflow as tf
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from uuid import uuid4  # For generating unique user IDs
import random
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import stripe


# Define constants
stripe.api_key = "sk_test_51Q7NqdGpfvzWqvI6jGNrRWB1VwQR1xMkmOiET678WRzvEHiG3JxZXqn0D5omjOhPhOFHpLKLOuTDSbgFpLi10AiG00x7J4HxKv"
SECRET_KEY = "your_secret_key"  # Change this to a secure value
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Define allowed origins (frontend URLs)
origins = [
    "http://localhost:3000",  # Frontend React URL
]

# Global variables
app = FastAPI()

# Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

model = None
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# MongoDB connection
client = AsyncIOMotorClient("mongodb+srv://satishbisa:HiThere!123@cluster0.vji5t.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['runwai_db']
collection = db['clothing_catalog']
feedback_collection = db['user_feedback']  # Feedback collection
user_collection = db['user']  # Users collection

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Cryptography for password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

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

# Models for user and clothing items
class ClothingItem(BaseModel):
    clothing_id: str
    item_name: str
    category: str
    image_url: str
    price: float
    colors_available: List[str]
    brand: str
    size: List[str]

    

class User(BaseModel):
    user_id: str
    name: str
    email: str
    password: str

class Login(BaseModel):
    email: str
    password: str

class FeedbackInput(BaseModel):
    item_id: str
    liked: bool

# Password verification and hashing utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# Token generation function
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/create-user")
async def create_user(user: User):
    existing_user = await user_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    user_data = {
        "user_id": str(uuid4()),  # Generate a unique user ID
        "name": user.name,
        "email": user.email,
        "password": hash_password(user.password)  # Hash the password before storing
    }
    
    result = await user_collection.insert_one(user_data)
    if result.acknowledged:
        return {"message": "User created successfully", "user_id": user_data["user_id"]}
    raise HTTPException(status_code=500, detail="Failed to create user")

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await user_collection.find_one({"email": form_data.username})
    
    if user and verify_password(form_data.password, user["password"]):
        access_token = create_access_token(data={"sub": user["user_id"]})
        headers = {"Access-Control-Allow-Origin": "http://localhost:3000", 
                   "Access-Control-Allow-Credentials": "true"}
        return JSONResponse(content={"access_token": access_token, "token_type": "bearer"}, headers=headers)
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

# Token verification dependency
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await user_collection.find_one({"user_id": user_id})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
@app.get("/protected-route")
async def protected_route(current_user: dict = Depends(get_current_user)):
    # Only accessible if the user is logged in
    return {"message": f"Hello, {current_user['name']}! You are authenticated."}

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


# Content-Based Recommendation System
def recommend_items(liked_items, clothing_data, cosine_sim):
    recommended_items = set()

    for item_id in liked_items:
        # Find the index of the liked item in the clothing_data DataFrame
        item_idx = clothing_data[clothing_data['clothing_id'] == item_id].index[0]

        # Get similarity scores for this item
        similarity_scores = list(enumerate(cosine_sim[item_idx]))

        # Sort items by similarity score
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get the index of the top similar item (excluding the liked item itself and already recommended/liked items)
        similar_items_indices = [i for i, _ in similarity_scores[1:] if clothing_data.iloc[i]['clothing_id'] not in liked_items]

        # Add the similar item(s) to the recommendation list (making sure it's not already liked)
        recommended_items.update(clothing_data.iloc[similar_items_indices]['clothing_id'].tolist())

    return list(recommended_items)


@app.options("/recommend")
async def options_recommend():
    """
    Handle the OPTIONS method for /recommend to allow CORS preflight requests.
    """
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    }
    return JSONResponse(content={}, headers=headers)

@app.get("/recommend")
async def recommend_clothing(current_user: dict = Depends(get_current_user)):
    # Get user feedback history
    user_feedback = await feedback_collection.find({"user_id": current_user["user_id"], "liked": True}).to_list(100)
    liked_items = [feedback["item_id"] for feedback in user_feedback]

    if not liked_items:
        raise HTTPException(status_code=404, detail="No liked items found for recommendation.")

    # Fetch clothing data
    clothing_items = await collection.find().to_list(100)
    clothing_data = pd.DataFrame(clothing_items)

    # Build the content-based recommendation engine
    clothing_data['combined_features'] = clothing_data.apply(lambda x: f"{x['item_name']} {x['brand']} {x['category']}", axis=1)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(clothing_data['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get recommendations based on liked items
    recommended_items = recommend_items(liked_items, clothing_data, cosine_sim)

    if not recommended_items:
        raise HTTPException(status_code=404, detail="No recommendations available.")

    # Convert the top recommended item to dict for JSONResponse (take only the first item)
    top_recommended_clothing = clothing_data[clothing_data['clothing_id'].isin(recommended_items[:1])].to_dict(orient="records")

    if top_recommended_clothing:
        # Ensure ObjectId is converted to string and handle NaN or infinite values for the top item
        top_item = top_recommended_clothing[0]
        if "_id" in top_item:
            top_item["_id"] = str(top_item["_id"])  # Convert ObjectId to string
        # Replace NaN or infinite values with a default value
        for key, value in top_item.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                top_item[key] = 0 if isinstance(value, float) else ""  # Replace with 0 for float or empty string

        # Set CORS headers for the response
        headers = {
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true",
        }

        print(top_item)
        return JSONResponse(content={"recommended_item": top_item}, headers=headers)
    
    # If no items were recommended, raise an exception
    raise HTTPException(status_code=404, detail="No recommendation could be made.")





@app.options("/feedback")
async def options_feedback():
    """
    Handle the OPTIONS method for /feedback to allow CORS preflight requests.
    """
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    }
    return JSONResponse(content={}, headers=headers)


# Route to submit feedback for a clothing item
@app.post("/feedback")
async def submit_feedback(feedback: FeedbackInput, current_user: dict = Depends(get_current_user)):
    feedback_data = {
        "user_id": current_user["user_id"],
        "item_id": feedback.item_id,
        "liked": feedback.liked,
        "timestamp": datetime.utcnow()
    }
    result = await feedback_collection.insert_one(feedback_data)
    
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",
        "Access-Control-Allow-Credentials": "true",
    }

    if result.acknowledged:
        return JSONResponse(content={"message": "Feedback recorded"}, headers=headers)
    
    raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.options("/saved-items")
async def options_feedback():
    """
    Handle the OPTIONS method for /feedback to allow CORS preflight requests.
    """
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    }
    return JSONResponse(content={}, headers=headers)


@app.get("/saved-items")
async def get_saved_items(current_user: dict = Depends(get_current_user)):
    """
    Endpoint to get all saved/liked items for the current user.
    """
    # Fetch all liked items for the user
    user_feedback = await feedback_collection.find({"user_id": current_user["user_id"], "liked": True}).to_list(100)
    liked_items = [feedback["item_id"] for feedback in user_feedback]

    if not liked_items:
        raise HTTPException(status_code=404, detail="No liked items found.")

    # Fetch clothing data for the liked items
    clothing_items = await collection.find({"clothing_id": {"$in": liked_items}}).to_list(len(liked_items))
    
    if not clothing_items:
        raise HTTPException(status_code=404, detail="No matching clothing items found.")

    # Convert clothing items to a list of dictionaries
    liked_clothing_data = pd.DataFrame(clothing_items).to_dict(orient="records")

    # Ensure ObjectId is converted to string for each item and handle NaN/infinite values
    for item in liked_clothing_data:
        if "_id" in item:
            item["_id"] = str(item["_id"])  # Convert ObjectId to string
        # Replace NaN or infinite values
        for key, value in item.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                item[key] = 0 if isinstance(value, float) else ""  # Replace with 0 for float or empty string
    
    # Set CORS headers for the response
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",
        "Access-Control-Allow-Credentials": "true",
    }

    # Return the liked items
    return JSONResponse(content={"liked_items": liked_clothing_data}, headers=headers)

