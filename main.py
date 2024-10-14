from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Request
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import List, Optional
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
from bson import ObjectId
import json


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
cart_collection = db['cart']
closet_collection = db['closet']

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

# Define a Pydantic model for the request body
class PurchaseSuccessRequest(BaseModel):
    stripe_session_id: str

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

# Define a Pydantic model for the request body
class RemoveSavedItemRequest(BaseModel):
    item_id: str

# Model for checkout item
class CheckoutItem(BaseModel):
    clothing_id: str
    item_name: str
    price: float
    quantity: int = 1

# Model for items that will be stored in the closet collection
class ClosetItem(BaseModel):
    clothing_id: str
    item_name: str
    image_url: Optional[str]  # Make the image URL optional
    purchase_date: Optional[datetime] = None  # Will be set at the time of insertion
    price: float

class ClosetItemResponse(BaseModel):
    clothing_id: str
    item_name: str
    image_url: Optional[str]
    purchase_date: datetime
    price: float


# Model for CartItem
class CartItem(BaseModel):
    clothing_id: str
    item_name: str
    price: float
    quantity: int = 1  # Default to 1 if not specified    

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

@app.options("/login")
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

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await user_collection.find_one({"email": form_data.username})
    
    if user and verify_password(form_data.password, user["password"]):
        access_token = create_access_token(data={"sub": user["user_id"]})
        # Adding CORS headers like a pro
        headers = {
            "Access-Control-Allow-Origin": "http://localhost:3000", 
            "Access-Control-Allow-Credentials": "true"
        }
        # Serve up that response with headers like a gourmet dish
        return JSONResponse(content={"access_token": access_token, "token_type": "bearer"}, headers=headers)
    
    # Oops, bad credentials! 🙅‍♂️
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
def recommend_items(liked_items, disliked_items, clothing_data, cosine_sim):
    recommended_items = set()

    for item_id in liked_items:
        # Find the index of the liked item in the clothing_data DataFrame
        item_idx = clothing_data[clothing_data['clothing_id'] == item_id].index[0]

        # Get similarity scores for this item
        similarity_scores = list(enumerate(cosine_sim[item_idx]))

        # Sort items by similarity score
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get the index of the top similar item (excluding liked and disliked items)
        similar_items_indices = [
            i for i, _ in similarity_scores[1:]  # Skip the first item since it's the liked item itself
            if clothing_data.iloc[i]['clothing_id'] not in liked_items
            and clothing_data.iloc[i]['clothing_id'] not in disliked_items  # Exclude disliked items as well
        ]

        # Add the similar items to the recommendation list (ensure no duplicates)
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
    # Get user feedback history (liked and disliked items)
    user_feedback = await feedback_collection.find({"user_id": current_user["user_id"]}).to_list(100)
    liked_items = [feedback["item_id"] for feedback in user_feedback if feedback["liked"]]
    disliked_items = [feedback["item_id"] for feedback in user_feedback if not feedback["liked"]]

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

    # Get recommendations based on liked items, excluding disliked items
    recommended_items = recommend_items(liked_items, disliked_items, clothing_data, cosine_sim)

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



@app.options("/saved-items/remove")
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


@app.post("/saved-items/remove")
async def remove_saved_item(body: RemoveSavedItemRequest, current_user: dict = Depends(get_current_user)):
    # Use the item_id from the request body
    result = await feedback_collection.delete_one({"user_id": current_user["user_id"], "item_id": body.item_id})

    if result.deleted_count == 1:
        headers = {
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true",
        }
        return JSONResponse(content={"message": "Item removed from saved items"}, headers=headers)

    raise HTTPException(status_code=404, detail="Item not found in saved items")



@app.options("/cart/add")
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


from fastapi.responses import JSONResponse

# Route to add an item to the cart
@app.post("/cart/add")
async def add_to_cart(cart_item: CartItem, current_user: dict = Depends(get_current_user)):
    # Check if the item is already in the cart
    existing_cart_item = await cart_collection.find_one({"user_id": current_user["user_id"], "clothing_id": cart_item.clothing_id})
    
    if existing_cart_item:
        # Update the quantity of the existing item
        new_quantity = existing_cart_item["quantity"] + cart_item.quantity
        await cart_collection.update_one(
            {"user_id": current_user["user_id"], "clothing_id": cart_item.clothing_id},
            {"$set": {"quantity": new_quantity}}
        )
        
        headers = {
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true",
        }
        return JSONResponse(content={"message": "Item quantity updated in cart", "quantity": new_quantity}, headers=headers)
    
    # Add the new item to the cart
    cart_data = {
        "user_id": current_user["user_id"],
        "clothing_id": cart_item.clothing_id,
        "item_name": cart_item.item_name,
        "price": cart_item.price,
        "quantity": cart_item.quantity
    }
    result = await cart_collection.insert_one(cart_data)
    
    if result.acknowledged:
        headers = {
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true",
        }
        return JSONResponse(content={"message": "Item added to cart", "cart_item_id": str(result.inserted_id)}, headers=headers)
    
    raise HTTPException(status_code=500, detail="Failed to add item to cart")



@app.options("/cart/remove")
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


@app.post("/cart/remove")
async def remove_from_cart(clothing_id: str, current_user: dict = Depends(get_current_user)):
    """
    Removes an item from the cart based on clothing_id for the current user.
    """
    # First, check if the item exists in the cart
    existing_cart_item = await cart_collection.find_one({"user_id": current_user["user_id"], "clothing_id": clothing_id})

    if not existing_cart_item:
        # Item not found in the cart, return a friendly message
        headers = {
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true",
        }
        return JSONResponse(
            content={"message": "Item not found in cart or already removed."}, 
            status_code=404, 
            headers=headers
        )

    # Now try to remove the item
    result = await cart_collection.delete_one({"user_id": current_user["user_id"], "clothing_id": clothing_id})

    if result.deleted_count == 1:
        headers = {
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true",
        }
        return JSONResponse(content={"message": "Item removed from cart"}, headers=headers)
    
    # If somehow it still fails, return an error
    raise HTTPException(status_code=500, detail="Failed to remove item from cart")





@app.options("/cart")
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


@app.get("/cart")
async def get_cart_items(current_user: dict = Depends(get_current_user)):
    cart_items = await cart_collection.find({"user_id": current_user["user_id"]}).to_list(100)
    
    if not cart_items:
        raise HTTPException(status_code=404, detail="Cart is empty")
    
    # Convert cart items to a list of dictionaries
    cart_data = pd.DataFrame(cart_items).to_dict(orient="records")
    
    # Ensure ObjectId is converted to string for each item
    for item in cart_data:
        if "_id" in item:
            item["_id"] = str(item["_id"])
    
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",
        "Access-Control-Allow-Credentials": "true",
    }
    
    return JSONResponse(content={"cart_items": cart_data}, headers=headers)



@app.options("/create-checkout-session")
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

# Route to create a Stripe checkout session
@app.post("/create-checkout-session")
async def create_checkout_session(items: List[CheckoutItem], current_user: dict = Depends(get_current_user)):
    try:
        # Convert your cart items into a format Stripe requires
        line_items = [{
            'price_data': {
                'currency': 'usd',
                'product_data': {
                    'name': item.item_name,
                },
                'unit_amount': int(item.price * 100),  # Stripe expects the price in cents
            },
            'quantity': item.quantity,
        } for item in items]

        # Create the checkout session with shipping and billing information
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=line_items,
            mode='payment',
            success_url="http://localhost:3000/success?session_id={CHECKOUT_SESSION_ID}",  # Frontend success URL
            cancel_url="http://localhost:3000/cancel",  # Frontend cancel URL

            # Collect customer shipping address (optional, required, or no preference)
            shipping_address_collection={
                'allowed_countries': ['US', 'CA', 'GB']  # Limit shipping to specific countries
            },
            
            # Collect billing information like email, address, and phone
            billing_address_collection='required',  # 'auto' or 'required'
            
            # Optional: Define customer details like name and email if known
            customer_email=current_user["email"],  # Using the email of the current user

            # Optional: Define shipping options
            shipping_options=[
                {
                    'shipping_rate_data': {
                        'type': 'fixed_amount',
                        'fixed_amount': {
                            'amount': 500,  # Shipping cost in cents (e.g., $5.00)
                            'currency': 'usd',
                        },
                        'display_name': 'Standard shipping',
                        # Delivery estimate
                        'delivery_estimate': {
                            'minimum': {'unit': 'business_day', 'value': 5},
                            'maximum': {'unit': 'business_day', 'value': 7},
                        },
                    },
                },
                {
                    'shipping_rate_data': {
                        'type': 'fixed_amount',
                        'fixed_amount': {
                            'amount': 1500,  # Shipping cost in cents (e.g., $15.00)
                            'currency': 'usd',
                        },
                        'display_name': 'Express shipping',
                        'delivery_estimate': {
                            'minimum': {'unit': 'business_day', 'value': 1},
                            'maximum': {'unit': 'business_day', 'value': 3},
                        },
                    },
                },
            ]
        )

        headers = {
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true",
        }

        # Return session ID with headers
        return JSONResponse(content={"sessionId": session.id}, headers=headers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.options("/digital-closet")
async def options_digital_closet():
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",  # Frontend URL
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    }
    return JSONResponse(content={}, headers=headers)

@app.get("/digital-closet")
async def get_digital_closet_items(current_user: dict = Depends(get_current_user)):
    try:
        # Fetch closet items based on user_id
        closet_items = await closet_collection.find({"user_id": current_user["user_id"]}).to_list(100)

        # Log the fetched items
        print(f"Closet Items Fetched: {closet_items}")

        # If no items are found, raise a 404 exception
        if not closet_items:
            raise HTTPException(status_code=404, detail="No items found in your digital closet.")

        # Convert ObjectId to string before returning the response
        for item in closet_items:
            if '_id' in item:
                item['_id'] = str(item['_id'])  # Convert ObjectId to string

        headers = {
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true",
        }

        # Log the success message and return the items
        print(f"Returning closet items for user {current_user['user_id']}")
        return JSONResponse(content=closet_items, headers=headers)

    except Exception as e:
        # Log the error and raise HTTPException
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


        
@app.options("/purchase-success")
async def options_purchase_success():
   
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",  # Allow requests from the frontend
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    }
    return JSONResponse(content={}, headers=headers)

@app.post("/purchase-success")
async def purchase_success(request: PurchaseSuccessRequest, current_user: dict = Depends(get_current_user)):
    stripe_session_id = request.stripe_session_id
    print(f"Received session ID: {stripe_session_id}")
    
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",  # Allow requests from the frontend
        "Access-Control-Allow-Credentials": "true",
    }

    try:
        # Retrieve the Stripe session using the session_id (without `await`)
        session = stripe.checkout.Session.retrieve(stripe_session_id)

        # Log the session to see what's returned
        print(f"Stripe Session: {session}")

        # Retrieve the cart items for the current user
        cart_items = await cart_collection.find({"user_id": current_user["user_id"]}).to_list(100)

        # Check if the cart is empty
        if not cart_items:
            raise HTTPException(status_code=404, detail="Cart is empty or no items found for the user.")

        # Loop through cart items and add them to the user's digital closet
        for item in cart_items:
            closet_item = {
                "user_id": current_user["user_id"],
                "clothing_id": item["clothing_id"],
                "item_name": item["item_name"],
                "price": float(item["price"]),
                "purchase_date": datetime.utcnow(),
                "image_url": item.get("image_url", None)  # Handle optional image URL
            }
            await closet_collection.insert_one(closet_item)

        # Clear the cart for the user after purchase
        await cart_collection.delete_many({"user_id": current_user["user_id"]})

        return JSONResponse(content={"message": "Purchase processed and closet updated successfully"}, headers=headers)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing purchase: {str(e)}")







@app.options("/stripe-session/{session_id}")
async def options_stripe_session():
    """
    Handle the OPTIONS method for /stripe-session to allow CORS preflight requests.
    """
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    }
    return JSONResponse(content={}, headers=headers)




@app.get("/stripe-session/{session_id}")
async def get_stripe_session(session_id: str):
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        return JSONResponse(content={"id": session.id, "amount_total": session.amount_total}, headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true"
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
