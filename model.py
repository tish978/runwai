import numpy as np
import tensorflow as tf
from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId
import requests
from PIL import Image
from io import BytesIO
import aiohttp

# Load the model
model = tf.keras.models.load_model("clothing_model.keras")

# MongoDB connection
client = AsyncIOMotorClient("mongodb+srv://satishbisa:HiThere!123@cluster0.vji5t.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['runwai_db']
collection = db['clothing_catalog']


# Fetch image and preprocess it for model prediction
async def fetch_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                image_data = await response.read()
                image = Image.open(BytesIO(image_data)).convert('L')  # Convert to grayscale
                image = image.resize((28, 28))
                return np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0
            raise Exception(f"Failed to fetch image from {url}")

async def download_image(image_url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            if resp.status == 200:
                image_data = await resp.read()
                image = Image.open(BytesIO(image_data)).convert('L')  # Convert to grayscale
                image = image.resize((28, 28))  # Resize to match the input of your model
                image = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0
                return image
            else:
                raise Exception("Failed to download image")

# In the get_image_for_prediction function
async def get_image_for_prediction(item_id: str, model):
    # Retrieve clothing item by ID from MongoDB
    item = await collection.find_one({"clothing_id": item_id})
    if not item:
        raise Exception("Item not found")

    # Fetch the image from the URL
    image_url = item["image_url"]
    image = await download_image(image_url)

    # Predict if the user will like it
    prediction = model.predict(image)
    return prediction[0][0], item