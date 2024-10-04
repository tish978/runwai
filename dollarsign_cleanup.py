from pymongo import MongoClient

# MongoDB connection URI (replace with your actual URI)
client = MongoClient("mongodb+srv://satishbisa:HiThere!123@cluster0.vji5t.mongodb.net/?retryWrites=true&w=majority")

# Connect to the specific database and collection
db = client['runwai_db']  # Replace with your database name
collection = db['clothing_catalog']  # Replace with your collection name

# Filter to delete documents where brand is 'H&M'
filter_query = {'brand': 'H&M'}

# Delete all documents matching the filter
delete_result = collection.delete_many(filter_query)

# Output how many documents were deleted
print(f"Deleted {delete_result.deleted_count} documents where brand is 'H&M'.")

# Close the connection
client.close()
