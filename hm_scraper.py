from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import uuid
from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB connection
client = AsyncIOMotorClient("mongodb+srv://satishbisa:HiThere!123@cluster0.vji5t.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['runwai_db']
collection = db['clothing_catalog']

def generate_unique_clothing_id():
    """Generate a unique clothing ID."""
    return str(uuid.uuid4())

def insert_product_to_db(product_data):
    """Insert product into MongoDB."""
    collection.insert_one(product_data)

def scrape_product_page(driver, product_url):
    # Open the product page
    driver.get(product_url)
    
    # Allow the page to load
    time.sleep(2)

    # Parse the product page to find the image URL
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Extract the main product image
    img_tag = soup.find('img')
    if img_tag and 'src' in img_tag.attrs:
        return img_tag['src']
    else:
        return 'N/A'

def scrape_page(driver):
    # Get the page source and parse it with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Find all the product containers (adjust the selector based on your page structure)
    product_containers = soup.find_all('div', class_='eed2a5')  # Update this with correct class for the product container
    if not product_containers:
        print("No product containers found.")
        return

    for product in product_containers:
        try:
            # Extract product name
            name = product.find('h2').text.strip() if product.find('h2') else 'N/A'

            # Find any span containing the price (text with '$')
            price_tag = product.find('span', string=lambda text: '$' in text if text else False)
            price = price_tag.text.strip().replace('$', '').strip() if price_tag else 'N/A'  # Remove the dollar sign

            # Extract product URL
            product_url = product.find('a', href=True)['href'] if product.find('a', href=True) else 'N/A'

            # Fetch the image URL from the product page
            image_url = scrape_product_page(driver, product_url) if product_url != 'N/A' else 'N/A'

            # Construct the product data
            product_data = {
                "clothing_id": generate_unique_clothing_id(),
                "item_name": name,
                "image_url": image_url,
                "price": price,  # Cleaned price without dollar sign
                "url": product_url,
                "category": "Men's Clothing",
                "brand": "H&M"
            }

            # Insert the product into MongoDB
            insert_product_to_db(product_data)

            # Print the scraped data for verification
            print(f"Inserted Product - Name: {name}, Price: {price}, URL: {product_url}")
            print("-" * 40)
        except Exception as e:
            print(f"Error extracting product details: {str(e)}")

def click_next_page(driver):
    try:
        # Find and click the "Load Next Page" button or pagination next arrow
        next_button = driver.find_element(By.CSS_SELECTOR, 'button[aria-label="Load Next Page"]')
        next_button.click()
        time.sleep(3)  # Allow time for the next page to load
        return True
    except Exception as e:
        print(f"Error clicking next page: {str(e)}")
        return False

def scroll_and_scrape(driver, max_scrolls=10):
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0

    while scroll_count < max_scrolls:
        # Scroll down to the bottom of the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait for new products to load
        time.sleep(5)

        # Scrape the currently visible products
        print(f"Scraping after scroll {scroll_count + 1}")
        scrape_page(driver)

        # Get the new page height after scrolling
        new_height = driver.execute_script("return document.body.scrollHeight")

        # If the new height is the same as the last height, we've reached the bottom of the page
        if new_height == last_height:
            print("Reached the bottom of the page.")
            break

        last_height = new_height
        scroll_count += 1

    print("Finished scrolling and scraping.")

def scrape_all_pages(driver, max_scrolls_per_page=10):
    while True:
        # Scroll and scrape the current page
        scroll_and_scrape(driver, max_scrolls=max_scrolls_per_page)
        
        # Try to click on the "Next Page" button
        if not click_next_page(driver):
            break  # If there are no more pages, break the loop

# Initialize Edge driver
driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()))

# Navigate to H&M men's product page
driver.get('https://www2.hm.com/en_us/men/shop-by-product/view-all.html?page=1')

# Give the page some time to load initially
time.sleep(5)

# Scrape all pages
scrape_all_pages(driver, max_scrolls_per_page=5)

# Close the browser after scraping
driver.quit()
