import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST data (simulating the catalog)
(_, _), (catalog_images, catalog_labels) = fashion_mnist.load_data()

# Class labels to simulate catalog items
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def get_random_catalog_item():
    idx = np.random.randint(0, len(catalog_images))
    item_image = catalog_images[idx]
    item_label = class_labels[catalog_labels[idx]]
    return item_image, item_label
