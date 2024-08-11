from PIL import Image
import numpy as np
import base64
from io import BytesIO
import json

# Create a random image of size 3072x3072
image_array = np.random.randint(0, 255, (3072, 3072, 3), dtype=np.uint8)
image = Image.fromarray(image_array)

# Convert image to bytes buffer
buffered = BytesIO()
image.save(buffered, format="PNG")
image_bytes = buffered.getvalue()

# Encode image bytes to base64
image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# Create the dictionary structure
data = {
    "input": {
        "x": 10,
        "y": 20,
        "Image_buffer": image_base64
    }
}

# Save the JSON to a file named test_input.json
with open('test_input.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("test_input.json has been created.")

