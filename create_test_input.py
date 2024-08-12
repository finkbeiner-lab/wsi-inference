from PIL import Image
import base64
from io import BytesIO
import json

# Read an existing image file
image_path = "path/to/your/image.png"  # Replace with your image path
image = Image.open(image_path)

# Convert image to bytes buffer
buffered = BytesIO()
image.save(buffered, format="PNG")
image_bytes = buffered.getvalue()

# Encode image bytes to base64
image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# Create the dictionary structure
data = {
    "input": {
        "x": 100,
        "y": 100,
        "Image_buffer": image_base64
    }
}

# Save the JSON to a file named test_input.json
with open('test_input.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("test_input.json has been created.")