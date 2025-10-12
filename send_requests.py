import requests
import base64

url = "http://localhost:8000/image/"

image_path = "images/image1.jpg"

with open(image_path, "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, files=files)
print(response.status_code)
print(response.json())

