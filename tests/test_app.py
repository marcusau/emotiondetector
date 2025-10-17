from fastapi.testclient import TestClient

from app import app
from models import EmotionsResponse

client = TestClient(app)


def test_healthcheck():
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"message": "connection successful"}


def _test_emotion_detection():
    image_path = "images/image1.jpg"
    with open(image_path, "rb") as image_file:
        files = {"image": image_file}
    response = client.post(
        "/image/",
        files=files,
    )
    assert response.status_code == 200
    assert response.json() is not None
    assert isinstance(response.json(), EmotionsResponse)
