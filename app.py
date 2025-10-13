import io
import logging
import os
import pickle
import sys

import numpy as np
from PIL import Image

#sys.path.append(os.getcwd())

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from config import Settings
from models import Emotion, EmotionsResponse
from predictor import BatchFeatureExtractor, EmotionPredictor

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()

emotion_predictor = EmotionPredictor(settings.model_path)
feature_extractor = BatchFeatureExtractor()

# Initialize FastAPI app
app = FastAPI(title="Emotion Detector API",
              description="A simple API to detect emotions from images",
              version="1.0.0")


@app.get("/healthcheck")
async def root():
    return {"message": "connection successful"}


# POST endpoint to detect emotions from image
@app.post(
    "/image/",
    response_model=EmotionsResponse,
    summary="detect emotions from image",
    description=
    "upload an image and then use the model to detect emotions from the image",
    tags=["emotion-detection"])  ##
async def detect_emotion(image: UploadFile = File(...)) -> EmotionsResponse:

    image_content = await image.read()
    if not image_content or not isinstance(image_content, bytes):
        logger.error(f"Invalid image content: {image_content}")
        raise HTTPException(status_code=400, detail="Invalid image content")

    # Convert to numpy array
    image_pil = Image.open(io.BytesIO(image_content))

    # Verify the image was loaded successfully
    if image_pil.size[0] == 0 or image_pil.size[1] == 0:
        logger.error(f"Invalid image dimensions: {image_pil.size}")
        raise HTTPException(status_code=400, detail="Invalid image dimensions")

    # Convert to RGB if necessary (handles RGBA, L, etc.)
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')

    image_np = np.array(image_pil)
    if image_np.size == 0:
        logger.error(f"Empty image array: {image_np.size}")
        raise HTTPException(status_code=400, detail="Empty image array")

    try:
        features = feature_extractor.process_batch(image_np)
        if not features:
            logger.error(
                f"No features detected: {features} from image: {image_content}"
            )
            raise HTTPException(status_code=500, detail="No features detected")
        predictions = emotion_predictor.predict(features)
    except Exception as e:
        logger.error(
            f"Error processing image: {e} from image: {image_content}")
        raise HTTPException(status_code=500,
                            detail=f"Error processing image: {e}")

    emotions = []
    for prediction in predictions:
        for face_idx, emotion in prediction.items():
            emotions.append(Emotion(face_num=face_idx, emotion=emotion))

    return EmotionsResponse(emotions=emotions)


# Run the application (for development purposes)
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
