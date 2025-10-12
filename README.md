# emotiondetector

## Setup Instructions

Follow these steps to set up and run the emotion detection API:

### 1. Set up Python Virtual Environment
```bash
python3 -m venv .venv
```

### 2. Activate the Virtual Environment
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the Model
- Attach the `svm_model.pkl` file to your project
- Specify the model path in the `config.yaml` file

### 5. Start the API Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Test the API
Use the Python script to test the API server with your picture:
```bash
python send_requests.py
```
