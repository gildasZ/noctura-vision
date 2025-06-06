#
## -> `requirements.txt`
```bash
pipenv install -r requirements.txt

# Specifically for PowerShell:
pipenv run pip freeze | Out-File -Encoding utf8 requirements.txt  
```

---
## Checking Python versions with Python Launcher (py.exe)
```bash
py --list  # Should list both 3.12 and 3.11
py -3.11 --version # Should output Python 3.11.9
py -3.12 --version # Should output Python 3.12.10
py --version       # Should output your default, Python 3.12.10
```

---
## Running the _list_structure.py file
```bash
# Show the built-in help
python _list_structure.py -h

# Basic invocation
## Execute with defaults (project_root defaults to . and depth defaults to 6):
python _list_structure.py
## To list the structure of the current directory, to 6 levels deep:
python _list_structure.py . --depth 6
## Or equivalently (since project_root defaults to .):
python _list_structure.py --depth 6

# Pointing at a specific project
## If your repo lives in /home/you/myproj, and you want full depth (8):
python _list_structure.py /home/you/myproj
## Or to limit to 3:
python _list_structure.py /home/you/myproj --depth 3
```

---
## The proper command to use for PyTorch packages with `pipenv`:
```bash
pipenv install torch torchvision torchaudio --index https://download.pytorch.org/whl/cu128
```

---
# -> Backend: Running the FastAPI Backend and Testing Endpoints

These are the primary commands for running and testing the `noctura-vision` backend application.

### 1. Start the Server

First, ensure your `pipenv` virtual environment is set up and all dependencies are installed.

1.  Navigate to the `backend` directory:
    ```bash
    cd path/to/noctura-vision/backend
    ```

2.  Run the Uvicorn server using `pipenv`:
    ```bash
    pipenv run uvicorn app.main:app --reload
    # or
    # pipenv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   This command starts the web server.
    *   The first time you run this, it will take several minutes to download the AI models into the `backend/models_cache` directory.
    *   `--reload` makes the server restart automatically when you save code changes, which is perfect for development.

### 2. Test the HTTP Endpoints (Image Processing)

Once the server is running, you can test the standard API endpoints.

1.  Open your web browser and go to the interactive documentation page:
    ```
    http://127.0.0.1:8000/docs
    ```
2.  From this page, you can test the following endpoints:
    *   `GET /health` (Or `http://localhost:8000/health`): A simple check to see if the server is running. You should see `{"status":"ok"}` as the JSON response.
    *   `GET /api/v1/models`: Returns a list of the available segmentation models.
    *   `POST /api/v1/process_image`: Upload an image file and choose processing options to get a result.

### 3. Test the WebSocket Endpoint (Video Streaming)

The WebSocket for real-time video cannot be tested from the `/docs` page.

1.  In your file explorer, navigate to the `noctura-vision/backend/` directory.
2.  Open the `wstest.html` file directly in your web browser (e.g., by double-clicking it).
3.  The browser will ask for permission to use your webcam. Click **Allow**.
4.  Click the **"Start Webcam"** button to begin the real-time processing stream. You can then interact with the model and enhancement options on the page.

---

I need to get the new formatted commands for running the backend...
