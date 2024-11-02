# Import necessary libraries from FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse,JSONResponse
import uvicorn
from fastapi.staticfiles import StaticFiles

# Import additional libraries for file handling
import sys
import os.path

# Add the parent directory to the system path for module imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

# Import the ImagePredictor class from the Butterfly_Classification module
from predictor import ImagePredictor

# Initialize the FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount static files directory to serve static assets
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the root endpoint to serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serve the index.html file as the main page of the application.
    """
    with open("/app/app/index.html", "r") as f:
        #/app/app/
        return f.read()

# Load the predictor configuration from the specified YAML file
predictor_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
predictor = ImagePredictor.init_from_config_path(predictor_config_path)

# Define the endpoint to upload files for scoring
@app.post("/scorefile/")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Handle file uploads and perform predictions using the ImagePredictor.
    
    Args:
        file (UploadFile): The uploaded file to be processed.

    Returns:
        The prediction result from the ImagePredictor.
    
    Raises:
        HTTPException: If there's an error during file processing.
    """
    try:
        # Perform prediction using the uploaded file
        return predictor.predict_from_file(file.file)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)