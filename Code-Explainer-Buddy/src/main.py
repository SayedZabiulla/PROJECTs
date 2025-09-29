from fastapi import FastAPI

# Create an instance of the FastAPI class
app = FastAPI()

# Define a path operation decorator
@app.get("/")
def read_root():
    """
    This is the root endpoint of the API.
    """
    return {"message": "Code Explainer Buddy API is running!"}