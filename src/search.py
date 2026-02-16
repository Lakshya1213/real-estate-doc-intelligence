from fastapi import FastAPI
from src.routes import router

app = FastAPI(
    title="Real Estate Doc Intelligence API",
    version="1.0.0"
)

# Include all routes from routes.py
app.include_router(router)

# # Optional root endpoint
# @app.get("/")
# def root():
#     return {"message": "API is running successfully 🚀"}
