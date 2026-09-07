from fastapi import FastAPI
from src.api.main import app as fastapi_app

app = FastAPI()

app.mount("/api", fastapi_app)