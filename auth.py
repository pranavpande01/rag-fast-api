import os
from dotenv import load_dotenv

class Config:
    def authenticate(self):
        load_dotenv()  # Loads variables from .env into os.environ
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("Missing GOOGLE_API_KEY in environment.")
