import os
from dotenv import load_dotenv

load_dotenv()  # Load the .env file

api_key = os.getenv("API_KEY")  # Get the API key from the environment

# Use the api_key variable in your code
print(api_key)