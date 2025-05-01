import os
from dotenv import load_dotenv


load_dotenv(override=True)

# Extract API keys and configuration
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')

