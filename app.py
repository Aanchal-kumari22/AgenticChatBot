from dotenv import load_dotenv

# Load local environment variables before importing the app.
load_dotenv()

from src.langGraphAgenticAi.main import load_langgraph_agenticai_app


if __name__ == "__main__":
    load_langgraph_agenticai_app()
