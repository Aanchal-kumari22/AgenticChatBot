from configparser import ConfigParser
from pathlib import Path


class Config:

    def __init__(self, config_file=None):

        self.config = ConfigParser()

        if config_file is None:
            config_file = (
                Path(__file__).resolve().parent
                / "uiconfigfile.ini"
            )

        config_file = Path(config_file)

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}"
            )

        self.config.read(config_file, encoding="utf-8")

    def _get_options(self, key):

        value = self.config["DEFAULT"].get(key, "")

        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    def get_llm_options(self):
        return self._get_options("LLM_OPTIONS")

    def get_usecase_options(self):
        return self._get_options("USECASE_OPTIONS")

    def get_groq_model_options(self):
        return self._get_options("GROQ_MODEL_OPTIONS")

    def get_page_title(self):
        return self.config["DEFAULT"].get(
            "PAGE_TITLE",
            "LangGraph: Agentic AI graph"
        )
