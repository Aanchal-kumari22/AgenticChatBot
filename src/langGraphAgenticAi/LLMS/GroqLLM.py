import os
from langchain_groq import ChatGroq


class GroqLLM:

    def __init__(self, user_controls_input):

        if not isinstance(user_controls_input, dict):
            raise TypeError(
                "Expected user_controls_input to be a dictionary."
            )

        self.user_controls_input = user_controls_input

    def get_llm_model(self):

        controls = self.user_controls_input

        # ------------------------------------------
        # GET SELECTED GROQ MODEL
        # ------------------------------------------

        selected_groq_model = controls.get(
            "selected_groq_model"
        )

        if not isinstance(selected_groq_model, str):
            raise ValueError(
                "Invalid Groq model selection."
            )

        selected_groq_model = selected_groq_model.strip()

        if not selected_groq_model:
            raise ValueError(
                "Please select a Groq model."
            )

        # ------------------------------------------
        # GET API KEY FROM ENVIRONMENT
        # ------------------------------------------

        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured on the server."
            )

        # ------------------------------------------
        # INITIALIZE GROQ MODEL
        # ------------------------------------------

        try:

            llm = ChatGroq(
                model=selected_groq_model,
                groq_api_key=groq_api_key,
                temperature=0
            )

            return llm

        except Exception as e:

            raise RuntimeError(
                f"Groq initialization failed: "
                f"{type(e).__name__}: {e}"
            ) from e
