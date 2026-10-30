
import os
import streamlit as st
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

        # Get and validate the selected model
        selected_groq_model = controls.get("selected_groq_model")

        if not isinstance(selected_groq_model, str):
            raise ValueError(
                f"Invalid Groq model: {selected_groq_model!r}"
            )

        selected_groq_model = selected_groq_model.strip()

        if not selected_groq_model:
            raise ValueError("Please select a Groq model.")

        # Get API key from the UI or environment
        groq_api_key = (
            controls.get("GROQ_API_KEY")
            or os.environ.get("GROQ_API_KEY")
        )

        # Reject unexpected API key types
        if not isinstance(groq_api_key, str):
            raise TypeError(
                "Groq API key must be a string. "
                "Check how the API key is stored."
            )

        groq_api_key = groq_api_key.strip()

        if not groq_api_key:
            raise ValueError("Please enter your Groq API key.")

        # Initialize the Groq chat model
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
