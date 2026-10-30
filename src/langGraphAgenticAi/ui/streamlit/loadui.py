import streamlit as st
import os

from src.langGraphAgenticAi.ui.uiconfigfile import Config


class LoadStreamlitUI:

    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit_ui(self):

        st.set_page_config(
            page_title=self.config.get_page_title(),
            layout="wide"
        )

        st.header("🤖 " + self.config.get_page_title())

        defaults = {
            "timeframe": "Daily",
            "IsFetchButtonClicked": False,
            "GROQ_API_KEY": "",
            "TAVILY_API_KEY": ""
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        with st.sidebar:

            st.subheader("LLM Configuration")

            # LLM provider
            llm_options = self.config.get_llm_options()

            selected_llm = st.selectbox(
                "Select LLM",
                llm_options
            )

            self.user_controls["selected_llm"] = selected_llm

            # Groq configuration
            if selected_llm == "Groq":

                model_options = self.config.get_groq_model_options()

                selected_model = st.selectbox(
                    "Select Model",
                    model_options,
                    key="selected_groq_model"
                )

                self.user_controls["selected_groq_model"] = selected_model

                groq_api_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    key="GROQ_API_KEY"
                )

                self.user_controls["GROQ_API_KEY"] = groq_api_key

                if not groq_api_key:
                    st.warning("Please enter your Groq API key.")

            # Use case
            usecase_options = self.config.get_usecase_options()

            selected_usecase = st.selectbox(
                "Select Usecase",
                usecase_options
            )

            self.user_controls["selected_usecase"] = selected_usecase

            normalized_usecase = " ".join(
                selected_usecase.strip().lower().split()
            )

            # Tavily configuration
            search_usecases = {
                "chatbot with web search",
                "ai news"
            }

            if normalized_usecase in search_usecases:

                st.subheader("🌐 Web Search Configuration")

                tavily_api_key = st.text_input(
                    "Tavily API Key",
                    type="password",
                    key="TAVILY_API_KEY"
                )

                self.user_controls["TAVILY_API_KEY"] = tavily_api_key

                if tavily_api_key:
                    os.environ["TAVILY_API_KEY"] = tavily_api_key
                else:
                    os.environ.pop("TAVILY_API_KEY", None)
                    st.warning("Please enter your Tavily API key.")

            # AI News Explorer
            if normalized_usecase == "ai news":

                st.subheader("📰 AI News Explorer")

                time_frame = st.selectbox(
                    "Select Time Frame",
                    ["Daily", "Weekly", "Monthly"],
                    key="news_timeframe"
                )

                self.user_controls["timeframe"] = time_frame

                if st.button(
                    "🔍 Fetch Latest AI News",
                    use_container_width=True
                ):
                    st.session_state.IsFetchButtonClicked = True
                    st.session_state.timeframe = time_frame

            else:
                st.session_state.IsFetchButtonClicked = False

        return self.user_controls
