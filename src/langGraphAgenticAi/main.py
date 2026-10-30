import streamlit as st

from src.langGraphAgenticAi.ui.streamlit.loadui import LoadStreamlitUI

from src.langGraphAgenticAi.LLMS.GroqLLM import GroqLLM

from src.langGraphAgenticAi.graph.graphbuilder import GraphBuilder

from src.langGraphAgenticAi.ui.streamlit.displayresult import (
    DisplayResultStreamlit
)


def load_langgraph_agenticai_app():

    # ------------------------------------------
    # LOAD UI
    # ------------------------------------------

    ui = LoadStreamlitUI()

    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Failed to load UI configuration.")
        return

    selected_usecase = user_input.get("selected_usecase")

    if not selected_usecase:
        st.warning("Please select a use case.")
        return

    normalized_usecase = " ".join(
        selected_usecase.strip().lower().split()
    )

    # ------------------------------------------
    # GET USER MESSAGE
    # ------------------------------------------

    if normalized_usecase == "ai news":

        if not st.session_state.get(
            "IsFetchButtonClicked", False
        ):
            return

        user_message = st.session_state.get(
            "timeframe", "Daily"
        )

    else:

        user_message = st.chat_input(
            "Enter your message:"
        )

        if not user_message:
            return

    # ------------------------------------------
    # VALIDATE API KEYS
    # ------------------------------------------

    selected_llm = user_input.get("selected_llm")

    if selected_llm != "Groq":
        st.error("Currently only Groq is supported.")
        return

    groq_api_key = user_input.get("GROQ_API_KEY")

    if not groq_api_key:
        st.error("Please enter your Groq API key.")
        return

    search_usecases = {
        "chatbot with web search",
        "ai news"
    }

    tavily_api_key = None

    if normalized_usecase in search_usecases:

        tavily_api_key = user_input.get("TAVILY_API_KEY")

        if not tavily_api_key:
            st.error("Please enter your Tavily API key.")
            return

    # ------------------------------------------
    # INITIALIZE LLM
    # ------------------------------------------

    try:

        llm_config = GroqLLM(
            user_controls_input=user_input
        )

        model = llm_config.get_llm_model()

        if model is None:
            st.error("Groq model initialization failed.")
            return

    except Exception as e:

        st.error(
            f"LLM initialization failed: {type(e).__name__}: {e}"
        )
        return

    # ------------------------------------------
    # BUILD GRAPH
    # ------------------------------------------

    try:

        graph_builder = GraphBuilder(
            model=model,
            tavily_api_key=tavily_api_key
        )

        graph = graph_builder.setup_graph(
            selected_usecase
        )

    except Exception as e:

        st.error(
            f"Graph setup failed: {type(e).__name__}: {e}"
        )
        return

    # ------------------------------------------
    # DISPLAY RESULT
    # ------------------------------------------

    try:

        display = DisplayResultStreamlit(
            selected_usecase,
            graph,
            user_message
        )

        display.display_result_on_ui()

    except Exception as e:

        st.error(
            f"Execution failed: {type(e).__name__}: {e}"
        )

    finally:

        if normalized_usecase == "ai news":
            st.session_state.IsFetchButtonClicked = False


if __name__ == "__main__":
    load_langgraph_agenticai_app()
