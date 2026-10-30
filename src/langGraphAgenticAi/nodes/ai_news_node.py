from pathlib import Path

from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate


class AINewsNode:

    def __init__(self, llm, tavily_api_key=None):

        if llm is None:
            raise ValueError("LLM model is not initialized.")

        if not tavily_api_key:
            raise ValueError("Tavily API key is missing.")

        self.llm = llm

        self.tavily = TavilyClient(
            api_key=tavily_api_key
        )

        self.output_dir = (
            Path(__file__).resolve().parents[3] / "AINews"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ------------------------------------------
    # GET FREQUENCY
    # ------------------------------------------

    def get_frequency(self, state):

        messages = state.get("messages", [])

        frequency = state.get("frequency", "daily")

        if isinstance(messages, str):
            frequency = messages

        elif isinstance(messages, list) and messages:

            last_message = messages[-1]

            if hasattr(last_message, "content"):
                frequency = last_message.content
            else:
                frequency = str(last_message)

        frequency = str(frequency).strip().lower()

        if frequency not in ["daily", "weekly", "monthly"]:
            raise ValueError(
                "Frequency must be Daily, Weekly or Monthly."
            )

        return frequency

    # ------------------------------------------
    # FETCH NEWS
    # ------------------------------------------

    def fetch_news(self, state: dict) -> dict:

        frequency = self.get_frequency(state)

        time_range_map = {
            "daily": "day",
            "weekly": "week",
            "monthly": "month"
        }

        response = self.tavily.search(
            query=(
                "Latest artificial intelligence news, "
                "LLMs, generative AI, Agentic AI and "
                "machine learning developments"
            ),
            topic="news",
            search_depth="basic",
            time_range=time_range_map[frequency],
            max_results=5,
            include_answer=False
        )

        news_data = response.get("results", [])

        if not news_data:
            raise ValueError(
                "No AI news articles were found."
            )

        return {
            "frequency": frequency,
            "news_data": news_data
        }

    # ------------------------------------------
    # SUMMARIZE NEWS
    # ------------------------------------------

    def summarize_news(self, state: dict) -> dict:

        news_items = state.get("news_data", [])

        if not news_items:
            raise ValueError("No news data available.")

        articles = []

        for index, item in enumerate(news_items[:5], start=1):

            articles.append(
                f"""
                Article {index}
                Title: {str(item.get('title', 'No title'))[:200]}
                Date: {item.get('published_date', 'N/A')}
                Content: {str(item.get('content', ''))[:600]}
                URL: {item.get('url', '')}
                """
            )

        articles_text = "\n".join(articles)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an AI technology news analyst.

                Summarize the provided news in Markdown.

                Include:
                - Article title
                - Publication date, if available
                - Short, informative summary
                - Original source URL

                Avoid duplicate news.
                Never invent facts.
                Keep the response concise.
                """
            ),
            (
                "human",
                "Summarize these AI news articles:\n{articles}"
            )
        ])

        formatted_prompt = prompt.format(
            articles=articles_text
        )

        response = self.llm.invoke(formatted_prompt)

        summary = response.content

        if not summary:
            raise ValueError(
                "The LLM returned an empty summary."
            )

        return {
            "summary": summary
        }

    # ------------------------------------------
    # SAVE RESULT
    # ------------------------------------------

    def save_result(self, state: dict) -> dict:

        frequency = state.get("frequency", "daily")
        summary = state.get("summary", "")

        if not summary:
            raise ValueError("No summary available to save.")

        output_file = (
            self.output_dir / f"{frequency}_summary.md"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                f"# {frequency.capitalize()} AI News Summary\n\n"
            )

            file.write(summary)

        return {
            "summary": summary,
            "output_file": str(output_file)
        }
