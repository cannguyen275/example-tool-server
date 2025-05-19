"""Example tool server!"""

import hmac
import os
import pandas as pd
from langchain_core.tools import tool
from universal_tool_server import Server, Auth

from app.tools.exchange_rate import get_exchange_rate
from app.tools.github import get_github_issues
from app.tools.hackernews import search_hackernews
from app.tools.reddit import search_reddit_news
from app.tools.retrieval_tool.retrieval_tool import retrieve_agroz_info
from langchain_core.tools import ToolException

DISABLE_AUTH = os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1")


def _get_app_secret() -> str:
    """Get the app secret from the environment.

    This is sufficient for a very simple authentication system that contains
    a single "user" with a single secret key.
    """
    secret = os.environ.get("APP_SECRET")

    if DISABLE_AUTH:
        if secret:
            raise ValueError("APP_SECRET is not needed when DISABLE_AUTH is enabled.")
        return ""

    if not secret:
        raise ValueError("APP_SECRET environment variable is required.")
    if secret != secret.strip():
        raise ValueError("APP_SECRET cannot have leading or trailing whitespace.")
    return secret


APP_SECRET = _get_app_secret()


app = Server()


@app.add_tool()
async def echo(msg: str) -> str:
    """Echo a message appended with an exclamation mark."""
    return msg + "!"


# Or add an existing langchain tool
@tool()
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is nice today with a high of 75°F."

@tool()
async def save_report(report_time: str, chat_history: str, summary_report: str, incident_type: str) -> str:
    """To save incident report provided by user, must use this tool."""
    try:
        new_row = {
            'Report Time': report_time,
            'Store Report': chat_history,
            'Report summary': summary_report,
            "Type of incident": incident_type
        }
        new_df_data = pd.DataFrame([new_row])

        file_path = 'user_reports.xlsx'
        if os.path.isfile(file_path):
            df = pd.read_excel(file_path)
            df = pd.concat([df, new_df_data], ignore_index=True)
        else:
            df = new_df_data
        # Save the updated DataFrame to the Excel file
        df.to_excel('user_reports.xlsx', index=False)

        return "Thanks for your information. The record is saved successfully\n"
    except Exception as e:
        raise ToolException(f"Error appending user reports to Excel file: {e}\n")
        # return f"Error appending user reports to Excel file: {e}"

app.add_tool(get_weather)

# Add some real tools
TOOLS = [
    search_hackernews,
    get_github_issues,
    get_exchange_rate,
    search_reddit_news,
    retrieve_agroz_info,
    save_report
]

for tool_ in TOOLS:
    app.add_tool(tool_)

# Add the authentication handler
auth = Auth()
app.add_auth(auth)


@auth.authenticate
async def authenticate(authorization: str) -> dict:
    """Authenticate the user based on the Authorization header."""
    if DISABLE_AUTH:
        return {
            "identity": "unauthenticated-user",
        }
    if not authorization or not hmac.compare_digest(authorization, APP_SECRET):
        raise Auth.exceptions.HTTPException(status_code=401, detail="Unauthorized")

    return {
        "identity": "authenticated-user",
    }
