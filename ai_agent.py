import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id, api_key=os.getenv("GROQ_API_KEY"))
    elif provider == "Google":
        llm = ChatGoogleGenerativeAI(model=llm_id, api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        raise ValueError("Invalid provider")

    tools = [TavilySearchResults(max_results=2, api_key=os.getenv("TAVILY_API_KEY"))] if allow_search else []

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )

    state = {"messages": query}
    result = agent.invoke(state)
    messages = result.get("messages", [])

    ai_messages = [m.content for m in messages if isinstance(m, AIMessage)]
    return {"response": ai_messages[-1] if ai_messages else "No response received."}
