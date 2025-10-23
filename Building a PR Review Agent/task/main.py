import asyncio
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from typing import Any
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.agent.workflow import AgentWorkflow
from tools import GithubTools
from github import Github, Auth
from agents import create_agents

repo_url = "https://github.com/MBorky/recipes-api.git"
load_dotenv(override=True)
git_token = os.getenv("GITHUB_TOKEN")
base_url = os.getenv("BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
auth = Auth.Token(git_token)
g = Github(auth=auth)

llm = OpenAI(model="gpt-4o-mini", api_base=base_url, api_key=api_key)
github_tools = GithubTools(g, "MBorky/recipes-api")

context_agent, commentor_agent, review_and_posting_agent = create_agents(github_tools=github_tools,
                                                llm=llm)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_context": "",
        "pr_number": "",
        "draft_comment": "",
        "final_review_comment": "",
    },
)


async def main():
    query = input().strip()
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    repo_url = "https://github.com/MBorky/recipes-api.git"
    asyncio.run(main())
    g.close()