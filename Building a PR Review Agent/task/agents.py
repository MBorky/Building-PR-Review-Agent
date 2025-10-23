import os
from dotenv import load_dotenv
from github import Github
from github import Auth
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from tools import GithubTools

def create_agents(llm: OpenAI, github_tools):
    """"""

    tools_context_agent = github_tools.to_function_tools(["get_pr_details",
                                            "get_file_content",
                                            "get_pr_commit_detail",
                                            "add_gathered_context_to_state"])
    
    context_agent = FunctionAgent(
        name="ContextAgent",
        tools=tools_context_agent,
        llm=llm,
        verbose=True,
        description="Gathers all the needed context and save it to the state.",
        can_handoff_to = ["CommentorAgent", "ReviewAndPostingAgent"],
        system_prompt= """
        You are the context gathering agent. When gathering context, you MUST gather \n:
      - The details: author, title, body, diff_url, state, and head_sha; \n
      - Changed files; \n
      - Any requested for files; \n
        Once you gather the requested info, you MUST hand control back to the Commentor Agent.
        """
    )
    
    commentor_agent = FunctionAgent(
        name="CommentorAgent",
        system_prompt='''
        You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n
    Ensure to do the following for a thorough review:
     - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
        - If you need any additional details, you must hand off to the
     ContextAgent, Do NOT ask user!. \n
     - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
        - What is good about the PR? \n
        - Did the author follow ALL contribution rules? What is missing? \n
        - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
        - Are new endpoints documented? - use the diff to determine this. \n
        - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
     - You should directly address the author. So your comments should sound like: \n
     "Thanks for fixing this. I think all places where we call quote should
     be fixed. Can you roll this fix out everywhere?".\n
     - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
        ''',
        llm=llm,
        verbose=True,
        description="Uses the context gathered by the context agent to draft a pull review comment comment.",
        tools=github_tools.to_function_tools(["add_draft_comment"]),
        can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
    )
    
    review_and_posting_agent = FunctionAgent(
        name="ReviewAndPostingAgent",
        system_prompt="""
        You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.
        """,
        llm=llm,
        verbose=True,
        description="Posts a review to GitHub once it is ready.",
        tools=github_tools.to_function_tools(["add_final_review_to_state", "post_final_review_to_github_pr"]),
        can_handoff_to=["CommentorAgent"]
        
    )
    
    return context_agent, commentor_agent, review_and_posting_agent