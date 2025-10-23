import os
from llama_index.core.tools import FunctionTool
from typing import Any
from github import Github
from llama_index.core.workflow import Context
from github import Auth
from dotenv import load_dotenv


class GithubTools:
    """
    A utility class designed for interacting with GitHub repositories and pull requests
    using a GitHub client object. This class encompasses methods to retrieve pull request
    details, commit information, file contents, and manipulate context state for integration
    purposes.

    This class simplifies GitHub operations by abstracting common tasks while providing
    capabilities for retrieving and saving GitHub data.

    :ivar _repo: The repository instance retrieved from the GitHub client
                 using the repository name or identifier.
    :type _repo: Repository
    """
    def __init__(self, github_client: Github, repo_name: str |
                                                                       int):
        self._g = github_client
        self._repo_name = repo_name
        self._repo = github_client.get_repo(repo_name)

    async def get_pr_details(self, pr_number: int)-> dict:
        """
           Get PR information: author, title, body, state, commit SHAs.

           IMPORTANT: The 'body' field contains PR description (unreliable for file names).
           To get actual changed files, you MUST call pr_commit_detail with each SHA from head_sha.

           Returns:
               dict with author, title, body, diff_url, state, head_sha (list of commit SHAs)
        """
        
        pr = self._repo.get_pull(pr_number)
        info_dict = {
            "user": pr.user,
            "title": pr.title,
            "body": pr.body,
            "diff_url": pr.diff_url,
            "state": pr.state,
            "commit_SHAs": [c.sha for c in pr.get_commits()],
        }
        return info_dict
    
    
    def get_pr_commit_detail(self, head_sha: str) -> list[dict[str, Any]]:
        """
        Given the commit SHA, this function can retrieve information about
        the commit, such as the files that changed, and return that information.
        """
        commit = self._repo.get_commit(head_sha)
        changed_files: list[dict[str, Any]] = []
        for f in commit.files:
            changed_files.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            })
        return changed_files if changed_files else None
    
    
    def get_file_content(self, file_path: str, ref: str) -> str | None:
        """
        Get file content from a given ref
        """
        try:
            content = self._repo.get_contents(file_path, ref=ref)
            if isinstance(content, list):
                return None
            
            if content.type != "file":
                return None
            
            return content.decoded_content.decode("utf-8")
        except Exception:
            return None
    
    async def add_gathered_context_to_state(self, ctx: Context, gathered_context: str) \
            -> None:
        """
        Save gathered context to state, to give a chance to other agents to use it.
        """
        current_state = await ctx.store.get("state") # type: ignore
        current_state["gathered_context"] = gathered_context
        await ctx.store.set("state", current_state) # type: ignore
    
    async def add_draft_comment(self, ctx: Context, draft_comment: str) \
            -> None:
        """
        Save context to state to give a chance to other agents to use it.
        """
        current_state = await ctx.store.get("state") # type: ignore
        current_state["draft_comment"] = draft_comment
        await ctx.store.set("state", current_state) # type: ignore
        
    async def add_final_review_to_state(self, ctx: Context,
                                        final_review_comment: str) -> None:
        """
        Save your final review to the state.
        """
        current_state = await ctx.store.get("state") # type: ignore
        current_state["final_review_comment"] = final_review_comment
        await ctx.store.set("state", current_state) # type: ignore
        
    async def post_final_review_to_github_pr(self, ctx: Context, pr_number: int) \
            -> None:
        """
        Post your final review to the PR comment on Github.
        """
        current_state = await ctx.store.get("state") # type: ignore
        self._g.get_repo(self._repo_name).get_pull(
            pr_number).create_review(body=current_state.get(
            "final_review_comment")) # type: ignore
        
    def to_function_tools(self, method_names: list[str] | None = None)-> list[FunctionTool]:
        tools = []
        for name in method_names:
            method = getattr(self, name, None)
            if method and callable(method):
                tools.append(FunctionTool.from_defaults(method))
                
        return tools