from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from .cv_tool import CVGeneratorTool

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


system_instruction = (
    "You are the a CV Generator Agent. Your goal is to maximize the user's ATS score. " +
    "Use the 'generate_optimized_cv' tool when the user provides a linkedin job offer link " +
    "Use the 'update_profile_with_new_experience' tool when the user provides new work experience or skills. " +
    "If the CV generation is successful, check the Gap Alert in the output. If a gap exists, " +
    "ask the user for relevant experience to address it, and then use the update tool."
)

root_agent = Agent(
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    name='root_agent',
    description='A helpful assistant for CV Generation.',
    instruction=system_instruction,
    tools=[], 
)


# 3. Initialize the Tool and pass the Agent instance to it! (THIS IS THE KEY FIX)
cv_generator_tool = CVGeneratorTool(agent=root_agent)

# 4. Update the Agent with the correctly initialized tools
root_agent.tools = [
    cv_generator_tool.generate_optimized_cv, 
    # Reference the update method as a separate callable tool
    cv_generator_tool.update_profile_with_new_experience
]