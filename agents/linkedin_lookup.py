from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, AgentType, Tool

from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")
    template = """given the fullname {name_of_person} I want you to get me a link to their Linkedin profile page. Your answer should contain ONLY a url"""

    tools_for_agent = [
        Tool(
            name="Crawl Google for linkedin profile page",
            func=get_profile_url,
            description="Useful for when you need to get the Linkedin page url",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))
    return linkedin_profile_url
