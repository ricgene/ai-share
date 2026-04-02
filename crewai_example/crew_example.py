"""
Simple two-agent CrewAI example: Researcher + Writer collaborating to produce a report.

Uses Groq if GROQ_API_KEY is set in .env (fast cloud inference).
Otherwise uses Ollama (FREE local inference — no API key, no internet required after setup).
See docs/groq-key-instructions.txt for Groq setup.
See docs/ollama-how-to.txt for Ollama setup.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

load_dotenv(dotenv_path='../.env')

# Model selection: Groq if API key available, else Ollama
if os.getenv("GROQ_API_KEY"):
    MODEL = "groq/llama3-70b-8192"  # Groq model (fast, cloud-based)
else:
    MODEL = "ollama/llama3.2:1b"   # Ollama model (local, free)


# --- Agents ---

researcher = Agent(
    role="Research Analyst",
    goal="Gather accurate, well-organized information on the given topic",
    backstory=(
        "You are a meticulous research analyst who excels at finding key facts, "
        "statistics, and insights on any subject. You present information in a "
        "structured, clear manner that writers can easily work with."
    ),
    llm=MODEL,
    verbose=True,
)

writer = Agent(
    role="Content Writer",
    goal="Transform research into a clear, engaging, well-structured report",
    backstory=(
        "You are an experienced writer who turns raw research into polished, "
        "reader-friendly content. You maintain accuracy while making complex "
        "topics accessible to a general audience."
    ),
    llm=MODEL,
    verbose=True,
)


# --- Tasks ---

research_task = Task(
    description=(
        "Research the topic: '{topic}'. "
        "Identify the 3-5 most important facts, current trends, and key challenges. "
        "Organize your findings with clear headings and bullet points."
    ),
    expected_output=(
        "A structured research brief with: key facts, current trends, "
        "main challenges, and any notable statistics."
    ),
    agent=researcher,
)

writing_task = Task(
    description=(
        "Using the research brief provided, write a concise 3-paragraph report "
        "on the topic. Include: an introduction, key findings, and a conclusion "
        "with implications. Keep it under 300 words."
    ),
    expected_output=(
        "A polished 3-paragraph report (under 300 words) suitable for "
        "a general audience, based on the research brief."
    ),
    agent=writer,
    context=[research_task],  # writer receives researcher's output
)


# --- Crew ---

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # researcher finishes before writer starts
    verbose=True,
)


if __name__ == "__main__":
    topic = "the impact of large language models on software development"

    print(f"\n{'='*60}")
    print(f"Topic: {topic}")
    print(f"Model: {MODEL}")
    print(f"{'='*60}\n")

    result = crew.kickoff(inputs={"topic": topic})

    print(f"\n{'='*60}")
    print("FINAL REPORT:")
    print(f"{'='*60}")
    print(result)
