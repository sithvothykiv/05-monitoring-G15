import json

from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from dotenv import load_dotenv

from evaluation_utils import llm_structured_retry


class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str


judge_instructions = """
You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the given question.

You must respond ONLY with a valid JSON object in this exact schema:
{
  "relevance": "RELEVANT",
  "explanation": "Short explanation of the verdict"
}

The "relevance" value must be exactly one of:
- "RELEVANT": the answer addresses the question
- "PARTLY_RELEVANT": the answer partially addresses the question
- "NON_RELEVANT": the answer does not address the question

Do not include markdown code block formatting (e.g. ```json). Return raw JSON only.
""".strip()

judge_prompt = """
Question: {question}
Generated Answer: {answer}
""".strip()


import os

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-max-2026-06-08"


def evaluate_relevance(question, answer, client=None, model=None, api_key=None, base_url=None):
    if client is None:
        load_dotenv()
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)

    prompt = judge_prompt.format(
        question=question,
        answer=answer
    )

    result, usage = llm_structured_retry(
        client,
        judge_instructions,
        prompt,
        RelevanceVerdict,
        model=model
    )

    return result.relevance, result.explanation

if __name__ == "__main__":
    load_dotenv()

    question = "Can I still join the course?"
    answer = "Yes, you can still join. The course is self-paced."

    relevance, explanation = evaluate_relevance(question, answer)
    print(relevance)
    print(explanation)