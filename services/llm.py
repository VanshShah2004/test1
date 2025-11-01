from __future__ import annotations

import os
from typing import List
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage

try:
	from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
	ChatGoogleGenerativeAI = None


class LLMService:
	def __init__(self, model: str = "gemini-2.0-flash-exp", temperature: float = 0.2):
		# Ensure environment loaded once
		load_dotenv()
		api_key = os.environ.get("GEMINI_API_KEY")
		if not api_key:
			raise ValueError("GEMINI_API_KEY not set")
		if ChatGoogleGenerativeAI is None:
			raise RuntimeError("langchain_google_genai not available")
		self._llm = ChatGoogleGenerativeAI(
			model=model, temperature=temperature, google_api_key=api_key
		)

	def generate(self, system_prompt: str, human_prompt: str) -> str:
		messages: List = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
		resp = self._llm.invoke(messages)
		return resp.content or ""


