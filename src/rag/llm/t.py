import asyncio
from src.rag.search.agent.agent_node import _get_bound_llm
from langchain_core.messages import HumanMessage, SystemMessage

async def t():
  llm = _get_bound_llm()
  r = await llm.ainvoke([SystemMessage(content='Echo what user says.'), HumanMessage(content='hello')])
  print('response:', r.content[:200])
  print('tool_calls:', r.tool_calls)

asyncio.run(t())