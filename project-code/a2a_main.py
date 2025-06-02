
from a2a.agent import Agent
from a2a.message import Message
from a2a.schema import ToolCall, ToolResult
from transformers import pipeline
import asyncio

TOOLS = {
    "iphone_sentiment": "iPhone-related issues or praise",
    "twitter_sentiment": "Twitter-related experiences or comments"
}

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

class Coordinator(Agent):
    def __init__(self):
        super().__init__("coordinator_agent")

    async def onInit(self):
        print("[Init] Tools registered:", TOOLS, flush=True)

    async def onMessage(self, msg: Message):
        prompt = msg.content
        result = classifier(prompt, list(TOOLS.values()))
        label = result["labels"][0]
        score = result["scores"][0]
        tool = next(k for k, v in TOOLS.items() if v == label)

        print(f"[Routing] '{prompt}' → {tool} (label='{label}', score={score:.2f})", flush=True)

        try:
            tool_call = ToolCall(tool=tool, name="analyze_prompt", arguments={"text": prompt})
            output: ToolResult = await self.call_tool(tool_call)
            sentiment = output.output.get("sentiment")
            text = output.output.get("text")
            print(f"[✓ {tool}] {sentiment}: {text}", flush=True)
            await self.send(msg.respond(f"[✓ {tool}] {sentiment}: {text}"))
        except Exception as e:
            print(f"[Coordinator ERROR] Could not reach tool '{tool}': {e}", flush=True)
            await self.send(msg.respond(f"[✗ {tool}] ERROR: {e}"))

if __name__ == "__main__":
    asyncio.run(Coordinator().run())
