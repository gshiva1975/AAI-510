
import asyncio

class Agent:
    def __init__(self, name="agent"):
        self.name = name
        print(f"[A2A Agent: {self.name}] Initialized")

    async def onInit(self):
        return []

    async def onMessage(self, msg):
        print(f"[{self.name}] Received message: {msg.content}")
        await self.send(self.respond(msg, "Default response"))

    async def send(self, response):
        print(f"[{self.name}] Sending response: {response.content}")

    async def call_tool(self, tool_call):
        print(f"[{self.name}] Calling tool: {tool_call.tool}")
        return type("ToolResult", (), {"output": {"text": "tool response", "sentiment": "POSITIVE"}})()

    def respond(self, msg, content):
        return type("Response", (), {"content": content})

    async def run(self):
        print(f"[{self.name}] Agent running... (stubbed)")
        while True:
            await asyncio.sleep(3600)
