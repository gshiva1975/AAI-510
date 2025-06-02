
import asyncio
from a2a.client import A2AClient

TEST_DATA = [
    {"prompt": "The iPhone battery drains too fast."},
    {"prompt": "The new iPhone design is stunning!"},
    {"prompt": "Twitter keeps crashing on my phone."},
    {"prompt": "I had an amazing time using Twitter Spaces."},
    {"prompt": "Why are people unhappy with the new iPhone?"},
    {"prompt": "Twitter is full of trolls lately."},
    {"prompt": "Are iPhone users complaining on Twitter?"},
    {"prompt": "I saw a tweet saying iPhones overheat easily."}
]

async def test_all():
    client = A2AClient()
    for item in TEST_DATA:
        prompt = item["prompt"]
        print(f"\n🧪 Prompt: {prompt}")
        response = await client.send(prompt) 
        print(f"➡️  {response.content}")

if __name__ == "__main__":
    asyncio.run(test_all())
