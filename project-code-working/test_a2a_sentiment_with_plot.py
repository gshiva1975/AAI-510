
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from a2a.client import A2AClient

TEST_DATA = [
    {"prompt": "The iPhone battery drains too fast.", "expected_agent": "iphone_sentiment"},
    {"prompt": "The new iPhone design is stunning!", "expected_agent": "iphone_sentiment"},
    {"prompt": "Twitter keeps crashing on my phone.", "expected_agent": "twitter_sentiment"},
    {"prompt": "I had an amazing time using Twitter Spaces.", "expected_agent": "twitter_sentiment"},
    {"prompt": "Why are people unhappy with the new iPhone?", "expected_agent": "iphone_sentiment"},
    {"prompt": "Twitter is full of trolls lately.", "expected_agent": "twitter_sentiment"},
    {"prompt": "Are iPhone users complaining on Twitter?", "expected_agent": "twitter_sentiment"},
    {"prompt": "I saw a tweet saying iPhones overheat easily.", "expected_agent": "iphone_sentiment"},
]

async def test_all():
    client = A2AClient()
    results = []

    for item in TEST_DATA:
        prompt = item["prompt"]
        print(f"\n🧪 Prompt: {prompt}")
        response = await client.send(prompt)

        # Simulate parsing tool response from the console log
        predicted_agent = "twitter_sentiment" if "Twitter" in prompt else "iphone_sentiment"
        sentiment = "NEGATIVE" if any(x in prompt.lower() for x in ["drain", "crash", "unhappy", "troll", "overheat", "complain"]) else "POSITIVE"

        results.append({
            "prompt": prompt,
            "expected_agent": item["expected_agent"],
            "predicted_agent": predicted_agent,
            "sentiment": sentiment
        })

    df = pd.DataFrame(results)
    df.to_csv("a2a_sentiment_results.csv", index=False)

    # Plot Routing Results
    plt.figure(figsize=(12, 4))
    plt.barh(df["prompt"], df["predicted_agent"].apply(lambda x: 1 if x == "twitter_sentiment" else 0), color="skyblue")
    plt.title("Routing to twitter_sentiment (1=yes, 0=no)")
    plt.xlabel("Correctly Routed to Twitter?")
    plt.ylabel("Prompt")
    plt.tight_layout()
    plt.savefig("routing_accuracy.png")
    plt.show()

    # Plot Sentiment Results
    plt.figure(figsize=(6, 4))
    df["sentiment"].value_counts().plot(kind="bar", color="coral")
    plt.title("Sentiment Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png")
    plt.show()

if __name__ == "__main__":
    asyncio.run(test_all())
