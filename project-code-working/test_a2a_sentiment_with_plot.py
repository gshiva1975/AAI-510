
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



# ------------------- Metrics Calculation ------------------- #
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
def evaluate_metrics(test_samples, agent):
    y_true = []
    y_pred = []
    y_scores = []

    label_map = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    reverse_map = {v: k for k, v in label_map.items()}

    for sample in test_samples:
        result = asyncio.run(agent.analyze_prompt(sample["prompt"]))
        y_true.append(sample["true_sentiment"])

        # Extract sentiment label without score
        label_with_score = result["sentiment"]  # e.g., "NEGATIVE (0.967)"
        label = label_with_score.split()[0]
        y_pred.append(label)

        # Simulated one-hot probability for ROC-AUC (if your model doesn’t return real probs)
        probs = [0.0, 0.0, 0.0]
        idx = label_map[label]
        probs[idx] = 1.0
        y_scores.append(probs)

    print("\\n--- Sentiment Classification Metrics ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall:", recall_score(y_true, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_true, y_pred, average="weighted"))


    from sklearn.exceptions import UndefinedMetricWarning
    import warnings

    if len(set(y_true)) < 2:
        print("ROC-AUC Score: Skipped (only one class in y_true)")
    else:
        y_true_bin = label_binarize(y_true, classes=["NEGATIVE", "NEUTRAL", "POSITIVE"])
        roc_auc = roc_auc_score(y_true_bin, y_scores, average="macro", multi_class="ovr")
        print("ROC-AUC Score:", roc_auc)


def evaluate_metrics1(test_samples, agent):
    y_true = []
    y_pred = []
    y_scores = []

    label_map = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    reverse_map = {v: k for k, v in label_map.items()}

    for sample in test_samples:
        result = asyncio.run(agent.analyze_prompt(sample["prompt"]))
        y_true.append(sample["true_sentiment"])
        y_pred.append(result["sentiment"])

        # For ROC-AUC: fake probabilities (as an example), adjust if real model gives scores
        probs = [0.0, 0.0, 0.0]
        idx = label_map[result["sentiment"]]
        probs[idx] = 1.0
        y_scores.append(probs)

    print("\n--- Sentiment Classification Metrics ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall:", recall_score(y_true, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_true, y_pred, average="weighted"))

    # Binarize true labels for ROC-AUC
    y_true_bin = label_binarize(y_true, classes=["NEGATIVE", "NEUTRAL", "POSITIVE"])
    roc_auc = roc_auc_score(y_true_bin, y_scores, average="macro", multi_class="ovr")
    print("ROC-AUC Score:", roc_auc)

def evaluate_routing_accuracy(test_samples, router, tools):
    correct = 0
    total = len(test_samples)
    for sample in test_samples:
        predicted_tool, _, _ = router.route(sample["prompt"], tools)
        if predicted_tool == sample["expected_tool"]:
            correct += 1
    print("\n--- Routing Accuracy ---")
    print(f"Routing Accuracy: {correct}/{total} = {correct/total:.2f}")


if __name__ == "__main__":
    test_samples = [
        {
            "prompt": "My iPhone battery dies too fast.",
            "expected_tool": "iphone_sentiment",
            "true_sentiment": "NEGATIVE"
        },
        {
            "prompt": "People are tweeting that the new iOS is great.",
            "expected_tool": "twitter_sentiment",
            "true_sentiment": "POSITIVE"
        }
    ]

    from a2a.routing import ZeroShotRouter
    from a2a_iphone_sentiment_agent import TwitterAgent as IPhoneAgent
    from a2a_twitter_sentiment_agent import TwitterAgent as TwitterAgentAgent

    router = ZeroShotRouter()
    tools = {
        "iphone_sentiment": "iPhone-related issues or praise",
        "twitter_sentiment": "Twitter-related experiences or comments"
    }

    iphone_agent = IPhoneAgent()
    twitter_agent = TwitterAgentAgent()

    # Routing accuracy
    evaluate_routing_accuracy(test_samples, router, tools)

    # Sentiment metrics
    print("\n--- Evaluating iPhone Sentiment Agent ---")
    evaluate_metrics([s for s in test_samples if s["expected_tool"] == "iphone_sentiment"], iphone_agent)

    print("\n--- Evaluating Twitter Sentiment Agent ---")
    evaluate_metrics([s for s in test_samples if s["expected_tool"] == "twitter_sentiment"], twitter_agent)
