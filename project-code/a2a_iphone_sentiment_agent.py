
from a2a.agent import Agent
from a2a.schema import ToolDefinition
from transformers import pipeline
import pandas as pd
from sentence_transformers import SentenceTransformer, util

class IPhoneAgent(Agent):
    def __init__(self):
        super().__init__("iphone_sentiment")
        self.sentiment_model = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        try:
            corpus = pd.read_csv("iphone.csv")["reviewDescription"].dropna().astype(str).tolist()[:50]
            self.corpus = corpus
            print(f"[{self.name}] Loaded corpus with {len(self.corpus)} entries", flush=True)
        except Exception as e:
            print(f"[{self.name}] Failed to load corpus: {e}", flush=True)

        self.embeddings = self.embedder.encode(corpus, convert_to_tensor=True)

    async def onInit(self):
        return [ToolDefinition(name="analyze_prompt", parameters={"text": {"type": "string"}})]

    async def analyze_prompt(self, text: str):
        sentiment_raw = self.sentiment_model(text[:512])[0]
        label = sentiment_raw["label"]
        if label == "LABEL_0":
            label = "NEGATIVE"
        elif label == "LABEL_1":
            label = "NEUTRAL"
        elif label == "LABEL_2":
            label = "POSITIVE"
        query_emb = self.embedder.encode(text, convert_to_tensor=True)
        match = util.pytorch_cos_sim(query_emb, self.embeddings)[0]
        top = match.argmax().item()
        return {"text": text, "sentiment": label, "similar_to": self.corpus[top]}

if __name__ == "__main__":
    import asyncio
    asyncio.run(IPhoneAgent().run())
