# On terminal 1

**Step 1: Create the virtual environment**

python3 -m venv a2a-venv

**Step 2: Activate the environment**
On macOS/Linux:

source a2a-venv/bin/activate

pip install --upgrade pip 

pip install transformers sentence-transformers pandas matplotlib seaborn 

pip install -e . python 

**Step 3: Activate the environment**

python ./test_a2a_sentiment_with_plot.py
