# On terminal 1

Step 1: Create the virtual environment

python3 -m venv a2a-venv

Step 2: Activate the environment
On macOS/Linux:

source a2a-venv/bin/activate

pip install --upgrade pip 
pip install transformers sentence-transformers pandas matplotlib seaborn 
pip install -e . python 
./a2a_iphone_sentiment_agent.py

# On terminal 2

Step 2: Activate the environment
On macOS/Linux:
source a2a-venv/bin/activate 
python ./a2a_twitter_sentiment_agent.py

# On terminal 3

Step 2: Activate the environment
On macOS/Linux:
source a2a-venv/bin/activate 
python ./a2a_main.py

# On terminal 4

Step 2: Activate the environment
On macOS/Linux:
source a2a-venv/bin/activate 
python ./test_a2a_sentiment_with_plot.py
