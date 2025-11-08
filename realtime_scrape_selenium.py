print("=== REALTIME SELENIUM SCRIPT LOADED ===")

import os; print("RUNNING FILE:", os.path.abspath(__file__))

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

from src.content_model import ContentModel
from src.behavior_model import BehaviorModel
from src.pipeline import FusionPipeline

def scrape_tweet(url):
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(5)  # allow dynamic JS to load

    # tweet text
    text_elem = driver.find_element(By.CSS_SELECTOR, "div[data-testid='tweetText']")
    text = text_elem.text

    # metrics :: likes + retweets
    metrics = driver.find_elements(By.CSS_SELECTOR, "div[data-testid='app-text-transition-container']")
    likes = 0
    retweets = 0
    if len(metrics) >= 2:
        retweets = int(metrics[0].text.replace(",", "") or "0")
        likes = int(metrics[1].text.replace(",", "") or "0")

    driver.quit()

    # dummy values for behaviour
    followers = 50
    following = 25
    tweets_count = 200
    account_age_days = 365
    posts_per_day = tweets_count / account_age_days

    return {
        "post_id": "0",
        "user_id": "0",
        "text": text,
        "followers": followers,
        "following": following,
        "tweets_count": tweets_count,
        "account_age_days": account_age_days,
        "retweets": retweets,
        "likes": likes,
        "posts_per_day": posts_per_day
    }

if __name__ == "__main__":
    url = input("Enter tweet URL: ")
    data = scrape_tweet(url)
    df = pd.DataFrame([data])

    cm = ContentModel.load("content_model.joblib")
    bm = BehaviorModel.load("behavior_model.joblib")
    fusion = FusionPipeline(cm, bm)

    result = fusion.score(df)

    content_prob = float(result["content_prob"][0])
    behavior_score = float(result["behavior_score"][0])
    fused = float(result["fused_score"][0])
    verdict = int(result["verdict"][0])

    print("\n=== FINAL PREDICTION REPORT ===\n")

    print(f"Content manipulation likelihood: {content_prob*100:.2f}%")
    print(f"Bot / abnormal behaviour likelihood: {behavior_score*100:.2f}%")
    print(f"Overall cyber-threat score: {fused*100:.2f}%\n")

    if verdict == 1:
        print("VERDICT: ⚠️ This post is likely part of a misinformation / social engineering threat.")
    else:
        print("VERDICT: ✅ This post looks normal / organic.")
