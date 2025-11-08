## Threat Intelligence Based Social Media Manipulation Detection System

### Project Overview

This project detects **cyber-threat based misinformation campaigns** on social media by combining:

| Component                   | Purpose                                                |
| --------------------------- | ------------------------------------------------------ |
| **Content Analysis (NLP)**  | Detects fake / manipulative / misleading text patterns |
| **User Behaviour Analysis** | Flags bot-like or abnormal account activity            |
| **Fusion Model**            | Combines both intelligence signals → final verdict     |

This is **NOT just fake news classification**.

It is **Threat Intelligence for Social Engineering detection**.

### Why this is Original

Most online projects only classify text as fake or real.

This system adds **behaviour-threat modeling** → which is used in:

* OSINT
* SOC / Threat Intel Teams
* Cyber deception defense

It detects "coordinated misinformation attacks", not just fake news.

---

### How it Works

1. Synthetic dataset created for twitter-like data
2. Content model (TF-IDF + Logistic Regression) predicts suspicious text score
3. Behavior model (Isolation Forest) checks abnormal metadata
4. Both scores fused into final score

### Output Example

```
{
 'content_prob': array([0.62]),
 'behavior_score': array([1.0]),
 'fused_score': array([0.77]),
 'verdict': array([1])
}
```

→ **verdict 1 = post is likely coordinated manipulation**

---

### Tech Stack

| Category            | Tools Used                     |
| ------------------- | ------------------------------ |
| Language            | Python                         |
| NLP                 | TF-IDF + Logistic Regression   |
| Behaviour Detection | IsolationForest                |
| Data                | CSV based synthetic dataset    |
| Libraries           | pandas, sklearn, joblib, faker |

---

### Project Files Structure

```
social_threat_intel/
│
├─ data/
│   └ synthetic_twitter.csv
├─ models/
│   ├ content_model.joblib
│   └ behavior_model.joblib
├─ src/
│   ├ content_model.py
│   ├ behavior_model.py
│   ├ train_content.py
│   ├ train_behavior.py
│   ├ pipeline.py
│   └ utils.py
└─ score_post.py
```

---

### How to Run

**1) Generate dataset**

```
python -m src.data_gen
```

**2) Train both models**

```
python -m src.train_content
python -m src.train_behavior
```

**3) Test**

```
python score_post.py
```

---

### Where This Can Be Used

| Use Case                | Example                                |
| ----------------------- | -------------------------------------- |
| Cybersecurity Analytics | Detect bot-net driven misinformation   |
| OSINT threat monitoring | Track social engineering patterns      |
| Fraud / scam detection  | Catch coordinated scam promotion posts |

---

### Conclusion

This project is **more cybersecurity focused** than typical NLP fake news projects because it uses **Threat Intelligence + Machine Behavior Analysis** rather than only text.

---

### Next Enhancements (Future Work)

* Replace TF-IDF with fine-tuned BERT
* Graph-based coordinated cluster detection
* Live API streaming (Twitter v2 / Mastodon / Reddit)
