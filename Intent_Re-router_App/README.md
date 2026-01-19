# Intent Re-router: Financial Logic Recovery Tool


### Project Overview

The **Intent Re-router** is a proof-of-concept Python application designed to enhance the accuracy of customer inquiry routing within a financial services context. In large institutions, third-party classification models sometimes struggle with specific financial jargon or nuances, leading to low-confidence predictions.

I developed this tool to act as a logic-based "second opinion" layer. It identifies low-confidence outputs and applies high-precision, adjustable keyword rules to reclassify the intent or filter out irrelevant queries.

Try the [Intent Re-router App](https://intentrerouterapppy-qtkwwybbteg6yb9foqkiwg.streamlit.app/)

### Key Features

* **Dynamic Reclassification:** Automatically intercepts data with low confidence scores and reapplies logic-based routing.
* **Irrelevant Intent Filtering:** Protects specialized service desks by identifying and isolating non-financial or "noise" inquiries.
* **User-Defined Thresholds:** Allows users to interactively set what level of confidence requires intervention.
* **Keyword Management:** A dedicated interface for supervisors to update business rules in real-time without touching the underlying code.

### Technical Stack

* **Language:** Python 3.x
* **Interface:** Streamlit (Web UI)
* **Data Handling:** Pandas (Dataframe processing)
* **Design Pattern:** Object-Oriented Programming (OOP)

### Installation & Local Usage

To run this project locally, follow these steps:

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Launch the application:**
```bash
streamlit run Intent_Router_App.py

```



### Project Context & Disclaimer

This is an individual student project created to demonstrate skills in application building and financial logic implementation.

**Note on Enterprise Deployment:** I recognize that in a production environment (such as at Fidelity Investments), an application of this nature would require enterprise-grade security (PII masking), high-scale data processing (Spark/Distributed computing), and integration into internal secure cloud infrastructures rather than a public Streamlit deployment.
