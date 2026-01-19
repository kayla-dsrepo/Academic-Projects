import streamlit as st
import pandas as pd
import string
import os
import io

# ====================================================================
# core classes (logic layer)
# ====================================================================

# simple list of stop words to filter out noise
STOP_WORDS = [
    'i', 'me', 'my', 'you', 'your', 'he', 'she', 'it', 'we', 'they', 
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'to', 'of', 'and', 
    'or', 'but', 'a', 'an', 'the', 'in', 'on', 'at', 'for', 'with', 
    'from', 'about', 'just', 'can', 'will', 'need', 'want', 'have', 'do'
]

class FidelityServiceDesk:
    """represents a specific fidelity department and its keywords."""

    def __init__(self, name):
        self.name = name.strip()
        self.keywords = [] 

    def add_keyword(self, word):
        # adds a keyword if it is unique
        clean_word = word.lower().strip()
        if clean_word and clean_word not in self.keywords:
            self.keywords.append(clean_word)

    def get_keywords(self):
        return self.keywords

    def score_text(self, text_tokens):
        # counts how many keywords appear in the text tokens
        score = 0
        for keyword in self.keywords:
            if keyword in text_tokens:
                score += 1
        return score

class RouterModel:
    """handles the core classification logic and keyword management."""
    
    FILE_PATH = "fidelity_router_config.txt"

    def __init__(self):
        self.desks = {}
        self._initialize_defaults()

    def _initialize_defaults(self):
        # sets up default desks including the new irrelevant category
        defaults = {
            "Trading": ["buy", "sell", "stock", "trade", "order", "limit", "market"],
            "Retirement": ["ira", "401k", "roth", "rollover", "distribution", "beneficiary"],
            "Service": ["login", "password", "locked", "address", "profile", "check"],
            "Tax": ["1099", "tax", "form", "deduction", "withholding"],
            "Irrelevant": ["weather", "hello", "lunch", "movie", "sports", "joke"]
        }
        for name, kws in defaults.items():
            desk = FidelityServiceDesk(name)
            for kw in kws:
                desk.add_keyword(kw)
            self.desks[name] = desk
        
        # tries to load saved work if it exists
        self.load_model()

    def predict_department(self, raw_text):
        tokens = self._preprocess(raw_text)
        scores = {name: desk.score_text(tokens) for name, desk in self.desks.items()}
        
        # if no keywords match at all, we classify as irrelevant
        if not scores or max(scores.values()) == 0:
            return "Irrelevant", 0
        
        best_dept = max(scores, key=scores.get)
        return best_dept, scores[best_dept]

    def _preprocess(self, text):
        if not isinstance(text, str): return []
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return [w for w in text.split() if w not in STOP_WORDS]

    def modify_keywords(self, desk_name, new_keywords_str):
        if desk_name in self.desks:
            words = [w.strip() for w in new_keywords_str.split(',') if w.strip()]
            for w in words:
                self.desks[desk_name].add_keyword(w)
            self.save_model()
            return True
        return False

    def save_model(self):
        try:
            with open(self.FILE_PATH, 'w') as f:
                for name, desk in self.desks.items():
                    f.write(f"{name}:{','.join(desk.keywords)}\n")
        except:
            pass 

    def load_model(self):
        if not os.path.exists(self.FILE_PATH): return
        try:
            with open(self.FILE_PATH, 'r') as f:
                for line in f:
                    if ":" in line:
                        name, kws = line.strip().split(":", 1)
                        if name in self.desks:
                            for kw in kws.split(','):
                                self.desks[name].add_keyword(kw)
        except:
            pass

class DataFrameProcessor:
    """handles pandas operations for bulk reclassification."""

    def __init__(self, router):
        self.router = router

    def process_dataframe(self, df, threshold):
        # processes the dataframe in memory using the user-defined threshold
        try:
            required_cols = ['customer_statement', 'department_routed', 'confidence_level']
            
            if not all(col in df.columns for col in required_cols):
                return None, "error: uploaded csv is missing required columns."

            final_depts = []
            indicators = []

            for index, row in df.iterrows():
                conf = pd.to_numeric(row['confidence_level'], errors='coerce')
                
                # Compare against the user-selected threshold
                if conf < threshold:
                    new_dept, score = self.router.predict_department(row['customer_statement'])
                    
                    # check if the reclassification is different from the original
                    if new_dept != row['department_routed']:
                        final_depts.append(new_dept)
                        indicators.append("Reclassified (Low Conf)")
                    else:
                        final_depts.append(row['department_routed'])
                        indicators.append("Original (Low Conf - Confirmed)")
                else:
                    final_depts.append(row['department_routed'])
                    indicators.append("Original")

            df['final_classification'] = final_depts
            df['processing_status'] = indicators
            return df, "success"
            
        except Exception as e:
            return None, f"processing error: {str(e)}"

# ====================================================================
# streamlit ui (application layer)
# ====================================================================

def main():
    st.set_page_config(page_title="Intent Re-router", layout="wide")

    if 'router' not in st.session_state:
        st.session_state.router = RouterModel()
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DataFrameProcessor(st.session_state.router)

    st.sidebar.title("App Navigation")
    option = st.sidebar.radio("Choose an option:", 
        ["1. Upload & Reclassify", "2. Modify Keywords", "3. About"])

    if option == "1. Upload & Reclassify":
        st.title("📂 Intent Re-router: Upload & Reclassify")
        st.markdown("Use this tool to fix **low-confidence** classifications from the third-party vendor.")

        st.info("Download the expanded sample file below to test diverse departments and irrelevant text.")
        
        # expanded sample data
        sample_data = {
            'customer_statement': [
                "I need to reset my password immediately", 
                "i want to buy 100 shares of apple", 
                "what is the limit for my 401k contribution",
                "where is the tax form 1099 for last year",
                "Can you tell me a joke or the weather?",
                "I want to sell my mutual funds",
                "How do I start a rollover for my old plan?",
                "Do you like pizza or movies?"
            ],
            'department_routed': ["Service", "Service", "Trading", "Tax", "Service", "Retirement", "Trading", "Service"],
            'confidence_level': [0.95, 0.40, 0.88, 0.45, 0.30, 0.92, 0.35, 0.25] 
        }
        sample_df = pd.DataFrame(sample_data)
        st.download_button(
            label="Download Expanded Sample CSV",
            data=sample_df.to_csv(index=False).encode('utf-8'),
            file_name='fidelity_sample_v2.csv',
            mime='text/csv',
        )

        st.markdown("---")
        
        st.subheader("1. Configuration")
        threshold = st.slider(
            "Select Low-Confidence Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.05,
            help="Any record with a confidence level BELOW this number will be re-evaluated."
        )

        st.subheader("2. File Upload")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head(10))

            if st.button("Run Reclassification Logic"):
                with st.spinner('Processing...'):
                    result_df, status = st.session_state.processor.process_dataframe(df, threshold)
                
                if result_df is not None:
                    st.success("Processing Complete!")
                    
                    def highlight_reclassified(row):
                        if "Reclassified" in str(row['processing_status']):
                            return ['background-color: #d4edda'] * len(row)
                        return [''] * len(row)

                    st.dataframe(result_df.style.apply(highlight_reclassified, axis=1))

                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Processed Results",
                        data=csv,
                        file_name='reclassified_results.csv',
                        mime='text/csv',
                    )
                else:
                    st.error(status)

    elif option == "2. Modify Keywords":
        st.title("⚙️ Intent Re-router: Modify Keywords")
        router = st.session_state.router
        desk_names = list(router.desks.keys())
        selected_desk = st.selectbox("Select Department (including Irrelevant):", desk_names)

        if selected_desk:
            current_kws = router.desks[selected_desk].get_keywords()
            st.write(f"**Current Keywords for {selected_desk}:**")
            st.code(", ".join(current_kws))
            new_input = st.text_input("Add new keywords (comma-separated):")
            
            if st.button("Update Keywords"):
                if new_input:
                    router.modify_keywords(selected_desk, new_input)
                    st.success(f"Updated {selected_desk}!")
                    st.rerun()

    elif option == "3. About":
        st.title("About Intent Re-router")
        st.markdown("### What is this app?")
        st.write("""
        The **Intent Re-router** is a specialized tool designed to improve data quality in customer service routing. 
        It acts as a 'second opinion' layer for external classification models. When a primary model outputs a 
        **low-confidence score**, this tool re-evaluates the text. If the text does not match any financial keywords, 
        it is moved to the **Irrelevant** category to keep specialized queues clean.
        """)

        st.markdown("### Key Features")
        st.markdown("- **Irrelevant Filtering:** Automatically identifies non-financial inquiries to prevent them from reaching specialized desks.")
        st.markdown("- **Automatic Reclassification:** Overrides third-party predictions using high-precision business rules.")
        st.markdown("- **Keyword Adjustment:** Allows for real-time updates to the logic.")

        st.markdown("---")
        st.markdown("### Student Project Disclaimer")
        st.info("""
        **Proof of Concept:** This application was developed as a student project to demonstrate technical proficiency in 
        Python programming, Object-Oriented Design, and Streamlit application building.
        
        **Real-World Context:** I acknowledge that in a large financial institution like Fidelity Investments, 
        an application like this would **not** be deployed on Streamlit due to enterprise constraints such as 
        Data Security, Scalability, and Infrastructure requirements.
        """)

if __name__ == "__main__":
    main()