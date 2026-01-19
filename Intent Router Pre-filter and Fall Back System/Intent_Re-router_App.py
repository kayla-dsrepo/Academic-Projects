
### **Final Streamlit Code (`Intent_Router_App.py`)**


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
        # sets up default desks
        defaults = {
            "Trading": ["buy", "sell", "stock", "trade", "order", "limit", "market"],
            "Retirement": ["ira", "401k", "roth", "rollover", "distribution", "beneficiary"],
            "Service": ["login", "password", "locked", "address", "profile", "check"],
            "Tax": ["1099", "tax", "form", "deduction", "withholding"]
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
        
        if not scores or max(scores.values()) == 0:
            return "Uncertain", 0
        
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
        self.confidence_threshold = 0.60 

    def process_dataframe(self, df):
        # processes the dataframe in memory
        try:
            required_cols = ['customer_statement', 'department_routed', 'confidence_level']
            
            if not all(col in df.columns for col in required_cols):
                return None, "error: uploaded csv is missing required columns."

            final_depts = []
            indicators = []

            for index, row in df.iterrows():
                conf = pd.to_numeric(row['confidence_level'], errors='coerce')
                
                if conf < self.confidence_threshold:
                    new_dept, score = self.router.predict_department(row['customer_statement'])
                    
                    if new_dept != "Uncertain":
                        final_depts.append(new_dept)
                        indicators.append("Reclassified (Low Conf)")
                    else:
                        final_depts.append(row['department_routed'])
                        indicators.append("Original (Low Conf - No Rule Match)")
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

    # initialize router in session state so it remembers changes
    if 'router' not in st.session_state:
        st.session_state.router = RouterModel()
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DataFrameProcessor(st.session_state.router)

    # sidebar menu
    st.sidebar.title("App Navigation")
    
    # Updated Menu Structure
    option = st.sidebar.radio("Choose an option:", 
        ["1. Upload & Reclassify", "2. Modify Keywords", "3. About"])

    # --- option 1: upload & reclassify ---
    if option == "1. Upload & Reclassify":
        st.title("📂 Intent Re-router: Upload & Reclassify")
        st.markdown("Use this tool to fix **low-confidence** classifications from the third-party vendor.")

        # helper to generate sample csv for the user
        st.info("Don't have a file? Download a sample below.")
        sample_data = {
            'customer_statement': [
                "I need to reset my password immediately", 
                "i want to buy 100 shares of apple", 
                "what is the limit for my 401k contribution",
                "where is the tax form 1099 for last year"
            ],
            'department_routed': ["Service", "Service", "Service", "Service"],
            'confidence_level': [0.95, 0.40, 0.35, 0.45] 
        }
        sample_df = pd.DataFrame(sample_data)
        st.download_button(
            label="Download Sample CSV",
            data=sample_df.to_csv(index=False).encode('utf-8'),
            file_name='third_party_sample.csv',
            mime='text/csv',
        )

        # file uploader
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            # read file
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            if st.button("Run Reclassification Logic"):
                with st.spinner('Processing...'):
                    result_df, status = st.session_state.processor.process_dataframe(df)
                
                if result_df is not None:
                    st.success("Processing Complete!")
                    
                    # highlight reclassified rows
                    def highlight_reclassified(row):
                        if "Reclassified" in str(row['processing_status']):
                            return ['background-color: #d4edda'] * len(row)
                        return [''] * len(row)

                    st.dataframe(result_df.style.apply(highlight_reclassified, axis=1))

                    st.warning("If the new classifications look wrong, go to **Option 2** to update the keywords.")

                    # download button for results
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Processed Results",
                        data=csv,
                        file_name='fidelity_reclassified_results.csv',
                        mime='text/csv',
                    )
                else:
                    st.error(status)

    # --- option 2: modify keywords ---
    elif option == "2. Modify Keywords":
        st.title("⚙️ Intent Re-router: Modify Keywords")
        st.markdown("Update the logic used to fix low-confidence tickets.")

        router = st.session_state.router
        
        # select desk
        desk_names = list(router.desks.keys())
        selected_desk = st.selectbox("Select Department:", desk_names)

        if selected_desk:
            current_kws = router.desks[selected_desk].get_keywords()
            st.write(f"**Current Keywords for {selected_desk}:**")
            st.code(", ".join(current_kws))

            # input for new keywords
            new_input = st.text_input("Add new keywords (comma-separated):")
            
            if st.button("Update Keywords"):
                if new_input:
                    router.modify_keywords(selected_desk, new_input)
                    st.success(f"Updated {selected_desk}! New keywords saved.")
                    st.rerun() # refreshes page to show new list
                else:
                    st.error("Please enter at least one keyword.")

    # --- option 3: about (updated - no emojis) ---
    elif option == "3. About":
        st.title("About Intent Re-router")
        
        st.markdown("### What is this app?")
        st.write("""
        The **Intent Re-router** is a specialized tool designed to improve data quality in customer service routing. 
        It acts as a 'second opinion' layer for external classification models. When a primary model outputs a 
        **low-confidence score** for a customer inquiry, this tool intervenes by using a controllable, 
        keyword-based rule set to reclassify the intent and route it to the correct department (e.g., Trading, Retirement, Tax).
        """)

        st.markdown("### Key Features")
        st.markdown("- **Automatic Reclassification:** Detects low-confidence predictions and overrides them based on specific business rules.")
        st.markdown("- **Keyword Adjustment:** Allows supervisors to manually add keywords to departments, instantly adapting the model to new trends.")
        st.markdown("- **Transparency:** Provides clear feedback on which records were changed and why.")

        st.markdown("---")
        st.markdown("### Student Project Disclaimer")
        st.info("""
        **Proof of Concept:** This application was developed as a student project to demonstrate technical proficiency in 
        Python programming, Object-Oriented Design, and Streamlit application building.
        
        **Real-World Context:** We acknowledge that in a large financial institution like Fidelity Investments, 
        an application like this would **not** be deployed on Streamlit due to enterprise constraints such as:
        * **Data Security & Privacy:** Strict protocols for handling PII (Personally Identifiable Information).
        * **Scalability:** The need to process millions of transactions/calls in real-time.
        * **Infrastructure:** Integration with internal secure cloud environments rather than public web frameworks.
        """)

if __name__ == "__main__":
    main()

