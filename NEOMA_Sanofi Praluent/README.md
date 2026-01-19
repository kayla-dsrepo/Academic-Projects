# Sanofi Praluent: Medicaid Drug Utilization Analysis

## Link to the dashboard for downloads
[Sanofi Praluent Drug Utilization Analysis.pbix] (https://neoma-my.sharepoint.com/:u:/g/personal/yongqing_wang_25_neoma-bs_com/IQDDCvlau2PlQIgU2X7WhJGgAX4bH7bViRQncTmUqKezv0s?e=YelBo1)


## Project Overview
This project was developed as the individual core assignment for the graduate course "Machine Learning and Deep Learning." 

The primary objective was to practice an end-to-end data analysis pipeline: learning how to identify and source data based on a self-selected theme and a real-world business problem, perform data cleaning and processing, and ultimately utilize Power BI for data visualization and business intelligence analytics.

Using drug utilization analysis of Praluent versus Repatha within the U.S. Medicaid system as a case study, the project explores data-driven market insights.

## Professional Context & Business Problem
The business context for this analysis stems from Sanofi seeking to advance health equity for its product Praluent among low-income (Medicaid) populations in the United States. 

Both Praluent and its primary competitor Repatha are PCSK9 inhibitors, a class of high-cost injectable biologics prescribed to reduce the risk of myocardial infarction and stroke. The analysis aims to uncover systemic barriers to drug access—such as cost-control mechanisms like prior authorization and step-therapy—which may prevent vulnerable populations from receiving optimal preventative care.

## Data Source & Processing
### Dataset and Source
* **Primary Data:** State Drug Utilization Data (SDUD), sourced from Data.gov, which records all outpatient drugs paid for by state Medicaid agencies.
* **Supplementary Data:** The FDA's National Drug Code (NDC) Directory, used to obtain detailed information on dosage form, strength, and delivery device.
* **Analysis Period:** Eight consecutive quarters from Q1 2023 through Q4 2024.
* **Key Variables:** Number of prescriptions, Medicaid reimbursement amount, state, and NDC code.

### Data Preparation Procedures
Data preparation followed a structured ETL (Extract, Transform, Load) pipeline to ensure analytical integrity:
* **Exploration & Verification:** Confirmed the dataset contained sufficient relevant variables and records.
* **Cleaning:** Addressed missing values, removed duplicate records, and standardized textual data in categorical fields.
* **Transformation & Merging:** Consolidated quarterly SDUD files into a unified dataset, which was joined with the FDA NDC Directory to accurately distinguish product and device types.
* **Feature Engineering:** Created new calculated metrics, such as average reimbursement per prescription and period-over-period growth rates.

## Analytics Tools & Techniques
* **Power Query:** Used for all data loading, transformation (e.g., type changes, filtering), and merging operations to build a relational data model.
* **DAX (Data Analysis Expressions):** Employed to create core calculated measures, including total prescriptions, average reimbursement amount, and year-over-year growth percentage.
* **Power BI Visualizations:** Constructed interactive dashboards utilizing various visualizations—line charts, bar charts, treemaps, and filled maps—to answer specific business questions.

## Conclusion and Strategic Recommendation
The analysis reveals that Praluent is the lowest net-cost choice in 44 out of 50 states. This data provides a concrete basis for Sanofi to advocate for the removal of restrictive access barriers. By leveraging this lowest net-cost status, the company can push for equitable access, ensuring that low-income populations are not unfairly denied best-in-class cardiovascular care.