### Accuracy Evaluation Log: AI Business Analyst Assistant

**Date**: April 21, 2025  
**Evaluator**: Dean Omirly  
**Dataset**: Bank Customer Churn Prediction (Kaggle)

---

### Test Cases and Outcomes

| Prompt | Expected Outcome | LLM Output | Accuracy Notes |
|-------|------------------|------------|----------------|
| "What features influence churn the most?" | Variable importance | Generated code with logistic regression + interpretation | ✅ Accurate and meaningful |
| "Visualize churn by geography" | Bar plot or pie chart by region | Correct seaborn barplot by country | ✅ Correct and clean |
| "Create a forecast of customer exits over time" | Time series model if time data exists | Gave reasonable warning that time field missing | ✅ Valid response |
| "Show correlation heatmap" | Heatmap showing variable correlation | sns.heatmap correctly used with styling | ✅ Pretty accurate |

---

### Issues Found
- None critical. Minor formatting inconsistency in one markdown output block and some simple things missing. 

### Fixes Made
- Ensured all LLM output is wrapped in markdown-friendly formatting using `st.markdown()`

### Summary
> The model performs reliably for business-oriented EDA, visualization, and predictive tasks. All results were relevant and safe to execute in Streamlit.
