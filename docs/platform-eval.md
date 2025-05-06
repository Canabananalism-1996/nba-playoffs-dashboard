# Platform Evaluation

## 1. Purpose

This document captures our systematic evaluation of candidate platforms for the “NBA Playoffs Analytics” project. We score each option against weighted criteria aligned to our goals of data ingestion, statistical modeling, interactive visualization, collaboration, and free deployment.

---

## 2. Evaluation Criteria

| Criterion                   | Description                                                                                      | Weight |
|-----------------------------|--------------------------------------------------------------------------------------------------|:------:|
| **Data Connectivity**       | Ease of connecting to REST APIs, CSVs, databases; handling pagination and large volumes          |  20%   |
| **Statistical & ML Support**| Native or seamless integration with Python/R libraries (scikit-learn, Prophet, statsmodels, etc.)|  25%   |
| **Interactive Visualization**| Ability to create dynamic dashboards, custom widgets, drill-downs, and tooltips                 |  20%   |
| **Ease of Use & Learning**  | Balance of GUI vs. code; onboarding ramp-up for new users                                        |  15%   |
| **Collaboration & Versioning** | Git-friendliness, code review workflows, CI/CD support                                        |  10%   |
| **Free Deployment & Sharing** | Availability of truly free hosting or desktop-only trade-offs                                  |  10%   |

---

## 3. Candidate Platforms & Scoring

### 3.1 Scores (0–5 scale)

| Platform        | Data (20%) | ML (25%) | Visuals (20%) | Ease (15%) | Git (10%) | Deploy (10%) | **Total** |
|-----------------|:----------:|:--------:|:-------------:|:----------:|:---------:|:------------:|:---------:|
| **Streamlit**   |     5      |    5     |       4       |     3      |     5     |      5       | **4.45**  |
| **Dash**        |     5      |    5     |       5       |     2      |     5     |      5       | **4.40**  |
| **Power BI**    |     4      |    3     |       5       |     5      |     2     |      3       | **3.90**  |
| **Tableau**     |     4      |    3     |       5       |     4      |     2     |      3       | **3.75**  |
|**Looker Studio**|     3      |    2     |       4       |     5      |     3     |      4       | **3.50**  |

> *Weights applied: Data×0.20, ML×0.25, Visuals×0.20, Ease×0.15, Git×0.10, Deploy×0.10*

---

## 4. Pros & Cons Summary

| Platform      | Pros                                                                                     | Cons                                                      |
|---------------|------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| **Streamlit** | • Full Python control for ETL & modeling<br>• Rich interactivity with widgets & Plotly<br>• Free Community Cloud hosting<br>• Plain-text code (Git-friendly) | • Requires coding; no drag-drop editor                    |
| **Dash**      | • Deep Plotly callback customization<br>• Pure-Python stack                               | • Boilerplate for UI layouts; steeper initial setup       |
| **Power BI**  | • GUI-driven data prep & visuals<br>• Built-in forecasting and “Quick Insights”            | • Desktop-only free tier; binary PBIX limits diffs<br>• Limited Python logic |
| **Tableau**   | • Highly polished dashboards and storyboards                                              | • License required for public sharing; steeper learning curve |
| **Looker Studio** | • Easy Google-ecosystem integration<br>• GUI-first analytics                         | • Limited advanced modeling; reliant on connectors        |

---

## 5. Recommendation

**Platform of choice: Streamlit**

- **Total score: 4.45 / 5**  
- Delivers end-to-end Python data pipeline, advanced modeling, and interactive UI.  
- Fits our learning goal to gain hands-on experience with code-centric analytics.  
- Integrates seamlessly with GitHub for version control and CI/CD, with free hosting on Streamlit Community Cloud.

---
