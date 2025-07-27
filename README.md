# ReviewSense - Fuzzy Logic Satisfaction Analyzer

A data science project that applies **fuzzy logic** to customer review analysis for e-commerce products. This project demonstrates how linguistic ambiguity and real-world customer feedback can be classified in a human-like way using interpretable, rule-based fuzzy logic.

## 🚀 Features

- Supports large, realistic datasets (such as the [Synthetic E-commerce Product Reviews Dataset](https://www.kaggle.com/datasets/aryan208/synthetic-e-commerce-product-reviews-dataset))
- Calculates review length and utilizes provided sentiment for robust feature engineering.
- Models customer satisfaction using transparent fuzzy logic rules.
- Outputs both a satisfaction score (0-10) and a categorical satisfaction level (Low, Medium, High).
- Saves enriched results to CSV for further analysis or dashboarding.
- Visualizes fuzzy membership functions for transparency.


## 📁 Files

- `main.py` — Main script for fuzzy logic analysis and visualization.
- `ecommerce_product_reviews_dataset.csv` — Dataset containing product reviews.
- `README.md` — Project documentation

## 📊 Citing the Dataset

Data source: [Synthetic E-commerce Product Reviews Dataset](https://www.kaggle.com/datasets/aryan208/synthetic-e-commerce-product-reviews-dataset) (CC0 1.0 Universal — Public Domain)

## 🧪 Example Output

| review                                | sentiment | review_length | satisfaction_score | fuzzy_satisfaction |
|----------------------------------------|-----------|--------------|-------------------|--------------------|
| Fast shipping and great packaging.     | Positive  | 5            | 5.0               | Medium             |
| Terrible experience, do not buy.       | Negative  | 5            | 1.51              | Low                |
| Highly recommend. Excellent quality.   | Positive  | 4            | 5.0               | Medium             |

## 🛠️ Customizing

- Adjust fuzzy rules in `main.py` to match your business logic or satisfaction criteria.
- Visualize and tweak membership functions as desired.

## 🤝 Contributing

Open to contributions! Issues & PRs welcome for new rules, features, better visualizations, or refactoring.

---

*Human-like, interpretable, scalable sentiment analysis with fuzzy logic!*

