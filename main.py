import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# LOAD THE DATA
df = pd.read_csv("ecommerce_product_reviews_dataset.csv")

# FEATURE ENGINEERING
df = df.rename(columns={"review_text": "review"})
# Using provided sentiment label to map to a numeric score
sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
df['sentiment_score'] = df['sentiment'].map(sentiment_map)
# Review length based on word count
df['review_length'] = df['review'].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)
df_small = df.head(1000).copy()         

# FUZZY VARIABLE DEFINITIONS
sentiment = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'sentiment')
review_length = ctrl.Antecedent(np.arange(1, 51, 1), 'review_length')   # Adjust range as needed
satisfaction = ctrl.Consequent(np.arange(0, 11, 1), 'satisfaction')

# MEMBERSHIP FUNCTIONS
sentiment['negative'] = fuzz.trimf(sentiment.universe, [-1, -1, 0])
sentiment['neutral'] = fuzz.trimf(sentiment.universe, [-0.2, 0, 0.2])
sentiment['positive'] = fuzz.trimf(sentiment.universe, [0, 1, 1])

review_length['short'] = fuzz.trimf(review_length.universe, [1, 1, 10])
review_length['medium'] = fuzz.trimf(review_length.universe, [5, 20, 35])
review_length['long'] = fuzz.trimf(review_length.universe, [25, 50, 50])

satisfaction['low'] = fuzz.trimf(satisfaction.universe, [0, 0, 4])
satisfaction['medium'] = fuzz.trimf(satisfaction.universe, [3, 5, 7])
satisfaction['high'] = fuzz.trimf(satisfaction.universe, [6, 10, 10])

# FUZZY RULES 
rules = [
    ctrl.Rule(sentiment['positive'] & review_length['long'], satisfaction['high']),
    ctrl.Rule(sentiment['positive'] & review_length['medium'], satisfaction['high']),
    ctrl.Rule(sentiment['positive'] & review_length['short'], satisfaction['medium']),
    ctrl.Rule(sentiment['neutral'] & review_length['long'], satisfaction['medium']),
    ctrl.Rule(sentiment['neutral'] & review_length['medium'], satisfaction['medium']),
    ctrl.Rule(sentiment['neutral'] & review_length['short'], satisfaction['medium']),
    ctrl.Rule(sentiment['negative'] & review_length['long'], satisfaction['low']),
    ctrl.Rule(sentiment['negative'] & review_length['medium'], satisfaction['low']),
    ctrl.Rule(sentiment['negative'] & review_length['short'], satisfaction['low']),
]

satisfaction_ctrl = ctrl.ControlSystem(rules)

def save_results(df, filename="fuzzy_results.csv"):
    columns_to_save = [
        "product_id", "product_title", "category", "review", "sentiment",
        "review_length", "satisfaction_score", "fuzzy_satisfaction"
    ]
    # Only keep columns that exist in this DataFrame
    columns_to_save = [col for col in columns_to_save if col in df.columns]
    df[columns_to_save].to_csv(filename, index=False)
    print(f"Saved fuzzy results to: {filename}")


# INFERENCE FUNCTION
def fuzzy_predict(row):
    s_in = float(np.clip(row['sentiment_score'], -1, 1))
    l_in = float(np.clip(row['review_length'], 1, 50))
    scalc = ctrl.ControlSystemSimulation(satisfaction_ctrl)
    try:
        scalc.input['sentiment'] = s_in
        scalc.input['review_length'] = l_in
        scalc.compute()
        return scalc.output['satisfaction']
    except Exception as e:
        print("Fuzzy predict ERROR for:", row.to_dict())
        print("  (sentiment, review_length) -> (%.2f, %.2f)" % (s_in, l_in))
        print("  Error:", e)
        return np.nan

df_small['satisfaction_score'] = df_small.apply(fuzzy_predict, axis=1)

def satisfaction_label(score):
    if pd.isnull(score):
        return 'Unknown'
    elif score <= 4:
        return 'Low'
    elif score <= 7:
        return 'Medium'
    else:
        return 'High'

df_small['fuzzy_satisfaction'] = df_small['satisfaction_score'].apply(satisfaction_label)


# SAMPLE RESULT
print(df_small[['review', 'sentiment', 'review_length', 'satisfaction_score', 'fuzzy_satisfaction']].head(10))
save_results(df, "fuzzy_results_full.csv")

# (OPTIONAL) VISUALIZE MEMBERSHIP FUNCTIONS
# sentiment.view()
# review_length.view()
# satisfaction.view()
# plt.show()
