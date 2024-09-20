# scripts/apriori_analysis.py
from mlxtend.frequent_patterns import apriori, association_rules

def apply_apriori(df, min_support=0.1):
    """Apply Apriori to accident dataset to find frequent patterns."""
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    return rules

def get_strong_rules(rules, min_lift=1.5):
    """Filter strong rules based on lift."""
    return rules[rules['lift'] >= min_lift]
