"""Domain reference tables used across the agent.

These encode domain knowledge (which merchant categories, countries and payment
channels are inherently higher risk) so the feature engineering and reporting
layers share one source of truth.
"""

CATEGORY_RISK = {
    "groceries": 0.05, "dining": 0.10, "transport": 0.08, "utilities": 0.05,
    "telecom": 0.10, "insurance": 0.05, "software": 0.15, "entertainment": 0.15,
    "health": 0.05, "retail": 0.15, "petrol": 0.05, "subsistence": 0.10,
    "travel": 0.35, "electronics": 0.30, "gift_cards": 0.70,
    "cryptocurrency": 0.90, "gambling": 0.85, "jewelry": 0.60,
    "money_transfer": 0.80,
}

COUNTRY_RISK = {
    "ZA": 0.05, "US": 0.08, "GB": 0.10, "CA": 0.09, "AU": 0.09, "DE": 0.12,
    "FR": 0.12, "NL": 0.11, "JP": 0.15, "SG": 0.18, "AE": 0.30, "MX": 0.40,
    "BR": 0.45, "IN": 0.40, "ID": 0.45, "PH": 0.50, "CN": 0.55, "HK": 0.45,
    "UA": 0.70, "RU": 0.80, "NG": 0.85, "VE": 0.85, "KE": 0.80, "GT": 0.75,
}

CHANNEL_RISK = {
    "card_present": 0.20, "pos": 0.35, "mobile_app": 0.55,
    "card_not_present": 1.00,
}

HIGH_RISK_CATEGORIES = ("gift_cards", "cryptocurrency", "gambling", "jewelry",
                        "money_transfer")