import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

def generate_ott_dataset(n=5000):
    customer_id = [f"OTT-{str(i).zfill(5)}" for i in range(1, n + 1)]

    age = np.random.normal(35, 12, n).clip(18, 70).astype(int)
    gender = np.random.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04])
    subscription_type = np.random.choice(["Basic", "Standard", "Premium"], n, p=[0.30, 0.45, 0.25])
    payment_method = np.random.choice(["Credit Card", "UPI", "Net Banking", "Wallet"], n, p=[0.35, 0.30, 0.20, 0.15])
    favorite_genre = np.random.choice(["Action", "Drama", "Comedy", "Thriller", "Romance", "Sci-Fi", "Documentary"], n)
    devices_used = np.random.randint(1, 6, n)
    number_of_profiles = np.random.randint(1, 6, n)
    account_age_months = np.random.randint(1, 60, n)
    auto_renew_enabled = np.random.choice([0, 1], n, p=[0.25, 0.75])
    ads_tolerance = np.random.choice(["Low", "Medium", "High"], n, p=[0.40, 0.40, 0.20])

    monthly_subscription_cost = np.where(
        subscription_type == "Basic", np.random.uniform(99, 149, n),
        np.where(subscription_type == "Standard", np.random.uniform(199, 299, n),
                 np.random.uniform(349, 499, n))
    ).round(2)

    monthly_watch_hours = np.random.gamma(shape=4, scale=8, size=n).clip(1, 200).round(1)
    avg_session_time = (monthly_watch_hours / np.random.uniform(8, 25, n)).clip(0.5, 8).round(2)
    binge_watch_frequency = np.random.choice(["Never", "Rarely", "Sometimes", "Often", "Always"], n,
                                              p=[0.10, 0.20, 0.30, 0.25, 0.15])
    content_rating_given = np.random.uniform(1, 5, n).round(1)
    support_tickets = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.50, 0.20, 0.15, 0.08, 0.05, 0.02])
    days_since_last_login = np.random.exponential(scale=10, size=n).clip(0, 120).astype(int)

    # Churn logic: realistic business rules
    churn_score = (
        (days_since_last_login > 30) * 0.35 +
        (monthly_watch_hours < 10) * 0.20 +
        (support_tickets >= 3) * 0.20 +
        (auto_renew_enabled == 0) * 0.15 +
        (subscription_type == "Basic") * 0.10 +
        (account_age_months < 6) * 0.10 +
        np.random.uniform(0, 0.15, n)
    )
    churn_prob = 1 / (1 + np.exp(-5 * (churn_score - 0.55)))
    churn = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "gender": gender,
        "subscription_type": subscription_type,
        "monthly_watch_hours": monthly_watch_hours,
        "favorite_genre": favorite_genre,
        "devices_used": devices_used,
        "avg_session_time": avg_session_time,
        "monthly_subscription_cost": monthly_subscription_cost,
        "payment_method": payment_method,
        "support_tickets": support_tickets,
        "ads_tolerance": ads_tolerance,
        "days_since_last_login": days_since_last_login,
        "number_of_profiles": number_of_profiles,
        "binge_watch_frequency": binge_watch_frequency,
        "content_rating_given": content_rating_given,
        "account_age_months": account_age_months,
        "auto_renew_enabled": auto_renew_enabled,
        "churn": churn
    })

    # Introduce ~2% missing values
    for col in ["monthly_watch_hours", "avg_session_time", "content_rating_given", "support_tickets"]:
        mask = np.random.random(n) < 0.02
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "ott_churn_data.csv"
    df = generate_ott_dataset(5000)
    df.to_csv(out, index=False)
    print(f"Dataset saved → {out}  |  shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
