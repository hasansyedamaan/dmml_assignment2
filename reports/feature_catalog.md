# Feature Catalog

Generated from `src/features/registry.py`. The same feature code path (`FeatureStore`) serves both training and prediction.

| Feature | Entity | Description | Formula / logic | Data type | Source table | Used by |
|---|---|---|---|---|---|---|
| avg_rating | user | Mean rating given by the user in the training window. | mean(rating) grouped by user_id | float | train interactions | training, prediction |
| activity_count | user | Number of training interactions for the user. | count(interaction) grouped by user_id | int | train interactions | training, prediction |
| preferred_category | user | Most frequent item genre in the user's training history. | mode(primary_genre) grouped by user_id | string | train interactions + items | training, prediction |
| days_since_last_interaction | user | Days between the as-of date and the user's latest training interaction. | (as_of_date - max(timestamp)) / 1 day | float | train interactions | training, prediction |
| rating_std | user | Standard deviation of the user's training ratings (0 for one rating). | std(rating) grouped by user_id | float | train interactions | training, prediction |
| distinct_items_count | user | Number of distinct items the user rated in training. | nunique(item_id) grouped by user_id | int | train interactions | training, prediction |
| popularity | item | Number of training interactions for the item. | count(interaction) grouped by item_id | int | train interactions | training, prediction |
| avg_rating | item | Mean rating received by the item in training. | mean(rating) grouped by item_id | float | train interactions | training, prediction |
| rating_count | item | Number of ratings received by the item in training. | count(rating) grouped by item_id | int | train interactions | training, prediction |
| category | item | Primary genre of the item. | first genre in the item's genre list | string | items | training, prediction |
| genre_* | item | One-hot genre indicators for the item. | 1 if the item belongs to the genre, else 0 | int (one-hot) | items | training, prediction, diversity |
| item_age_years | item | Age of the item at the as-of date, when a release date is known. | as_of_year - release_year | float | items | training, prediction |
| rating | interaction | Observed explicit rating (missing at prediction time). | raw rating | float | train interactions | training |
| timestamp | interaction | UTC timestamp of the interaction. | unix timestamp converted to UTC | datetime | train interactions | training, prediction |
| hour | interaction | Hour of day of the interaction. | timestamp.hour | int | train interactions | training, prediction |
| day_of_week | interaction | Day of week of the interaction (Monday=0). | timestamp.dayofweek | int | train interactions | training, prediction |
| user_item_genre_match | interaction | Whether the item's category matches the user's preferred category. | user.preferred_category == item.category | bool | user and item features | training, prediction |
| recency_rank | interaction | Chronological rank of the interaction within the user's history (1=oldest). | rank(timestamp) grouped by user_id | int | train interactions | training, prediction |
