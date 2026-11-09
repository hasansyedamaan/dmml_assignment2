# Model Comparison

Primary K = 10; secondary K = 5. Relevant means a test rating >= 4.

Coverage = unique recommended items / training catalog size. Diversity = mean intra-list pairwise cosine distance between item genre vectors.

| Model | precision_at_10 | recall_at_10 | ndcg_at_10 | coverage_at_10 | diversity_at_10 | rmse | mae | precision_at_5 | recall_at_5 | ndcg_at_5 | coverage_at_5 | diversity_at_5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0754 | 0.0749 | 0.0943 | 0.0447 | 0.7478 | 1.0735 | 0.8587 | 0.0769 | 0.0402 | 0.0880 | 0.0273 | 0.7812 |
| matrix_factorization | 0.1320 | 0.1543 | 0.1763 | 0.2862 | 0.7128 | 1.0082 | 0.7915 | 0.1566 | 0.0941 | 0.1746 | 0.2228 | 0.6970 |

Ranking metrics are averaged over test users with at least one relevant item. Cold-start users receive the popularity fallback; unseen test items are not recommendable and therefore count as misses.
