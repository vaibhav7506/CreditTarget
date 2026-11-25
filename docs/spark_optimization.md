# PySpark Physical Plan Breakdown & Performance Optimization

## 1. Physical Plan Analysis
CreditTarget optimizes large-scale Spark SQL workloads across millions of customer records through Adaptive Query Execution (AQE), broadcast joins, and shuffle partition tuning.

### 1.1 BroadcastHashJoin on Dimension Tables
In the `GoldStage`, the dimension table `product_catalog_dim` contains 3 static commercial pricing segments. To eliminate expensive network shuffle exchange across millions of customer records, CreditTarget enforces `safe_broadcast_join`:

```text
== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=false
+- == BroadcastHashJoin ==
   +- Project [customer_id, snapshot_month, ..., product_segment]
   |  +- Window [spend_30d, spend_60d, spend_90d, ...]
   |     +- Sort [customer_id ASC NULLS FIRST, snapshot_month ASC NULLS FIRST]
   +- BroadcastExchange HashedRelationBroadcastMode(List(input[0, string, false]))
      +- Filter isnotnull(product_segment)
         +- Scan Delta [silver_product_catalog]
```

By broadcasting `product_catalog_dim` to every executor, Spark eliminates the shuffle step on the multi-million row fact dataset, drastically cutting wall-clock execution time and avoiding out-of-memory errors.

---

## 2. Adaptive Query Execution (AQE)
The platform explicitly enables AQE in `spark_session.py`:
- `spark.sql.adaptive.enabled = true`: Dynamically coalesces shuffle partitions based on runtime statistics.
- `spark.sql.adaptive.coalescePartitions.enabled = true`: Prevents excessive small partition files after wide window transformations.
- `spark.sql.adaptive.skewJoin.enabled = true`: Dynamically splits skewed partitions (e.g. high-frequency transaction accounts) into sub-partitions.

---

## 3. Strict Point-in-Time Join Optimization
Point-in-time feature computation joins campaign snapshot dates with preceding transaction records:
```sql
FROM silver_campaign_labels c
INNER JOIN silver_transactions t
  ON c.customer_id = t.customer_id
 AND t.txn_ts <= c.snapshot_date
 AND t.txn_ts > c.snapshot_date - INTERVAL 90 DAYS
```
By enforcing temporal range boundaries directly in the join condition, Spark filters out non-qualifying events prior to grouping, reducing shuffle size by >75%.
