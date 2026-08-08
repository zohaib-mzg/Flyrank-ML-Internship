# Capstone Report, Refresh / Content Opportunity Scoring

- **Author:** Muhammad Zohaib
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/zohaib-mzg/Flyrank-ML-Internship
- **Date:** 2026-08-07

## 1. Problem framing

The decision this project supports: given limited review time, which pages should a content team look at first this week. The unit of analysis is one page, one row per `content_id`. The output is a ranked queue, a priority score plus a reason code and an action label, `refresh_content`, `ctr_review`, `refresh_and_ctr_review`, or `monitor`. The human in the loop is a content editor, who takes the top of the queue and manually reviews it, this project never acts on a page directly.

The cost of a wrong call is asymmetric and low-stakes in either direction. A false positive costs an editor a few minutes reviewing a page that did not need it. A false negative means a genuinely declining page goes unreviewed for another cycle, more costly, but recoverable next week. That asymmetry, not a hard accuracy requirement, is why Precision@50 (does the top of the list hold up) was chosen as the headline metric over stricter measures like recall or F1.

Why ML instead of a fixed rule: the underlying signals, staleness, demand, position, and content depth, trade off against each other in ways a flat rule can only approximate with fixed thresholds and weights. Section 3 shows this baseline is a real, validated rule, not a strawman, and Section 5 shows where a trained model actually earns its place over it, and where the honest margin is smaller than a first look suggests.

## 2. Data safety

**Source:** `data/raw/content_refresh_anonymized.csv`, a 30,000-page anonymized slice from the FlyRank ML Internship warehouse release. One row per page. `content_id` and `client_id` are pseudonymized hashes throughout this project, no client names, domains, URLs, or private queries appear anywhere in `work/`.

**Deliberately excluded columns:** `trend_direction` and `trend_pct` (the label itself), plus `impressions_last_30d`, `impressions_prev_30d`, `clicks_last_30d`, `clicks_prev_30d`, `sessions_last_30d`, `sessions_prev_30d` (the exact two 30-day windows the label is computed from). Including any of these would let the model partly read the label off the input, the same mistake deliberately caused and caught in an earlier leakage exercise on this dataset before modeling began.

**Pseudonymous IDs:** `client_id` is used only for split grouping (Section 5), never as a model feature. `content_id` is used only as a row identifier in the output queue, never as a feature.

**Leakage check performed:** correlation between every remaining candidate feature and the label topped out at 0.19 (`days_with_impressions`), far below what a genuine leak would produce.

## 3. Baseline

`low_ctr_visible_page`, a hand-written rule: pages with at least 500 impressions in 90 days, a position between 1 and 20, and a CTR under 0.5. Its ranking score is the gap between a page's CTR and its position tier's average CTR, multiplied by impression volume, an estimate of clicks left on the table.

This rule was validated on its own terms before being used as a comparison point. Average CTR falls cleanly as position tier weakens across the full dataset (top_3 highest, deep lowest), confirming the assumption the rule depends on. It is a fair baseline, not a strawman.

**Important honest correction found while writing this report:** the baseline's Precision@50, 0.520 on a random split and 0.531 averaged across 20 client-grouped splits, is actually *at or slightly below* the dataset's base rate for the label, 54.2% of pages are declining. A ranking with zero real skill would be expected to score close to the base rate on this metric, so the baseline shows effectively **no measurable lift for predicting decline** (-0.022 and -0.011 respectively). This makes sense in hindsight, the rule was built to catch CTR opportunities, not decline, so its earlier framing as "a baseline that beats or nearly beats the model" was misleading. The baseline is a legitimate, validated rule for its own purpose, it is just not a strong competitor on this specific metric, and this report should have said so from the start.

| Split | Baseline P@50 | Base rate | Lift over base rate |
|---|---:|---:|---:|
| Random (single) | 0.520 | 0.542 | -0.022 |
| Client-grouped (20-split mean) | 0.531 | 0.542 | -0.011 |

## 4. Model / analysis

Random Forest classifier predicting `is_declining` (`trend_direction == "down"`), chosen over Logistic Regression because the signals genuinely interact, staleness, demand, position, and content depth trade off against each other in ways a flat or linear combination does not fully capture.

**Feature list**, 22 numeric and 9 categorical columns: `search_volume`, `competition`, `cpc`, `word_count`, `char_count`, `impressions_90d`, `clicks_90d`, `pageviews_90d`, `sessions_90d`, `users_90d`, `engaged_sessions_90d`, `ai_sessions_90d`, `scroll_events_90d`, `days_with_impressions`, `days_with_sessions`, `content_age_days`, `days_since_last_update`, `ctr`, `avg_position`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`, plus `competition_level`, `content_type`, `main_intent`, `age_tier`, `freshness_tier`, `word_count_tier`, `char_count_tier`, `impression_tier`, `position_tier`. Missing values in `search_volume`, `competition`, `cpc`, `word_count`, `char_count` filled with column medians, a named choice, not a silent default.

**Left out on purpose:** the label-adjacent last30/prev30 columns and `trend_pct`, per Section 2.

**Target, in one sentence:** `is_declining`, whether a page's 30-day impression trend is currently negative, an observed comparison already present in the data, not a proxy invented after the fact.

## 5. Evaluation

Two validation passes, deliberately, not one. First, a plain stratified random 75/25 split. Second, a client-grouped split (`GroupShuffleSplit`, every page from a given client entirely in train or entirely in test), because many pages share a client and a random split can let the model partly learn account-level patterns instead of a generalizable content signal.

With only 32 clients, a single grouped split is itself unstable, so the grouped result is reported as a mean and standard deviation across 20 different client groupings, not one number.

| Split | Base rate | Baseline P@50 | Model P@50 | Model lift over base rate |
|---|---:|---:|---:|---:|
| Random (single) | 0.542 | 0.520 | 0.880 | +0.338 |
| Client-grouped (20-split mean) | 0.542 | 0.531 | 0.761 &plusmn; 0.147 | +0.219 |

The random split overstated the model's edge. The honest, client-grouped number is a real, measured advantage over base rate, +0.219 on average, but meaningfully smaller than +0.338, and it comes with real spread (0.147 std) that the random split hid entirely.

**Error analysis:** in the Week 5 top-50 review under the random split, 6 of 50 top-ranked pages were false positives. Five of those six shared the exact same `days_since_last_update` value (104 days), a likely batch-update artifact rather than five independent mistakes, the model correctly learned a real pattern that simply does not hold for every page matching it. The sixth false positive had unusually high impression volume (134,567 in 90 days) and a recent update (20 days), a genuinely uncertain, high-visibility case, exactly the kind of page a review queue should surface rather than something to treat as a clean failure.

## 6. Interpretation

Top feature importances: `days_with_impressions`, `impressions_90d`, `avg_position`, `content_age_days`, in that order. In plain words, pages with thinner, less consistent visibility over 90 days, and older pages, are both more likely to be caught declining, a reasonable, intuitive story, and none of the top features touch the excluded last30/prev30 columns.

**Two negative or surprising results, reported as real findings, not hidden:**

1. Staleness alone (`days_since_last_update >= 180`) is *not* a reliable decline signal, 47.1% observed decline rate among stale pages, barely above the 54.2% base rate, and actually below it. This directly informed Week 7's playbook, staleness was demoted from a trigger to supporting context.
2. The baseline's Precision@50 sitting at or below the base rate (Section 3), only discovered while writing this report. A well-understood "the baseline has no real skill here" is treated as a valid, reportable result, not something to quietly drop.

## 7. Recommendation

Every page receives one of four actions: `refresh_and_ctr_review` (2,348 pages, high decline confidence and a real CTR gap), `refresh_content` (3,136, high decline confidence alone), `ctr_review` (7,411, a real CTR gap without confirmed decline), `monitor` (17,105, no action this cycle).

An editor would open the ranked queue, take the top N that fit this week's review capacity, and use the reason code to know what kind of review each page needs. **Confidence:** directional, not precise, model lift over base rate is real and measured (+0.219 on average under an honest split) but has real spread across which clients are involved, and the label describes a current snapshot, not a checked future outcome. **Explicit limits:** only 32 clients in this slice (3 to 7,000+ pages each), no causal claim about why any page declines, no automated actions triggered directly from a score (see Week 7's no-go list for the full set of manual-review rules).

## 8. Reproducibility

Clone and re-run from scratch:

```bash
git clone https://github.com/zohaib-mzg/Flyrank-ML-Internship
cd Flyrank-ML-Internship
pip install -r requirements.txt
jupyter execute work/notebooks/w04_baseline_score.ipynb --output=w04_baseline_score.ipynb
jupyter execute work/notebooks/w05_model.ipynb --output=w05_model.ipynb
jupyter execute work/notebooks/w06_validation_audit.ipynb --output=w06_validation_audit.ipynb
jupyter execute work/notebooks/w07_action_playbook.ipynb --output=w07_action_playbook.ipynb
jupyter execute work/notebooks/capstone.ipynb --output=capstone.ipynb
```

**Random seeds:** `random_state=42` fixed on every `train_test_split`, `GroupShuffleSplit`, and `RandomForestClassifier` call. The 20-split grouped evaluation in Section 5 sweeps seeds 0 through 19 deliberately, to measure variance, not to hide it.

**Environment**, from `requirements.txt`: `pandas>=2.2`, `numpy>=1.26`, `scikit-learn>=1.4`, `matplotlib>=3.8`, `duckdb>=1.0`, `huggingface_hub>=0.24`.

---

**Claims checklist**

- [x] Observed / measured / directional / decision-support language throughout, including this report's own correction in Section 3.
- [x] Base rate (54.2%) reported next to every Precision@50 number, added in this pass after finding it was missing.
- [x] No causal claims, no "predicted Google's algorithm."
- [x] No client-identifying details, pseudonymized hashes only.
- [x] Numbers in this report match a fresh re-run, base rate, baseline P@50, and model P@50 recomputed directly from `data/raw/content_refresh_anonymized.csv` while writing this report, not carried over from memory.
