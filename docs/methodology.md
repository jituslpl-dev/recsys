# Modeling and evaluation methodology

## Feedback interpretation

MovieLens 100K contains explicit ratings. This project discards the numeric rating and treats each observed user–movie event as a positive **consumption proxy**. Thus the embedding task uses binary implicit labels but the source is not a click/exposure dataset. A low rating still indicates observed consumption; this model predicts consumption, not explicit satisfaction. Real experiments ignore demographics, tags and titles.

Duplicate user-item events are reduced to the earliest timestamp before splitting, creating a novel-item task. Dates are globally sorted. The empirical 60%, 80% and 90% timestamp boundaries define four phases, with equal timestamps kept together:

1. Retrieval train: learn user/item ID embeddings and both MLP towers; fit all statistics and the item catalog.
2. Rank train: retrieve unseen candidates from the frozen first-stage model and fit LambdaRank query groups using only relevance in this period.
3. Validation: report both stages and popularity; no settings are selected using these outcomes.
4. Test: report final metrics without refitting or changing settings.

The 60/20/10/10 fractions may vary slightly because ties stay together. Global splits can create substantial cold-user cohorts; these are retained and reported rather than silently dropping them. ID mappings and index membership use only phase 1.

## Neural training

Each tower contains a learned ID embedding and an MLP with Linear–ReLU–Linear layers. Output vectors are L2-normalized. For a user u, observed positive i and uniform sampled unobserved item j, optimize softplus((s(u,j)-s(u,i))/temperature), with cosine scores and four negatives per event by default. Negatives exclude every known phase-1 positive for that user, not merely the current item. Future positives are not inspected when sampling; sampled negatives may consequently be false negatives. AdamW trains fixed epochs with a seed and deterministic CPU operations. No pretrained embeddings are used.

## Retrieval

FAISS `IndexFlatIP` searches normalized item vectors, so inner product is cosine similarity. This is exact search, not approximate ANN. Request candidate count plus known seen count, bounded by catalog size, then exclude known items. A known user queries the learned user tower. Unknown users use phase-1 popularity. Item identifiers are mapped separately from FAISS row offsets and invalid `-1` search results are excluded.

The recommendation catalog is the phase-1 observed catalog. New later items cannot be retrieved and remain in relevance denominators, counting as misses. The model has no content-based new-item encoder.

## Learning to rank

LightGBM LambdaRank uses one query group per user in the rank-training period. Features include first-stage score, training popularity/activity/relative item recency, affinity to the user's frozen training-history mean item vector, elementwise user/item embedding products, and both vector components. Every feature statistic is frozen at the first cutoff. Item/user numeric identifiers are not ranking features.

Labels are binary membership in that user's later rank-training relevance set. Candidates are the actual retrieval output; held-out positives are never forcibly injected. Groups with no positive or no negative are uninformative and skipped when fitting, with counts recorded. Evaluation retains users and targets even when retrieval misses everything. Cold users bypass neural reranking and use popularity.

The feature order is identical in fitting and serving and is verified against the persisted booster schema. Ranker scores are ordering signals, not calibrated probabilities.

## Offline metrics

For a user with a nonempty future-period relevant set R, Recall@K=hits/|R|; HitRate@K=1 if at least one relevant item appears; MRR@K is reciprocal position of the first relevant recommendation, or zero; NDCG@K uses binary gains and log2(position+1) discounts, divided by ideal DCG with min(K,|R|) hits. All metrics are macro means over users. Duplicate recommendations are invalid; shorter lists are permitted and unavailable items count as misses. MRR is explicitly truncated at K.

Both stages share the same candidate pool and targets. Reranking cannot exceed candidate Recall@100 in final Recall@K; it only changes ordering. Report popularity, stage-1 and two-stage results for all users and separate warm/cold cohorts. Catalog coverage is unique top-20 recommendations divided by training catalog size; it is not relevance.

Evaluation predicts a whole later block from one phase-boundary snapshot. Seen-item filters include all events before that boundary; profile features remain frozen at phase 1. Within-phase events are targets, not newly updated histories. This is block recommendation evaluation, not per-event online replay or an unbiased counterfactual click estimate. Rating observations have selection/exposure bias and no propensity correction is claimed.

## Serving and persistence

Save PyTorch weights, explicit FAISS index, LightGBM text model, JSON mappings and non-pickled NumPy arrays. The local recommendation service uses cached user vectors and saved statistics, and verifies file hashes before binary loading. Hashes detect accidental corruption; they are not signatures or trust for externally supplied artifacts. Load only your own trusted training output. The exported seen snapshot ends at the validation cutoff, matching test-phase inference. No raw data, real user-level predictions or model artifacts are published to Git.

## References

- [PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [FAISS index selection guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [LightGBM ranking examples](https://github.com/lightgbm-org/LightGBM/tree/master/examples/lambdarank)
- [GroupLens MovieLens datasets](https://grouplens.org/datasets/movielens/)
- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS 5(4), Article 19. [DOI](https://doi.org/10.1145/2827872).
