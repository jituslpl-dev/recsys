# Interview walkthrough

1. Explain retrieval's candidate budget and ranking's order improvement. Why can a ranker never recover an item absent from its candidate pool?
2. Derive the sampled pairwise logistic loss and explain temperature, embedding normalization and false negatives.
3. Show the two learned towers and why exact FAISS inner product equals cosine for normalized embeddings.
4. Explain global chronological splits versus per-user holdouts, tied timestamps and cold cohorts.
5. Show why injecting evaluation positives would inflate apparent ranking quality. This implementation never injects them.
6. Explain LambdaRank query groups, binary relevance, uninformative group exclusion and identical feature order in serving.
7. Compute Recall, NDCG, MRR and HitRate on a three-item example. Discuss macro averaging and multiple relevant items.
8. Explain why ratings-as-consumption proxies do not measure clicks or satisfaction, and why exposure bias limits offline inference.
9. Compare both stages to training popularity using measured results, including cases where the neural system loses.
10. Discuss new users, new items, stale user features and the limits of ID-only embeddings.
11. Explain `CandidateRetriever` Strategy/Protocol, data adapters, immutable training configuration, shared features and service composition.
12. Describe a realistic extension: content-conditioned towers, HNSW/IVF retrieval with measured ANN recall, online event ingestion, independent tuning, A/B testing and service reliability. These are future work, not claims about the implemented system.
