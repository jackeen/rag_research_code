# ============================================================================
# RAGAS Evaluation Metrics
# ============================================================================
# All metrics below return a score in the range [0, 1].
# For most metrics, higher is better. NoiseSensitivity is the exception:
# lower is better (it measures an error rate).
#
# The metrics are organized into four diagnostic groups:
#   1. Retriever side       -> ContextPrecision, ContextRecall,
#                              ContextEntityRecall
#   2. Generator side       -> Faithfulness, ResponseRelevancy
#   3. End-to-end accuracy  -> SemanticSimilarity
#   4. Robustness diagnosis -> NoiseSensitivity (relevant + irrelevant)
# ============================================================================


# ----------------------------------------------------------------------------
# [Retriever] ContextPrecision
# ----------------------------------------------------------------------------
# Range: [0, 1]   |   Direction: higher is better
#
# Principle:
#   Measures the signal-to-noise ratio of the retrieved context. For each
#   retrieved chunk, the judge LLM decides whether it is relevant to the
#   ground-truth answer. The metric then computes a rank-aware score
#   (similar to Mean Average Precision) so that relevant chunks appearing
#   at the top of the retrieval list contribute more than those at the
#   bottom. A high score means the retriever returns relevant chunks early
#   and avoids polluting the context with unrelated material.
#
# Example:
#   Question:     "When did the Apollo 11 mission land on the Moon?"
#   Ground truth: "July 20, 1969"
#   Retrieved chunks:
#     1. "Apollo 11 landed on the Moon on July 20, 1969."   (relevant)
#     2. "Neil Armstrong was born in Ohio in 1930."         (irrelevant)
#     3. "The Saturn V rocket was used for the mission."    (loosely related)
#   The relevant chunk is ranked first, so ContextPrecision is high
#   (close to 1.0). If the irrelevant chunk had been ranked first,
#   the score would drop significantly.


# ----------------------------------------------------------------------------
# [Retriever] ContextRecall
# ----------------------------------------------------------------------------
# Range: [0, 1]   |   Direction: higher is better
#
# Principle:
#   Measures how much of the information required by the ground-truth
#   answer is actually present in the retrieved context. The judge LLM
#   breaks the ground truth into individual claims and checks, for each
#   claim, whether it can be attributed to the retrieved context. The
#   score is the fraction of claims that are supported. A low score means
#   the retriever is missing information the model needs to answer
#   correctly, regardless of how good the generator is.
#
# Example:
#   Question:     "What are the health benefits of green tea?"
#   Ground truth: "Green tea boosts metabolism, contains antioxidants,
#                  and may reduce the risk of heart disease."
#   Retrieved context:
#     "Green tea is rich in antioxidants and is known to boost metabolism."
#   Two of the three claims are covered (antioxidants, metabolism), but
#   the heart-disease claim is missing, so ContextRecall is roughly 0.67.


# ----------------------------------------------------------------------------
# [Retriever] ContextEntityRecall
# ----------------------------------------------------------------------------
# Range: [0, 1]   |   Direction: higher is better
#
# Principle:
#   A finer-grained variant of ContextRecall that focuses specifically on
#   named entities (people, places, organizations, dates, numbers, etc.).
#   The judge LLM extracts entities from the ground-truth answer and
#   checks how many of them appear in the retrieved context. This metric
#   is especially useful in domains where missing a single entity is a
#   critical failure (finance, legal, medical, factual QA), since overall
#   semantic recall can look acceptable while a key name or number is
#   silently missing.
#
# Example:
#   Question:     "Who founded Microsoft and in what year?"
#   Ground truth: "Microsoft was founded by Bill Gates and Paul Allen
#                  in 1975."
#   Entities in ground truth: {Microsoft, Bill Gates, Paul Allen, 1975}
#   Retrieved context:
#     "Microsoft was founded by Bill Gates in the mid-1970s."
#   Entities found: {Microsoft, Bill Gates}
#   Missing: {Paul Allen, 1975}
#   ContextEntityRecall = 2 / 4 = 0.5


# ----------------------------------------------------------------------------
# [Generator] Faithfulness
# ----------------------------------------------------------------------------
# Range: [0, 1]   |   Direction: higher is better
#
# Principle:
#   Measures whether the generated answer is grounded in the retrieved
#   context, i.e. whether the model is hallucinating. The judge LLM
#   extracts the individual claims made in the answer and verifies each
#   one against the context. The score is the fraction of claims that
#   can be inferred from the context. A high score means the answer
#   stays faithful to the source material; a low score means the model
#   is fabricating or extrapolating beyond what the context supports.
#
# Example:
#   Question: "Who wrote the novel 1984?"
#   Context:  "1984 is a dystopian novel written by George Orwell,
#              published in 1949."
#   Answer A: "1984 was written by George Orwell and published in 1949."
#             -> All claims supported by the context. Faithfulness = 1.0
#   Answer B: "1984 was written by George Orwell, who also wrote
#              Brave New World."
#             -> The second claim is false (Brave New World is by Aldous
#                Huxley) and is not supported by the context.
#                Faithfulness drops to ~0.5.


# ----------------------------------------------------------------------------
# [Generator] ResponseRelevancy
# ----------------------------------------------------------------------------
# Range: [0, 1]   |   Direction: higher is better
#
# Principle:
#   Measures whether the generated answer actually addresses the user's
#   question, regardless of factual correctness. The judge LLM is asked
#   to reverse-engineer N possible questions that the answer could be
#   responding to, and the metric computes the average cosine similarity
#   between those reverse-generated questions and the original question
#   (using the provided embedding model). If the answer is on-topic and
#   complete, the reverse-generated questions will closely match the
#   original. If the answer is evasive, off-topic, or only partially
#   addresses the question, the similarity drops.
#
#   This metric complements Faithfulness: Faithfulness catches answers
#   that hallucinate, while ResponseRelevancy catches answers that are
#   well-grounded but fail to actually answer the question.
#
# Example:
#   Question: "What is the capital of France, and what is it famous for?"
#   Answer A: "The capital of France is Paris, famous for the Eiffel
#              Tower and the Louvre."
#             -> Reverse questions cluster around the original. ~0.95
#   Answer B: "France is a country in Europe with a long history."
#             -> Doesn't answer either part of the question.
#                Reverse questions diverge from the original. ~0.40


# ----------------------------------------------------------------------------
# [End-to-end] SemanticSimilarity
# ----------------------------------------------------------------------------
# Range: [0, 1]   |   Direction: higher is better
#
# Principle:
#   Measures the semantic closeness between the generated answer and the
#   ground-truth answer. Both texts are encoded into vectors using the
#   provided embedding model, and their cosine similarity is computed
#   and normalized into [0, 1]. This metric does not check factual
#   correctness; it only checks whether the answer expresses similar
#   meaning to the reference. It is useful for catching answers that
#   are phrased differently but convey the same idea, and for flagging
#   answers that are off-topic. It should NOT be used as a stand-alone
#   correctness signal -- pair it with Faithfulness / ResponseRelevancy.
#
# Example:
#   Ground truth: "The Eiffel Tower is located in Paris, France."
#   Answer A: "The Eiffel Tower stands in Paris, the capital of France."
#             -> Same meaning, different wording. Similarity ~0.95
#   Answer B: "The Eiffel Tower is a famous landmark in Europe."
#             -> Related but vaguer. Similarity ~0.75
#   Answer C: "The Great Wall of China is in Beijing."
#             -> Off-topic. Similarity ~0.30


# ----------------------------------------------------------------------------
# [Robustness] NoiseSensitivity:relevant
# ----------------------------------------------------------------------------
# Range: [0, 1]   |   Direction: LOWER is better (this is an error rate)
#
# Principle:
#   Measures the generator's baseline error rate when given ONLY the
#   relevant portion of the context (i.e. a clean, noise-free input).
#   The judge LLM examines each claim in the generated answer and counts
#   how many are incorrect with respect to the relevant context. A low
#   score means the generator handles clean information correctly; a
#   high score means the generator itself is unreliable -- it
#   hallucinates or misuses information even when the context is ideal.
#
#   This is the "generator-only" baseline. It tells you the floor of
#   error rate that no retriever improvement can fix.
#
# Example:
#   Question:     "What is the capital of Australia?"
#   Relevant ctx: "Canberra is the capital of Australia."
#   Answer A: "The capital of Australia is Canberra."
#             -> Correctly uses the clean context. Score = 0.0
#   Answer B: "The capital of Australia is Sydney."
#             -> Wrong even though the context is clean.
#                Score = 1.0 (a pure generator failure).


# ----------------------------------------------------------------------------
# [Robustness] NoiseSensitivity:irrelevant
# ----------------------------------------------------------------------------
# Range: [0, 1]   |   Direction: LOWER is better (this is an error rate)
#
# Principle:
#   Measures the generator's error rate when irrelevant chunks are mixed
#   into the retrieved context (i.e. realistic, noisy input). The judge
#   LLM examines each claim in the generated answer and counts how many
#   are incorrect under the noisy condition. A low score means the
#   generator is robust to retrieval noise; a high score means the
#   generator gets distracted by irrelevant chunks and produces wrong
#   claims.
#
#   Interpreting "relevant" + "irrelevant" together:
#     Let R = NoiseSensitivity(mode="relevant")
#         I = NoiseSensitivity(mode="irrelevant")
#     Generator error          ~= R         (baseline)
#     Retriever-induced error  ~= I - R     (extra harm from noise)
#
#     R low,  I low   -> system healthy
#     R low,  I high  -> retriever bottleneck (too much noise / bad ranking)
#     R high, I high  -> generator bottleneck (hallucinates even on clean ctx)
#     R high, I higher-> both broken; fix generator first
#
# Example:
#   Question: "What is the capital of Australia?"
#   Retrieved context (noisy):
#     - "Canberra is the capital of Australia."           (relevant)
#     - "Sydney is the most populous city in Australia."  (irrelevant noise)
#   Answer A: "The capital of Australia is Canberra."
#             -> Not misled by noise. Score = 0.0
#   Answer B: "The capital of Australia is Sydney."
#             -> Misled by the irrelevant chunk. Score = 1.0


# ============================================================================
# Quick reference
# ============================================================================
#   Group         Metric                           Direction   What it tells you
#   -----------   ------------------------------   ---------   ------------------
#   Retriever     ContextPrecision                 higher      ranking quality
#   Retriever     ContextRecall                    higher      claim-level coverage
#   Retriever     ContextEntityRecall              higher      entity-level coverage
#   Generator     Faithfulness                     higher      grounded, no hallucination
#   Generator     ResponseRelevancy                higher      actually answers the question
#   End-to-end    SemanticSimilarity               higher      meaning matches ground truth
#   Robustness    NoiseSensitivity (relevant)      LOWER       generator baseline error
#   Robustness    NoiseSensitivity (irrelevant)    LOWER       error under realistic noise
#
# Aggregation note:
#   When computing a composite score, invert the NoiseSensitivity metrics
#   (use 1 - score) so that "higher is better" holds uniformly. Otherwise
#   the average will be misleading.
# ============================================================================
