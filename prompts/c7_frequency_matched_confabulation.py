#!/usr/bin/env python3
"""
Campaign 2 — Control C7: Frequency-Matched Confabulation
=========================================================

Tests whether KV-cache effective rank differences for confabulation prompts
are driven by rare/unusual tokens rather than the confabulation itself.

Background:
  Campaign 1 showed that confabulation prompts ("Zephyr Cloudwalker",
  "Etherealium", "quantum bicycle") produce distinct geometric signatures
  in KV-cache — higher effective rank than grounded facts. Campaign 1's
  Control C1 demonstrated that token frequency drives *norm* differences
  (d ~ 0 when frequency is matched). But effective rank showed a persistent
  signal even when norms were flat (d = 0.46-0.67 at 1.1B-7B).

  This control asks: does that geometric signal survive when we eliminate
  the token frequency confound entirely?

Design:
  - 30 confabulation prompts using ONLY common, high-frequency English words
  - 30 matched factual prompts with same vocabulary profile, length, and structure
  - The ONLY systematic difference is truth value
  - No exotic names, invented substances, sci-fi scenarios, or technical jargon

If effective rank still differs between these sets → the geometric signal is
about confabulation *processing*, not token frequency. This would be the
strongest evidence yet for the paper's central claim.

Liberation Labs / THCoalition — February 2026
"""

# ================================================================
# CONFABULATION PROMPTS — Common tokens only, all false claims
# ================================================================

CONFABULATION_COMMON_TOKENS = [
    # --- History ---
    {
        "id": "confab_common_01",
        "text": "The first president of the United States was born in 1800 in Boston.",
        "matched_factual_id": "fact_common_01",
        "false_element": "Washington was born 1732 in Westmoreland County VA, not 1800 in Boston",
        "domain": "history",
    },
    {
        "id": "confab_common_02",
        "text": "The American Civil War ended in 1840 after lasting only two years.",
        "matched_factual_id": "fact_common_02",
        "false_element": "Ended 1865 after four years, not 1840 after two",
        "domain": "history",
    },
    {
        "id": "confab_common_03",
        "text": "The Great Wall of China was built in the 1700s to keep out wild animals.",
        "matched_factual_id": "fact_common_03",
        "false_element": "Construction began ~7th century BC for military defense, not 1700s for animals",
        "domain": "history",
    },
    {
        "id": "confab_common_04",
        "text": "The first person to walk on the moon was a British man in 1954.",
        "matched_factual_id": "fact_common_04",
        "false_element": "Neil Armstrong (American) in 1969, not British man in 1954",
        "domain": "history",
    },
    {
        "id": "confab_common_05",
        "text": "World War Two started because of a trade deal between France and Brazil.",
        "matched_factual_id": "fact_common_05",
        "false_element": "Started with Germany's invasion of Poland, not a France-Brazil trade deal",
        "domain": "history",
    },
    # --- Geography ---
    {
        "id": "confab_common_06",
        "text": "The largest ocean on Earth is the Atlantic Ocean, which covers most of the planet.",
        "matched_factual_id": "fact_common_06",
        "false_element": "Pacific Ocean is largest, not Atlantic",
        "domain": "geography",
    },
    {
        "id": "confab_common_07",
        "text": "Russia is a small country located in South America near the coast of Brazil.",
        "matched_factual_id": "fact_common_07",
        "false_element": "Russia is the largest country, in Europe/Asia, not small in South America",
        "domain": "geography",
    },
    {
        "id": "confab_common_08",
        "text": "The longest river in the world is the River Thames in England.",
        "matched_factual_id": "fact_common_08",
        "false_element": "Nile or Amazon is longest, Thames is only 346 km",
        "domain": "geography",
    },
    {
        "id": "confab_common_09",
        "text": "Africa is the smallest of all the seven main land areas on Earth.",
        "matched_factual_id": "fact_common_09",
        "false_element": "Africa is second largest continent, not smallest",
        "domain": "geography",
    },
    {
        "id": "confab_common_10",
        "text": "The capital city of Japan is a town called Dallas in the southern part of the country.",
        "matched_factual_id": "fact_common_10",
        "false_element": "Capital of Japan is Tokyo, not Dallas",
        "domain": "geography",
    },
    # --- Science ---
    {
        "id": "confab_common_11",
        "text": "Water boils at fifty degrees when heated on a normal kitchen stove.",
        "matched_factual_id": "fact_common_11",
        "false_element": "Water boils at 100C / 212F, not 50 degrees",
        "domain": "science",
    },
    {
        "id": "confab_common_12",
        "text": "The Earth goes around the Sun once every ten years in a long slow path.",
        "matched_factual_id": "fact_common_12",
        "false_element": "Earth orbits the Sun once per year, not every ten years",
        "domain": "science",
    },
    {
        "id": "confab_common_13",
        "text": "Humans have three hundred bones in their body when they are fully grown adults.",
        "matched_factual_id": "fact_common_13",
        "false_element": "Adult humans have 206 bones, not 300",
        "domain": "science",
    },
    {
        "id": "confab_common_14",
        "text": "Sound travels faster than light, which is why we hear thunder before we see it.",
        "matched_factual_id": "fact_common_14",
        "false_element": "Light is faster than sound; we see lightning before hearing thunder",
        "domain": "science",
    },
    {
        "id": "confab_common_15",
        "text": "The Moon is bigger than the Earth and gives off its own bright white light.",
        "matched_factual_id": "fact_common_15",
        "false_element": "Moon is much smaller than Earth and reflects sunlight",
        "domain": "science",
    },
    # --- Animals ---
    {
        "id": "confab_common_16",
        "text": "Dogs were first kept as pets by people living in England around the year 1600.",
        "matched_factual_id": "fact_common_16",
        "false_element": "Dogs domesticated ~15,000 years ago, likely in Asia, not 1600 England",
        "domain": "animals",
    },
    {
        "id": "confab_common_17",
        "text": "The biggest animal that has ever lived on Earth is the African elephant.",
        "matched_factual_id": "fact_common_17",
        "false_element": "Blue whale is the biggest animal ever, not African elephant",
        "domain": "animals",
    },
    {
        "id": "confab_common_18",
        "text": "Cats can live for about fifty years if they are well taken care of at home.",
        "matched_factual_id": "fact_common_18",
        "false_element": "Domestic cats typically live 12-18 years, not 50",
        "domain": "animals",
    },
    {
        "id": "confab_common_19",
        "text": "Fish need to come to the top of the water to breathe air through their mouths.",
        "matched_factual_id": "fact_common_19",
        "false_element": "Most fish breathe through gills underwater, do not need surface air",
        "domain": "animals",
    },
    {
        "id": "confab_common_20",
        "text": "Birds fly south in winter because they are afraid of the cold snow and ice.",
        "matched_factual_id": "fact_common_20",
        "false_element": "Birds migrate for food availability, not fear of snow/ice",
        "domain": "animals",
    },
    # --- Food / Daily Life ---
    {
        "id": "confab_common_21",
        "text": "Bread is made by mixing water and salt together without using any kind of flour.",
        "matched_factual_id": "fact_common_21",
        "false_element": "Bread requires flour as its primary ingredient",
        "domain": "food",
    },
    {
        "id": "confab_common_22",
        "text": "Orange juice comes from apples that are grown in cold places like Canada.",
        "matched_factual_id": "fact_common_22",
        "false_element": "Orange juice comes from oranges, typically grown in warm climates",
        "domain": "food",
    },
    {
        "id": "confab_common_23",
        "text": "Rice is mostly grown in very dry places like the middle of the desert.",
        "matched_factual_id": "fact_common_23",
        "false_element": "Rice is grown in flooded paddies in wet climates, not deserts",
        "domain": "food",
    },
    {
        "id": "confab_common_24",
        "text": "Drinking cold water on a hot day is bad for your health and should be avoided.",
        "matched_factual_id": "fact_common_24",
        "false_element": "Drinking water on hot days is important for hydration, not harmful",
        "domain": "food",
    },
    {
        "id": "confab_common_25",
        "text": "Coffee was first made in France around the year 1900 using tea leaves.",
        "matched_factual_id": "fact_common_25",
        "false_element": "Coffee originated in Ethiopia centuries earlier, made from coffee beans not tea leaves",
        "domain": "food",
    },
    # --- Culture / General Knowledge ---
    {
        "id": "confab_common_26",
        "text": "The game of football was first played in China about three hundred years ago.",
        "matched_factual_id": "fact_common_26",
        "false_element": "Modern football rules from England 1863; ancient Chinese cuju was ~2000 years ago",
        "domain": "culture",
    },
    {
        "id": "confab_common_27",
        "text": "Most children start school at the age of twelve in the United States.",
        "matched_factual_id": "fact_common_27",
        "false_element": "US children typically start school at age 5-6, not 12",
        "domain": "culture",
    },
    {
        "id": "confab_common_28",
        "text": "Books were first made by printing words on thin sheets of metal in Germany.",
        "matched_factual_id": "fact_common_28",
        "false_element": "Gutenberg printed on paper, not metal; earlier books existed in other forms",
        "domain": "culture",
    },
    {
        "id": "confab_common_29",
        "text": "The summer months in Australia fall between June and August every single year.",
        "matched_factual_id": "fact_common_29",
        "false_element": "Australian summer is December-February (Southern Hemisphere)",
        "domain": "culture",
    },
    {
        "id": "confab_common_30",
        "text": "There are about fifty days in every month of the year on our current calendar.",
        "matched_factual_id": "fact_common_30",
        "false_element": "Months have 28-31 days, not 50",
        "domain": "culture",
    },
]


# ================================================================
# FACTUAL PROMPTS — Frequency-matched controls (all true claims)
# ================================================================

FACTUAL_COMMON_TOKENS = [
    # --- History ---
    {
        "id": "fact_common_01",
        "text": "The first president of the United States was born in 1732 in Virginia.",
        "matched_confab_id": "confab_common_01",
        "domain": "history",
    },
    {
        "id": "fact_common_02",
        "text": "The American Civil War ended in 1865 after lasting about four years.",
        "matched_confab_id": "confab_common_02",
        "domain": "history",
    },
    {
        "id": "fact_common_03",
        "text": "The Great Wall of China was built over many years to keep out enemy armies.",
        "matched_confab_id": "confab_common_03",
        "domain": "history",
    },
    {
        "id": "fact_common_04",
        "text": "The first person to walk on the moon was an American man in 1969.",
        "matched_confab_id": "confab_common_04",
        "domain": "history",
    },
    {
        "id": "fact_common_05",
        "text": "World War Two started because of Germany's attack on Poland in the fall.",
        "matched_confab_id": "confab_common_05",
        "domain": "history",
    },
    # --- Geography ---
    {
        "id": "fact_common_06",
        "text": "The largest ocean on Earth is the Pacific Ocean, which covers most of the planet.",
        "matched_confab_id": "confab_common_06",
        "domain": "geography",
    },
    {
        "id": "fact_common_07",
        "text": "Russia is a large country located in both Europe and Asia near the north.",
        "matched_confab_id": "confab_common_07",
        "domain": "geography",
    },
    {
        "id": "fact_common_08",
        "text": "The longest river in the world is the River Nile in Africa.",
        "matched_confab_id": "confab_common_08",
        "domain": "geography",
    },
    {
        "id": "fact_common_09",
        "text": "Africa is the second largest of all the seven main land areas on Earth.",
        "matched_confab_id": "confab_common_09",
        "domain": "geography",
    },
    {
        "id": "fact_common_10",
        "text": "The capital city of Japan is a place called Tokyo in the eastern part of the country.",
        "matched_confab_id": "confab_common_10",
        "domain": "geography",
    },
    # --- Science ---
    {
        "id": "fact_common_11",
        "text": "Water boils at one hundred degrees when heated on a normal kitchen stove.",
        "matched_confab_id": "confab_common_11",
        "domain": "science",
    },
    {
        "id": "fact_common_12",
        "text": "The Earth goes around the Sun once every single year in a long round path.",
        "matched_confab_id": "confab_common_12",
        "domain": "science",
    },
    {
        "id": "fact_common_13",
        "text": "Humans have two hundred and six bones in their body when they are fully grown adults.",
        "matched_confab_id": "confab_common_13",
        "domain": "science",
    },
    {
        "id": "fact_common_14",
        "text": "Light travels faster than sound, which is why we see things before we hear them.",
        "matched_confab_id": "confab_common_14",
        "domain": "science",
    },
    {
        "id": "fact_common_15",
        "text": "The Moon is smaller than the Earth and shines by reflecting the light of the Sun.",
        "matched_confab_id": "confab_common_15",
        "domain": "science",
    },
    # --- Animals ---
    {
        "id": "fact_common_16",
        "text": "Dogs were first kept as pets by people living in Asia thousands of years ago.",
        "matched_confab_id": "confab_common_16",
        "domain": "animals",
    },
    {
        "id": "fact_common_17",
        "text": "The biggest animal that has ever lived on Earth is the great blue whale.",
        "matched_confab_id": "confab_common_17",
        "domain": "animals",
    },
    {
        "id": "fact_common_18",
        "text": "Cats can live for about fifteen years if they are well taken care of at home.",
        "matched_confab_id": "confab_common_18",
        "domain": "animals",
    },
    {
        "id": "fact_common_19",
        "text": "Fish use special parts on the sides of their heads to breathe air from the water.",
        "matched_confab_id": "confab_common_19",
        "domain": "animals",
    },
    {
        "id": "fact_common_20",
        "text": "Birds fly south in winter because they need to find warm places to get food.",
        "matched_confab_id": "confab_common_20",
        "domain": "animals",
    },
    # --- Food / Daily Life ---
    {
        "id": "fact_common_21",
        "text": "Bread is made by mixing flour and water together with a small amount of salt.",
        "matched_confab_id": "confab_common_21",
        "domain": "food",
    },
    {
        "id": "fact_common_22",
        "text": "Orange juice comes from oranges that are grown in warm places like Florida.",
        "matched_confab_id": "confab_common_22",
        "domain": "food",
    },
    {
        "id": "fact_common_23",
        "text": "Rice is mostly grown in very wet places like the fields of southern Asia.",
        "matched_confab_id": "confab_common_23",
        "domain": "food",
    },
    {
        "id": "fact_common_24",
        "text": "Drinking cold water on a hot day is good for your health and helps you stay cool.",
        "matched_confab_id": "confab_common_24",
        "domain": "food",
    },
    {
        "id": "fact_common_25",
        "text": "Coffee was first made in Africa many hundreds of years ago using roasted beans.",
        "matched_confab_id": "confab_common_25",
        "domain": "food",
    },
    # --- Culture / General Knowledge ---
    {
        "id": "fact_common_26",
        "text": "The game of football was first given its rules in England about two hundred years ago.",
        "matched_confab_id": "confab_common_26",
        "domain": "culture",
    },
    {
        "id": "fact_common_27",
        "text": "Most children start school at the age of five or six in the United States.",
        "matched_confab_id": "confab_common_27",
        "domain": "culture",
    },
    {
        "id": "fact_common_28",
        "text": "Books were first made by printing words on thin sheets of paper in Germany.",
        "matched_confab_id": "confab_common_28",
        "domain": "culture",
    },
    {
        "id": "fact_common_29",
        "text": "The summer months in Australia fall between December and February every single year.",
        "matched_confab_id": "confab_common_29",
        "domain": "culture",
    },
    {
        "id": "fact_common_30",
        "text": "There are about thirty days in most months of the year on our current calendar.",
        "matched_confab_id": "confab_common_30",
        "domain": "culture",
    },
]


# ================================================================
# DESIGN NOTES
# ================================================================

DESIGN_NOTES = """
Control C7: Frequency-Matched Confabulation — Design Notes
============================================================

PURPOSE:
    Campaign 1 found that confabulation prompts show elevated effective rank
    (dimensionality) in KV-cache compared to factual prompts. Campaign 1's
    Control C1 showed that token frequency drives NORM differences — when
    confab and factual prompts were matched on word frequency, norm differences
    disappeared (d ~ 0). However, the GEOMETRIC signal (effective rank) persisted
    even when norms were flat (d = 0.46-0.67 at 1.1B-7B scales).

    This raises a critical question: is the effective rank signal also driven by
    token frequency, just more subtly? Campaign 1's confabulation prompts used
    obviously rare tokens ("Zephyr Cloudwalker", "Etherealium", "quantum bicycle",
    "47th president of Mars"). Rare tokens might produce unusual activation
    patterns that inflate dimensionality regardless of truth value.

    C7 eliminates this confound entirely by using ONLY common English words
    in both the confabulation and factual prompt sets.

MATCHING METHODOLOGY:
    Each confabulation prompt is paired with a factual prompt matched on:

    1. VOCABULARY FREQUENCY: All words in both sets drawn from high-frequency
       English vocabulary. No proper nouns beyond well-known places (England,
       Africa, Japan, etc.) and no technical/scientific terminology. Words like
       "the", "is", "in", "was", "about", "years" dominate both sets equally.

    2. SENTENCE LENGTH: Matched pairs are within +/- 3 words of each other.
       Both sets have similar overall length distributions.

    3. SYNTACTIC STRUCTURE: Paired prompts use the same grammatical frame.
       E.g., "The [superlative] [noun] on Earth is the [noun]" appears in
       both the confab and factual versions.

    4. DOMAIN: Each pair shares the same topic domain (history, geography,
       science, animals, food, culture). 5 prompts per domain per set.

    5. SEMANTIC NATURALNESS: All prompts are complete, natural-sounding
       English sentences. No sentence fragments, no unusual constructions.

    The ONLY systematic difference is truth value: confabulation prompts
    make false claims, factual prompts make true claims.

WHAT A POSITIVE RESULT MEANS (effective rank differs):
    If effective rank is significantly higher for confabulation prompts even
    with matched token frequencies, this confirms that the geometric signal
    reflects something about how the model PROCESSES false claims — not an
    artifact of unusual tokens creating unusual cache patterns.

    This would be the strongest evidence for the paper's central thesis: that
    confabulation has a geometric signature in the model's internal state that
    is independent of surface-level features like token frequency or length.
    The model is doing something computationally different when it generates
    false content, and that difference is visible in the dimensionality of
    its key-value cache representations.

    Combined with Campaign 1 results, the narrative becomes:
    - Norms: driven by token frequency (C1 nullified the signal)
    - Effective rank: driven by truth value (C7 confirms the signal survives)
    - The signal "lives in geometry, not magnitude" — the paper's core claim

WHAT A NEGATIVE RESULT MEANS (effective rank does NOT differ):
    If effective rank shows no significant difference (d < 0.2) between these
    matched sets, then the Campaign 1 geometric signal was indeed a more subtle
    frequency artifact. Rare tokens may create more diverse activation patterns
    that inflate dimensionality, and the "confabulation geometry" finding was
    actually measuring "unusual token geometry."

    This would NOT invalidate the entire paper — the deception and refusal
    findings from other experiments have separate controls. But it would require
    reframing the confabulation result as a frequency artifact and removing it
    from the geometric framework's evidence base.

    Honestly reporting this negative result would strengthen the paper's
    credibility overall: it shows we tested every confound and reported what
    we found, even when it complicated the narrative.

STATISTICAL PLAN:
    - Primary metric: effective rank (eigenvalue-based dimensionality)
    - Paired analysis: Wilcoxon signed-rank on matched pairs (30 pairs)
    - Unpaired backup: Mann-Whitney U, Welch's t (30 vs 30)
    - Effect size: Cohen's d with bootstrap 95% CIs
    - Multiple scales: run at 1.1B, 3B, 7B minimum (14B if feasible)
    - Power: 30 pairs gives ~80% power to detect d >= 0.52 (paired test)
    - Bonferroni correction across scales

PROMPT DESIGN CONSTRAINTS:
    - Every word should appear in a standard 5,000-word English vocabulary
    - No invented words, no neologisms, no brand names
    - No proper nouns beyond top-100 most recognized globally
    - Numbers written as words where possible to avoid tokenizer artifacts
    - All prompts are declarative statements (no questions, no imperatives)
    - Length range: 10-18 words per prompt
"""


# ================================================================
# VALIDATION UTILITIES
# ================================================================

def validate_prompts():
    """Verify structural integrity of the prompt sets."""
    errors = []

    # Check counts
    if len(CONFABULATION_COMMON_TOKENS) != 30:
        errors.append(f"Expected 30 confab prompts, got {len(CONFABULATION_COMMON_TOKENS)}")
    if len(FACTUAL_COMMON_TOKENS) != 30:
        errors.append(f"Expected 30 factual prompts, got {len(FACTUAL_COMMON_TOKENS)}")

    # Check ID uniqueness
    confab_ids = [p["id"] for p in CONFABULATION_COMMON_TOKENS]
    fact_ids = [p["id"] for p in FACTUAL_COMMON_TOKENS]
    if len(set(confab_ids)) != len(confab_ids):
        errors.append("Duplicate confabulation IDs found")
    if len(set(fact_ids)) != len(fact_ids):
        errors.append("Duplicate factual IDs found")

    # Check cross-references
    fact_id_set = set(fact_ids)
    confab_id_set = set(confab_ids)
    for p in CONFABULATION_COMMON_TOKENS:
        if p["matched_factual_id"] not in fact_id_set:
            errors.append(f"{p['id']}: matched_factual_id '{p['matched_factual_id']}' not found")
    for p in FACTUAL_COMMON_TOKENS:
        if p["matched_confab_id"] not in confab_id_set:
            errors.append(f"{p['id']}: matched_confab_id '{p['matched_confab_id']}' not found")

    # Check length matching (within +/- 3 words)
    confab_by_id = {p["id"]: p for p in CONFABULATION_COMMON_TOKENS}
    fact_by_id = {p["id"]: p for p in FACTUAL_COMMON_TOKENS}
    length_diffs = []
    for cp in CONFABULATION_COMMON_TOKENS:
        fp = fact_by_id.get(cp["matched_factual_id"])
        if fp:
            c_len = len(cp["text"].split())
            f_len = len(fp["text"].split())
            diff = abs(c_len - f_len)
            length_diffs.append(diff)
            if diff > 3:
                errors.append(
                    f"Length mismatch: {cp['id']} ({c_len} words) vs "
                    f"{fp['id']} ({f_len} words) — diff={diff}"
                )

    # Check domain matching
    for cp in CONFABULATION_COMMON_TOKENS:
        fp = fact_by_id.get(cp["matched_factual_id"])
        if fp and cp.get("domain") != fp.get("domain"):
            errors.append(
                f"Domain mismatch: {cp['id']} ({cp.get('domain')}) vs "
                f"{fp['id']} ({fp.get('domain')})"
            )

    # Summary statistics
    confab_lengths = [len(p["text"].split()) for p in CONFABULATION_COMMON_TOKENS]
    fact_lengths = [len(p["text"].split()) for p in FACTUAL_COMMON_TOKENS]

    print("=" * 60)
    print("  C7 Prompt Validation Report")
    print("=" * 60)
    print(f"  Confab prompts:  {len(CONFABULATION_COMMON_TOKENS)}")
    print(f"  Factual prompts: {len(FACTUAL_COMMON_TOKENS)}")
    print(f"  Confab word count:  mean={sum(confab_lengths)/len(confab_lengths):.1f}, "
          f"range=[{min(confab_lengths)}, {max(confab_lengths)}]")
    print(f"  Factual word count: mean={sum(fact_lengths)/len(fact_lengths):.1f}, "
          f"range=[{min(fact_lengths)}, {max(fact_lengths)}]")
    if length_diffs:
        print(f"  Pair length diffs:  mean={sum(length_diffs)/len(length_diffs):.1f}, "
              f"max={max(length_diffs)}")
    print()

    # Domain distribution
    from collections import Counter
    confab_domains = Counter(p.get("domain", "unknown") for p in CONFABULATION_COMMON_TOKENS)
    fact_domains = Counter(p.get("domain", "unknown") for p in FACTUAL_COMMON_TOKENS)
    print("  Domain distribution (confab / factual):")
    for domain in sorted(set(list(confab_domains.keys()) + list(fact_domains.keys()))):
        print(f"    {domain:12s}: {confab_domains.get(domain, 0):2d} / {fact_domains.get(domain, 0):2d}")
    print()

    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  All checks passed.")
    print("=" * 60)

    return len(errors) == 0


if __name__ == "__main__":
    validate_prompts()
