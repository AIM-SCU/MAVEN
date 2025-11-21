from itertools import permutations
import pandas as pd
import json

# ───────────────────  CONFIG  ────────────────────
cultures = ["Chinese", "American", "Romanian"]

actions = {
    "Chinese": {
        "food":  ["eating Peking duck", "eating mooncakes", "eating dumplings"],
        "music": ["playing guzheng", "playing erhu", "playing dizi"],
        "dance": ["dancing fan dance", "dancing ribbon dance", "dancing umbrella dance"],
    },
    "American": {
        "food":  ["eating hot dogs", "eating burgers", "eating pizza slice"],
        "music": ["playing banjo", "playing electric guitar", "playing saxophone"],
        "dance": ["dancing hip-hop", "dancing moonwalk", "dancing tap dance"],
    },
    "Romanian": {
        "food":  ["eating sarmale", "eating mici", "eating mămăligă"],
        "music": ["playing nai", "playing cobză", "playing țambal"],
        "dance": ["dancing Hora", "dancing Sârba", "dancing Brâul"],
    },
}

locations = {
    "Chinese":  ["the Forbidden City", "West Lake", "the Potala Palace"],
    "American": ["the Statue of Liberty", "the Grand Canyon", "Mount Rushmore"],
    "Romanian": ["Bran Castle", "the Palace of Parliament", "the Wooden Churches of Maramureș"],
}
# ─────────────────────────────────────────────────


def build_prompts(experiment_tag: str = "pal_template_v1") -> pd.DataFrame:
    """Return a dataframe with same-culture + mixed-culture prompts plus
    explicit person/action/location segments."""
    rows = []

    def add_row(*, alignment, person_culture, action_culture, location_culture,
                action_category, action_text, location_text):
        # Derive prompt pieces
        person_text = f"a {person_culture} person"
        prompt_text = f"{person_text} {action_text} at {location_text}."

        rows.append(
            {
                # metadata
                "alignment": alignment,
                "person_culture": person_culture,
                "action_culture": action_culture,
                "location_culture": location_culture,
                "action_category": action_category,
                "experiment_tag": experiment_tag,
                # segments
                "person_segment": person_text,
                "action_segment": action_text,
                "location_segment": location_text,
                # full prompt
                "prompt": prompt_text,
            }
        )

    # 1️⃣  same-culture combinations
    for c in cultures:
        for cat, acts in actions[c].items():
            for act in acts:
                for loc in locations[c]:
                    add_row(
                        alignment="same_culture",
                        person_culture=c,
                        action_culture=c,
                        location_culture=c,
                        action_category=cat,
                        action_text=act,
                        location_text=loc,
                    )

    # 2️⃣  mixed-culture permutations
    for pc, ac, lc in permutations(cultures, 3):
        for cat, acts in actions[ac].items():
            for act in acts:
                for loc in locations[lc]:
                    add_row(
                        alignment="mixed_culture",
                        person_culture=pc,
                        action_culture=ac,
                        location_culture=lc,
                        action_category=cat,
                        action_text=act,
                        location_text=loc,
                    )

    return pd.DataFrame(rows)


# ─── build & save ─────────────────────────────────
df = build_prompts()

df.to_json("/workspace/t2v_self/iter_t2v/templates/pal_prompts_v1.jsonl", orient="records", lines=True, force_ascii=False)

summary = {
    "total_prompts": len(df),
    "by_alignment": df["alignment"].value_counts().to_dict(),
    "by_action_category": df["action_category"].value_counts().to_dict(),
}

with open("/workspace/t2v_self/iter_t2v/templates/pal_prompt_summary_v1.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("Saved jsonl & summary.")
