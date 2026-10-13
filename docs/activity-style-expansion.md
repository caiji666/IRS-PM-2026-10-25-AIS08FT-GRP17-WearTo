# Activity styles and dressing needs — v0.2.0

The KG returns alternative style directions and independently scoped dressing needs. Item selection, outfit construction and final feasibility belong to downstream modules.

## Implemented scope

- 16 activities, 22 Style nodes (10 aesthetics, 11 dress-code directions, 1 outdoor functional direction), 34 attributes, 90 nodes, 203 edges and 13 sources.
- The original 35 assertions and three source records are preserved. New assertions remain pending. Existing IDs and node/edge types retain their meaning.
- All 11 Place KG activity IDs now resolve to an Activity KG node. Shared identity is a `place_activity_id` plus explicit aliases; importing the two seeds does not automatically create cross-graph relationships.
- New activities: theme park, religious visit, beach visit, photography outing, food market, shopping, aquarium visit, nightlife/fine dining, cafe visit and travel transit. Nightlife/fine dining is not silently merged with ordinary dinner.
- New aesthetic directions: minimalist, maximalist, preppy, streetwear, artistic, bohemian, utility, romantic, retro and sporty.
- New explicit dress-code interpretations: dressy casual, black tie, creative black tie, black tie optional, festive and white tie. Smart casual and dressy casual remain distinct. Wedding/dinner/business/nightlife queries still require explicit context where configured.

## Evidence and acquisition plan

| Research question | Primary material read | What it supports |
|---|---|---|
| What does a supplied dress code mean? | [Emily Post attire guide](https://emilypost.com/advice/attire-guide-dress-codes-from-casual-to-white-tie) | Dress-code interpretation; not the claim that every dinner or wedding uses it |
| Which functional needs apply to ordinary hiking? | [REI hiking clothing guide](https://www.rei.com/learn/expert-advice/how-to-choose-hiking-clothes.html) | Mobility, moisture management, durability and trail traction |
| How can restrained and expressive aesthetics be described? | [FIT Minimalism/Maximalism](https://exhibitions.fitnyc.edu/minimalism-maximalism/) | Visual descriptors grouped by alternative style |
| What characterizes collegiate classic references? | [FIT Ivy Style](https://sites.fitnyc.edu/depts/museum/Ivy_Style/) | Preppy/Ivy references, not workplace requirements |
| How can contemporary streetwear be described? | [V&A Samuel Ross](https://www.vam.ac.uk/articles/fashion-design-samuel-ross-a-cold-wall) | One example combining sports, industrial and tailored references |
| How do artistic/bohemian references enter dress? | [FIT Art X Fashion](https://www.fitnyc.edu/museum/exhibitions/art-fashion/index.php) | Art and historical bohemian references |
| How does utility become an aesthetic? | [FIT Beyond Utility](https://www.fitnyc.edu/museum/exhibitions/beyond-utility/index.php) | Workwear/function-inspired design; appearance does not certify performance |
| How can romantic references be represented? | [FIT Ravishing](https://www.fitnyc.edu/museum/exhibitions/ravishing-the-rose-in-fashion.php) | Optional romantic/floral references across genders |
| What does retro mean? | [FIT RetroSpective](https://sites.fitnyc.edu/depts/museum/RetroSpective/) | Historical silhouette references, not vintage authenticity |
| How do sporting references relate to casual fashion? | [FIT Sporting Life](https://sites.fitnyc.edu/depts/museum/sporting_life/home.html) | Sports-inspired aesthetic, distinct from sport-specific function |

Access date: 2026-10-13. Only short paraphrases, source URLs, sections, scope and reading status are retained. Sources define concepts; **activity-to-aesthetic suitability edges are project suggestions**, not claims established by these sources. Descriptors abstract examples and are not exhaustive definitions. Dress-code sources mainly describe Western etiquette, not global cultural norms.

The acquisition register is `student1/research/style_expansion_2026-10-13.json` in the parent workspace. Local maintenance rules and controlled IDs are registered in `student1/kg/KG_SPEC.md` and `vocabulary.json`. Place KG facts, rules and item tables were not expanded in this batch.

## Query and output

```bash
.venv/bin/python -m travel_agent.understanding.activity_engine hiking \
  --preferred-style minimalist --allow-draft
.venv/bin/python -m travel_agent.understanding.activity_engine wedding \
  --dress-code 'creative black tie' --allow-draft
```

The first query returns outdoor, minimalist, sporty and utility directions; independent needs include mobility, moisture management, breathability, durability and traction. Each aesthetic has its own descriptors and evidence. Choosing minimalist does not remove functional needs.

New fields:

- `style_candidates`: style ID/name/kind/description, grouped descriptors, matched preferences and evidence paths.
- `dressing_requirements`: activity/dress-code needs, should/nice strength, scope and evidence paths.
- `preference_match`: exact vocabulary/alias matches and unmatched original inputs. Unsupported preferences are retained as unmatched, not invented as compatible styles.
- `place_activity_id`: explicit identity bridge to the Place KG vocabulary.

The shared service exposes `activity_style_options` and `activity_dressing_requirements` by input activity. Legacy `activity_style_requirements` remains limited to occasion/function directions and needs. Retrieval text labels aesthetic directions as alternatives and uses matched preferences when available. Do not flatten the candidate list into simultaneous hard constraints.

The seven full examples in `examples/understanding/activity_style_options.json` are executable draft demonstrations, not independent evaluation results. New English style vocabulary is recognized by the source-evidence rules; this does not prove that raw Qwen reliably extracts new activities or preferences. New activities can be passed through structured requests.

## Next improvements

1. Human review a small activity/style subset. Review style definitions separately from suggested activity compatibility; preserve conditional dress codes and distinguish functional needs from aesthetic descriptors.
2. Author an independent scenario set covering activity, venue constraints and style preference combinations. Measure inappropriate suggestions and missing needs, not just graph reachability.
3. Add more evidence-backed needs by activity: walking intensity/duration, environment and explicit host policies. Register new condition keys before using them; do not guess weather or universal religious-site rules.
4. Enrich style descriptions after checking additional museum/education sources, using optional descriptors rather than mandatory palettes, gender assumptions or fixed item recipes.
5. Validate the regenerated Activity Neo4j seed against a real development instance. The optional backend still uses Python reasoning plus database path verification; this batch does not claim live database validation.

All new knowledge remains draft. Default reviewed-only queries exclude pending assertions. No new item nodes or item mappings were added.
