# User Requirement Understanding and Contextual Reasoning

## System Development & Implementation

The project implemented a travel requirement understanding and contextual reasoning module that converts natural-language requests into structured, editable and traceable clothing requirements. The module combines a pretrained language model, deterministic rules and knowledge graphs to provide a consistent input for downstream inspiration retrieval, product search and outfit recommendation.

### Requirement Understanding and Dialogue Management

A locally deployed Qwen3-0.6B model extracts the destination, travel dates, budget, currency, activities and style preferences. Model outputs are checked using Pydantic schemas, business rules and evidence from the user's original message. User-provided facts are stored separately from inferred requirements to reduce the risk of introducing unsupported preferences or contextual information. Missing, ambiguous or invalid information triggers a focused clarification while preserving details already collected.

The dialogue manager distinguishes proposed information from confirmed requirements. Users can provide details incrementally and revise existing requirements through set, add and remove operations. Unspecified fields are retained, and invalid updates do not overwrite the previous state. Price preferences are also distinguished from budget limits: a request for a cheaper option does not create an arbitrary new budget.

Date handling supports standard numeric dates, slash-separated dates, abbreviated English months, Chinese date expressions and selected relative expressions. Users can supply a missing year without repeating the entire request. Ambiguous dates require clarification rather than silent interpretation. Activities and styles are optional; the system can offer knowledge-graph-supported choices, but suggestions become request facts only after explicit selection. These interactions were integrated into the team's web interface through multiple-selection controls, selection feedback and trip confirmation.

### Weather Enrichment and Knowledge-Based Reasoning

The weather component integrates Open-Meteo geocoding and forecasts with deterministic rules that produce clothing categories and functional requirements, including insulation, layering and rain protection. For trips outside or crossing the forecast window, the system uses explicitly labelled historical climate references rather than presenting historical statistics as trip-date forecasts.

The historical dataset contains 180 monthly summaries for 15 cities across 12 countries, derived from ERA5 daily data for 2015–2024. A shareable JSON snapshot and a local SQLite database retain location, provenance and statistical-method information. Country-level inputs require city clarification, and unsupported cities have an explicit data acquisition path. Unavailable data remains unknown rather than being replaced by assumed clear weather or a neighbouring city's conditions.

The Activity Knowledge Graph contains 16 activity types and 22 Style nodes, with 90 nodes and 203 relationships overall. Bounded one-hop and multi-hop traversal connects activities, occasions, formality, styles and attributes while retaining conditions and supporting evidence. Alternative aesthetics are represented separately from functional clothing needs. Activities such as dining require additional dress-code context where appropriate, rather than implying a universal level of formality.

A complementary Place Knowledge Graph covers 100 places across eight countries and represents venue requirements, activity choices and conditions such as entering an interior space. Its 753 base records are exported as 601 nodes and 1,214 relationships, including aliases and provenance structures. Outputs from both graphs are combined with weather requirements and user preferences to produce structured context and an ExpandedFashionContext representation for downstream retrieval.

### Knowledge Review and Traceability

Knowledge maintenance distinguishes source-supported statements from project heuristics and preserves applicability conditions, source reading levels and review status. The project owner completed the review of the existing knowledge, and approval was recorded for the activity nodes, relationships and sources, together with all 753 Place Knowledge Graph records.

The review process retains record-level snapshots, reviewer identity, timestamps, content fingerprints and specification fingerprints. Validation precedes the application of review decisions, and backups support recovery. Dependency checks prevent approval of a source or place from automatically approving every associated assertion. Changes to reviewed content invalidate the corresponding approval and require further review. Runtime snapshots and export files are kept consistent with the reviewed data.

The system supports both local Python reasoning and a Neo4j backend for the Activity Knowledge Graph. Python performs conditional reasoning, while Neo4j provides storage and bounded-path verification. Live queries confirmed agreement between the two backends for the evaluated scenarios. The Place Knowledge Graph currently uses local reasoning.

## Findings and Discussion

### Evaluation Dataset Design

Evaluation was organised into complementary layers covering language understanding, dialogue state, knowledge reasoning, weather processing and interface behaviour. Test cases included incomplete information, ambiguity, negation, invalid inputs and successive requirement changes, alongside complete requests.

| Evaluation set | Design and coverage |
|---|---|
| Multilingual requirement understanding set | 32 scenarios comprising 40 turns: 25 English, 13 Chinese and two mixed-language turns. Covers extraction, missing information, currency ambiguity, cross-year dates, negation, exclusions, invalid inputs and three continuous dialogues. |
| English development subset | 25 English turns selected from the multilingual set, used for development and regression evaluation of extraction, updates, explanation requests and confirmation. |
| Supplementary English diagnostic set | 16 additional turns covering new destinations and styles, monetary precision, missing currencies, date changes, exclusion removal and ambiguous confirmation. |
| Dialogue controller scenario set | Six scenarios comprising 19 turns, covering date formats, year-only clarification, ambiguous-date protection, incremental information collection, activity selection and optional-field skipping. |
| Place reasoning scenario set | 16 scenarios with predefined expectations for seasonal differences, indoor and outdoor settings, entry conditions, venue requirements and missing information. |

Raw model outputs and outputs after deterministic processing were scored separately. Full correctness required agreement on expected facts, empty fields, intent, validation behaviour and state transitions. This prevented numerous unchanged fields from masking a missed update or an unsupported addition. Safe rejection of invalid inputs was recorded separately from successful extraction.

Single-turn evaluation supplied the expected previous state to isolate extraction errors. Continuous dialogue evaluation instead carried forward the actual generated state, allowing errors to accumulate. The English development subset includes two of the multilingual set's three dialogues; these are overlapping observations, not additional independent examples. Model versions and prompt and dataset fingerprints were retained to support reproducibility.

### Development Progress and Metric Changes

Early evaluation identified omitted facts, invented preferences, incorrect handling of negation and full-request responses where a local update was required. Development therefore focused on prompt refinement, source-evidence checks, vocabulary normalisation, patch-based updates and explicit confirmation. Date normalisation and dialogue controls addressed additional interaction problems.

| Dataset | Initial raw model | Initial rule-augmented system | Improved raw model | Improved rule-augmented system |
|---|---:|---:|---:|---:|
| Multilingual understanding set | 3/40 | 14/40 | 4/40 | **23/40** |
| English development subset | 4/25 | 21/25 | 4/25 | **22/25** |
| Supplementary English diagnostic set | 1/16 | 15/16 | 1/16 | **16/16** |

Full correctness on the multilingual set increased from 35.0% to 57.5%, a gain of 22.5 percentage points. The English development subset reached 88.0%, while the supplementary English diagnostic set reached 100%. These results indicate improved handling of the covered cases, principally through deterministic evidence checks and state management. The much lower raw-model scores show that this improvement should not be attributed to model training or fine-tuning.

The number of continuous dialogues reaching the expected final state increased from zero of three to two of three. Both English dialogues succeeded, while the Chinese dialogue remained unsuccessful. Follow-up intent classification and intended changed-field correctness were both 8/8 in the English development subset; the corresponding results were 7/7 and 5/5 in the supplementary English set. Neither English evaluation recorded additional changes to fields the user had not requested to modify.

All six dialogue controller scenarios, comprising 19 turns, met their predefined expectations. These scenarios used real-model extraction with deterministic reference responses, assessing requirement and state correctness rather than the naturalness of generated wording.

### Functional and Knowledge Review Validation

| Validation | Result | Interpretation |
|---|---|---|
| Shared-repository automated regression suite | **613 passed; 63 skipped** | Tests code and interface behaviour; skipped cases are excluded from passed results. |
| Activity Knowledge Graph source-project tests | **14/14 passed** | Covers scenarios, paths, conditions, review filtering and export consistency. |
| Place Knowledge Graph source-project tests | **60/60 passed** | Includes review persistence, dependency checks, stale approvals, re-review, conflict detection, rollback and recovery. |
| Place reasoning scenarios | **16/16 passed** | Executed through the command-line interface with draft knowledge disabled; checks required outputs, prohibited inferences and missing-information responses. |
| Neo4j and local reasoning comparison | **4/4 matched** | Evaluates hiking, museum visits and dining under different dress-code and traversal-depth settings; retained paths passed live Cypher verification. |
| Strict review-chain validation | **Complete chains for 12 requirements across three scenarios** | Checks the dependencies supporting each result rather than an isolated approval flag. |
| Weather processing checks | **2/2 passed** | Verifies a live city forecast, a future-trip historical lookup and the distinction between the two data types. |

Review completion also exposed test assumptions that all knowledge remained pending. These tests were revised to use explicit pending-review fixtures, preserving checks that unreviewed or rejected knowledge is excluded. Additional checks verified that approval did not change underlying facts and that review records, runtime snapshots and file fingerprints remained consistent.

### Findings and Limitations

The principal finding is that structured generation alone is insufficient for reliable requirement understanding. Raw-model performance remained substantially below the rule-augmented system. Output formatting also did not improve consistently: strict JSON objects in the multilingual set decreased from 30/40 to 8/40, although a parser accepting a single code fence could still parse all 40 outputs. The prompt changed from a general extraction configuration to an English-focused configuration, so the overall improvement should not be interpreted as uniform progress across all metrics or attributed to a single component.

English improvements have not transferred reliably to Chinese. None of the 13 Chinese turns met the strict full-correctness criterion, including one invalid-input case that was correctly rejected. Remaining errors involved currency omission, activity normalisation, unsupported exclusions, update intent and dialogue-state evolution. English is therefore the primary supported workflow.

Safe rejection and extraction success must also remain distinct. Three strict failures in the English development subset involved an impossible date, a negative budget and unsupported currency precision. The system rejected these inputs correctly but did not retain a complete candidate satisfying the extraction criterion. Validation-outcome correctness was therefore 25/25, while full correctness remained 22/25.

Knowledge approval improves accountability but does not establish professional consensus. Review was completed by the project owner, without a measured independent two-reviewer or expert agreement study. Of the Place Knowledge Graph's 62 sources, seven retain a search-excerpt-only classification, 30 record relevant-section reading, 24 record page-text reading and one is a local project policy. Approval does not upgrade these evidence levels or convert project heuristics into universally applicable facts.

The evaluation primarily demonstrates correctness within the development scenarios. Synthetic labels still require independent human review, and previously inspected datasets cannot establish held-out generalisation. Dataset overlap and the absence of a controlled ablation study also limit conclusions about the individual contribution of prompts, rules and knowledge graphs. Field-level F1, independent reasoning-quality ratings and reviewer agreement remain areas for further evaluation.

Finally, component validation does not establish complete recommendation quality. The 63 skipped automated cases largely reflect missing dependencies or disabled live integration and paid-preview tests. Weather thresholds remain project heuristics; the ten-year historical reference is neither a standard 30-year climate normal nor a prediction for specific travel dates. City-level weather also cannot directly represent mountain or indoor conditions. Final outfit quality and user satisfaction require separate evaluation.

## Differences from the Proposal

The following points are provided separately for integration into the overall project report.

| Area | Implementation difference or clarification |
|---|---|
| Language model | Qwen3-0.6B follows the Stage 1 design. LoRA was optional and was not implemented. |
| Requirement understanding | The implementation explicitly combines model extraction, source-evidence rules, schema validation and user confirmation. Its performance should not be described as raw-model performance. |
| Mandatory information | Compared with the early demonstration in Section 7.2, activities are optional. The workflow can proceed once destination, start and end dates, budget and currency are available. |
| Dates and interaction | Added flexible date parsing, year-only clarification, activity and style suggestions, multiple selection and skipping. |
| Historical weather | Implemented the planned fallback through a provenance-preserving database, coverage management and an explicit acquisition workflow. |
| Place Knowledge Graph | Extended the original Activity Knowledge Graph with place-specific requirements and contextual conditions. |
| Neo4j | Supports both local reasoning and Neo4j. Python performs conditional reasoning; Neo4j provides storage and path verification. |
| Human review | Existing knowledge has project-owner approval and traceable review records; independent reviewer agreement and expert validation have not been measured. |
| Evaluation | Implemented full-correctness, formatting, intent, changed-field, dialogue-state, path-consistency and scenario checks. Field-level F1, independent human quality ratings and held-out generalisation evaluation remain incomplete. |
| Weather coverage | Some attributes, including humidity, independent validation of thresholds and special-environment handling remain limited. |
