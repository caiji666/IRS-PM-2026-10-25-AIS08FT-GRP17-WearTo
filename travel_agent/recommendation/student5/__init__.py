"""Student 5 recommendation core: Stages 10, 12, 13 and 14 of the proposal pipeline.

Hard constraint filtering (Stage 10), final outfit ranking and the Top-3 RecommendationSet
(Stage 12), the user decision loop (Stage 13) and the AI-generated outfit preview integration
(Stage 14), plus the Student 4 -> Student 5 adapter and the stage orchestrator.

The public team entry point is `travel_agent.recommendation.service.run(payload)`. Importing
this package reads no environment variable or `.env`, opens no network connection and loads no
model; the optional real preview provider is only created explicitly.
"""
