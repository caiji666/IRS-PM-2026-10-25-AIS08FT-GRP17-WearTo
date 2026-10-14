from pathlib import Path

from travel_agent.products.retrieval.upstream.adapter import adapt_files

fixture_dir = Path("tests/fixtures/inspiration")
output_dir = Path("tests/inspiration/outputs")

request, fallback, metadata = adapt_files(
    pt_path=output_dir / "phase_2_output.pt",
    text_path=fixture_dir / "expanded_fashion_context.txt",
    context_path=fixture_dir / "context.json",
    model_info_path=output_dir / "model_info.json",
)

print("Adapter status:", metadata["status"])
print("Model identity:", metadata["local_model_identity"])
print("Context embedding shape:", metadata["tensor_shapes"]["context_embedding"])
print("Image embeddings shape:", metadata["tensor_shapes"]["top_image_embeddings"])
print("Reference order:", metadata["reference_vector_order"])

assert metadata["status"] == "declared_compatible"
assert len(request.top_social_references) == 5
assert request.context_embedding is not None
assert all(ref.embedding is not None for ref in request.top_social_references)

print("\nStudent 2 → Student 3 handoff compatibility test passed.")
