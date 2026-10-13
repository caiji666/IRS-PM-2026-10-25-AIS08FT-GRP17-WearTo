import json
import torch
from pathlib import Path

from travel_agent.inspiration.handoff import save_inspiration_handoff
from travel_agent.products.retrieval.upstream.adapter import adapt_files


# Existing standalone test files
fixture_dir = Path("tests/fixtures/inspiration")
output_dir = Path("tests/inspiration/outputs")

# Load previously generated Phase 2 output
phase2_output = torch.load(
    output_dir / "phase_2_output.pt",
    map_location="cpu",
    weights_only=True
)

expanded_context = (
    fixture_dir / "expanded_fashion_context.txt"
).read_text(encoding="utf-8").strip()

structured_context = json.loads(
    (fixture_dir / "context.json").read_text(encoding="utf-8")
)

# Generate Student 3 handoff files
handoff = save_inspiration_handoff(
    phase2_output=phase2_output,
    expanded_context=expanded_context,
    structured_context=structured_context
)

print("\nGenerated Student 3 Handoff:")
print(json.dumps(handoff, indent=4))


# Resolve paths using Student 3's configuration
from travel_agent.products.retrieval.integration import Settings

settings = Settings.from_environment()

upstream = handoff["upstream"]

pt_path = settings.input_path(upstream["pt"])
text_path = settings.input_path(upstream["text"])
context_path = settings.input_path(upstream["context"])
model_info_path = settings.input_path(upstream["model_info"])

# Verify all four files exist
for path in [pt_path, text_path, context_path, model_info_path]:
    assert path.is_file(), f"Missing handoff file: {path}"

print("\nAll four handoff files created successfully.")


# Validate through Student 3's actual adapter
request, fallback, metadata = adapt_files(
    pt_path=pt_path,
    text_path=text_path,
    context_path=context_path,
    model_info_path=model_info_path
)

assert metadata["status"] == "declared_compatible"
assert len(request.top_social_references) == len(phase2_output["top_images"])
assert request.context_embedding is not None

assert all(
    ref.embedding is not None
    for ref in request.top_social_references
)

# Verify original ranking order is preserved
expected_ids = [
    image["image_id"]
    for image in phase2_output["top_images"]
]

assert metadata["reference_vector_order"] == expected_ids

print("Model compatibility:", metadata["status"])
print("Reference order:", metadata["reference_vector_order"])
print("\nStudent 2 -> Student 3 file-based handoff test passed.")