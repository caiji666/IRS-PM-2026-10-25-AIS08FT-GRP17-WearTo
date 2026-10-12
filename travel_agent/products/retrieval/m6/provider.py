"""Optional M1-compatible ranked provider; never replaces the existing M2 provider."""

from ..m5.images import load_image
from ..m5.models import vector
from ..schemas import Garment, ProductRetrievalResponse, RecordIssue
from .models import GroupedRankingRequest, RankingConfig, RankingRequest
from .ranking import rank, rank_by_category


def external_embedding(embedding, encoder, modality):
    if embedding.model != encoder.identity or len(embedding.values) != encoder.dimension:
        raise ValueError(
            "external embedding model/revision/dimension incompatible; no silent reuse"
        )
    return vector(embedding.model, modality, embedding.values)


def prepare_request(request, encoder, image_root, config=None):
    warnings = []
    if request.context_embedding is not None:
        context = external_embedding(request.context_embedding, encoder, "text")
    else:
        query = request.query if request.query is not None else request.context.query
        # Trip dates are not semantic fashion attributes. Supplied weather is used; never invented.
        text = ". ".join(
            x
            for x in [
                query,
                request.context.trip.destination,
                ", ".join(request.context.activities),
                ", ".join(request.context.style_preferences),
                request.context.weather_notes,
            ]
            if x
        )
        context = encoder.encode_texts([text])[0]
    social = []
    for reference in request.top_social_references:
        if reference.embedding is not None:
            social.append(external_embedding(reference.embedding, encoder, "image"))
        elif reference.image_path:
            try:
                social.append(
                    encoder.encode_images([load_image(reference.image_path, image_root)])[0]
                )
            except (OSError, ValueError) as exc:
                warnings.append(f"Social reference {reference.reference_id} image skipped: {exc}")
                if reference.description and reference.description.strip():
                    social.append(encoder.encode_texts([reference.description])[0])
                    warnings.append(
                        f"Social reference {reference.reference_id}: description text fallback used."
                    )
        elif reference.description and reference.description.strip():
            social.append(encoder.encode_texts([reference.description])[0])
            warnings.append(
                f"Social reference {reference.reference_id}: description text fallback used."
            )
        else:
            warnings.append(
                f"Social reference {reference.reference_id} has no usable local data; remote URL not fetched."
            )
    ranking = RankingRequest(
        context=context,
        social_references=social,
        config=config or RankingConfig(),
        categories=request.categories,
        size=request.size,
        currency=request.currency,
        max_price=request.max_price,
        limit=request.limit,
    )
    return ranking, warnings


class RankedProductProvider:
    def __init__(self, index, encoder, image_root, config=None, processing_issues=None):
        if index.model != encoder.identity or index.dimension != encoder.dimension:
            raise ValueError("encoder/index mismatch")
        self.index, self.encoder, self.image_root = index, encoder, image_root
        self.config = config or RankingConfig()
        self.processing_issues = processing_issues or []

    def retrieve_ranked(self, request):
        prepared, warnings = prepare_request(request, self.encoder, self.image_root, self.config)
        report = rank(self.index, prepared)
        report.warnings.extend(warnings)
        report.warnings.extend(i.reason for i in self.index.issues)
        return report

    def retrieve_grouped(self, request, k_per_category=5):
        prepared, warnings = prepare_request(request, self.encoder, self.image_root, self.config)
        grouped = GroupedRankingRequest.model_validate(
            {**prepared.model_dump(), "k_per_category": k_per_category}
        )
        report = rank_by_category(self.index, grouped)
        report.warnings.extend(warnings)
        report.warnings.extend(i.reason for i in self.index.issues)
        return report

    def retrieve(self, request):
        report = self.retrieve_ranked(request)
        # Select original Garment fields to satisfy the strict unchanged M1 contract.
        products = []
        for result in report.products:
            data = result.item.product.model_dump()
            fields = {key: data[key] for key in Garment.model_fields}
            fields["embedding"] = {
                "model": result.item.vector.model,
                "values": result.item.vector.values,
            }
            products.append(Garment.model_validate(fields))
        issues = [
            RecordIssue(record_index=i.record_index, reason=f"{i.stage}: {i.reason}")
            for i in self.processing_issues
        ]
        return ProductRetrievalResponse(
            products=products,
            total_matches=report.total_matches,
            wardrobe_garments=request.wardrobe_garments,
            pending_wardrobe_uploads=request.wardrobe_uploads,
            issues=issues,
            warnings=report.warnings,
        )
