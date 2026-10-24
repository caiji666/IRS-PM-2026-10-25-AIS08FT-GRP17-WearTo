"""Atomic proposal-style request patches and latest-round outfit confirmation."""

from travel_agent.schemas import ConversationState, TripShoppingRequest, TripShoppingRequestPatch

LIST_FIELDS = {"activities", "style", "excluded_items", "owned_item_ids", "places"}


def apply_patch(request: TripShoppingRequest, patch: TripShoppingRequestPatch):
    values = request.model_dump()
    values.update(patch.set)
    for key, additions in patch.add.items():
        values[key] = list(dict.fromkeys([*values[key], *additions]))
    for key, removals in patch.remove.items():
        if key in LIST_FIELDS:
            values[key] = [v for v in values[key] if v not in removals] if removals else []
        else:
            values[key] = {} if key == "dress_codes" else None
    return TripShoppingRequest.model_validate(values)


def update_state(state: ConversationState, request: TripShoppingRequest):
    updated = state.model_copy(deep=True)
    if request != state.current_request:
        updated.current_request = request
        updated.request_version += 1
        updated.selected_outfit_id = None
        updated.last_recommended_outfit_ids = []
        updated.session_status = "COLLECTING_REQUIREMENTS"
    return updated


def confirm_outfit(state: ConversationState, outfit_id=None, display_position=None):
    if state.session_status != "AWAITING_CONFIRMATION":
        raise ValueError("No current recommendation round is awaiting confirmation")
    ids = state.last_recommended_outfit_ids
    if display_position is not None:
        if type(display_position) is not int or not 1 <= display_position <= len(ids):
            raise ValueError("Choose a position in the latest displayed recommendation round")
        resolved = ids[display_position - 1]
        if outfit_id is not None and outfit_id != resolved:
            raise ValueError("Outfit ID and display position disagree")
        outfit_id = resolved
    if outfit_id is None or outfit_id not in ids:
        raise ValueError("Choose an outfit from the latest displayed recommendation round")
    updated = state.model_copy(deep=True)
    updated.selected_outfit_id = outfit_id
    updated.session_status = "CONFIRMED"
    return updated
