def generate_search_queries(context):

    destination = context["destination"].strip()
    season = context["season"].strip()

    wcr = context.get("weather_clothing_requirements", {})
    asr = context.get("activity_style_requirements", {})
    user_styles = context.get("user_style", [])

    queries = []

    # 1. General user-style query
    if user_styles:
        style_text = " ".join(style.strip() for style in user_styles if style.strip())
        if style_text:
            queries.append(f"{destination} {season} {style_text} outfit")

    # 2. Activity-specific queries
    for activity, styles in asr.items():
        style_text = " ".join(style.strip() for style in styles if style.strip())
        queries.append(f"{destination} {season} {activity.strip()} {style_text} outfit")

    # 3. Weather-based outerwear queries
    outerwear = wcr.get("outerwear", [])
    for garment in outerwear:
        if garment.strip():
            queries.append(f"{destination} {season} {garment.strip()} outfit")

    # 4. Clothing combination query
    from itertools import product

    tops = wcr.get("top", [])
    bottoms = wcr.get("bottom", [])
    footwear = wcr.get("footwear", [])

    if tops and bottoms and footwear:
        for top, bottom, shoe in product(tops, bottoms, footwear):
            query = f"{destination} {season} {top} {bottom} {shoe} outfit"
            queries.append(query)

    # 5. Clean whitespace and remove duplicate queries and return
    queries = [" ".join(query.split()) for query in queries]
    return list(dict.fromkeys(queries))
