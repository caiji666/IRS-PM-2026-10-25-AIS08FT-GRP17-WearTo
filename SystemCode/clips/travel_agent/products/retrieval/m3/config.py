"""Exact category aliases; no title inference or visual classification."""

CATEGORY_ALIASES = {
    "TOP": ["top", "shirt", "t-shirt", "t shirt", "tee", "blouse", "sweater", "上衣", "衬衫"],
    "BOTTOM": ["bottom", "trousers", "pants", "jeans", "shorts", "skirt", "裤子"],
    "FOOTWEAR": ["footwear", "shoes", "sneakers", "boots", "sandals", "鞋"],
    "OUTERWEAR": ["outerwear", "jacket", "coat", "blazer", "外套"],
}
COLOR_ALIASES = {"grey": "gray", "灰色": "gray", "黑色": "black", "白色": "white", "蓝色": "blue"}
SIZE_ALIASES = {
    "small": "S",
    "medium": "M",
    "large": "L",
    "extra large": "XL",
    "one size": "ONE SIZE",
}
STOCK_ALIASES = {
    "in stock": "IN_STOCK",
    "in_stock": "IN_STOCK",
    "out of stock": "OUT_OF_STOCK",
    "out_of_stock": "OUT_OF_STOCK",
    "unknown": "UNKNOWN",
}
