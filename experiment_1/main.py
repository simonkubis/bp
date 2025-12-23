never_use = []
could_use = []
always_use = []

neutral_keywords = [
    "see also",
    "used with",
    "apply before",
    "apply after",
    "related to",
    "related pattern",
    "similar pattern",
    "alternative to",
    "alternatively used pattern"
]

negative_keywords = [
    "do not use",
    "never use",
    "avoid using",
    "not recommended",
    "not to be used"
]

positive_keywords = [
    "always use",
    "must use",
    "recommended",
    "best practice",
    "should use",
    "this pattern uses",
    "pattern is an alternative"
]

KEYWORD_WEIGHTS = {
    "negative": 0.0,
    "neutral": 0.5,
    "positive": 1.0
}
