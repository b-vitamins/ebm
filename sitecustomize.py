import re
try:
    import pydantic.color
    if not hasattr(pydantic.color, "r_rgba"):
        pydantic.color.r_rgba = re.compile(".*")
except Exception:
    pass
