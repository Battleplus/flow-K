"""Matplotlib font helpers used by chart-rendering endpoints."""

from pathlib import Path

import matplotlib
import matplotlib.font_manager as font_manager


FONT_FAMILIES = [
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "SimHei",
    "DejaVu Sans",
]

FONT_PATHS = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path.home() / ".local/share/fonts/NotoSansCJK-Regular.ttc",
    Path.home() / ".local/share/fonts/NotoSansCJK-Bold.ttc",
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
]


def configure_chinese_font() -> font_manager.FontProperties:
    """Configure Matplotlib for Chinese labels without assuming a platform."""
    font_prop = None
    font_name = None

    for path in FONT_PATHS:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            font_prop = font_manager.FontProperties(fname=str(path))
            font_name = font_prop.get_name()
            break

    if font_prop is None:
        for family in FONT_FAMILIES:
            try:
                found = font_manager.findfont(
                    font_manager.FontProperties(family=family),
                    fallback_to_default=False,
                )
            except ValueError:
                continue
            if found:
                font_prop = font_manager.FontProperties(fname=found)
                font_name = font_prop.get_name()
                break

    if font_prop is None:
        font_prop = font_manager.FontProperties()

    sans_serif = ([font_name] if font_name else []) + FONT_FAMILIES
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = sans_serif
    matplotlib.rcParams["axes.unicode_minus"] = False
    return font_prop
