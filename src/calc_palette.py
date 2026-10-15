"""候选背景色对比度试算（用于配色标定，非交付功能）。

用法：
    python -m src.calc_palette
"""
from __future__ import annotations


def lum(h: str) -> float:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def f(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def cr(a: str, b: str) -> float:
    la, lb = lum(a), lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def main() -> None:
    print("候选浅色背景 vs 白色卡片（目标 ≥ 1.20）")
    for c in ["#eef0f4", "#ecedf2", "#e9eaf0", "#e8e9ee", "#e6e6eb", "#e6e8ef", "#e7e8ee", "#e5e7ee"]:
        print(f"  {c}   {cr(c, '#ffffff'):.3f}")

    print("\n候选深色背景 vs 卡片 #242428（目标 ≥ 1.20）")
    for c in ["#08080a", "#0a0a0d", "#0c0c10", "#101014"]:
        print(f"  {c}   {cr(c, '#242428'):.3f}")


if __name__ == "__main__":
    main()
