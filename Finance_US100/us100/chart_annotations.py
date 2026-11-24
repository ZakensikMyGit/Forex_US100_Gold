"""Warstwy cenowe dla wykresu świecowego ECharts."""

from __future__ import annotations

import pandas as pd

from . import advisor, data, harmonics, price_structures, signals

LAYERS = {
    "entry": "Sygnał Doradcy",
    "harmonics": "Układy AB=CD",
    "volume": "Wolumen",
    "levels": "Wsparcie i opór",
    "fib": "Fibonacci H4",
    "pof": "POF · pomiar ostatniej fali",
    "zzb": "ZZB · zmiana roli poziomu",
    "candles": "Formacje świec",
}
DEFAULT_LAYERS = {"entry", "harmonics", "volume"}
ENTRY_SERIES_ID = "advisor_entry"


def candle_ends(view: pd.DataFrame, interval: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime([data.bar_end(interval, t) for t in view.index], utc=True))


def build(
    view: pd.DataFrame, interval: str, off: float, layers: set[str],
    levels: list[tuple[str, float]], frames: dict[str, pd.DataFrame], now: pd.Timestamp,
    patterns: list[tuple[harmonics.Projection, harmonics.Status]], proposal: advisor.Proposal | None,
    *, show_exit_levels: bool = False, pof: price_structures.WaveMeasurement | None = None,
    zones: tuple[price_structures.PolarityZone, ...] = (),
) -> tuple[dict, list[float], dict | None]:
    """Zwraca symbole, linie i strefy oraz ceny potrzebne do objęcia ich skalą wykresu."""
    lo, hi = float(view["low"].min()) + off, float(view["high"].max()) + off
    span = max(hi - lo, abs(hi) * 0.001, 0.01)
    ends = candle_ends(view, interval)
    lines, points, areas = [], [], []
    extent = []
    entry = None

    def nearby(price: float) -> bool:
        return lo - 0.25 * span <= price <= hi + 0.25 * span

    def slot(instant: pd.Timestamp, *, at_close: bool = False) -> int | None:
        # Czas zamknięcia należy do świecy kończącej się w tej chwili.
        index = int(ends.searchsorted(instant, side="left" if at_close else "right"))
        if index >= len(view):
            return None
        if interval != "D1" and instant < view.index[index]:
            return None
        if interval != "D1" and at_close and instant == view.index[index]:
            return None
        if interval == "D1" and ends[index] - instant > pd.Timedelta(hours=25):
            return None
        return index

    def horizontal(name: str, price: float, color: str, *, dashed: bool = True, side: str = "insideStartTop") -> None:
        lines.append({
            "name": name, "yAxis": round(float(price), 2), "lineStyle": {"color": color, "type": "dashed" if dashed else "solid"},
            "label": {"show": True, "position": side, "color": color, "formatter": "{b}  {c}"},
        })
        extent.append(price)

    def segment(a: tuple[int, float], b: tuple[int, float], color: str, *, projected: bool = False) -> None:
        lines.append([
            {"coord": list(a), "lineStyle": {"color": color, "width": 2, "type": "dashed" if projected else "solid"}},
            {"coord": list(b)},
        ])

    def marker(name: str, index: int, price: float, color: str, *, label: str | None = None,
               symbol: str = "circle", size: int = 14) -> None:
        point = {
            "name": name, "coord": [index, price], "value": round(price, 2),
            "symbol": symbol, "symbolSize": size, "itemStyle": {"color": color},
        }
        if label:
            point["label"] = {"show": True, "position": "top", "formatter": label, "color": color}
        points.append(point)

    if "levels" in layers:
        price = float(view["close"].iloc[-1]) + off
        clean = [(name, level) for name, level in levels if nearby(level)]
        supports = sorted((item for item in clean if item[1] <= price), key=lambda item: price - item[1])[:3]
        resistances = sorted((item for item in clean if item[1] > price), key=lambda item: item[1] - price)[:3]
        for label, color, group in (("Wsparcie", "#0f9c85", supports), ("Opór", "#e65363", resistances)):
            used = set()
            for name, level in group:
                if level not in used:
                    horizontal(f"{label} · {name}", level, color, dashed=False)
                    used.add(level)
        daily = signals.last_closed(frames["D1"], "D1", now)
        if daily is not None:
            for field, label, color in (("low", "Minimum sesji D1", "#0f9c85"), ("high", "Maksimum sesji D1", "#e65363")):
                level = float(daily[field]) + off
                if nearby(level):
                    horizontal(label, level, color)

    if "fib" in layers:
        wave = next(reversed(list(harmonics.confirmed_waves(frames["H4"], now))), None)
        if wave is not None:
            a, b = wave
            for ratio in (0.382, 0.5, 0.618, 0.786):
                price = b.price - ratio * (b.price - a.price) + off
                if nearby(price):
                    horizontal(f"Fibo H4 {ratio:.1%}".replace(".", ","), price, "#b77816", side="insideEndTop")

    if "pof" in layers and pof is not None:
        color = "#348dcf"
        anchors = []
        for label, pivot in (("Początek fali", pof.start), ("Koniec fali", pof.end)):
            index = int(view.index.get_indexer([pivot.time])[0])
            if index >= 0:
                price = pivot.price + off
                marker(f"POF · {label}", index, price, color, size=9)
                anchors.append((index, price))
                extent.append(price)
        if len(anchors) == 2:
            segment(anchors[0], anchors[1], color)
        for ratio, raw_price in pof.levels():
            if nearby(raw_price + off):
                label = f"POF {ratio * 100:g}%".replace(".", ",")
                horizontal(label, raw_price + off, color, dashed=ratio not in (0.5, 0.618), side="insideEndTop")

    if "zzb" in layers:
        for zone in zones:
            price = zone.pivot.price + off
            if not nearby(price):
                continue
            support = zone.pivot.high
            color = "#0d927c" if support else "#d76650"
            fill = "rgba(13,146,124,0.12)" if support else "rgba(215,102,80,0.12)"
            label = f"ZZB · {zone.role.lower()}"
            lower, upper = (value + off for value in zone.bounds)
            start = int(ends.searchsorted(zone.broken_at, side="left"))
            if start < len(view) - 1:
                areas.append([{"name": label, "xAxis": start, "yAxis": lower, "itemStyle": {"color": fill}},
                              {"xAxis": len(view) - 1, "yAxis": upper}])
                extent.extend((lower, upper))
            horizontal(label, price, color, dashed=zone.retested_at is None)
            if zone.retested_at is not None and (idx := slot(zone.retested_at, at_close=True)) is not None:
                marker(f"{label} · retest", idx, price, color, symbol="diamond", size=11)

    if "harmonics" in layers:
        for i, (projection, status) in enumerate(patterns):
            current = i == len(patterns) - 1
            active = current and status.stage in ("building", "reached")
            completed = status.stage == "rejected"
            color = "#a476f4" if active else ("#8757ce" if completed else "#80859a")
            path = []
            for label, pivot in zip("ABC", (projection.a, projection.b, projection.c)):
                idx = int(view.index.get_indexer([pivot.time])[0])
                price = pivot.price + off
                coord = (idx, price) if idx >= 0 else None
                path.append(coord)
                if coord is not None:
                    marker(f"AB=CD · punkt {label}", *coord, color, label=label if current else None, size=10)
                    extent.append(price)
            for left, right in zip(path, path[1:]):
                if left is not None and right is not None and left[0] < right[0]:
                    segment(left, right, color)
            target = projection.d + off
            if active and nearby(target):
                lower, upper = (p + off for p in projection.zone)
                areas.append([{"name": "Strefa D", "yAxis": lower}, {"yAxis": upper}])
                extent.extend((lower, upper))
                horizontal("AB=CD · projekcja D", target, color, side="insideEndTop")
                if path[-1] is not None and path[-1][0] < len(view) - 1:
                    segment(path[-1], (len(view) - 1, target), color, projected=True)
            elif completed:
                idx = slot(status.at, at_close=True)
                if idx is not None and nearby(target):
                    marker("AB=CD · D potwierdzony", idx, target, color, label="D", size=12)
                    extent.append(target)
                    if path[-1] is not None and path[-1][0] < idx:
                        segment(path[-1], (idx, target), color)

    if "candles" in layers:
        closed_count = int(ends.searchsorted(now, side="right"))
        for i in range(max(0, closed_count - 25), closed_count):
            candle = view.iloc[i]
            names = signals.candle_patterns(candle, view.iloc[i - 1] if i else None)
            if names:
                marker(", ".join(names), i, float(candle["high"]) + off, "#37aabf", symbol="diamond", size=10)

    if "entry" in layers and proposal is not None:
        idx = slot(proposal.at, at_close=True)
        if idx is not None:
            direction = proposal.level.direction.upper()
            price = proposal.entry + off
            color = "#0e9c68" if direction == "LONG" else "#e34b63"
            name = f"{direction} · {proposal.level.name} · R:R {proposal.rr:.2f}:1"
            entry = {
                "name": name, "value": [idx, price], "symbolRotate": 0 if direction == "LONG" else 180,
                "itemStyle": {"color": color},
                "label": {"show": True, "position": "top", "formatter": f"{direction} {price:.2f}", "color": color},
            }
            extent.append(price)
            if show_exit_levels:
                horizontal("SL", proposal.stop + off, "#e65363", side="insideEndTop")
                horizontal("TP", proposal.target + off, "#0e9c68", side="insideEndTop")

    return {
        "markLine": {"symbol": "none", "data": lines, "label": {"show": False, "fontSize": 10}},
        "markPoint": {"data": points, "label": {"show": False}, "tooltip": {"trigger": "item", "formatter": "{b}"}},
        "markArea": {"data": areas, "silent": True, "label": {"show": False}, "itemStyle": {"color": "rgba(164,118,244,0.14)"}},
    }, extent, entry


def recent_patterns(
    bars: pd.DataFrame, interval: str, now: pd.Timestamp, *, lookback: int,
) -> list[tuple[harmonics.Projection, harmonics.Status]]:
    """Ostatnia projekcja i najwyżej jeden wcześniejszy ukończony układ."""
    recent = bars.tail(lookback + 60)
    patterns = harmonics.projections(recent, now, interval=interval)
    if not patterns:
        return []
    latest = patterns[-1]
    status = harmonics.track(latest, recent, now, interval=interval)
    candidates = patterns[-8:]
    for earlier, following in reversed(list(zip(candidates, candidates[1:]))):
        earlier_status = harmonics.track(earlier, recent, following.c.confirmed_at, interval=interval)
        if earlier_status.stage == "rejected":
            return [(earlier, earlier_status), (latest, status)]
    return [(latest, status)]
