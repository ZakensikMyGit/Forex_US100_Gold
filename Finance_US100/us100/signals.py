"""Opis stanu technicznego dla każdego interwału i sprawdzanie reguł poziomów."""

from __future__ import annotations

import math

import pandas as pd

from .data import bar_closed, bar_end

NO_DATA = "brak danych"
RULE_COLUMNS = ["Spełniona", "Reguła", "Warunek", "Świeca", "Ostatnie zamknięcie", "Dystans [pkt]", "Znaczenie"]


def last_closed(df: pd.DataFrame, interval: str, now: pd.Timestamp | None = None) -> pd.Series | None:
    """Ostatnia zamknięta świeca. None, gdy nie ma żadnej zamkniętej (np. jedna otwarta świeca)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    now = min(now, df.attrs.get("pobrano", now))
    for i in range(len(df) - 1, -1, -1):
        if bar_closed(interval, df.index[i], now):
            return df.iloc[i]
    return None


def _ok(*vals) -> bool:
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in vals)


def _vs(c: float, ref: float) -> str:
    return ("nad" if c > ref else "pod") if _ok(ref) else NO_DATA


def summarize(df: pd.DataFrame, interval: str) -> dict:
    """Stan techniczny na ostatniej (także otwartej) świecy. Brakujące wskaźniki
    przy krótkiej historii są opisywane jako "brak danych" zamiast błędnego porównania."""
    if df.empty:
        return {"Interwał": interval, "Trend": NO_DATA}
    last = df.iloc[-1]
    c = float(last["close"])
    ema9, ema26, ema50, ema200 = (float(last[x]) for x in ("ema9", "ema26", "ema50", "ema200"))

    # Trend: położenie ceny względem średnich i ich ułożenie
    if not _ok(ema9, ema26, ema50):
        trend = NO_DATA
    else:
        flags = [c > ema50, ema9 > ema26] + ([c > ema200] if _ok(ema200) else [])
        trend = "wzrostowy" if all(flags) else ("spadkowy" if not any(flags) else "mieszany")

    r = float(last["rsi"])
    if not _ok(r):
        rsi_txt = NO_DATA
    elif r < 30:
        rsi_txt = f"{r:.1f} wyprzedanie"
    elif r > 70:
        rsi_txt = f"{r:.1f} wykupienie"
    else:
        rsi_txt = f"{r:.1f} neutralnie"

    hist = df["macd_hist"].dropna()
    if len(hist) < 2:
        macd_txt = NO_DATA
    else:
        hist_now, hist_rising = hist.iloc[-1], hist.iloc[-1] > hist.iloc[-2]
        side = "nad sygnałem" if last["macd"] > last["macd_signal"] else "pod sygnałem"
        if hist_now < 0:
            momentum = "słabnie spadek" if hist_rising else "rośnie spadek"
        else:
            momentum = "rośnie wzrost" if hist_rising else "słabnie wzrost"
        macd_txt = f"{side}, {momentum}"

    bb_low, bb_up = float(last["bb_low"]), float(last["bb_up"])
    if not _ok(bb_low, bb_up):
        bb = NO_DATA
    elif c < bb_low:
        bb = "pod dolną wstęgą"
    elif c > bb_up:
        bb = "nad górną wstęgą"
    else:
        bb = "w kanale"

    atr = float(last["atr"])
    stretch = round((c - ema9) / atr, 2) if _ok(atr, ema9) and atr else None

    closed = bar_closed(interval, df.index[-1])
    return {
        "Interwał": interval,
        "Świeca": "zamknięta" if closed else "w trakcie",
        "Trend": trend,
        "Cena vs EMA 50 / 200": f"{_vs(c, ema50)} / {_vs(c, ema200)}",
        "EMA 9 vs 26": ("9 nad 26" if ema9 > ema26 else "9 pod 26") if _ok(ema9, ema26) else NO_DATA,
        "RSI 14": rsi_txt,
        "MACD": macd_txt,
        "Bollinger": bb,
        "Odległość od EMA 9 (ATR)": stretch,
    }


def trend_series(df: pd.DataFrame) -> pd.Series:
    """Ta sama ocena trendu co w summarize(), policzona dla każdej świecy."""
    import numpy as np

    c, e9, e26, e50, e200 = (df[x].to_numpy(dtype=float) for x in ("close", "ema9", "ema26", "ema50", "ema200"))
    f1, f2 = c > e50, e9 > e26
    has200 = np.isfinite(e200)
    f3 = c > e200
    all_up = f1 & f2 & np.where(has200, f3, True)
    none_up = ~f1 & ~f2 & np.where(has200, ~f3, True)
    out = np.where(all_up, "wzrostowy", np.where(none_up, "spadkowy", "mieszany")).astype(object)
    out[~(np.isfinite(e9) & np.isfinite(e26) & np.isfinite(e50))] = NO_DATA
    return pd.Series(out, index=df.index)


def level_hits(low, high, open_, prev_close, level: float):
    """Wspólna definicja dla monitora i odtwarzania historii (tablice w skali XTB).

    dotknięcie: poziom mieści się w zakresie low-high świecy;
    przeskok luką: świeca go nie dotknęła, a poziom leży między zamknięciem poprzedniej
    świecy a otwarciem tej świecy.
    """
    import numpy as np

    low, high, open_, prev_close = (np.asarray(x, dtype=float) for x in (low, high, open_, prev_close))
    touched = (low <= level) & (high >= level)
    with np.errstate(invalid="ignore"):
        gap = ~touched & (np.fmin(prev_close, open_) < level) & (np.fmax(prev_close, open_) > level) & np.isfinite(prev_close)
    return touched, gap


def rejects_level(open_, high, low, close, level, direction: str):
    """Otwarcie i zamknięcie po tej samej stronie, ekstremum po przeciwnej."""
    if direction == "long":
        return (open_ > level) & (low < level) & (close > level)
    return (open_ < level) & (high > level) & (close < level)


def candle_patterns(bar: pd.Series, previous: pd.Series | None = None) -> list[str]:
    """Nazwy formacji na zamkniętej świecy i jej poprzedniczce."""
    opening, high, low, close = (float(bar[col]) for col in ("open", "high", "low", "close"))
    span = high - low
    if span <= 0:
        return []
    body = abs(close - opening)
    lower = min(opening, close) - low
    upper = high - max(opening, close)
    found = []
    if body <= 0.1 * span:
        found.append("Doji")
    elif lower >= 2 * body and upper <= 0.15 * span:
        found.append("Długi dolny cień")
    elif upper >= 2 * body and lower <= 0.15 * span:
        found.append("Długi górny cień")
    if previous is not None:
        before_open, before_close = float(previous["open"]), float(previous["close"])
        if close > opening and before_open > before_close and opening <= before_close and close >= before_open:
            found.append("Objęcie hossy")
        elif close < opening and before_open < before_close and close <= before_open and opening >= before_close:
            found.append("Objęcie bessy")
    return found


def evaluate_rules(
    rules: list[dict], frames: dict[str, pd.DataFrame], offset: float, *, include_closed_at: bool = False,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reguły oceniane wyłącznie na zamkniętych świecach; bez zamkniętej świecy reguła jest pomijana."""
    rows = []
    for rule in rules:
        interval = rule.get("interwał", "H1")
        try:
            level = float(rule.get("poziom", 0) or 0)
        except (TypeError, ValueError):
            continue
        cond = rule.get("warunek", "zamknięcie poniżej")
        df = frames.get(interval)
        if df is None or not level:
            continue
        bar = last_closed(df, interval, now)
        if bar is None:
            continue
        close = float(bar["close"]) + offset
        hit = close > level if "powyżej" in cond else close < level
        rows.append(
            {
                "Spełniona": hit,
                "Reguła": rule.get("nazwa", ""),
                "Warunek": f"{cond} {level:,.0f} ({interval})".replace(",", " "),
                "Świeca": f"{bar.name:%Y-%m-%d %H:%M}" if interval != "D1" else f"{bar.name:%Y-%m-%d}",
                "Ostatnie zamknięcie": round(close, 1),
                "Dystans [pkt]": round(level - close, 1),
                "Znaczenie": rule.get("znaczenie", ""),
            }
        )
        if include_closed_at:
            rows[-1]["Czas zamknięcia"] = bar_end(interval, bar.name)
    columns = [*RULE_COLUMNS, "Czas zamknięcia"] if include_closed_at else RULE_COLUMNS
    return pd.DataFrame(rows, columns=columns)
