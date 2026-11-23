"""Widok panelu analizy technicznej dla jednego instrumentu (US100, złoto)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import backtest, data, delivery, engine, harmonics, indicators as ind, journal, notify, paper, plan, risk, signals, storage
from .instruments import Instrument


@st.cache_data(ttl=300, show_spinner="Pobieram notowania…")
def get_frames(ticker: str) -> dict[str, pd.DataFrame]:
    return engine.load_frames(ticker)


@st.cache_data(ttl=300, show_spinner="Pobieram świece do wykresu…")
def get_intraday_chart(ticker: str, interval: str) -> pd.DataFrame:
    return ind.add_indicators(data.load_intraday(ticker, interval))


@st.cache_data(ttl=300, show_spinner=False)
def get_quote(ticker: str) -> tuple[float, float]:
    return data.last_value(ticker)


def _num(x: float, dec: int = 0) -> str:
    return f"{x:,.{dec}f}".replace(",", " ")


def _table(rows: list[dict], kind: str) -> pd.DataFrame:
    """Tabela do edytora - z kolumnami także wtedy, gdy lista jest pusta."""
    return pd.DataFrame(rows, columns=storage.COLUMNS[kind])


def build_chart(
    df: pd.DataFrame, interval: str, n: int, off: float, levels: list[tuple[str, float]], name: str,
    abcd: harmonics.Projection | None = None,
) -> dict:
    view = df.iloc[-n:].copy()
    for col in ("open", "high", "low", "close", "ema9", "ema26", "ema50", "ema200", "bb_up", "bb_low"):
        view[col] = view[col] + off
    fmt = "%Y-%m-%d" if interval == "D1" else "%m-%d %H:%M"
    view["czas"] = [t.strftime(fmt) for t in view.index]
    view["hist_plus"] = view["macd_hist"].where(view["macd_hist"] >= 0)
    view["hist_minus"] = view["macd_hist"].where(view["macd_hist"] < 0)
    src = view[
        ["czas", "open", "close", "low", "high", "ema9", "ema26", "ema50", "ema200", "bb_up", "bb_low",
         "rsi", "macd", "macd_signal", "hist_plus", "hist_minus"]
    ].round(2).reset_index(drop=True)

    lo, hi = float(view["low"].min()), float(view["high"].max())
    pad = (hi - lo) * 0.05
    marks = [{"name": nm, "yAxis": lvl} for nm, lvl in levels if lo - pad <= lvl <= hi + pad]
    pivots = []
    if abcd is not None and interval == "H4":
        pivots = [{"name": label, "coord": [point.time.strftime(fmt), point.price + off]}
                  for label, point in zip("ABC", (abcd.a, abcd.b, abcd.c)) if point.time in view.index]

    def line(col: str, label: str, color: str, axis: int = 0, dashed: bool = False) -> dict:
        return {
            "type": "line", "name": label, "encode": {"x": "czas", "y": col},
            "xAxisIndex": axis, "yAxisIndex": axis, "showSymbol": False,
            "lineStyle": {"width": 1.2, "color": color, "type": "dashed" if dashed else "solid"},
            "itemStyle": {"color": color},
        }

    axes = [0, 1, 2]
    return {
        "animation": False,
        "legend": {"top": 0, "data": ["EMA 9", "EMA 26", "EMA 50", "EMA 200", "Bollinger"]},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "axisPointer": {"link": [{"xAxisIndex": "all"}]},
        "dataset": {"source": src},
        "grid": [
            {"left": 60, "right": 70, "top": 30, "height": "52%"},
            {"left": 60, "right": 70, "top": "64%", "height": "13%"},
            {"left": 60, "right": 70, "top": "81%", "height": "13%"},
        ],
        "xAxis": [{"type": "category", "gridIndex": g, "axisLabel": {"show": g == 2}, "boundaryGap": True} for g in axes],
        "yAxis": [
            {"scale": True, "gridIndex": 0, "position": "right", "splitLine": {"lineStyle": {"opacity": 0.3}}},
            {"gridIndex": 1, "min": 0, "max": 100, "position": "right", "splitNumber": 2, "name": "RSI"},
            {"gridIndex": 2, "scale": True, "position": "right", "splitNumber": 2, "name": "MACD"},
        ],
        "dataZoom": [
            {"type": "inside", "xAxisIndex": axes, "start": 0, "end": 100},
            {"type": "slider", "xAxisIndex": axes, "bottom": 0, "height": 18},
        ],
        "series": [
            {
                "type": "candlestick", "name": name,
                "encode": {"x": "czas", "y": ["open", "close", "low", "high"]},
                "itemStyle": {"color": "#26a69a", "color0": "#ef5350", "borderColor": "#26a69a", "borderColor0": "#ef5350"},
                "markLine": {
                    "symbol": "none",
                    "data": marks,
                    "label": {"formatter": "{b}: {c}", "position": "insideStartTop", "fontSize": 10},
                    "lineStyle": {"color": "#7e57c2", "type": "dashed", "width": 1},
                },
                "markPoint": {"data": pivots, "symbol": "circle", "symbolSize": 20,
                              "itemStyle": {"color": "#7e57c2"}, "label": {"formatter": "{b}", "color": "#fff"}},
            },
            line("ema9", "EMA 9", "#2962ff"),
            line("ema26", "EMA 26", "#ff9800"),
            line("ema50", "EMA 50", "#43a047"),
            line("ema200", "EMA 200", "#e53935"),
            line("bb_up", "Bollinger", "#9e9e9e", dashed=True),
            line("bb_low", "Bollinger", "#9e9e9e", dashed=True),
            {
                **line("rsi", "RSI", "#7e57c2", axis=1),
                "markLine": {
                    "symbol": "none", "silent": True,
                    "data": [{"yAxis": 30}, {"yAxis": 70}],
                    "lineStyle": {"color": "#9e9e9e", "type": "dashed"},
                    "label": {"show": False},
                },
            },
            line("macd", "MACD", "#2962ff", axis=2),
            line("macd_signal", "Sygnał", "#ff9800", axis=2),
            {"type": "bar", "name": "Histogram +", "encode": {"x": "czas", "y": "hist_plus"}, "xAxisIndex": 2, "yAxisIndex": 2, "itemStyle": {"color": "#26a69a"}},
            {"type": "bar", "name": "Histogram −", "encode": {"x": "czas", "y": "hist_minus"}, "xAxisIndex": 2, "yAxisIndex": 2, "itemStyle": {"color": "#ef5350"}},
        ],
    }


@st.cache_data(ttl=3600, show_spinner="Sprawdzam podobne sygnały w historii…", max_entries=64)
def holding_cached(ticker: str, direction: str, sl_atr: float, tp_atr: float, limit_h: float):
    stats, trades = backtest.holding_times(get_frames(ticker)["H1"], direction, sl_atr, tp_atr, limit_h)
    return stats, trades


def _h(x: float | None) -> str:
    return "–" if x is None else f"{x:.0f}"


def render_holding(inst: Instrument, direction: str, ev: dict, entry: float, tp: float, atr_h1: float | None, limit_h: float) -> None:
    st.markdown("**Czas trzymania – podobne sygnały z historii H1**")
    if not ev["poprawny"] or not atr_h1 or ev["sl_atr_h1"] is None:
        st.caption("Uzupełnij poprawne SL i TP, aby porównać z historią.")
        return
    tp_atr = abs(tp - entry) / atr_h1
    stats, _ = holding_cached(inst.ticker, direction, round(ev["sl_atr_h1"], 2), round(tp_atr, 2), float(limit_h))
    if not stats.wystarczy:
        st.caption(f"Za mało danych: {stats.przypadki} porównywalnych przypadków (potrzeba co najmniej {backtest.MIN_SAMPLE}).")
        return
    with st.container(horizontal=True):
        for label, n, med, p25, p75 in (
            ("Do TP: mediana", stats.tp, stats.tp_mediana_h, stats.tp_p25_h, stats.tp_p75_h),
            ("Do SL: mediana", stats.sl, stats.sl_mediana_h, stats.sl_p25_h, stats.sl_p75_h),
        ):
            if n == 0:
                st.metric(label, "brak wyjść")
            else:
                st.metric(label, f"{_h(med)} godz.", f"przedział {_h(p25)}–{_h(p75)} godz.", delta_color="off", delta_arrow="off")
    st.caption(
        f"{stats.przypadki} przypadków: TP {stats.tp} ({stats.odsetek_tp:.0%}), SL {stats.sl}, "
        f"bez wyjścia przed limitem {limit_h:g} godz.: {stats.bez_wyjscia} ({stats.odsetek_bez_wyjscia:.0%}). "
        "Przedział to 25.–75. percentyl. To statystyka z przeszłości, nie prognoza."
    )


def render_plan_history(direction: str, res: "backtest.PlanHistory", vol: float, usd_pp: float, usdpln: float | None, limit_h: float) -> None:
    st.markdown(f"**{direction.capitalize()}**")
    if res.powod and res.trades.empty:
        st.info(res.powod, icon=":material/info:")
        return
    s = res.stats()
    st.caption(f"Komplet warunków wejścia spełniony {res.sygnaly} razy; ocenionych pozycji (bez nakładania się, z pełnym wynikiem): {s.przypadki}.")
    if s.przypadki == 0:
        st.info("Żaden przypadek nie dał pełnego wyniku (np. zamknięcie było już za TP albo za SL).", icon=":material/info:")
        return
    if not s.wystarczy:
        st.warning(f"Mała próba ({s.przypadki} < {backtest.MIN_SAMPLE}) – czasy i wyniki traktuj wyłącznie orientacyjnie.", icon=":material/warning:")
    with st.container(horizontal=True):
        st.metric("TP", f"{s.tp}", f"{s.odsetek_tp:.0%}", delta_color="off", delta_arrow="off")
        st.metric("SL", f"{s.sl}", f"{s.sl / s.przypadki:.0%}", delta_color="off", delta_arrow="off")
        st.metric(f"Limit {limit_h:g} godz.", f"{s.bez_wyjscia}", f"{s.odsetek_bez_wyjscia:.0%}", delta_color="off", delta_arrow="off")
    times = []
    for label, n, med, p25, p75 in (("TP", s.tp, s.tp_mediana_h, s.tp_p25_h, s.tp_p75_h), ("SL", s.sl, s.sl_mediana_h, s.sl_p25_h, s.sl_p75_h)):
        times.append(f"do {label}: " + (f"mediana {_h(med)} godz. ({_h(p25)}–{_h(p75)})" if n else "brak"))
    st.caption("Czas: " + "; ".join(times) + ".")
    pts = res.trades["wynik [pkt]"]
    pln = (lambda x: x * vol * usd_pp * usdpln) if usdpln else None
    with st.container(horizontal=True):
        st.metric("Wynik na pozycję (mediana)", f"{pts.median():+.1f} pkt", f"{pln(pts.median()):+.2f} PLN" if pln else None, delta_color="off", delta_arrow="off")
        st.metric("Suma wszystkich", f"{pts.sum():+.1f} pkt", f"{pln(pts.sum()):+.2f} PLN" if pln else None, delta_color="off", delta_arrow="off")
    with st.expander("Przypadki"):
        st.dataframe(res.trades, hide_index=True, alt=f"Historyczne przypadki planu {direction}",
                     column_config={"wejście": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")})


def render_journal_tab(inst: Instrument, cfg: dict, frames: dict[str, pd.DataFrame], off: float, usdpln: float | None) -> None:
    k, dec = inst.key, inst.decimals
    step = 0.1 if dec > 1 else 1.0
    usd_pp = float(cfg["usd_za_punkt"])
    st.caption(
        "Faktyczne transakcje: z zamkniętych planów i dodane ręcznie. „Plan” pokazuje, jak skończyłaby się pozycja przy "
        "trzymaniu się SL, TP i limitu czasu z planu; „bez limitu” – przy trzymaniu do SL lub TP (maks. 30 dni). "
        "Porównanie używa ciągłej historii zamkniętych świec H1, bez spreadu i swapów. Ekstrema świecy rozpoczętej "
        "przed wejściem są pomijane. Jeśli limit wypada wewnątrz świecy i nie ma w niej SL ani TP, cenę wyjścia "
        "szacuje się z poprzedniego zamknięcia. Wynik w PLN z xStation ma pierwszeństwo przed szacunkiem."
    )
    df = journal.entries(cfg["plany"], frames["H1"], off, usd_pp, usdpln)
    if df.empty:
        st.info("Dziennik jest pusty. Zamknij plan w zakładce „Plan pozycji” albo dodaj transakcję spoza planu poniżej.", icon=":material/menu_book:")
    else:
        s = journal.summary(df)
        broker_rate = s["xStation: skuteczność"]
        with st.container(horizontal=True):
            st.metric("Skuteczność w punktach", f"{s['skuteczność']:.0%}",
                      f"{s['zyskowne']} z {s['transakcje']} transakcji", delta_color="off", delta_arrow="off")
            st.metric("Skuteczność wg xStation", f"{broker_rate:.0%}" if broker_rate is not None else "–",
                      f"{s['xStation: zyskowne']} z {s['xStation: n']} transakcji" if broker_rate is not None else "brak wyników",
                      delta_color="off", delta_arrow="off")
            st.metric("Suma", f"{_num(s['suma PLN'], 2)} PLN" if s["suma PLN"] is not None else "brak USD/PLN")
            st.metric("Średnie R", f"{s['średnie R']:+.2f}" if s["średnie R"] is not None else "–", help="Wynik w wielokrotnościach początkowego ryzyka (wejście–SL).")
            st.metric("Mediana czasu", f"{s['mediana godzin']:.0f} godz.")
        notes = []
        if "ręczne: n" in s:
            notes.append(
                f"Ręczne wyjścia: {s['ręczne: n']}, lepiej niż plan w {s['ręczne: lepiej niż plan']} przypadkach; "
                f"łącznie {s['ręczne: różnica do planu [pkt]']:+.1f} pkt względem trzymania się planu."
            )
        if "limit: n" in s:
            notes.append(
                f"Zamknięcia z powodu limitu czasu: {s['limit: n']}, lepiej niż trzymanie dalej w {s['limit: lepiej niż trzymanie dalej']}; "
                f"łącznie {s['limit: różnica do trzymania dalej [pkt]']:+.1f} pkt względem trzymania do SL/TP."
            )
        if "trzymane po limicie: n" in s:
            inside = s.get("w limicie: średni wynik [pkt]")
            notes.append(
                f"Trzymane dłużej niż limit: {s['trzymane po limicie: n']}, średnio {s['trzymane po limicie: średni wynik [pkt]']:+.1f} pkt"
                + (f" (w limicie: {s['w limicie: n']}, średnio {inside:+.1f} pkt)." if inside is not None else ".")
            )
        if notes:
            st.markdown("\n".join(f"- {n}" for n in notes))
        elif len(df) < 5:
            st.caption("Porównania z planem pojawią się, gdy w dzienniku będą ręczne wyjścia albo zamknięcia z powodu limitu.")
        st.dataframe(df, hide_index=True, alt=f"Dziennik transakcji {inst.name}",
                     column_config={"po limicie": st.column_config.CheckboxColumn(width="small")})
        st.download_button("Pobierz dziennik CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name=f"dziennik_{k}.csv",
                           mime="text/csv", icon=":material/download:", key=f"jr_csv_{k}")

    with st.expander("Dodaj transakcję spoza planu"):
        with st.form(f"jr_add_{k}"):
            c1, c2, c3, c4 = st.columns(4)
            direction = c1.selectbox("Kierunek", ["long", "short"])
            entry = c2.number_input("Cena wejścia", min_value=0.0, step=step)
            exit_px = c3.number_input("Cena wyjścia", min_value=0.0, step=step)
            vol = c4.number_input("Wolumen [lot]", min_value=0.001, value=0.01, step=0.001, format="%.3f")
            c5, c6, c7, c8 = st.columns(4)
            now_ts = pd.Timestamp.now()
            d_in = c5.date_input("Data wejścia", value=now_ts.date())
            t_in = c6.time_input("Godzina wejścia", value=now_ts.time().replace(second=0, microsecond=0))
            d_out = c7.date_input("Data wyjścia", value=now_ts.date())
            t_out = c8.time_input("Godzina wyjścia", value=now_ts.time().replace(second=0, microsecond=0))
            c9, c10 = st.columns(2)
            reason = c9.selectbox("Powód wyjścia", journal.REASONS)
            pln_real = c10.number_input("Wynik w PLN z xStation (opcjonalnie)", value=None, step=1.0)
            note = st.text_input("Notatka")
            if st.form_submit_button("Dodaj do dziennika", icon=":material/add:"):
                opened, closed_iso = f"{d_in:%Y-%m-%d}T{t_in:%H:%M}", f"{d_out:%Y-%m-%d}T{t_out:%H:%M}"
                entry_rec = journal.new_offplan(direction, entry, exit_px, vol, opened, closed_iso, reason, pln_real, note)
                errs = (["cena wejścia musi być większa od zera"] if entry <= 0 else []) + journal.validate_close(entry_rec, exit_px, closed_iso, reason)
                if errs:
                    st.error("\n".join(f"- {e}" for e in errs), title="Nie dodano transakcji")
                else:
                    cfg["plany"].append(entry_rec)
                    storage.save_settings(cfg, k)
                    st.toast("Dodano transakcję do dziennika.", icon=":material/check:")
                    st.rerun()


def render_replay_tab(inst: Instrument, cfg: dict, frames: dict[str, pd.DataFrame]) -> None:
    k = inst.key
    st.caption(
        "Te same reguły co monitor – reguły zamknięć, dotknięcia i przeskoki poziomów (przerwa 4 godz.), strefy RSI H1/H4 – "
        "uruchomione po kolei na historycznych, zamkniętych świecach z obecnymi ustawieniami. Pozwala sprawdzić, "
        "czy alerty przychodziłyby za rzadko albo za często. Dywergencje nie są odtwarzane."
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        days = st.select_slider("Okres", options=[7, 14, 30, 60, 90, 180], value=30, format_func=lambda d: f"{d} dni", key=f"rp_days_{k}")
        run = st.button("Odtwórz alerty", icon=":material/history:", key=f"rp_run_{k}")
    if run:
        st.session_state[f"rp_res_{k}"] = (days, backtest.replay_alerts(frames, cfg, days=days))
    res = st.session_state.get(f"rp_res_{k}")
    if not res:
        st.caption("Kliknij „Odtwórz alerty”.")
        return
    days_run, df = res
    if df.empty:
        st.info(f"W ostatnich {days_run} dniach nie byłoby żadnego alertu.", icon=":material/info:")
        return
    per_day = len(df) / days_run
    with st.container(horizontal=True):
        st.metric("Alerty", str(len(df)), f"{per_day:.1f} dziennie", delta_color="off", delta_arrow="off")
        for src, n in df["źródło"].value_counts().items():
            s = str(src)
            st.metric(s[:1].upper() + s[1:], str(n))
    st.dataframe(
        df, hide_index=True, alt=f"Odtworzone alerty {inst.name} z ostatnich {days_run} dni",
        column_config={"czas": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")},
    )
    st.download_button(
        "Pobierz CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name=f"alerty_{k}_{days_run}d.csv",
        mime="text/csv", icon=":material/download:", key=f"rp_csv_{k}",
    )


def render_entry_watches(
    inst: Instrument, cfg: dict, scenarios: dict[str, tuple[float, float, float]],
    volume: float, limit_h: float, cost: float, min_rr: float, frames: dict[str, pd.DataFrame], off: float,
) -> None:
    k, dec = inst.key, inst.decimals
    watches = cfg["obserwacje wejścia"]
    sent = notify.load_state(k).get("obserwacje wejścia", {})
    now = pd.Timestamp.now(tz=data.TZ_LOCAL)
    h1 = frames["H1"]
    latest = h1.iloc[-1]
    latest_closed = signals.last_closed(h1, "H1", now)
    st.markdown("#### Obserwuj wejście")
    st.caption(
        "Zapisz poziomy z karty Long lub Short. Monitor sprawdzi świecę H1 zamkniętą po zapisaniu obserwacji "
        "i wyśle jeden alert, gdy warunki będą spełnione, a cena będzie między SL i TP. "
        "Zysk/ryzyko w alercie jest przeliczone dla ceny sygnału."
    )
    for direction, col in zip(plan.DIRECTIONS, st.columns(2)):
        with col, st.container(border=True):
            st.markdown(f"**{direction.capitalize()}**")
            watch = watches.get(direction)
            if watch:
                expires = pd.Timestamp(watch["ważny do"]).tz_convert(data.TZ_LOCAL)
                signal = sent.get(watch["id"])
                if signal:
                    closed = pd.Timestamp(signal["świeca"]).tz_convert(data.TZ_LOCAL)
                    if "ekstremum odrzucenia" in signal:
                        reaction = plan.watch_rejection_text(direction, watch["próg"], signal["ekstremum odrzucenia"], signal["cena"], dec)
                        st.success(f"OTWÓRZ {direction.upper()} · {closed:%d.%m %H:%M}\n\n{reaction} Zysk do ryzyka {signal['rr']:.2f}:1 po kosztach.")
                    else:
                        st.success(f"Sygnał H1 {closed:%d.%m %H:%M} po {_num(signal['cena'], dec)}. Zysk/ryzyko {signal['rr']:.2f}.")
                elif now < expires and watch.get("odrzucenie H1"):
                    created = pd.Timestamp(watch["utworzono"])
                    sign = 1 if direction == "long" else -1
                    level = watch["próg"]
                    closed_since_creation = latest_closed is not None and latest_closed.name >= created
                    if (latest.name >= created and not data.bar_closed("H1", latest.name, now)
                            and plan.watch_rejects_level(direction, watch, latest, off)):
                        st.info("CZEKAJ NA ZAMKNIĘCIE H1 · Cena przebiła poziom i wróciła. Świeca nadal trwa.")
                    elif closed_since_creation and plan.watch_rejects_level(direction, watch, latest_closed, off):
                        when = data.bar_end("H1", latest_closed.name)
                        price = float(latest_closed["close"]) + off
                        candidate = plan.entry_signal(direction, watch, frames, price, when)
                        extreme = float(latest_closed["low" if direction == "long" else "high"]) + off
                        reaction = plan.watch_rejection_text(direction, watch["próg"], extreme, price, dec)
                        if candidate is None:
                            st.info(f"CZEKAJ · {reaction} Trend H4/D1 lub parametry planu nie potwierdzają wejścia.")
                        elif candidate["rr"] < watch["min rr"]:
                            st.info(f"CZEKAJ · {reaction} Zysk do ryzyka {candidate['rr']:.2f}:1 przy wymaganym {watch['min rr']:g}:1.")
                        else:
                            st.success(f"OTWÓRZ {direction.upper()} · {reaction} Zysk do ryzyka {candidate['rr']:.2f}:1 po kosztach.")
                    elif (closed_since_creation and (latest_closed["open"] + off - level) * sign > 0
                          and (latest_closed["close"] + off - level) * sign < 0):
                        st.info("CZEKAJ · H1 zamknęła się po drugiej stronie poziomu. Brak odrzucenia.")
                    else:
                        st.caption("CZEKAJ NA ODRZUCENIE POZIOMU · Monitor czeka na przebicie i powrót na zamkniętej H1.")
                else:
                    st.caption("Obserwacja wygasła." if now >= expires else "Czeka na spełnienie warunków.")
                st.caption(
                    f"Próg {_num(watch['próg'], dec)} · SL {_num(watch['sl'], dec)} · TP {_num(watch['tp'], dec)} · "
                    f"ważny do {expires:%d.%m.%Y %H:%M} (czas polski)"
                )
                if st.button("Usuń obserwację", key=f"watch_cancel_{k}_{direction}"):
                    del watches[direction]
                    storage.save_settings(cfg, k)
                    st.rerun()
            else:
                with st.form(f"watch_{k}_{direction}"):
                    require_rejection = st.checkbox(
                        "Wymagaj odrzucenia progu na H1", key=f"watch_rejection_{k}_{direction}",
                        help="Świeca musi przebić próg i zamknąć się po tej samej stronie, po której się otworzyła.",
                    )
                    expires_at = st.datetime_input(
                        "Ważny do (czas polski)", value=(now.tz_localize(None).floor("min") + pd.Timedelta(days=1)).isoformat(),
                        format="DD/MM/YYYY", required=True, key=f"watch_expiry_{k}_{direction}",
                    )
                    if st.form_submit_button("Obserwuj wejście"):
                        entry, sl, tp = scenarios[direction]
                        errors = plan.validate_plan(direction, entry, sl, tp, volume, limit_h)
                        if expires_at is None:
                            st.error("Podaj datę i godzinę ważności obserwacji.")
                        else:
                            expires = pd.Timestamp(expires_at).tz_localize(data.TZ_LOCAL, ambiguous="NaT", nonexistent="NaT")
                            if not isinstance(expires, pd.Timestamp) or expires <= now:
                                errors.append("wybierz przyszłą, jednoznaczną godzinę w czasie polskim")
                            if isinstance(expires, pd.Timestamp) and not errors:
                                watches[direction] = plan.new_watch(
                                    entry, sl, tp, volume, limit_h, cost, min_rr, now, expires,
                                    require_rejection=require_rejection,
                                )
                                storage.save_settings(cfg, k)
                                st.rerun()
                            st.error("Popraw obserwację: " + ", ".join(errors))


def render_rejection_history(inst: Instrument, frames: dict[str, pd.DataFrame], limit_h: float, cost: float) -> None:
    details = st.expander("Test H1 · Dotknięcie a odrzucenie", on_change="rerun", key=f"rejection_history_{inst.key}")
    if not details.open:
        return
    with details:
        st.caption(
            "Poziomy to minimum i maksimum ostatniej zakończonej sesji D1. Wejście na następnym otwarciu H1. "
            f"SL 1 ATR · TP 2 ATR · limit {limit_h:g} godz. · koszt {cost:g} pkt na pozycję. "
            "Każdy przypadek jest oceniany osobno, także gdy pozycje nakładają się w czasie."
        )
        trades = backtest.rejection_history(frames, limit_h, cost)
        if trades.empty:
            st.info("Brak zakończonych przypadków do porównania.")
            return
        rows = []
        for direction in ("LONG", "SHORT"):
            by_direction = trades[trades["kierunek"] == direction]
            for label, sample in (("Dotknięcie", by_direction), ("Odrzucenie", by_direction[by_direction["odrzucenie"]])):
                count = len(sample)
                wins = int((sample["wynik"] == "TP").sum())
                rows.append({
                    "Kierunek": direction, "Warunek": label, "Przypadki": count, "TP": wins,
                    "SL": int((sample["wynik"] == "SL").sum()),
                    "Limit": int((sample["wynik"] == "limit").sum()),
                    "TP przed SL": f"{wins / count:.1%}" if count >= backtest.MIN_SAMPLE else "Za mało danych",
                    "Średni wynik po kosztach [pkt]": sample["wynik [pkt]"].mean(),
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Porównanie dotknięć i odrzuceń na H1",
                     column_config={"Średni wynik po kosztach [pkt]": st.column_config.NumberColumn(format="%.2f")})
        st.caption(f"Dotknięcia obejmują odrzucenia. Odsetek TP pokazujemy od {backtest.MIN_SAMPLE} rozstrzygniętych przypadków.")


def render_plan_tab(
    inst: Instrument, cfg: dict, frames: dict[str, pd.DataFrame], off: float, price: float,
    levels: list[tuple[str, float]], usdpln: float | None,
) -> dict[str, tuple[float, float, float]]:
    k, dec = inst.key, inst.decimals
    step = 0.1 if dec > 1 else 1.0
    ps = cfg["plan_ustawienia"]
    st.caption(
        "Porównanie warunków, które sam ustawiasz – aplikacja nie wybiera kierunku. Alerty dotyczą poziomów wykrytych "
        f"w danych panelu ({inst.source}, opóźnienie do ok. 15 min, z przesunięciem). Cenę wykonania sprawdzasz w xStation; "
        "XTB zastrzega, że cena realizacji SL może się różnić od ustawionego poziomu."
    )

    with st.container(horizontal=True, vertical_alignment="bottom"):
        vol = st.number_input("Wolumen [lot]", min_value=0.001, value=float(ps["wolumen"]), step=0.001, format="%.3f", key=f"pl_vol_{k}")
        cost = st.number_input("Koszty [pkt]", min_value=0.0, value=float(ps["koszt_pkt"]), step=step, key=f"pl_cost_{k}", help="Spread i inne koszty w punktach, odejmowane od zysku i doliczane do ryzyka.")
        min_rr = st.number_input("Min. zysk/ryzyko", min_value=0.1, value=float(ps["min_rr"]), step=0.1, key=f"pl_rr_{k}")
        limit_h = st.number_input("Limit czasu [godz.]", min_value=1.0, value=float(ps["limit_h"]), step=1.0, key=f"pl_lim_{k}")
    new_ps = {"wolumen": vol, "koszt_pkt": cost, "min_rr": min_rr, "limit_h": limit_h}
    if new_ps != ps:
        cfg["plan_ustawienia"] = new_ps
        storage.save_settings(cfg, k)

    def _reset() -> None:
        for d in plan.DIRECTIONS:
            for f in ("in", "sl", "tp"):
                st.session_state.pop(f"pl_{k}_{d}_{f}", None)

    st.button("Przywróć propozycję poziomów", icon=":material/restart_alt:", key=f"pl_reset_{k}", on_click=_reset)

    usd_pp = float(cfg["usd_za_punkt"])
    results: dict[str, dict] = {}
    scen: dict[str, tuple[float, float, float]] = {}
    col_l, col_s, col_w = st.columns(3)
    for direction, col in (("long", col_l), ("short", col_s)):
        d = plan.default_scenario(direction, frames, levels, price, off, dec)
        with col, st.container(border=True):
            st.subheader(direction.capitalize())
            side = "powyżej" if direction == "long" else "poniżej"
            entry = st.number_input(f"Wejście: zamknięcie H1 {side}", value=d["wejście"], step=step, key=f"pl_{k}_{direction}_in")
            sl = st.number_input("Stop-loss", value=d["sl"], step=step, key=f"pl_{k}_{direction}_sl",
                                 help="Domyślnie za ostatnim dołkiem (long) / szczytem (short) H4, z buforem 0,1 ATR H4.")
            tp = st.number_input("Take-profit", value=d["tp"], step=step, key=f"pl_{k}_{direction}_tp",
                                 help="Domyślnie kolejny poziom z listy w kierunku pozycji.")
            ev = plan.evaluate_scenario(direction, entry, sl, tp, frames, off, vol, usd_pp, usdpln, cost, min_rr)
            results[direction], scen[direction] = ev, (entry, sl, tp)
            st.dataframe(
                pd.DataFrame(ev["checks"]), hide_index=True,
                column_config={"Spełniony": st.column_config.CheckboxColumn(width="small")},
                alt=f"Warunki scenariusza {direction}",
            )
            if not ev["poprawny"]:
                st.warning("SL i TP muszą leżeć po przeciwnych stronach wejścia.", icon=":material/warning:")
            atr_txt = " / ".join(f"{v:.1f}" if v is not None else "–" for v in (ev["sl_atr_h1"], ev["sl_atr_d1"]))
            with st.container(horizontal=True):
                st.metric("Ryzyko", f"{_num(ev['ryzyko_pkt'], 1)} pkt", f"{atr_txt} ATR H1/D1", delta_color="off", delta_arrow="off")
                st.metric("Zysk/ryzyko", f"{ev['rr']:.2f}" if ev["rr"] is not None else "–")
            with st.container(horizontal=True):
                st.metric("Strata przy SL", f"{_num(-ev['strata_pln'], 2)} PLN" if ev["strata_pln"] is not None else "brak USD/PLN")
                st.metric("Wynik przy TP", f"{_num(ev['zysk_pln'], 2)} PLN" if ev["zysk_pln"] is not None else "brak USD/PLN")
            atr_h1 = frames["H1"]["atr"].iloc[-1]
            render_holding(inst, direction, ev, entry, tp, float(atr_h1) if pd.notna(atr_h1) else None, limit_h)

    with col_w, st.container(border=True):
        st.subheader("Czekaj")
        complete = [d for d, ev in results.items() if ev["spełnione"] == len(ev["checks"])]
        for d, ev in results.items():
            missing = [c["Warunek"] for c in ev["checks"] if not c["Spełniony"]]
            st.markdown(f"**{d.capitalize()}**: spełnione {ev['spełnione']}/{len(ev['checks'])}")
            if missing:
                st.markdown("\n".join(f"- brakuje: {m}" for m in missing))
        if not complete:
            st.info("Żaden scenariusz nie ma kompletu ustawionych warunków.", icon=":material/hourglass_top:")
        with st.expander("Jak liczony jest czas trzymania"):
            st.markdown(
                "Podobny sygnał to zamknięcie H1 powyżej najwyższego szczytu (long) albo poniżej najniższego dołka "
                "(short) z 20 poprzednich świec. SL i TP stoją w tej samej odległości w ATR H1 co w scenariuszu. "
                "Każdy przypadek śledzimy do TP, SL albo limitu czasu (liczonego zegarowo, jak w monitorze). "
                "Gdy w jednej świecy padł i SL, i TP, liczymy SL. Bez spreadu i poślizgu. "
                f"Przy mniej niż {backtest.MIN_SAMPLE} przypadkach pokazujemy „za mało danych”."
            )

    render_entry_watches(inst, cfg, scen, vol, limit_h, cost, min_rr, frames, off)
    render_rejection_history(inst, frames, limit_h, cost)

    # -------------------------------------------------------- ten plan na historii
    st.markdown("#### Sprawdź ten plan na historii")
    st.caption(
        "Dokładnie warunki z kart powyżej: zamknięcie H1 za poziomem wejścia, trend H4 i D1 nie przeciwny "
        "(z zamkniętych świec, bez zaglądania w przyszłość), zysk/ryzyko po kosztach ≥ minimum. Wejście na zamknięciu "
        "świecy, w której komplet warunków pojawił się po raz pierwszy; SL i TP to ceny z planu; wynik po kosztach. "
        "Ponieważ poziomy są stałymi cenami, liczą się tylko okresy, gdy rynek był w ich pobliżu – próba bywa mała."
    )
    params = (round(off, 4), vol, cost, min_rr, limit_h, tuple(scen["long"]), tuple(scen["short"]))
    if st.button("Sprawdź ten plan na historii", icon=":material/manage_search:", key=f"pl_hist_{k}"):
        st.session_state[f"pl_hist_res_{k}"] = (
            params,
            {d: backtest.plan_history(frames, d, *scen[d], off, cost, min_rr, limit_h) for d in plan.DIRECTIONS},
        )
    saved = st.session_state.get(f"pl_hist_res_{k}")
    if saved:
        saved_params, res = saved
        if saved_params != params:
            st.caption("Poziomy lub ustawienia zmieniły się od ostatniego sprawdzenia – kliknij przycisk ponownie.")
        cols = st.columns(2)
        for d, col in zip(plan.DIRECTIONS, cols):
            with col, st.container(border=True):
                render_plan_history(d, res[d], vol, usd_pp, usdpln, limit_h)

    # -------------------------------------------------------- zapis planu po otwarciu pozycji
    st.markdown("#### Zapisz plan po otwarciu pozycji w xStation")
    direction = st.segmented_control("Kierunek otwartej pozycji", plan.DIRECTIONS, default="long", key=f"pl_dir_{k}") or "long"
    s_entry, s_sl, s_tp = scen[direction]
    watches = cfg["obserwacje wejścia"]
    watch = watches.get(direction)
    use_watch = watch is not None and st.checkbox("Użyj zapisanej obserwacji wejścia", key=f"pl_use_watch_{k}_{direction}")
    volume, holding_limit = vol, limit_h
    source = "scenariusz"
    if use_watch:
        s_entry, s_sl, s_tp = watch["próg"], watch["sl"], watch["tp"]
        volume, holding_limit = watch["wolumen"], watch["limit godzin"]
        source = watch["id"]
        st.caption(f"Limit trzymania z obserwacji: {holding_limit:g} godz.")
    with st.form(f"pl_save_{k}_{direction}_{source}"):
        c1, c2, c3, c4 = st.columns(4)
        real_entry = c1.number_input("Faktyczna cena otwarcia z xStation", value=None if use_watch else float(s_entry),
                                     step=step, format=f"%.{dec}f")
        p_sl = c2.number_input("SL", value=float(s_sl), step=step, format=f"%.{dec}f")
        p_tp = c3.number_input("TP", value=float(s_tp), step=step, format=f"%.{dec}f")
        real_volume = c4.number_input("Wolumen transakcji [lot]", min_value=0.001, value=float(volume), step=0.001,
                                      format="%.3f", key=f"pl_open_vol_{k}_{direction}_{source}",
                                      help="Wpisz faktyczny wolumen transakcji z xStation.")
        note = st.text_input("Notatka (opcjonalnie)")
        if st.form_submit_button("Zapisz plan", icon=":material/save:"):
            errs = plan.validate_plan(direction, real_entry, p_sl, p_tp, real_volume, holding_limit) if real_entry is not None else []
            if real_entry is None:
                st.error("Podaj faktyczną cenę otwarcia z xStation.")
            elif errs:
                st.error("\n".join(f"- {e}" for e in errs), title="Nie zapisano planu")
            else:
                cfg["plany"].append(plan.new_plan(direction, real_entry, p_sl, p_tp, real_volume, holding_limit, s_entry, note))
                if use_watch:
                    del watches[direction]
                storage.save_settings(cfg, k)
                st.toast("Zapisano plan. Monitor zacznie go śledzić w najbliższym cyklu.", icon=":material/check:")
                st.rerun()

    open_plans = [p for p in cfg["plany"] if p.get("status") == "otwarty"]
    st.markdown("#### Otwarte plany")
    if not open_plans:
        st.caption("Brak otwartych planów.")
    else:
        fx = f" · USD/PLN {_num(usdpln, 4)}" if usdpln is not None else " · brak kursu USD/PLN"
        st.caption(f"Wycena wg danych panelu, bez spreadu, swapów i prowizji. 1 lot: {usd_pp:g} USD za punkt{fx}.")
        st.dataframe(
            pd.DataFrame([plan.plan_status(p, price, usd_pp, usdpln) for p in open_plans]),
            column_config={"Wolumen [lot]": st.column_config.NumberColumn(format="%.3f")},
            hide_index=True, alt="Otwarte plany pozycji z odległością do SL i TP",
        )
        with st.expander("Skoryguj wolumen otwartej pozycji"):
            selected = st.selectbox(
                "Pozycja do korekty", list(range(len(open_plans))), key=f"pl_volume_select_{k}",
                format_func=lambda i: (
                    f"{open_plans[i]['kierunek'].upper()} · wejście {_num(open_plans[i]['cena wejścia'], dec)} · "
                    f"{open_plans[i]['wolumen']:.3f} lota · {open_plans[i]['id']}"
                ),
            )
            p = open_plans[selected]
            with st.form(f"pl_correct_volume_{k}_{p['id']}"):
                corrected = st.number_input("Faktyczny wolumen [lot]", min_value=0.001, value=float(p["wolumen"]),
                                            step=0.001, format="%.3f", key=f"pl_volume_corrected_{k}_{p['id']}")
                if st.form_submit_button("Zapisz poprawiony wolumen"):
                    if 0.001 <= corrected < float("inf"):
                        p["wolumen"] = corrected
                        storage.save_settings(cfg, k)
                        st.rerun()
                    st.error("Wpisz prawidłowy wolumen od 0,001 lota.")
        with st.form(f"pl_close_form_{k}"):
            st.markdown("**Zamknij plan – wpis do dziennika**")
            c1, c2, c3 = st.columns(3)
            pid = c1.selectbox("Plan", [p["id"] for p in open_plans], key=f"pl_close_id_{k}")
            exit_px = c2.number_input("Cena wyjścia z xStation", value=float(round(price, dec)), step=step, key=f"pl_close_px_{k}")
            reason = c3.selectbox("Powód wyjścia", journal.REASONS, key=f"pl_close_reason_{k}")
            c4, c5, c6 = st.columns(3)
            now_ts = pd.Timestamp.now()
            d = c4.date_input("Data wyjścia", value=now_ts.date(), key=f"pl_close_d_{k}")
            t = c5.time_input("Godzina wyjścia", value=now_ts.time().replace(second=0, microsecond=0), key=f"pl_close_t_{k}")
            pln_real = c6.number_input("Wynik w PLN z xStation (opcjonalnie)", value=None, step=1.0, key=f"pl_close_pln_{k}",
                                       help="Wynik z historii transakcji XTB, ze spreadem i swapami. Bez niego dziennik pokaże szacunek.")
            note = st.text_input("Notatka (dlaczego tak, czego się nauczyłem)", key=f"pl_close_note_{k}")
            if st.form_submit_button("Zamknij plan i zapisz w dzienniku", icon=":material/check_circle:"):
                p = next(x for x in cfg["plany"] if x["id"] == pid)
                closed_iso = f"{d:%Y-%m-%d}T{t:%H:%M}"
                errs = journal.validate_close(p, exit_px, closed_iso, reason)
                if errs:
                    st.error("\n".join(f"- {e}" for e in errs), title="Nie zamknięto planu")
                else:
                    journal.close_plan(p, exit_px, closed_iso, reason, pln_real, note)
                    storage.save_settings(cfg, k)
                    st.toast("Plan zamknięty i zapisany w dzienniku.", icon=":material/check:")
                    st.rerun()
    n_closed = sum(1 for p in cfg["plany"] if p.get("status") == "zamknięty")
    if n_closed:
        st.caption(f"Zamknięte transakcje ({n_closed}) są w zakładce „Dziennik”.")
    return scen


def render_simulation(
    inst: Instrument, cfg: dict, scenarios: dict[str, tuple[float, float, float]], off: float,
) -> None:
    k = inst.key
    active = cfg["symulacja"]
    settings = cfg["plan_ustawienia"]
    volume = paper.VOLUME_BY_INSTRUMENT[k]
    st.subheader("Symulacja na bieżących notowaniach")
    st.button("Odśwież symulację", key=f"paper_refresh_{k}")
    state = paper.load(k)
    st.caption(
        "Monitor otwiera wirtualne pozycje na zamknięciu H1 według warunków z „Planu pozycji”. "
        "Nowe wejście wymaga ponownego spełnienia warunków po świecy, na której nie były spełnione. "
        "SL, TP i limit rozliczane są na podstawie kolejnych świec H1. Pozycje i wyniki są wyłącznie wirtualne."
    )
    st.caption(f"Stały wolumen nowych pozycji w symulacji: {volume:.3f} lota.")
    for direction, col in zip(plan.DIRECTIONS, st.columns(2)):
        with col, st.container(border=True):
            st.markdown(f"**{direction.capitalize()}**")
            saved = active.get(direction)
            entry, sl, tp = (saved["próg"], saved["sl"], saved["tp"]) if saved else scenarios[direction]
            st.caption(f"Próg {_num(entry, inst.decimals)} · SL {_num(sl, inst.decimals)} · TP {_num(tp, inst.decimals)}")
            if saved:
                st.caption(f"Aktywna · limit {saved['limit godzin']:g} godz. · koszty {saved['koszt pkt']:g} pkt")
                if st.button("Wstrzymaj nowe wejścia", key=f"paper_pause_{k}_{direction}"):
                    del active[direction]
                    storage.save_settings(cfg, k)
                    st.rerun()
            else:
                if st.button("Włącz z Planu pozycji", key=f"paper_enable_{k}_{direction}"):
                    errors = plan.validate_plan(direction, entry, sl, tp, volume, settings["limit_h"])
                    if errors:
                        st.error("Popraw wartości w „Planie pozycji”. " + ", ".join(errors))
                    else:
                        active[direction] = paper.new_scenario(k, entry, sl, tp, settings, off)
                        storage.save_settings(cfg, k)
                        st.rerun()

    if last := state.get("sprawdzono"):
        stamp = pd.Timestamp(last).tz_convert(data.TZ_LOCAL)
        st.caption(f"Ostatnia przetworzona świeca zakończyła się {stamp:%d.%m.%Y o %H:%M} (czas polski).")
    elif active:
        st.info("Oczekiwanie na pierwszą zamkniętą świecę H1 po włączeniu. Monitor musi działać w tle.")

    positions = paper.positions(state)
    opened = positions[positions["status"].eq("otwarta")] if not positions.empty else positions
    if not opened.empty:
        st.markdown("#### Otwarte pozycje wirtualne")
        columns = ["kierunek", "otwarto", "cena wejścia", "wolumen", "sl początkowy", "sl", "tp",
                   "zmiany SL", "wynik otwarty pkt", "maks. wynik pkt", "limit godzin", "description"]
        st.dataframe(opened.reindex(columns=columns),
                     hide_index=True, alt="Otwarte pozycje wirtualne i ich poziomy wyjścia")
        st.caption("Wstrzymanie nowych wejść pozostawia otwarte pozycje pod opieką monitora.")

    closed = positions[positions["status"].eq("zamknięta")] if not positions.empty else positions
    st.markdown("#### Zamknięte pozycje wirtualne")
    if closed.empty:
        st.caption("Brak zamkniętych pozycji w symulacji.")
    else:
        with st.container(horizontal=True):
            st.metric("Pozycje", len(closed))
            st.metric("Dodatni wynik", f"{(closed['wynik pkt'] > 0).mean():.0%}")
            st.metric("Suma po kosztach", f"{closed['wynik pkt'].sum():+.2f} pkt")
        columns = ["kierunek", "otwarto", "zamknięto", "cena wejścia", "cena wyjścia", "wolumen", "sl początkowy", "sl", "tp",
                   "zmiany SL", "powód", "wynik pkt", "wynik R", "godziny", "description"]
        st.dataframe(closed.iloc[::-1].reindex(columns=columns), hide_index=True, alt="Historia zamkniętych pozycji wirtualnych")
    if not positions.empty:
        st.caption(f"Monitor zapisuje otwarte i zamknięte pozycje w dane/{paper.csv_path(k).name}.")
        st.download_button("Pobierz dane symulacji CSV", paper.csv_bytes(positions),
                           file_name=paper.csv_path(k).name, mime="text/csv", key=f"paper_export_{k}")

    if records := [*state.get("otwarte", {}).values(), *reversed(state.get("zamknięte", []))]:
        with st.expander("Pełny opis decyzji"):
            selected = st.selectbox(
                "Pozycja wirtualna", range(len(records)), key=f"paper_description_{k}",
                format_func=lambda i: (
                    f"{records[i]['kierunek'].upper()} · {pd.Timestamp(records[i]['otwarto']).tz_convert(data.TZ_LOCAL):%d.%m.%Y %H:%M} · "
                    f"{records[i].get('powód', 'otwarta')}"
                ),
            )
            if selected is not None:
                description = records[selected].get("description")
                if description:
                    st.markdown("\n\n".join(description.splitlines()))
                else:
                    st.caption("Opis decyzji nie był zapisywany w tej wersji symulacji.")

    with st.expander("Jak symulacja rozlicza pozycje"):
        st.markdown(
            "- Przy wejściu zapisujemy poziomy, koszty, przesunięcie ceny i wskaźniki H1 widoczne w chwili sygnału.\n"
            "- TP pozostaje na poziomie z planu. SL przesuwa się tylko po nowym dodatnim maksimum wyniku po kosztach "
            "na zamknięciu H1. Zachowuje początkowy dystans od ceny i obowiązuje od następnej świecy.\n"
            "- Jeśli w jednej świecy osiągnięto SL i TP, przyjmujemy SL. Przy otwarciu świecy za SL "
            "przyjmujemy jej cenę otwarcia, a przy luce za TP przyjmujemy cenę TP.\n"
            "- Limit czasu rozliczamy na końcu świecy obejmującej limit lub po cenie otwarcia pierwszej świecy po przerwie.\n"
            "- CSV obejmuje otwarte i zamknięte pozycje, aktualny SL, liczbę jego przesunięć i opis decyzji. "
            "Czas wyjścia wewnątrz świecy jest przybliżony jej zamknięciem."
        )


def render_harmonics(
    inst: Instrument, cfg: dict, off: float, price: float, h1: pd.DataFrame,
    patterns: list[harmonics.Projection], status: harmonics.Status | None,
) -> None:
    st.subheader("Układ harmoniczny AB=CD")
    enabled = bool(cfg.get(harmonics.ALERT_SETTING))
    alerts = st.checkbox("Powiadamiaj o zmianie etapu AB=CD", value=enabled, key=f"abcd_alerts_{inst.key}")
    if alerts != enabled:
        cfg[harmonics.ALERT_SETTING] = alerts
        storage.save_settings(cfg, inst.key)

    if not patterns or status is None:
        st.info("Brak potwierdzonych punktów A, B i C spełniających warunki AB=CD.")
        return
    projection = patterns[-1]
    title, body = harmonics.describe(projection, status, off, inst.decimals)
    st.info(body, title=title)
    lower, upper = projection.zone
    if status.stage == "building":
        distance = max(lower + off - price, price - upper - off, 0.0)
        st.caption(f"Odległość bieżącej ceny od strefy D · {_num(distance, inst.decimals)}.")
    table = pd.DataFrame([
        {"Punkt": label, "Cena": _num(point.price + off, inst.decimals),
         "Świeca H4": point.time.tz_convert(data.TZ_LOCAL).strftime("%d.%m.%Y %H:%M"),
         "Potwierdzenie": point.confirmed_at.tz_convert(data.TZ_LOCAL).strftime("%d.%m.%Y %H:%M")}
        for label, point in zip("ABC", (projection.a, projection.b, projection.c))
    ] + [{"Punkt": "D · projekcja", "Cena": _num(projection.d + off, inst.decimals), "Świeca H4": "—", "Potwierdzenie": "—"}])
    st.dataframe(table, hide_index=True, alt="Punkty i czasy potwierdzenia układu AB=CD")
    st.caption(f"Ostatnia zmiana etapu · {status.at.tz_convert(data.TZ_LOCAL):%d.%m.%Y %H:%M} (czas polski).")

    history = st.expander("Poprzednie układy AB=CD", key=f"abcd_history_{inst.key}", on_change="rerun")
    if history.open:
        with history:
            rows = []
            recent = patterns[-31:]
            for old, next_pattern in reversed(list(zip(recent, recent[1:]))):
                old_status = harmonics.track(old, h1, next_pattern.c.confirmed_at)
                label, _ = harmonics.describe(old, old_status, off, inst.decimals)
                rows.append({
                    "Potwierdzenie C (czas polski)": old.c.confirmed_at.tz_convert(data.TZ_LOCAL).strftime("%d.%m.%Y %H:%M"),
                    "Cel D": _num(old.d + off, inst.decimals),
                    "Stan przed kolejną projekcją": label.removeprefix("AB=CD · "),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Ostatnie projekcje AB=CD na historii świec")
            else:
                st.caption("Brak wcześniejszych układów w dostępnej historii.")


def render(inst: Instrument) -> None:
    k = inst.key
    dec = inst.decimals
    skey = f"settings_{k}"
    if skey not in st.session_state:
        st.session_state[skey] = storage.load_settings(k)
    cfg: dict = st.session_state[skey]

    # ------------------------------------------------------------ nagłówek i panel boczny
    st.title(f"{inst.icon} {inst.name} – panel analizy technicznej")
    st.caption(
        f"Dane: {inst.source} z Yahoo Finance, opóźnione do ok. 10–15 min. W xStation: {inst.xtb_symbol}. "
        "Narzędzie pokazuje dane i sygnały techniczne – nie jest poradą inwestycyjną, decyzje podejmujesz sam."
    )

    step = 0.1 if dec > 1 else 1.0

    def _calibrate() -> None:
        """Callback przycisku Kalibruj - działa przed utworzeniem widżetów, więc może ustawić
        wartość pola przesunięcia (zmiana stanu widżetu po jego utworzeniu rzuca wyjątek)."""
        xtb = float(st.session_state.get(f"calib_price_{k}") or 0)
        if xtb <= 0:
            st.session_state[f"calib_msg_{k}"] = ("warning", "Podaj cenę z xStation większą od zera.")
            return
        try:
            h1_now = get_frames(inst.ticker)["H1"]
        except Exception as exc:
            st.session_state[f"calib_msg_{k}"] = ("error", f"Nie udało się pobrać notowań: {exc}")
            return
        if h1_now.empty:
            st.session_state[f"calib_msg_{k}"] = ("error", "Brak świec H1 do porównania.")
            return
        src = float(h1_now["close"].iloc[-1])
        cfg["przesunięcie"] = round(xtb - src, dec)
        cfg["kalibracja"] = {
            "czas": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "cena_xtb": xtb,
            "cena_zrodla": round(src, dec),
            "świeca": f"{h1_now.index[-1]:%Y-%m-%d %H:%M}",
        }
        storage.save_settings(cfg, k)
        st.session_state[f"off_{k}"] = cfg["przesunięcie"]
        st.session_state[f"calib_msg_{k}"] = ("success", f"Przesunięcie: {cfg['przesunięcie']:+.{dec}f}")

    with st.sidebar:
        st.header(f"Ustawienia: {inst.name}")
        if f"off_{k}" not in st.session_state:
            st.session_state[f"off_{k}"] = float(cfg["przesunięcie"])
        manual_off = st.number_input(
            f"Przesunięcie względem XTB [{'pkt' if dec <= 1 else 'USD'}]",
            step=step,
            key=f"off_{k}",
            help=f"Różnica: cena {inst.xtb_symbol} w xStation minus cena {inst.ticker}. Zmiana zapisuje się od razu.",
        )
        with st.form(f"calib_{k}", border=True):
            st.number_input(
                "Kalibracja: aktualna cena z xStation",
                min_value=0.0,
                value=0.0,
                step=step,
                key=f"calib_price_{k}",
                help="Przesunięcie zostanie wyliczone raz, w chwili kliknięcia, i zapisane razem z czasem obu cen.",
            )
            st.form_submit_button("Kalibruj", icon=":material/tune:", width="stretch", on_click=_calibrate)
        msg = st.session_state.pop(f"calib_msg_{k}", None)
        if msg:
            kind, text = msg
            {"success": st.success, "warning": st.warning, "error": st.error}[kind](text)
        cal = cfg.get("kalibracja") or {}
        if cal:
            st.caption(f"Ostatnia kalibracja: {cal.get('czas', '?')} – xStation {cal.get('cena_xtb')} vs {inst.ticker} {cal.get('cena_zrodla')} (świeca {cal.get('świeca', '?')}).")
        bars = st.slider("Liczba świec na wykresie", 60, 600, 200, step=20, key=f"bars_{k}")
        if st.button("Odśwież dane", icon=":material/refresh:", width="stretch", key=f"refresh_{k}"):
            get_frames.clear()
            get_intraday_chart.clear()
            get_quote.clear()

    try:
        frames = get_frames(inst.ticker)
    except Exception as exc:  # brak sieci, limit Yahoo itp. - bez notowań nie ma czego pokazać
        st.error(f"Nie udało się pobrać notowań {inst.ticker}: {exc}", title="Błąd pobierania")
        st.stop()

    # Dane pomocnicze są opcjonalne - ich brak nie zatrzymuje wykresu ani reguł
    usdpln: float | None
    try:
        usdpln = get_quote(data.TICKER_USDPLN)[0]
    except Exception:
        usdpln = None
    context = []
    for label, t, inv in inst.context:
        try:
            context.append((label, *get_quote(t), inv))
        except Exception:
            context.append((label, None, None, inv))

    if any(frames[i].empty for i in data.INTERVALS):
        st.warning("Źródło zwróciło za mało notowań, aby pokazać panel. Spróbuj odświeżyć dane później.", icon=":material/hourglass_empty:")
        st.stop()

    last = float(frames["H1"]["close"].iloc[-1])
    if manual_off != float(cfg["przesunięcie"]):
        cfg["przesunięcie"] = manual_off
        storage.save_settings(cfg, k)  # monitor w tle korzysta z tego samego przesunięcia
    off = float(cfg["przesunięcie"])
    price = last + off
    # przy jednej świecy D1 nie ma poprzedniego zamknięcia - zmiana dzienna jest wtedy pomijana
    prev_close = float(frames["D1"]["close"].iloc[-2]) + off if len(frames["D1"]) >= 2 else None
    d1, h4, h1 = frames["D1"].iloc[-1], frames["H4"].iloc[-1], frames["H1"].iloc[-1]
    levels = [(str(x["nazwa"]), float(x["poziom"])) for x in cfg["poziomy"] if x.get("poziom")]
    pattern_now = pd.Timestamp.now(tz="UTC")
    abcd_patterns = harmonics.projections(frames["H4"], pattern_now)
    abcd = abcd_patterns[-1] if abcd_patterns else None
    abcd_status = harmonics.track(abcd, frames["H1"], pattern_now) if abcd else None

    # ------------------------------------------------------------ KPI
    with st.container(horizontal=True):
        day_delta = f"{_num(price - prev_close, dec)} ({(price / prev_close - 1) * 100:+.2f}%)" if prev_close else None
        st.metric(f"{inst.name} (+ przesunięcie)", _num(price, dec), day_delta, border=True)
        for label, val, prev, inv in context:
            if val is None or prev is None:
                st.metric(label, "brak danych", border=True)
            else:
                st.metric(label, f"{val:.2f}", f"{val - prev:+.2f}", delta_color="inverse" if inv else "normal", border=True)
        st.metric("RSI H1 / H4 / D1", f"{h1['rsi']:.0f} / {h4['rsi']:.0f} / {d1['rsi']:.0f}", border=True)
        st.metric("ATR D1 / H1", f"{_num(d1['atr'], 0 if dec <= 1 else 1)} / {_num(h1['atr'], 0 if dec <= 1 else 1)}", border=True)
        st.metric("EMA 200 D1", _num(d1["ema200"] + off, dec), f"{_num(price - d1['ema200'] - off, dec)} od ceny", delta_color="off", border=True)
        st.metric("USD/PLN", f"{usdpln:.4f}" if usdpln else "brak danych", border=True)
    last_bar = frames["H1"].index[-1]
    age_min = (pd.Timestamp.now(tz=last_bar.tz) - last_bar).total_seconds() / 60 - 60  # koniec świecy H1
    age_txt = "świeca w trakcie" if age_min < 0 else (f"{age_min:.0f} min temu" if age_min < 180 else f"{age_min / 60:.0f} godz. temu (rynek zamknięty lub brak danych)")
    st.caption(f"Ostatnia świeca H1: {last_bar:%Y-%m-%d %H:%M} (czas polski), {age_txt}. Przesunięcie: {off:+.{dec}f}.")

    # ------------------------------------------------------------ stan techniczny i reguły
    summary = pd.DataFrame([signals.summarize(frames[i], i) for i in data.INTERVALS])
    rules_df = signals.evaluate_rules(cfg["reguły"], frames, off)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Stan techniczny")
        st.dataframe(summary, hide_index=True, alt=f"Trend, RSI, MACD i Bollinger {inst.name} dla D1, H4 i H1")
    with right:
        st.subheader("Sygnały z reguł")
        if rules_df.empty:
            st.info("Brak reguł. Dodaj je w zakładce Ustawienia poziomów.", icon=":material/info:")
        else:
            hits = rules_df[rules_df["Spełniona"]]
            for _, r in hits.iterrows():
                st.warning(f"{r['Warunek']} – {r['Znaczenie']}", title=r["Reguła"], icon=":material/notifications_active:")
            if hits.empty:
                st.success("Żadna reguła nie jest spełniona na zamkniętych świecach.", icon=":material/check_circle:")

    div_rows = [{"Interwał": i, **d} for i in data.INTERVALS for d in ind.detect_divergences(frames[i])]
    st.subheader("Dywergencje (ostatnie ~80 świec)")
    if div_rows:
        st.dataframe(pd.DataFrame(div_rows), hide_index=True, alt="Wykryte dywergencje RSI i MACD")
    else:
        st.caption("Brak dywergencji – ruch ceny jest potwierdzany przez RSI i MACD.")

    # ------------------------------------------------------------ wykres
    st.subheader("Wykres")
    options = (*data.INTERVALS, *data.CHART_INTRADAY)
    interval = st.segmented_control("Interwał wykresu", options, default="H1", key=f"interval_{k}") or "H1"
    chart_frame = frames.get(interval)
    if interval in data.CHART_INTRADAY:
        try:
            chart_frame = get_intraday_chart(inst.ticker, interval)
        except Exception as exc:
            st.warning(f"Nie udało się pobrać świec {interval} z Yahoo Finance: {exc}")
        else:
            if chart_frame.empty:
                st.warning(f"Yahoo Finance nie zwróciło kompletnych świec {interval}.")
            else:
                st.caption(f"{interval} · początek ostatniej świecy: {chart_frame.index[-1]:%d.%m.%Y %H:%M} (czas polski).")
    if chart_frame is not None and not chart_frame.empty:
        chart_levels = levels
        if abcd and interval in ("H1", "H4"):
            chart_levels = [*levels, ("AB=CD · D", abcd.d + off)]
        st.echarts_chart(
            build_chart(chart_frame, interval, bars, off, chart_levels, inst.name, abcd),
            height=640,
            alt=f"Wykres świecowy {inst.name} {interval} z EMA, Bollingerem, RSI i MACD",
        )

    # ------------------------------------------------------------ zakładki
    tab_plan, tab_simulation, tab_journal, tab_rules, tab_fib, tab_risk, tab_notify, tab_replay, tab_edit = st.tabs(
        ["Plan pozycji", "Symulacja", "Dziennik", "Reguły", "Fibonacci", "Ryzyko pozycji", "Powiadomienia", "Historia alertów", "Ustawienia poziomów"]
    )

    with tab_plan:
        scenarios = render_plan_tab(inst, cfg, frames, off, price, levels, usdpln)
    with tab_simulation:
        render_simulation(inst, cfg, scenarios, off)
    with tab_journal:
        render_journal_tab(inst, cfg, frames, off, usdpln)
    with tab_replay:
        render_replay_tab(inst, cfg, frames)

    with tab_rules:
        if rules_df.empty:
            st.caption("Brak reguł.")
        else:
            st.dataframe(
                rules_df,
                hide_index=True,
                alt="Reguły poziomów i ich stan na ostatnich zamkniętych świecach",
                column_config={"Spełniona": st.column_config.CheckboxColumn(width="small")},
            )

    with tab_fib:
        render_harmonics(inst, cfg, off, price, frames["H1"], abcd_patterns, abcd_status)
        lo_v, lo_t, hi_v, hi_t = ind.auto_fib_swing(frames["D1"])
        top_v, top_t, bot_v, bot_t = ind.auto_fib_decline(frames["D1"])
        st.markdown(
            f"Automatycznie z D1 (ostatni rok): wzrost **{_num(lo_v + off, dec)}** ({lo_t:%Y-%m-%d}) → "
            f"szczyt **{_num(hi_v + off, dec)}** ({hi_t:%Y-%m-%d}), potem spadek do **{_num(bot_v + off, dec)}** ({bot_t:%Y-%m-%d})."
        )
        swings = [
            ("Wzrost do szczytu (auto)", lo_v + off, hi_v + off, True),
            ("Spadek od szczytu (auto) – opory", bot_v + off, top_v + off, False),
        ] + [(str(s["nazwa"]), float(s["dołek"]), float(s["szczyt"]), True) for s in cfg["fibo"]]
        fib_rows = []
        for nm, lo, hi, down in swings:
            if hi <= lo:
                continue
            for ratio, lvl in ind.fib_levels(lo, hi, downtrend=down):
                fib_rows.append({"Fala": nm, "Poziom": ratio, "Cena": round(lvl, dec), "Dystans": round(lvl - price, dec)})
        fib_df = pd.DataFrame(fib_rows, columns=["Fala", "Poziom", "Cena", "Dystans"])
        near = fib_df[fib_df["Dystans"].abs() <= 2 * float(d1["atr"])].sort_values("Cena", ascending=False).reset_index(drop=True)
        st.markdown("**Poziomy w zasięgu 2× ATR D1 od ceny**")
        st.dataframe(near, hide_index=True, alt="Poziomy Fibonacciego blisko bieżącej ceny")
        with st.expander("Wszystkie poziomy"):
            st.dataframe(fib_df, hide_index=True, alt="Wszystkie poziomy Fibonacciego")

    with tab_risk:
        usd_pp = st.number_input(
            "Wartość 1 punktu dla 1 lota [USD]",
            value=float(cfg["usd_za_punkt"]),
            step=1.0,
            key=f"usdpp_{k}",
            help=f"Domyślnie: {inst.lot_note}. Sprawdź w specyfikacji instrumentu w xStation.",
        )
        if usd_pp != cfg["usd_za_punkt"]:
            cfg["usd_za_punkt"] = usd_pp
            storage.save_settings(cfg, k)
        if usdpln is None:
            st.warning("Brak kursu USD/PLN – wyniki w PLN są chwilowo niedostępne. Wykres i reguły działają.", icon=":material/currency_exchange:")
        else:
            pv = risk.point_value_pln(1.0, usdpln, usd_pp)
            st.caption(f"1 lot ≈ {pv:.2f} PLN za punkt. Wynik bez spreadu, swapów i prowizji – dokładne liczby pokazuje xStation.")
        if not cfg["pozycje"]:
            st.caption("Brak pozycji. Dodaj je w zakładce Ustawienia poziomów.")
        elif usdpln is None:
            pass
        else:
            scen = risk.scenario_table(cfg["pozycje"], levels, price, usdpln, usd_pp)
            st.dataframe(scen, hide_index=True, alt="Łączny wynik pozycji przy kluczowych poziomach")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Odległość stop-lossa w ATR H1**")
            st.dataframe(risk.atr_stops(float(h1["atr"]), price), hide_index=True, alt="Stop-loss w wielokrotnościach ATR H1")
        with c2:
            st.markdown("**Odległość stop-lossa w ATR D1**")
            st.dataframe(risk.atr_stops(float(d1["atr"]), price), hide_index=True, alt="Stop-loss w wielokrotnościach ATR D1")

    with tab_notify:
        status = delivery.read_json(notify.DATA_DIR / "monitor_status.json", {})
        mon = notify.load_state(k)
        if not status.get("cykl"):
            st.info(
                "Monitor w tle jeszcze nie zgłosił żadnego cyklu. Uruchom go: `.\\uruchom_monitor.ps1` "
                "albo zadanie Windows „US100 monitor sygnałów”.",
                title="Brak danych z monitora",
                icon=":material/notifications_off:",
            )
        else:
            cycle = pd.Timestamp(status["cykl"])
            age = (pd.Timestamp.now() - cycle).total_seconds() / 60
            with st.container(horizontal=True):
                st.metric("Ostatni cykl monitora", f"{age:.0f} min temu", border=True)
                st.metric("Czeka na ponowienie", str(len(notify.pending(k))), border=True)
                dl = status.get("pobrania", {})
                st.metric("Pobrania (pełne / przyrostowe)", f"{dl.get('pełne', 0)} / {dl.get('przyrostowe', 0)}", border=True)
            if age > 15:
                st.warning(
                    "Monitor nie zgłosił się od ponad 15 minut – mógł zostać zatrzymany. "
                    "Sprawdź zadanie Windows albo dane/monitor.log.",
                    icon=":material/warning:",
                )
            for src, err in (status.get("błędy") or {}).items():
                st.error(f"{src}: {err}", title="Błąd w ostatnim cyklu", icon=":material/error:")
            if mon.get("sprawdzono"):
                st.caption(f"{inst.name} sprawdzony: {mon['sprawdzono'].replace('T', ' ')}.")

        st.markdown(
            "Monitor zgłasza: spełnienie reguły, nową dywergencję RSI/MACD, wyjście RSI z wyprzedania lub "
            "wykupienia (H1, H4), dotknięcie poziomu z listy oraz przeskok poziomu luką (raz na 4 godziny dla poziomu). "
            f"Nieudana wysyłka jest ponawiana co cykl, do {delivery.MAX_ATTEMPTS} prób lub 6 godzin."
        )
        waiting = notify.pending(k)
        if waiting:
            st.markdown("**Czekają na wysyłkę**")
            st.dataframe(
                pd.DataFrame(
                    [{"czas": e["czas"], "tytuł": e["tytuł"], "próby": e.get("próby", 0), **e.get("kanały", {})} for e in waiting]
                ),
                hide_index=True,
                alt="Powiadomienia oczekujące na ponowienie wysyłki",
            )
        log, bad = notify.read_log(instrument=k)
        st.markdown("**Historia**")
        if bad:
            st.warning(f"Pominięto {bad} uszkodzonych wierszy w dane/powiadomienia.jsonl.", icon=":material/warning:")
        if log.empty:
            st.caption("Brak powiadomień.")
        else:
            st.dataframe(log, hide_index=True, alt=f"Historia powiadomień {inst.name} ze statusem wysyłki")

        with st.form(f"telegram_{k}"):
            st.markdown("**Telegram (opcjonalnie, wspólny dla wszystkich instrumentów)**")
            st.caption(
                "Utwórz bota u @BotFather, wklej jego token, napisz do bota dowolną wiadomość, "
                "a chat_id odczytaj z https://api.telegram.org/bot<token>/getUpdates. "
                f"Token jest przechowywany w: {delivery.telegram_storage()}."
            )
            tg = notify.load_telegram()
            token = st.text_input("Token bota", value=tg.get("token", ""), type="password")
            chat_id = st.text_input("chat_id", value=tg.get("chat_id", ""))
            if st.form_submit_button("Zapisz Telegram", icon=":material/save:"):
                where = notify.save_telegram({"token": token.strip(), "chat_id": chat_id.strip()})
                st.toast(f"Zapisano ustawienia Telegrama ({where})", icon=":material/check:")

        if st.button("Wyślij powiadomienie testowe", icon=":material/notifications:", key=f"test_{k}"):
            ev = notify.make_event("Test powiadomień", "Jeśli to widzisz, powiadomienia działają.", k)
            notify.dispatch([ev])
            ch = ev.get("kanały", {})
            st.toast(f"Windows: {ch.get('windows')}, Telegram: {ch.get('telegram')}")

    with tab_edit:
        st.caption(f"Zmiany zapisują się w pliku dane/{storage.settings_path(k).name} po kliknięciu Zapisz.")
        with st.form(f"edit_{k}"):
            price_col = st.column_config.NumberColumn(min_value=0.01, required=True)
            st.markdown("**Poziomy na wykresie**")
            poz = st.data_editor(
                _table(cfg["poziomy"], "poziomy"), num_rows="dynamic", hide_index=True, key=f"ed_poz_{k}",
                column_config={"poziom": price_col},
            )
            st.markdown("**Reguły**")
            reg = st.data_editor(
                _table(cfg["reguły"], "reguły"),
                num_rows="dynamic",
                hide_index=True,
                key=f"ed_reg_{k}",
                column_config={
                    "poziom": price_col,
                    "warunek": st.column_config.SelectboxColumn(options=list(storage.CONDITIONS), required=True),
                    "interwał": st.column_config.SelectboxColumn(options=list(data.INTERVALS), required=True),
                },
            )
            st.markdown("**Fale Fibonacciego** (dołek → szczyt)")
            fib_ed = st.data_editor(
                _table(cfg["fibo"], "fibo"), num_rows="dynamic", hide_index=True, key=f"ed_fib_{k}",
                column_config={"dołek": price_col, "szczyt": price_col},
            )
            st.markdown("**Pozycje** (wpisz według „Moich Transakcji” w xStation)")
            pos = st.data_editor(
                _table(cfg["pozycje"], "pozycje"),
                num_rows="dynamic",
                hide_index=True,
                key=f"ed_pos_{k}",
                column_config={
                    "kierunek": st.column_config.SelectboxColumn(options=["long", "short"], required=True),
                    "wolumen": st.column_config.NumberColumn(min_value=0.001, step=0.001, format="%.3f", required=True),
                    "cena otwarcia": price_col,
                },
            )
            if st.form_submit_button("Zapisz", icon=":material/save:"):
                tables = {"poziomy": poz, "reguły": reg, "fibo": fib_ed, "pozycje": pos}
                clean, errors = storage.validate({name: df.to_dict("records") for name, df in tables.items()})
                if errors:
                    st.error("\n".join(f"- {e}" for e in errors), title="Nie zapisano – popraw dane")
                else:
                    cfg.update(clean)
                    storage.save_settings(cfg, k)
                    st.toast("Zapisano ustawienia", icon=":material/check:")
                    st.rerun()
