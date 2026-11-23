# Pełny potencjał Pythona + AI w analizie technicznej, interpretacji wykresów i probabilistycznym prognozowaniu kursów instrumentów finansowych

**Stan researchu: 23 listopada 2026 r., Europa/Warszawa.** Punktem wyjścia był dołączony 31‑stronicowy raport „Analiza Techniczna_ Python i AI - Dokumenty Google.pdf”. Dokument jest ambitny i znacznie lepszy od typowych zestawień „bibliotek do tradingu”, bo prawidłowo łączy dane, obliczenia, prognozowanie, walidację i wizualizację. Jednocześnie kilka jego tez jest zbyt kategorycznych, część benchmarków pochodzi od autorów samych projektów, jedna ważna funkcjonalność walidacyjna jest przypisana bibliotece, która jej jeszcze nie implementuje, a obszary takie jak mikrostruktura, opcje, point‑in‑time data, breadth, kalibracja prognoz, koszty wykonania, zarządzanie eksperymentami i pełna architektura produkcyjna są potraktowane zbyt skrótowo. fileciteturn0file0

## Wnioski najważniejsze dla projektu

Python w 2026 r. jest wyjątkowo dobrym językiem orkiestrującym kompletny warsztat ilościowy: może pobierać ticki i order book, obrabiać miliardy rekordów przez Polars/DuckDB/Arrow, wykonywać wskaźniki w kodzie C/Rust/Numba/GPU, budować modele ML i modele probabilistyczne, uruchamiać backtesty zdarzeniowe, tworzyć interaktywne wykresy, a na końcu przekazywać zweryfikowane liczby modelowi LLM, który zamienia je w czytelny komentarz. Istotne jest jednak, że wydajność nie bierze się z „szybkiego Pythona”, lecz z delegowania ciężkich operacji do bibliotek natywnych. Polars ma rdzeń w Ruście, DuckDB wykonuje analityczne SQL bezpośrednio na Parquet/Arrow, TA‑Lib korzysta z biblioteki C, a NautilusTrader ma rdzeń w Ruście. citeturn353852search4turn353852search6turn353852search1turn353852search10

Najważniejsza decyzja projektowa nie brzmi zatem „który wskaźnik przewiduje cenę?”, ale **jak oddzielić trzy zupełnie różne problemy**. Analiza opisowa odpowiada „co dzieje się teraz?” — np. trend, zmienność, odległość od AVWAP, struktura wolumenu, imbalance, stan reżimu. Predykcja odpowiada „jaki jest rozkład możliwego wyniku za określony horyzont?” — np. P(r₅d > 0), kwantyle P10/P50/P90 albo prawdopodobieństwo dotknięcia bariery +2% przed −1,5%. Sygnał transakcyjny jest dopiero regułą decyzyjną, która łączy predykcję z kosztami, płynnością, ryzykiem, wielkością pozycji, ograniczeniami portfela i zasadą wykonania. Dobra predykcja nie musi być dobrym sygnałem, a atrakcyjny wskaźnik opisowy nie musi mieć żadnej wartości prognostycznej.

Najbardziej dojrzały projekt nie powinien więc próbować „użyć wszystkich wskaźników jednocześnie”. Powinien mieć **katalog wskaźników**, który można filtrować według rodziny, rynku, interwału, wymaganych danych, kosztu obliczeniowego i statusu empirycznego. Setki silnie skorelowanych transformacji tej samej ceny mogą zwiększyć ryzyko data snoopingu i przeuczenia zamiast zwiększyć informację.

Najlepszy model AI również nie powinien bezpośrednio „oglądać wykresu i wydawać werdyktu”. Własny raport słusznie zmierza w stronę przekazywania LLM‑owi ustrukturyzowanego JSON‑a z liczbowymi cechami, ale wniosek ze stron 18–19 jest sformułowany zbyt absolutnie. Badania z 2026 r. wskazują, że VLM nie są wystarczająco wiarygodnym samodzielnym silnikiem prognozowania na podstawie świec, szczególnie poza wyraźnymi trendami; nie oznacza to, że wizja jest bezużyteczna. Może służyć do inspekcji wykresów, kontroli renderingu, klasyfikacji nietypowych układów czy interakcji z użytkownikiem, lecz sygnały produkcyjne powinny być zakotwiczone w liczbach. fileciteturn1file2 citeturn464515academia10

W praktyce najbardziej wartościowe połączenie „Python + AI” wygląda następująco:

| Warstwa | Python powinien robić deterministycznie | AI/ML powinno robić |
|---|---|---|
| Dane | normalizacja, korekty, strefy czasowe, PIT, corporate actions | wykrywanie anomalii jakości |
| TA | wszystkie wskaźniki i geometria wykresu | ranking/ważenie cech |
| Mikrostruktura | spread, imbalance, CVD, OFI, footprint | klasyfikacja reżimu / krótkoterminowych prawdopodobieństw |
| Forecast | budowa targetów, leakage control | rozkład przyszłych wyników |
| Wykres | współrzędne, strefy, markery | opis językowy |
| Risk | VaR/CVaR, exposure, sizing, limity | pomocnicza ocena scenariuszy |
| Backtest | egzekucja i koszty | nie powinno „dopisywać” wyników |
| Raport | liczby i provenance | synteza, objaśnienie, pytania użytkownika |

To rozdzielenie jest ważniejsze niż wybór między konkretnym LLM czy TSFM.

## Audyt merytoryczny dołączonego raportu

Raport poprawnie proponuje architekturę pięciowarstwową na stronach 1–4 oraz od razu umieszcza walidację między badaniem a prezentacją. To dobry fundament. fileciteturn1file7 Do projektu produkcyjnego dodałbym jednak jako jawne komponenty **Data Quality & Point‑in‑Time Layer**, **Execution/Cost/Risk Layer** oraz **Experiment & Model Governance**. Pięć bloków z diagramu na stronie 3 dobrze wygląda jako model koncepcyjny, ale ukrywa odpowiedzialności, które w realnym systemie finansowym są krytyczne.

| Miejsce w PDF | Ocena | Co wymaga korekty lub doprecyzowania |
|---|---|---|
| s.1: „żaden inny język nie spina…” | retoryczne | Nie jest to twierdzenie falsyfikowalne. Lepiej: Python ma wyjątkowo szeroki, dojrzały ekosystem integracyjny. |
| s.4–6: Massive i Databento | częściowo poprawne | Ceny i zakres historii zależą od planu, licencji i datasetu. Databento Standard nie daje jednakowo głębokiego L2/L3 dla całej historii. |
| s.5: Databento OPRA od 2013 | wymaga ważnego zastrzeżenia | Dane OPRA sprzed 28.03.2023 nie są równoważne nowszemu packet capture; dokumentacja opisuje wcześniejsze dane jako subsamplowane i o niższej granularności. citeturn755743search1 |
| s.6–7: Polars/DuckDB „~10×” | benchmark lokalny | Test na pojedynczym MacBooku i określonym workloadzie nie powinien być przedstawiany jako uniwersalny wynik. Oficjalne cechy architektury są mocne, ale rzeczywistą przewagę trzeba mierzyć na własnych danych. citeturn353852search4turn353852search6 |
| s.7: streaming TA‑Lib = O(1) | zbyt szerokie | Nowszy interfejs streamingowy jest realny, ale dokument projektu pokazuje, że nie wszystkie funkcje mają koszt O(1); część rodzin nadal wykonuje pracę zależną od okresu. citeturn585990search0turn585990search7 |
| s.8: TA‑Lib 0.8.3.post1 | potwierdzone | PyPI pokazuje 0.8.3.post1 opublikowane 21.11.2026; oficjalne wheels zawierają natywną bibliotekę, co rzeczywiście usuwa dawny problem ręcznej kompilacji w typowych instalacjach. citeturn353852search1turn353852search2 |
| s.8: pandas-ta-classic 0.9.0 | potwierdzone, benchmark ostrożnie | 0.9.0, MIT, 224 wskaźniki + 62 natywne formacje są zgodne z projektem; deklarowane przyspieszenia Numba pochodzą jednak od projektu, nie z niezależnego benchmarku. citeturn947620search0 |
| s.9: polars-talib ~150× | marketing/project benchmark | Biblioteka jest aktywna i ma 201 funkcji, ale ~150× pochodzi z notebooka autora i nie powinno być traktowane jako niezależny wynik. citeturn947620search2 |
| s.10–11: SMC/ICT | użyteczne jako cechy, nie „udowodniona przewaga” | smartmoneyconcepts implementuje FVG/BOS/CHoCH/OB itd., lecz istnienie implementacji nie jest dowodem prognostycznej skuteczności koncepcji. Projekt należy traktować jako generator cech do falsyfikacji OOS. citeturn947620search3 |
| s.11: Hurst H>0,5 = trend | nadmierne uproszczenie | Hurst jest estymatorem własności skalowania; w krótkich, niestacjonarnych seriach finansowych jest podatny na bias. Nie należy przekładać pojedynczego H na automatyczny sygnał trend/mean reversion. |
| s.11: wyniki Volume Profile | ciekawy case study, nie dowód ogólny | Wyniki konkretnego repozytorium nie mogą zostać uogólnione na cały rynek. Używałbym profilu jako kontekstu/cechy, nie jako samodzielnego dowodu alfa. |
| s.13: „pełnoprawna aplikacja wyłącznie w Pythonie” | skrót myślowy | Programista może pisać praktycznie wyłącznie Python, ale frontend Streamlit/Dash/Reflex/Lightweight Charts nadal kończy się kodem webowym wykonywanym w przeglądarce. |
| s.16–17: Chronos‑2 „21 kwantyli P1–P99” | wymaga korekty | Oficjalny pipeline Chronos‑2 domyślnie operuje na kwantylach 0.1…0.9 i pozwala określać poziomy kwantyli. Nie znalazłem podstawy w oficjalnej dokumentacji dla stałej, natywnej reguły „21 kwantyli P1–P99”. citeturn464515search0turn464515search1turn464515search2 |
| s.17: TimesFM 2.5/3.0 | zasadniczo poprawne | Kod projektu jest dostępny, lecz pretrained weights TimesFM 3.0 mają warunki non-commercial; TimesFM 2.5 pozostaje dużo prostszy licencyjnie dla własnego wdrożenia. citeturn464515search5 |
| s.16–17: rankingi TSFM | zbyt statyczne | LiveHouse‑TS pokazuje, że rankingi modeli mogą zmieniać się w protokole „żywym”; pojedyncza tabela win-rate nie powinna decydować o architekturze. citeturn464515academia9 |
| s.17: TSFM jako prognoza rynku | raport słusznie ostrożny | Najnowsze testy finansowe pokazują niewielkie i niestabilne przewagi zero-shot nad naiwnymi benchmarkami; świeże badanie ES/NQ z listopada 2026 dodatkowo ostrzega przed przenoszeniem benchmarków ogólnych na niskosygnałowe intraday returns. citeturn464515academia10turn464515academia11 |
| s.18–19: VLM | kierunek dobry, język za mocny | „VLM nie potrafią czytać wykresów” należy zastąpić „nie są dostatecznie niezawodne jako samodzielny silnik prognostyczny”. |
| s.19: PydanticAI „gwarantuje” poprawne etykiety | nie | Pydantic może wymusić strukturę i walidację typów. Nie gwarantuje semantycznej poprawności ceny, czasu, kierunku sygnału ani sensowności strefy bez dodatkowych validatorów domenowych. citeturn234658search0turn234658search3 |
| s.19–20: LangGraph/TradingAgents | architektonicznie ciekawe | Popularność projektu ani liczba gwiazdek nie są dowodem wartości prognostycznej. Agenci powinni orkiestracyjnie wywoływać sprawdzone funkcje, a nie stanowić źródło danych lub „intuicji rynkowej”. fileciteturn1file1 |
| s.20–24: NautilusTrader | dobry wybór, wersja już nieaktualna | Na 23.11.2026 widoczny jest 2.0.0rc9 z 10.11.2026; PDF wskazuje rc8. Rdzeń backtest/live, fill models, latency i order book są mocną stroną. citeturn353852search8turn353852search10turn353852search11 |
| s.23: skfolio zapewnia PBO | **błąd** | skfolio ma CPCV i Deflated Sharpe, ale issue z października 2026 proponuje dopiero dodanie PBO jako funkcji. PBO trzeba dziś zaimplementować osobno lub wykorzystać inne narzędzie. citeturn158813search0turn158813search5 |
| s.23: CPCV jako „lepszy walk-forward” | wymaga niuansu | Dokumentacja skfolio podkreśla, że CPCV może trenować również na obserwacjach późniejszych od danego bloku testowego. Jest świetne do dystrybucji ścieżek/OOS i selection stress, ale nie zastępuje chronologicznej symulacji deploymentu. citeturn158813search2 |
| s.24: „najlepszy” quantstats-lumi | zbyt kategoryczne | Aktywność projektu jest zaletą, lecz „najlepszy” zależy od potrzeb. Metryki trzeba weryfikować niezależnie; raportowanie nie zastępuje własnego modułu statystycznego. |
| s.24–25: stack | dobry MVP | Do produkcji brakuje PIT data, catalog/provenance, event bus, model registry, drift/calibration monitor, cost model i security boundary. fileciteturn1file8 |

Dodatkowo należy skorygować sposób prezentowania bibliotek licencyjnych. **vectorbt nie jest po prostu „Apache 2.0”**: jego publiczne warunki wskazują Apache 2.0 z Commons Clause, które ogranicza sprzedaż produktu/usługi będącej zasadniczo tym oprogramowaniem. Dla komercyjnej platformy jest to ważna różnica względem czystego Apache 2.0. citeturn822282search3turn822282search9 LEAN natomiast deklaruje permissive Apache 2.0 i obsługę wielu klas aktywów, więc jest ciekawym kandydatem tam, gdzie ważniejsza jest infrastruktura tradingowa niż notebookowy research. citeturn822282search4

Raport bardzo dobrze zrobił, że cytuje fora i społeczności, ale **Reddit, HN czy Quant StackExchange powinny służyć do odkrywania problemów**, a nie do udowadniania skuteczności. Relacja „u mnie Polars był 20× szybszy”, „ten model działa na BTC” albo „ten wskaźnik jest najlepszy” może zainspirować test, lecz nie powinna wejść do tabeli dowodów obok pracy prerejestrowanej czy dokumentacji producenta.

## Pełny zakres analityczny: dane, wskaźniki, mikrostruktura i różnice między rynkami

Warstwa wskaźników powinna być zbudowana jako **taksonomia cech**, a nie lista nazw. Wtedy system może obliczać setki miar, ale model i użytkownik wiedzą, które niosą podobną informację.

| Rodzina | Przykłady | Biblioteki / implementacja | Realna rola |
|---|---|---|---|
| Trend | SMA/EMA/WMA/HMA, ALMA, DEMA/TEMA, MACD, ADX/DMI, Ichimoku, Donchian, SuperTrend | TA‑Lib, pandas-ta-classic, polars-talib | opis kierunku i siły trendu |
| Momentum | RSI, Stoch, StochRSI, ROC, CMO, Williams %R, TSI, Fisher Transform | TA‑Lib / pandas-ta-classic | tempo zmiany, ekstremalne położenie |
| Zmienność | ATR/NATR, Bollinger width, Donchian width, Parkinson, Garman–Klass, Rogers–Satchell, realized volatility | TA‑Lib + własne Polars/Numba + `arch` | sizing, regime, forecast volatility |
| Wolumen | OBV, CMF, MFI, Acc/Dist, VWAP, RVOL | TA‑Lib / własne | potwierdzenie aktywności |
| Price action | HH/HL/LH/LL, pivots, gaps, inside/outside bars, range expansion | scipy.signal + własne | deterministyczna geometria |
| SMC/ICT | FVG, BOS, CHoCH, OB, liquidity zones | smartmoneyconcepts | eksperymentalne cechy struktury, nie potwierdzone alfa |
| Profile | TPO, POC, VAH/VAL, HVN/LVN, developing POC | własny histogram/tick data | kontekst aukcyjny |
| Anchored metrics | AVWAP, anchored volume, anchored std | własne Polars | kontekst od eventu/pivotu |
| Cykl | autocorrelation, spectral density, FFT, Hilbert | scipy.signal | wykrywanie okresowości; ostrożnie przy niestacjonarności |
| Wavelet | DWT/CWT, wavelet energy | PyWavelets / ssqueezepy | wieloskalowe cechy, denoising |
| Fraktale/chaos | Hurst, fractal dimension, entropy, Lyapunov proxies | antropy/nolds/własne | cechy reżimu; eksperymentalne |
| Adaptacyjne filtry | Kalman, KAMA, FRAMA, adaptive EMA | statsmodels/filterpy/TA‑Lib | wygładzanie/reżimy |
| Statystyka | z-score, rolling beta, skew, kurtosis, autocorr, cointegration | scipy/statsmodels | normalizacja i relacje |
| Regime | Markov switching, HMM, change points | statsmodels, hmmlearn, ruptures | klasyfikacja środowiska |
| Analogi | Matrix Profile motifs/discords | STUMPY | podobne historyczne trajektorie/anomalie |
| Cross-sectional | relative strength, ranks, breadth | Polars/NumPy | porównanie instrumentów |

TA‑Lib 0.8.3.post1 jest obecnie bardzo solidnym rdzeniem klasycznej TA, a nowsze wydania rozszerzyły katalog o m.in. ALMA, FRAMA i HMA. [TA‑Lib PyPI](https://pypi.org/project/TA-Lib/) i [repozytorium](https://github.com/TA-Lib/ta-lib-python). citeturn353852search1turn353852search2 pandas-ta-classic jest wygodniejsze do eksploracji i ma obecnie 284 unikalne obliczenia/formacje według projektu. [pandas-ta-classic](https://github.com/xgboosted/pandas-ta-classic). citeturn947620search0 polars-talib jest naturalnym wyborem do skanowania wielu symboli przez Polars, lecz jego benchmarki szybkości trzeba traktować jako benchmark autora. [polars-talib](https://github.com/Yvictor/polars_ta_extension). citeturn947620search2

Dla waveletów PyWavelets daje DWT/CWT, ale jego dokumentacja sama zaznacza ograniczenia implementacji CWT i wskazuje ssqueezepy jako szybszą/niekiedy dokładniejszą alternatywę. To dobry przykład, dlaczego „nowocześniejsze” nie znaczy automatycznie „bardziej predykcyjne”. citeturn272234search3turn272234search12

STUMPY jest szczególnie interesujące, bo Matrix Profile odpowiada na inne pytanie niż klasyczny wskaźnik: „czy fragment ostatnich 80 świec przypomina coś, co wydarzyło się wcześniej?”. Biblioteka ma także wersję przyrostową `stumpi`, więc nadaje się do strumienia. citeturn272234search23turn272234search27 Taką funkcję warto wykorzystać jako **historical analog search**: po znalezieniu K najbliższych motifów można pokazać użytkownikowi empiryczną dystrybucję późniejszych 5/20/60‑bar returns, zamiast udawać, że analog „przepowiada” przyszłość.

Jeszcze większą luką w raporcie jest **order flow i market microstructure**. OHLCV nie pozwala odtworzyć pełnej informacji o przebiegu aukcji. Dla futures, aktywnych akcji, opcji i krypto wartościowe stają się bid/ask, trades at bid/ask, depth, add/cancel/modify, queue, imbalance i order-flow imbalance. Minimalny katalog mikrostruktury powinien zawierać spread absolutny i względny, microprice, top-of-book imbalance, depth imbalance na N poziomach, signed volume, cumulative volume delta, trade intensity, cancel/add ratio, realized spread, effective spread, Kyle-type price impact proxy, VPIN-like measures traktowane ostrożnie, book slope/convexity, queue pressure i footprint per price level. Do tego potrzebne są jednak rzeczywiste dane tick/L2/L3, a nie rekonstruowane OHLCV.

Databento nadaje się do tego znacznie lepiej niż typowe darmowe API. Oficjalny cennik z listopada 2026 pokazuje, że zakres L0/L1/L2/L3 zależy od planu i datasetu; np. plan Standard może mieć bardzo głęboką historię L0, ale znacznie krótszą dla depth. [Databento pricing](https://databento.com/pricing). citeturn755743search0 Dla OPRA koniecznie trzeba zapisać provenance: dokumentacja Databento mówi, że dane sprzed 28 marca 2023 r. pochodzą z innego procesu i są subsamplowane, więc badanie mikrostruktury przez cały okres 2013–2026 bez rozróżnienia regime'u danych byłoby metodologicznie błędne. citeturn755743search1

Massive, dawniej Polygon.io, jest wygodniejszy jako zunifikowane API dla rynku USA. Aktualne plany Stocks obejmują m.in. bezpłatny EOD z ograniczeniami, a płatne poziomy zwiększają historię i redukują opóźnienie; ceny typu $29/$79/$199 dotyczą konkretnych planów i nie powinny być wpisywane do architektury jako stała. [Massive pricing](https://massive.com/pricing). citeturn755743search6turn755743search7 Massive udostępnia również MCP, ale MCP jest warstwą dostępu dla agenta, a nie substytutem własnej warstwy danych. citeturn755743search4turn755743search5

Dane powinny mieć co najmniej taki model jakości:

| Problem | Co się psuje bez obsługi | Rozwiązanie |
|---|---|---|
| Splity | fałszywe −50%/+100% ruchy | raw + split-adjusted |
| Dywidendy | błędny total return | osobny cashflow/adjusted series |
| Delisting | survivorship bias | historyczne universe + delisted |
| Symbol changes | przerwane szeregi | stabilny instrument_id |
| Futures rolls | sztuczne luki | raw contracts + jawna metoda continuous |
| Bid/ask | zawyżona rentowność | symulować spread/side |
| Missing bars | fałszywe wskaźniki | session calendar + quality flags |
| Timezones/DST | przesunięte sygnały | UTC storage + exchange timezone |
| Makro revisions | look-ahead | vintage / point-in-time |
| Fundamental revisions | look-ahead | PIT filing timestamps |
| News timestamps | leakage | first_seen timestamp |
| Universe membership | survivorship | PIT constituents |
| Vendor corrections | niereprodukowalność | dataset version/hash |

Dla makro **ALFRED** jest dużo ważniejszy niż zwykłe pobieranie obecnej historii z FRED, ponieważ pozwala odzyskać wartości dostępne w konkretnym dniu historycznym — czyli vintage. Bez tego model może nieświadomie „znać” rewizję PKB, która nie była dostępna w momencie sygnału. [ALFRED/FRED](https://fred.stlouisfed.org/docs/api/fred/). citeturn604896search6turn604896search8

W przypadku poszczególnych rynków zakres cech powinien się różnić:

| Rynek | Szczególnie ważne dane | Czego nie przenosić bezmyślnie |
|---|---|---|
| Akcje/ETF | corporate actions, constituents PIT, breadth, short interest, fundamentals | 24/7 crypto assumptions |
| Indeksy | breadth, futures basis, constituents, options | wolumen spot indeksu jako prawdziwy wolumen |
| FX | sesje, spot vs futures, rate differentials, COT | centralized order book — spot FX go nie ma |
| Futures | contract calendar, roll, open interest, term structure, COT, L2 | continuous contract jako „prawdziwa” cena |
| Opcje | full chain, IV surface, Greeks, OI, volume, skew, term structure | sama cena opcji bez implied vol |
| Krypto | 24/7, funding, basis, OI, liquidation, exchange-specific books, on-chain | jedna giełda jako cały rynek |
| GPW | kalendarz GPW, corporate actions, fixing, płynność, PIT WIG/WIG20 | amerykańskie dane jako proxy jakości |

Dla **opcji** raport trzeba zdecydowanie rozszerzyć. Cboe oferuje historyczne quote intervals zawierające NBBO oraz opcjonalnie implied volatility i Greeks, a produkty Hanweck dostarczają w czasie rzeczywistym m.in. delta, gamma, vega, theta i rho. citeturn604896search0turn604896search11 Powinny wejść co najmniej ATM IV, realized-vs-implied, IV rank/percentile, 25Δ risk reversal, butterfly/skew, term structure, put/call volume i OI, gamma/vanna/charm oraz agregowane exposure. „Gamma Exposure” trzeba jednak oznaczać jako **modelowaną wielkość zależną od założeń**, a nie bezpośrednio obserwowalny stan dealerów.

Dane **COT** powinny być osobną klasą danych dla futures/FX/commodities. CFTC publikuje Legacy COT sięgające 1986 r., a bardziej szczegółowe Disaggregated/TFF od 2006 r.; aktualne pliki są publikowane przez CFTC. [CFTC COT](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm). citeturn604896search4turn604896search5

Dla krypto warto dołączyć funding rates, futures basis, perpetual basis, open interest, liquidation flows, stablecoin flows oraz on-chain. Glassnode ma REST API z live i historical on-chain metrics, ale dostęp API jest elementem planów komercyjnych, więc nie należy projektować architektury tak, jakby te dane były bezpłatnym standardem. citeturn604896search9turn604896search10

Dla polskiego rynku trzeba rozróżniać **WIG i WIG20**. WIG jest indeksem dochodowym, natomiast metodologia WIG20 opisuje indeks cenowy; różnica ma bezpośrednie znaczenie przy długookresowych benchmarkach. GPW Benchmark publikuje oficjalne metodologie i dane dotyczące rewizji składu, więc historyczna analiza breadth lub rotacji musi używać składów obowiązujących w danym momencie, a nie obecnych spółek cofniętych w przeszłość. citeturn100394search1turn100394search2

Breadth i relacje międzyrynkowe powinny obejmować m.in. advance/decline, % instrumentów powyżej MA50/MA200, new highs/lows, equal-weight vs cap-weight, sektorowe relative strength, rolling correlation/beta, PCA/factor exposure, credit vs equity, rates vs growth, USD vs commodities i volatility complex. To często daje modelowi więcej informacji o stanie rynku niż dodanie piętnastego oscylatora RSI.

## ML, modele bazowe, probabilistyka i AI opisujące rynek

Najważniejszą zmianą wobec tradycyjnego „forecast next close” powinno być przejście na **target probabilistyczny**. Cena za pięć dni jest jednym z najtrudniejszych i najmniej użytecznych targetów. Dużo lepsze są: stopa zwrotu, rank cross-sectional, prawdopodobieństwo przekroczenia bariery, expected shortfall, przyszła zmienność, max adverse excursion, max favorable excursion albo etykieta triple-barrier.

Dla danych tabelarycznych LightGBM, XGBoost i CatBoost nadal powinny być obowiązkowymi baseline'ami, zanim zacznie się używać dużych modeli foundation. LightGBM wspiera objective `quantile`, XGBoost 3.4.2 ma `reg:quantileerror` i wiele wartości `quantile_alpha`, a CatBoost posiada Quantile i MultiQuantile. citeturn446076search5turn446076search6turn446076search7 Dzięki temu można otrzymać P10/P50/P90 bez wielkiego modelu językowego.

Przykładowo model nie powinien zwracać:

> „Cena za 5 dni wyniesie 112,40.”

Powinien zwrócić coś bliższego:

> `P10=-3.1%, P50=+0.4%, P90=+4.0%, P(return>0)=0.56, interval_coverage_est=0.81`

i dopiero potem warstwa decyzyjna ocenia, czy ten rozkład daje sensowny stosunek oczekiwanego wyniku do kosztów i ryzyka.

Nixtla nadal jest jednym z najlepszych ekosystemów do budowy takiej warstwy: StatsForecast, MLForecast i NeuralForecast są aktywnie rozwijane i publikowane na Apache 2.0. [Nixtla repositories](https://github.com/orgs/Nixtla/repositories). citeturn446076search0 W raporcie ze stron 14–15 kierunek ten jest trafny. fileciteturn1file5

Istotna architektura eksperymentalna powinna porównywać:

| Klasa | Modele | Status |
|---|---|---|
| Naiwne | last value, random walk, seasonal naive | **obowiązkowy baseline** |
| Statystyczne | ARIMA/ETS, exponential smoothing | dojrzałe |
| Volatility | GARCH/EGARCH/GJR, HAR-RV | dojrzałe dla zmienności |
| Tree ML | LightGBM/XGBoost/CatBoost | bardzo praktyczne |
| Deep TS | NBEATS, NHiTS, TFT, PatchTST | dojrzałe badawczo |
| Regime | HMM, Markov switching | użyteczne jako latent state |
| TSFM | Chronos‑2, TimesFM, TiRex, Toto, Moirai | obiecujące, ale nie gwarantują alfa |
| Conformal | split/rolling/adaptive conformal | bardzo wartościowe dla coverage |
| Online | River / incremental learners | wartościowe przy drift |
| RL | PPO/SAC/DQN itp. | **eksperymentalne i bardzo łatwe do przeuczenia** |

Statsmodels ma bezpośrednio modele Markov Switching/Markov Regression, a hmmlearn daje GaussianHMM, dzięki czemu reżim może stać się zmienną wejściową zamiast luźnej etykiety typu „rynek byka”. citeturn272234search0turn272234search28

River jest wart dodania jako brakujący element raportu, ponieważ realizuje **online machine learning, drift detection i progressive validation**. To nie znaczy, że należy trenować wszystko tick po ticku; oznacza, że system może monitorować zmianę rozkładu i aktualizować wybrane lekkie modele. [River](https://riverml.xyz/). citeturn158813search6

Triple barrier i meta-labeling są również istotną luką. Metody te nie magicznie tworzą alfa, ale lepiej dopasowują problem ML do decyzji transakcyjnej niż „czy następna świeca będzie zielona”. Open-source implementacje istnieją m.in. w mlfinpy/pochodnych mlfinlab i nowych projektach AFML. citeturn822282search0turn822282search5 Nowy projekt quantowl implementuje również triple-barrier, meta-labels, purged CV, PBO i Deflated Sharpe, ale jest młody i nie ma dojrzałości dużych bibliotek, więc traktowałbym go jako referencję/źródło testów, nie od razu jako krytyczną zależność produkcyjną. citeturn822282search7

W **TSFM** najbardziej rozsądnym kandydatem do własnego hostowania pozostaje obecnie Chronos‑2: jest otwarty na Apache 2.0 i wspiera univariate, multivariate oraz covariates. [Chronos](https://github.com/amazon-science/chronos-forecasting). citeturn464515search0turn464515search2 TimesFM wymaga dokładnego rozróżnienia między kodem, wersją modelu i warunkami wag; dla 3.0 pretrained weights nie są po prostu swobodnym Apache 2.0 do dowolnego komercyjnego wdrożenia. [TimesFM](https://github.com/google-research/timesfm). citeturn464515search5

Kluczowy wniosek empiryczny jest mniej spektakularny niż marketing: na finansach TSFM trzeba traktować jako **challenger**, nie jako automatycznego championa. Badanie finansowe z czerwca 2026 na pięciu płynnych akcjach USA wykazało ograniczone i nierówne korzyści względem prostych benchmarków. citeturn464515academia10 Jeszcze świeższe badanie z 9 listopada 2026 r., przeprowadzone na kontraktach ES/NQ, ostrzega, że dobre wyniki modeli na ogólnych benchmarkach nie przekładają się automatycznie na bardzo niskosygnałowe intraday returns. citeturn464515academia11

Dlatego benchmark modeli w Twoim systemie powinien mieć formę „champion–challenger”:

| Target | Baseline | Challenger |
|---|---|---|
| return 1D/5D | zero/random walk | LightGBM/TSFM |
| realized vol | HAR/GARCH | NeuralForecast/Chronos |
| barrier hit | logistic/LightGBM | ensemble |
| cross-sectional rank | linear/LightGBM | neural ranker |
| interval | historical quantile | quantile ML/conformal |
| anomaly | z-score | IsolationForest/Matrix Profile |

**Conformal prediction** warto potraktować jako warstwę kalibracji, a nie kolejny „model”. Model przewiduje rozkład; conformal sprawdza, czy przedział rzeczywiście osiąga np. 80% pokrycia poza próbą. W nowszej literaturze forecastowej podkreśla się rozdzielenie calibration od sharpness — bardzo wąski przedział jest bezużyteczny, jeżeli rzeczywiste wyniki często z niego wypadają. citeturn471153search12

Nie wystarczy też CRPS/WQL. System powinien zapisywać dla każdego horyzontu co najmniej MAE/RMSE dla punktowych prognoz, pinball loss dla kwantyli, coverage i average interval width dla przedziałów, Brier/log-loss dla probabilistycznej klasyfikacji, reliability/calibration curves, rank IC dla rankingów i finalnie metryki ekonomiczne po kosztach.

LLM powinien dostać **snapshot analityczny**, nie obraz i nie surową historię miliona ticków. Przykładowy kontrakt:

```json
{
  "instrument": "WIG20",
  "as_of": "2026-11-23T16:50:00+01:00",
  "timeframe": "1D",
  "data_provenance": {
    "vendor": "official_or_licensed_source",
    "adjustment": "price_index",
    "dataset_version": "sha256:..."
  },
  "market_state": {
    "trend": "up",
    "adx14": 27.4,
    "rsi14": 61.2,
    "atr_pct": 1.37,
    "distance_avwap_pct": 2.1
  },
  "structure": {
    "support": [2450.0, 2395.0],
    "resistance": [2528.0],
    "regime": "trend_medium_vol"
  },
  "forecast": {
    "horizon_bars": 5,
    "q10_return": -0.028,
    "q50_return": 0.004,
    "q90_return": 0.034,
    "p_positive": 0.56,
    "calibration_window": 500
  },
  "validation": {
    "model_id": "lgbm_q_2026_11_17",
    "oos_coverage_80": 0.79,
    "last_drift_test": "ok"
  }
}
```

LLM może następnie otrzymać instrukcję: „Opisuj wyłącznie pola wejściowe, rozdziel fakt od interpretacji, nie dodawaj poziomów nieobecnych w JSON, pokaż konflikty sygnałów, zawsze podaj horyzont i niepewność”.

PydanticAI nadaje się do wymuszania struktury odpowiedzi; jego dokumentacja obejmuje structured outputs oraz walidację Pydantic. [PydanticAI](https://ai.pydantic.dev/). citeturn234658search0turn234658search3 LangGraph jest dobrym kandydatem, gdy potrzebny jest trwały graf stanów, checkpointing, replay i human-in-the-loop. [LangGraph](https://docs.langchain.com/oss/python/langgraph/). citeturn234658search6turn234658search7

MCP może wystawiać agentowi np. `get_market_snapshot`, `compute_indicator`, `get_forecast`, `run_backtest`, `get_event_calendar`, ale jest również nową powierzchnią ataku. Oficjalna specyfikacja MCP wskazuje, że serwery i narzędzia mogą otwierać ścieżki do wykonania kodu lub dostępu do danych; wywołania o skutkach ubocznych wymagają wyraźnej kontroli i możliwości odmowy. [MCP specification](https://modelcontextprotocol.io/specification/2026-07-28). citeturn234658search8turn234658search9

Najbezpieczniejszy podział uprawnień to: LLM może odczytać przygotowany snapshot, wywołać whitelistowane funkcje analityczne i utworzyć propozycję; **nie powinien mieć swobodnego dostępu do shell, kluczy API, brokera ani funkcji składania zleceń**. Agent generujący własny Python powinien działać w ephemerycznym sandboxie bez sekretów, z limitem CPU/RAM/czasu i odłączonym brokerem.

RL zostawiłbym na koniec. Problemem nie jest brak bibliotek, lecz to, że agent łatwo uczy się błędów symulatora: zbyt taniej płynności, nierealistycznego fillu, stałego spreadu albo informacji niedostępnej w chwili decyzji. Jeżeli nie potrafimy udowodnić przewagi prostego policy rule lub supervised modelu po kosztach, RL zwykle zwiększa przestrzeń przeuczenia.

## Walidacja, wykresy i architektura produkcyjna

Największą przewagą profesjonalnego systemu nad „aplikacją ze wskaźnikami” będzie prawdopodobnie nie AI, lecz **metodologia walidacji**.

Zwykły losowy train/test split jest w finansowych szeregach zazwyczaj niewłaściwy. Walk-forward powinien odtwarzać prawdziwą kolejność: trenujemy wyłącznie na przeszłości, prognozujemy przyszłość, przesuwamy zegar. CPCV pozwala natomiast budować wiele kombinatorycznych ścieżek OOS i badać stabilność selekcji; nie powinno być mylone z czystym „jak system działałby dzień po dniu”. Dokumentacja skfolio sama opisuje tę różnicę. [skfolio CPCV](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html). citeturn158813search0turn158813search2

Do raportu warto obowiązkowo dodać **Deflated Sharpe Ratio**, który koryguje klasyczny Sharpe o selection bias i nienormalność zwrotów, oraz **Probability of Backtest Overfitting**. citeturn158813search10turn158813search11 White's Reality Check i Hansen SPA odpowiadają na problem wielokrotnego testowania dużej liczby strategii, czyli dokładnie to, co wystąpi, gdy narzędzie pozwala kombinować setki wskaźników. citeturn471153search0turn471153search2

Pełna macierz testowa powinna wyglądać tak:

| Test | Co wykrywa |
|---|---|
| expanding/rolling walk-forward | rzeczywiste pogorszenie w czasie |
| purge + embargo | leakage przez nakładające się etykiety |
| CPCV | dystrybucję możliwych ścieżek OOS |
| untouched final holdout | ostateczną kontrolę selekcji |
| Deflated Sharpe | zawyżony Sharpe po wielu eksperymentach |
| PBO | prawdopodobieństwo przeuczenia selekcji |
| Reality Check / SPA | data snooping wielu reguł |
| block/bootstrap | niepewność przy autokorelacji |
| Monte Carlo trade order | path dependency |
| parameter perturbation | kruchość optimum |
| slippage stress | wrażliwość na execution |
| regime split | stabilność w trend/range/high-vol |
| submarket split | generalizacja cross-asset |
| delayed signal | wrażliwość na latencję |
| missing data stress | odporność operacyjna |

Model kosztów powinien być osobnym modułem. Backtest „close → następny close” bez spreadu i slippage jest szczególnie niebezpieczny przy krótkich horyzontach. Minimalny koszt transakcyjny to prowizja + spread + slippage; przy większym kapitale dochodzi market impact, ograniczona dostępność na poziomach book i udział w wolumenie. NautilusTrader jest interesujący właśnie dlatego, że jego architektura zdarzeniowa ma fill/fee/latency/order-book models i stara się zachować wspólny rdzeń między backtestem a live. citeturn353852search8turn353852search10

Do szybkiego researchu wektorowego vectorbt nadal bywa bardzo produktywny, ale jego warunki licencyjne muszą wejść do przeglądu prawnego komercyjnego produktu. citeturn822282search3 LEAN jest alternatywą Apache 2.0 z obsługą wielu klas aktywów. citeturn822282search4

Warstwa wykresu powinna przejść z „renderowania wskaźników” do **versioned analytical canvas**. Lightweight Charts ma primitives pozwalające budować własne obiekty/rysunki, a wrapper Python potrafi aktualizować dane i markery. [Lightweight Charts](https://tradingview.github.io/lightweight-charts/) i [lightweight-charts-python](https://github.com/louisnw01/lightweight-charts-python). citeturn234658search10turn234658search11

Każda adnotacja powinna mieć własną tożsamość:

```json
{
  "annotation_id": "ann_01J...",
  "instrument_id": "WIG20",
  "timeframe": "1D",
  "type": "support_zone",
  "start_ts": "2026-10-05T00:00:00Z",
  "end_ts": null,
  "price_low": 2412.5,
  "price_high": 2431.0,
  "method": "pivot_cluster_v3",
  "confidence": 0.72,
  "created_as_of": "2026-11-23T15:00:00Z",
  "feature_version": "features_37",
  "source_data_hash": "sha256:...",
  "status": "active"
}
```

To rozwiązuje bardzo praktyczny problem: użytkownik może wrócić do wykresu po miesiącu i zobaczyć **co system wiedział wtedy**, a nie tylko dzisiejszą, przeliczoną wersję strefy.

Interfejs powinien mieć multi-timeframe synchronization, wspólny crosshair, replay, możliwość „freeze as-of”, przełączanie raw/adjusted, panel provenance, panel forecast calibration oraz możliwość kliknięcia adnotacji i zobaczenia algorytmu, który ją wygenerował. Plotly jest świetny do badań i raportów; Lightweight Charts lepiej odpowiada doświadczeniu terminalowemu. Raport trafnie rozróżnia te zastosowania na stronach 12–14. fileciteturn0file0

Architektura produkcyjna mogłaby wyglądać następująco:

```text
                       ┌──────────────────────┐
Market APIs/WebSockets │ Ingestion adapters   │
───────────────►       │ Massive/DB/CCXT/GPW │
                       └──────────┬───────────┘
                                  ▼
                        Raw immutable storage
                      Parquet / object storage
                                  │
                     quality + PIT normalization
                                  ▼
                   ┌─────────────────────────────┐
                   │ Arrow / Polars / DuckDB     │
                   │ feature + indicator engine  │
                   └─────────────┬───────────────┘
                                 │
             ┌───────────────────┼──────────────────┐
             ▼                   ▼                  ▼
        TA/structure       ML/forecast       microstructure
             │                   │                  │
             └───────────────────┼──────────────────┘
                                 ▼
                         Analytical Snapshot
                                JSON
                     ┌───────────┴───────────┐
                     ▼                       ▼
             Chart annotations          LLM commentary
                     │                       │
                     └───────────┬───────────┘
                                 ▼
                     Streamlit/Dash/Reflex UI
                                 │
                     audit + experiment store
                                 │
                 MLflow / model registry / metrics
```

Polars powinien być głównym silnikiem dataframe przy nowych dużych potokach, a DuckDB warstwą ad-hoc SQL nad Parquet. Polars oferuje parallel execution i streaming, a DuckDB bezpośrednio czyta Parquet, Pandas, Polars i Arrow. [Polars](https://docs.pola.rs/) i [DuckDB](https://duckdb.org/docs/stable/clients/python/overview). citeturn353852search4turn353852search6

Do dużej liczby danych live można rozważyć dwa różne modele bazodanowe. TimescaleDB daje relacyjny PostgreSQL i continuous aggregates, które są materializowane i odświeżane w tle. citeturn577025search18turn577025search24 QuestDB jest mocno zoptymalizowany pod szereg czasowy i oferuje obecnie oficjalne ścieżki Python/Arrow/Polars oraz protokoły ingest/query. citeturn577025search7turn577025search10turn577025search11 W małym MVP nie potrzebowałbym żadnej z nich: Parquet + DuckDB wystarczy zaskakująco długo.

FastAPI nadaje się na API pomiędzy silnikiem a UI i ma obsługę WebSocket/async. citeturn577025search9turn577025search31 Dłuższe backtesty i retraining nie powinny blokować procesu webowego; trafiają do workerów/task queue. Cache powinien rozróżniać immutable history od mutable latest bars. Event bus staje się potrzebny dopiero wtedy, gdy dochodzą live ticks, wiele usług i wykonanie.

MLflow Model Registry daje model lineage, wersje, aliasy i tagi, więc nadaje się do modelu champion/challenger. [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/). citeturn577025search3turn577025search5 Do tego dochodzi własny manifest eksperymentu: git SHA, lockfile, wersja datasetu, zakres czasu, universe, target, feature schema, random seed i hardware.

Minimalny kod integracyjny może wyglądać tak; celowo używa klasycznego modelu kwantylowego zamiast udawać, że foundation model jest zawsze lepszy:

```python
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import talib
from lightgbm import LGBMRegressor
from pydantic import BaseModel, Field

# 1. OHLCV
df = load_point_in_time_ohlcv("INSTRUMENT")
df = df.sort_index()

# 2. Deterministyczne cechy
df["rsi14"] = talib.RSI(df["close"], timeperiod=14)
df["atr14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
df["ema20"] = talib.EMA(df["close"], timeperiod=20)
df["ema50"] = talib.EMA(df["close"], timeperiod=50)
df["ret1"] = df["close"].pct_change()
df["rv20"] = df["ret1"].rolling(20).std() * np.sqrt(252)

# Anchored VWAP / pivots / profile / regime:
df = add_avwap(df)
df, zones = detect_support_resistance(df)
df["regime"] = estimate_regime(df)

# 3. Target: przyszła stopa zwrotu, nie przyszła cena
h = 5
df["target"] = df["close"].shift(-h) / df["close"] - 1

features = ["rsi14", "atr14", "ema20", "ema50", "rv20", "ret1"]
sample = df.dropna(subset=features + ["target"])

# 4. Chronologiczny split; w realnym systemie rolling walk-forward
cut = int(len(sample) * 0.8)
train, test = sample.iloc[:cut], sample.iloc[cut:]

# 5. Probabilistyczne kwantyle
models = {}
pred = {}
for q in (0.10, 0.50, 0.90):
    m = LGBMRegressor(
        objective="quantile",
        alpha=q,
        n_estimators=400,
        learning_rate=0.03
    )
    m.fit(train[features], train["target"])
    models[q] = m
    pred[q] = m.predict(test[features])

# 6. Forecast snapshot na ostatniej dostępnej obserwacji
i = -1
forecast = {
    "q10": float(pred[0.10][i]),
    "q50": float(pred[0.50][i]),
    "q90": float(pred[0.90][i]),
}

# 7. Backtest MUSI korzystać z sygnałów dostępnych as-of,
# z rzeczywistym spread/slippage/cost model.
bt = run_walk_forward_backtest(
    forecasts=pred,
    bars=test,
    cost_model="spread_slippage_v2"
)

# 8. Ustrukturyzowany kontrakt dla LLM
class Commentary(BaseModel):
    regime: str
    facts: list[str]
    conflicts: list[str]
    risk_notes: list[str]
    summary: str
    confidence: float = Field(ge=0, le=1)

snapshot = build_snapshot(
    bars=df,
    zones=zones,
    forecast=forecast,
    backtest_validation=bt.validation_metrics
)

commentary: Commentary = llm_generate_structured(
    schema=Commentary,
    input_data=snapshot,
    rule="Use only supplied numeric facts; invent no levels."
)

# 9. Wykres
annotations = snapshot_to_chart_annotations(snapshot)
render_chart(df, annotations, commentary)
```

LightGBM oficjalnie wspiera quantile objective, więc ten szkielet nie opiera się na nieistniejącej funkcjonalności. citeturn446076search5 W wersji zaawansowanej LightGBM można zamienić lub zestawić z Chronos‑2, NHiTS, PatchTST czy ensemble, ale **interfejs forecastu pozostaje ten sam**. To bardzo ważne: UI i LLM nie powinny wiedzieć, czy kwantyl pochodzi z GARCH, drzewa czy TSFM.

Demonstracyjny scenariusz dla **WIG20** wyglądałby następująco. System pobiera oficjalnie/licencjonowane OHLCV i przechowuje historię składu indeksu PIT; WIG20 jest indeksem cenowym, co musi być uwzględnione przy porównywaniu z WIG i strategiami total-return. citeturn100394search1turn100394search2 Na interwale 1D oblicza trend, ATR, realized volatility, RSI, ADX, AVWAP od ostatniego istotnego pivotu i strefy S/R; równolegle oblicza breadth z historycznie właściwego składu. Na 60 min mierzy zmianę intraday structure i RVOL. Moduł reżimu klasyfikuje stan jako np. `trend_low_vol`, `trend_high_vol`, `range_low_vol` lub `stress`. Forecast zwraca pięciodniowe P10/P50/P90, a drugi model przewiduje 20-dniową realized volatility. UI pokazuje obie prognozy, ale nie nazywa ich „celem ceny”. LLM dostaje wszystkie liczby i generuje komentarz w rodzaju: „trend dzienny pozostaje dodatni, lecz cena jest X ATR nad AVWAP; przedział P10–P90 jest asymetryczny; breadth potwierdza/nie potwierdza indeksu”. Backtester sprawdza następnie **konkretną regułę decyzyjną**, np. wejście tylko wtedy, gdy p_positive przekracza ustalony próg i prognozowana zmienność mieści się w limicie. To oddziela analizę od strategii.

## Roadmapa, decyzje make/buy i co dokładnie dopisać do istniejącego raportu

Nie budowałbym od razu „Bloomberga + TradingView + funduszu quant + ChatGPT”. Największą wartość osiąga się, gdy kolejne poziomy nie zmieniają kontraktów danych.

| Etap | Zakres | Orientacyjny wysiłek | Główny rezultat |
|---|---|---:|---|
| MVP osobisty | EOD/1h, 50–200 instrumentów, TA, Plotly/LWC, LightGBM quantile, komentarz LLM | 4–8 osobotygodni | działający warsztat |
| Research platform | PIT, breadth, options/COT, walk-forward/CPCV, registry, TSFM challengers | kolejne 8–16 | wiarygodne eksperymenty |
| Live analytical | WebSocket, incremental indicators, live annotations, drift monitor | kolejne 8–16 | monitoring rynku |
| Produkcja multi-asset | L2, event bus, HA, auth, audit, execution simulation, observability | 16–32+ | platforma produkcyjna |
| Institutional-grade | exchange licenses, entitlement, compliance, HA/DR, execution | zależne od zakresu | system organizacyjny |

To są **estymacje projektowe**, nie cenniki rynkowe. Koszt danych może zaczynać się niemal od zera przy EOD/MVP, ale szybko przechodzi w setki lub tysiące dolarów miesięcznie przy real-time, opcjach, depth i prawach do profesjonalnego użycia/redystrybucji. Massive ma obecnie kilka poziomów od darmowych/detalicznych do biznesowych, a Databento ma plany zaczynające się m.in. od Standard $199/mies., z wyższymi poziomami dla głębszej historii/uprawnień. citeturn755743search0turn755743search6turn755743search7 Koszty giełdowe i licencyjne trzeba liczyć osobno od ceny API.

Make/buy powinno wyglądać pragmatycznie:

| Element | Make | Buy/use library | Rekomendacja |
|---|---|---|---|
| Feed exchange | nie | vendor | buy |
| Normalizacja/PIT | tak | częściowo | make core |
| Standard TA | nie | TA‑Lib | buy/open-source |
| AVWAP/profile | tak | prosta implementacja | make |
| S/R/pattern engine | tak | scipy jako baza | make |
| Order-flow features | tak | feed + własne | make |
| Forecast baselines | nie | LightGBM/Nixtla | library |
| TSFM | nie | Chronos etc. | challenger |
| Backtest execution | raczej nie | Nautilus/LEAN | library |
| Experiment tracking | nie | MLflow | library |
| LLM orchestration | częściowo | PydanticAI/LangGraph | hybrid |
| Chart annotation model | tak | Lightweight Charts | hybrid |
| Risk policy | **tak** | biblioteki pomocnicze | make |
| Data provenance | **tak** | storage tools | make |

Najważniejsze jest, aby nie uzależnić kluczowych formatów danych od jednej biblioteki. `MarketBar`, `OrderBookSnapshot`, `IndicatorValue`, `ForecastDistribution`, `Annotation`, `Signal`, `BacktestRun` i `ModelVersion` powinny być własnymi schematami domenowymi. Wtedy TA‑Lib można wymienić, LightGBM zastąpić Chronos, a Streamlit zastąpić Dash bez przepisywania całości.

W aktualnym dokumencie bibliografia ma 124 pozycje, ale liczba źródeł nie gwarantuje jakości dowodów; szczególnie benchmarki z Medium, GitHub Gist, blogów producentów i Reddita powinny zostać wyraźnie oddzielone od dokumentacji pierwotnej i badań niezależnych. fileciteturn1file9 Proponuję oznaczać każde twierdzenie tagiem dowodowym: **DOC** – oficjalna dokumentacja, **PEER/PREPRINT** – literatura, **INDEPENDENT BENCH**, **VENDOR BENCH**, **COMMUNITY**, **OWN TEST**. To od razu uniemożliwia traktowanie „autor projektu twierdzi 150×” tak samo jak niezależnego eksperymentu.

Dokładnie do obecnego raportu dopisałbym, w kolejności wpływu na jakość systemu:

- **„Data Integrity, Point-in-Time i mikrostruktura”** — corporate actions, delistings, survivorship, PIT universe, vintage makro, futures rolls, calendars/timezones, bid/ask, L2/L3, OPRA regime change, provenance i dataset hashes. Następnie **„Walidacja statystyczna po data snoopingu”** — walk-forward vs CPCV, PBO, DSR, Reality Check/SPA, bootstrap, Monte Carlo, transaction cost stress i final untouched holdout. Dalej **„Probabilistyczny forecasting i kalibracja”** — quantile regression, CRPS/WQL, pinball, coverage, conformal prediction, reliability i calibration drift. Następnie **„Market microstructure/order flow”** — CVD, OFI, microprice, imbalance, footprint, depth, impact i queue. Potem **„Opcje, futures i alternatywne dane”** — IV surface, skew, term structure, Greeks/GEX, COT, breadth, macro vintage, news/NLP, on-chain. Następnie **„Financial ML”** — triple-barrier, meta-labeling, sample uniqueness/weights, regime models, online learning River, ensembles i causal inference z jasno zaznaczonym ograniczeniem obserwacyjnych danych finansowych. Potem **„AI governance”** — structured JSON, domain validators, agent permissions, MCP security, sandbox, testy halucynacji, eval dataset i provenance komentarza. Dalej **„Produkcyjna architektura danych”** — Arrow/Parquet/Polars/DuckDB, Timescale/QuestDB, FastAPI/WebSocket, workers/event bus, MLflow, observability, reproducibility, secrets i dependency/licence SBOM. Następnie **„Versioned chart annotations i replay”** — annotation schema, as-of view, crosshair/multi-timeframe sync, freeze/replay i historia sygnałów. Dalej **„GPW jako osobny przypadek”** — WIG vs WIG20, historyczne składy, corporate actions, kalendarz, płynność i licencjonowanie danych. Na końcu **„Macierz dowodów”** — dla każdej metody status `dojrzała / umiarkowanie uzasadniona / eksperymentalna / czysto opisowa`, z rozdzieleniem benchmarków niezależnych od vendorowych.

Po tych zmianach proponowany stos docelowy byłby następujący:

| Obszar | Rekomendowany rdzeń na 23.11.2026 | Alternatywa |
|---|---|---|
| Python | 3.13, `uv`, lockfile | 3.14 po testach zależności |
| Tablice | Polars + Arrow | Pandas compatibility |
| Query/history | DuckDB + Parquet | TimescaleDB |
| Live TS | QuestDB lub TimescaleDB | Kafka/Redpanda + object store przy większej skali |
| TA klasyczne | TA‑Lib 0.8.3.post1 | pandas-ta-classic |
| Multi-symbol TA | polars-talib | własne expressions Polars |
| Incremental | TA‑Lib stream/talipp + własne stateful funcs | Numba |
| Geometry | scipy.signal + własne | SMC jako dodatkowa cecha |
| Motifs/anomaly | STUMPY | custom ANN embeddings |
| Regime | statsmodels/HMM/ruptures | tree classifier |
| Forecast baseline | LightGBM/XGBoost + StatsForecast | CatBoost |
| Deep TS | NeuralForecast | Darts |
| TSFM challenger | Chronos‑2 | TimesFM/TiRex/Toto z kontrolą licencji |
| Uncertainty | quantiles + conformal | distributional nets |
| Online | River | custom incremental |
| Backtest research | vectorized własny/vectorbt po review licencji | backtesting.py |
| Event-driven | NautilusTrader | LEAN |
| Validation | własny walk-forward + skfolio CPCV/DSR | custom AFML |
| Charts | Lightweight Charts + Plotly | Dash charts |
| UI | Streamlit MVP → Dash/Reflex produkcja | własny frontend |
| API | FastAPI | — |
| Registry | MLflow | Weights & Biases / custom |
| LLM output | PydanticAI | native structured outputs |
| Agent graph | LangGraph dopiero gdy potrzebny | proste function calling |
| Agent protocol | MCP z whitelistą | zwykłe REST/functions |
| Deployment | Docker + CI/CD | Kubernetes dopiero przy skali |

Najważniejsza korekta kierunku względem pierwotnego raportu jest więc subtelna, ale fundamentalna. **Nie chodzi o to, by Python policzył „wszystkie możliwe wskaźniki” i AI wskazało, czy kurs wzrośnie.** Potencjał Pythona polega na zbudowaniu laboratorium, które potrafi obliczyć niemal dowolną hipotezę rynkową, zapisać dokładnie z jakich danych powstała, zweryfikować ją bez leakage i po kosztach, zmierzyć niepewność, porównać ją z naiwnym baseline'em, a dopiero potem przedstawić wynik człowiekowi w formie wykresu i języka naturalnego. Obecne badania TSFM, w tym najświeższe finansowe testy z 2026 r., nie uzasadniają obietnicy automatycznej przewagi prognostycznej; uzasadniają natomiast budowę bardzo mocnej platformy **probabilistycznego badania i monitorowania rynku**. citeturn464515academia9turn464515academia10turn464515academia11

W tym sensie załączony raport jest dobrym **pierwszym tomem o ekosystemie bibliotek**, lecz po uzupełnieniu o point‑in‑time data, mikrostrukturę, opcje, probabilistyczną kalibrację, financial ML, statystyczną kontrolę data snoopingu, agent security i produkcyjne versioning/provenance stanie się projektem rzeczywiście profesjonalnego narzędzia, a nie jedynie rozbudowanej aplikacji do analizy technicznej. fileciteturn0file0