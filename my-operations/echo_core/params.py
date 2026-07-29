"""Parametry produkcyjne — GENERATOR_PARAMS.md §1, §2, §3.2.

NIE ZMIENIAC bez zmiany dokumentu. Kazda wartosc ma tam odwolanie do
eksperymentu, ktory ja rozstrzygnal.
"""

SCRIPT_VERSION = "1.0.0"

INDIRECT_RAY_COUNT = 500       # §1, e2_bias_orientation / e2_rays_vs_renders
THREAD_COUNT = 1               # §1, e2_thread_budget_confirm (watki DZIELA budzet promieni)
SENSOR_HEIGHT = 1.25           # §1, listener_height + PKL_FORMAT.md (kamera i audio w jednym punkcie)
AVERAGING_DOMAIN = "mag"       # §1, e3_averaging_domain: estymata = (1/N) * suma |STFT|

# Rendery odrzucane zaraz po konstrukcji Simulatora, zanim ruszy pierwsza
# lokalizacja. DLACZEGO: pierwsze ~10 renderow w swiezej instancji ma
# systematycznie WYZSZY szum — zmierzone 2026-07-29 na trzech scenach
# (estymator wariancyjny, bloki po 10 renderow, N=100 na pozycje):
#
#   office_1/33        blok 1 = 0.11286 wobec 0.10130 w stanie ustalonym  (+11.4 %, +4.4 SD)
#   frl_apartment_5/186 blok 1 = 0.03945 wobec 0.03289                     (+19.9 %, +16.1 SD)
#   room_0/43          blok 1 = 0.06984 wobec 0.06329                      (+10.4 %, +7.3 SD)
#
# Efekt jest per KONSTRUKCJA Simulatora, nie per pozycja: po przeniesieniu
# agenta na druga pozycje W TEJ SAMEJ instancji pierwszy blok NIE jest
# podwyzszony (stosunek pierwszy/pozostale = 1.004, 0.999, 0.993 wobec 1.114,
# 1.199, 1.104 na pozycji pierwszej). Dotyczy wiec 18 lokalizacji — po jednej
# na scene — a nie wszystkich 1740.
#
# Dlaczego to naprawiamy, skoro kierunek bledu jest "bezpieczny": sonda pierwszej
# lokalizacji kazdej sceny wypada w calosci w rozgrzewce, wiec zawyza sigma_1,
# a przez N ~ sigma_1^2 zawyza N o ~20-45 %. Te lokalizacje dostalyby WIECEJ
# renderow niz potrzeba, czyli SNR wyzszy od pozostalych — jednorodnosc szumu
# w zbiorze zepsulaby sie dokladnie tak samo, jak przy nieodrzucaniu nadmiaru
# sondy dla orientacji 0 stopni (patrz komentarz w petli po orientacjach).
# WARTOSC 500 (rewizja 2026-07-29, po pelnym census sondy).
# Zmierzony punkt osiadania jest DUZO nizszy — w blokach po 5 renderow nadwyzka
# spada z +16..19 % (r0-4) do ponizej rozrzutu blokowego od r20, a estymata
# skumulowana od renderu 20 do konca lezy 0.35-0.75 % od odniesienia (r50-99).
# Bierzemy jednak 500, bo:
#   - `gpu_memory_scale` udokumentowal 500 renderow jako koniec fazy rozgrzewki
#     dla RSS, pamieci GPU i czasu renderu — spojnosc z tamtym pomiarem;
#   - to 25x zmierzony punkt osiadania, wiec zaden realny transient sie nie zmiesci;
#   - kosztuje 500 x 0.1412 s = 71 s na scene, czyli 21 min na caly zbior (+1.1 %),
#     co jest cena znikoma wobec ryzyka systematycznego biasu w 18 lokalizacjach.
#
# UWAGA na interpretacje: census wykazal, ze PIERWSZA sondowana lokalizacja sceny
# jest systematycznie glosniejsza (mediana percentyla 92 %, Wilcoxon p=0.001), ale
# domiar w stanie ustalonym pokazal, ze to gownie efekt PRZESTRZENNY, nie rozgrzewka
# — `loc_id` rosnie wzdluz siatki punktow, wiec id 0 to rog sceny, czesto przy
# scianach. Np. frl_apartment_3/0 ma w stanie ustalonym 0.09444 przy medianie sceny
# 0.04550. Rozgrzewka dokladala do tego najwyzej kilka procent.
WARMUP_DISCARD = 500

N_PROBE = 8                    # §3.3 pkt 1: sonda 8 renderow przy 0 stopni, podzial 4+4
N_MIN, N_MAX = 6, 64           # §3.2 (24 -> 40 rewizja 2026-07-26; 40 -> 64 po pelnym
                               # census sondy 2026-07-29, patrz WARMUP_DISCARD nizej)
TARGET_SNR = 3.5               # §3.2
SIGNAL_10DEG = 0.0644          # §3.2, mediana z noise_floor_scenes (zakres 0.0639-0.0662)

ANGLES_DEG = tuple(range(0, 360, 10))   # 36 orientacji co 10 stopni, §2
N_ANGLES = len(ANGLES_DEG)

S_PER_RENDER_SPEC = 0.2606     # §4, srednia wazona po 18 scenach
MEAN_N_SPEC = 9.83             # §3.1, srednia po 12 zmierzonych pozycjach

CAMERA_RESOLUTION = (128, 128)  # PKL_FORMAT.md
CAMERA_HFOV = 90.0              # PKL_FORMAT.md (kontrola negatywna przy 70 st.: RGB RMSE 33.6)

BYTES_PER_SAMPLE = 2 * 257 * 166 * 2 + 128 * 128 * 4 + 128 * 128 * 4
