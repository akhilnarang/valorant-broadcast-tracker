# Recognizer assets

## digits_lr.bin

The score-plate digit classifier. `vision/tools/train_digits.py` rebuilds it byte for byte.

## strip_templates.json

The `CURRENT:` anchor and one template per map name, all 720p grayscale.

The anchor is rows 1-12, columns 18-65 of the autocontrasted 26 Sep 2026 frame at second 14790.

Each map template is the per-pixel median of the 12x50 window after the anchor (columns
`anchor_x + current_w + 2` on, rows `anchor_y` to `anchor_y + 11`) over frames where the score
read 0-0 and the anchor matched at 0.75 or more. The 0-0 score and the map were checked against
the on-screen strip and the series' map order.

| Map | Broadcast | Source | Frames |
|---|---|---|---|
| Abyss, Ascent, Lotus | 26 Sep 2026, live capture | | |
| Summit | 16 Oct 2026, VIT vs LOUD, map 2 | twitch.tv/valorant VOD 2902056335, 7251-7322 s | 72 |
| Haven | 16 Oct 2026, VIT vs LOUD, map 3 | same VOD | |
| Split | 8 Oct 2026, VIT vs NS, map 1 | twitch.tv/valorant VOD 2895001673, 10290-10620 s | 108 |
| Sunset | 10 Oct 2026, NRG vs LOUD, map 2 | twitch.tv/valorant VOD 2896799484, 5676-6006 s | 79 |
| Breeze | 22 Aug 2026, BESTIA vs G2, map 1 | twitch.tv/valorant_americas VOD 2853632850, 2032-2412 s (720p60) | 65 |
| Bind | 11 May 2026, LOUD vs LEVIATÁN, map 1 | youtube.com/watch?v=HL6dGXZ9-tc, 360-478 s | 112 |
| Corrode | 15 Mar 2026, NS vs PRX, map 1 | youtube.com/watch?v=NR1AEXusjac, 474-552 s | 68 |
| Pearl | 8 Jun 2026, VIT vs FUT, map 1 | youtube.com/watch?v=pL4dzDvZkLs, 135-210 s | 62 |
| Fracture | 21 Jun 2026, PRX vs LEV, map 1 | youtube.com/watch?v=yWqPVOvSJPg, 126-231 s | 74 |
| Icebox | 31 Aug 2025, SEN vs NRG, map 2 | youtube.com/watch?v=VdsIJnK_kss, 22-110 s | 89 |

Empty cells were not recorded. Icebox is the only template from the 2025 HUD; its plates are
unreadable to the classifier, so 0-0 was confirmed from the round timer.

## pause_templates.json

Four phrases, each binarized from the phrase region of one real frame:

| Phrase | Region | Broadcast | Test frame |
|---|---|---|---|
| `HALFTIME` | header | 26 Sep 2026 | `halftime-6000.jpg` |
| `HALFTIME` | card | | |
| `TECH PAUSE` | header | 29 Sep 2026, Champions Shanghai | |
| `TIMEOUT` | header | Champions Shanghai, `TIMEOUT - TEAM VITALITY` | `timeout-8385.jpg` |

The scripts that built these templates were not kept, so they cannot be rebuilt exactly. A new
template needs a row here.
