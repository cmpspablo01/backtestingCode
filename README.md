# ORB + 3-Candle Imbalance Backtest

**Status:** MVP Mécanique (ossature du système, filtres discrétionnaires manquants)

Backtest du modèle **15-minute Opening Range Breakout + 3-Candle Imbalance** sur NQ/MNQ avec données Databento.

---

## 📊 Résultats (Juin 2024)

| Métrique | Valeur |
|----------|--------|
| Trades | 31 |
| Win Rate | 35.5% |
| Expectancy | +0.032R/trade |
| Profit Factor | 1.06 |
| Max Drawdown | -7R |
| Best Trade | +2R |
| Worst Trade | -1R |

---

## ✅ Ce qui est implémenté

### 1. Opening Range (09:30 - 09:45)
- Calcule ORH/ORL des 15 premières minutes
- Détecte breakout : close hors du range

### 2. 3-Candle Imbalance
- **Bullish:** `low(c3) > high(c1)` + c2 up (optionnel)
- **Bearish:** `high(c3) < low(c1)` + c2 down (optionnel)
- Entrée au close de c3

### 3. Risk/Reward Asymétrique
- Si `stop < 30 pts` → TP = Entry + 2R
- Si `stop >= 30 pts` → TP = Entry + 1R

### 4. Gestion du Trade
- Stop au min/max des 3 bougies
- Breakeven après +1R
- Max 2 trades/jour, stop après 1er win

### 5. Sortie
- TP ou SL hit, ou EOD
- CSV export avec stats détaillées

---

## ❌ Ce qui manque (Impact probable sur résultats)

### A. **Gap Confirmation** ⚠️ IMPORTANT
Le système original filtre les setups par gap (taille, direction, etc.).
- **Code:** Aucun filtre de gap
- **Impact:** Probablement +faux signaux

**À ajouter:** Vérifier gap > X points dans le sens du break

### B. **Session Bias + Liquidity Draw**
- PDH/PDL (Previous Day High/Low)
- ONH/ONL (Overnight High/Low)
- Bias bullish/bearish/mixed
- Draw vers HOD/LOD
- News filter (CPI, FOMC, etc.)

**Code:** Rien de tout ça
**Impact:** Perte d'une grosse partie de l'edge discrétionnaire

### C. **"Structure Clears" pour Breakeven**
Post original : "move stop to BE once structure clears"
- **Code:** Trigger simple : `high >= entry + 1R`
- **Réalité:** Structure clears = mini swing high break / BOS sur 1m

**Impact:** Peut-être un meilleur BE trigger

### D. **"Clean Break" Filter**
Post original : "clean break of ORH/ORL"
- **Code:** Accepte un close de 0.25 pts au-dessus = pas clean
- **Réalité:** Body > 50% du range, ou close > ORH + 2-3 pts minimum

**Impact:** Filtre les micro-breaks (false breakouts)

### E. **Fenêtre de Trading**
- **Code:** 09:45 - 11:00 (1h15)
- **Réalité:** Peut-être plus stricte (ex: 09:45 - 10:15 seulement)

---

## 🚀 Améliorations Prioritaires

Pour rendre ce backtest plus fidèle au système original :

### 1️⃣ Gap Confirmation (facile à coder)
```
if breakout_direction == "up":
    gap = open - yesterday_close
    if gap < 5 points: skip trade  # Gap trop petit
    if gap < 0: skip trade  # Gap à la baisse = pas bon pour long
```

### 2️⃣ Clean Break Filter (facile à coder)
```
if breakout_direction == "up":
    if close - ORH < 2: skip trade  # Pas assez clean
```

### 3️⃣ Structure Clears pour BE (plus complexe)
Nécessite détection mini swing high / BOS sur 1m
```
# Après entry, cherche une petite structure à 2-3 bougies
# Si break = move stop to BE
```

### 4️⃣ Session Bias (complexe)
Nécessite PDH/PDL/ONH/ONL
```
# À tester sur 3-6 mois : les setups dans le bias du jour
# gagnent-ils plus que contre le bias?
```

---

## 📝 Utilisation

### 1. Setup Python
```bash
pip install pandas numpy databento pytz
export DATABENTO_API_KEY="your_api_key"
```

### 2. Configuration
Modifier `orb_imbalance_backtest.py` :
```python
CFG = Config(
    symbol="NQ.v.0",        # ou "MNQ.v.0"
    start="2024-06-01",
    end="2024-06-30",
    max_trades_per_day=2,
    stop_after_first_win=True,
)
```

### 3. Run
```bash
python orb_imbalance_backtest.py
```

### Output
- `trades_orb_imbalance.csv` : Détail de chaque trade
- Console : Stats (winrate, PF, max DD, expectancy)

---

## 🎯 Interprétation des Résultats

**+0.032R avec 35.5% winrate = légèrement positif**

Mais avant de trader :
1. ✅ Tester sur **6-12 mois** (100+ trades)
2. ✅ Ajouter **gap confirmation** (devrait améliorer WR)
3. ✅ Ajouter **clean break filter** (devrait réduire faux signaux)
4. ✅ Vérifier si **session bias** existe (peut améliorer PF)
5. ✅ Backtest sur **MNQ** aussi (scaling different)

---

## 📚 Données

- **Source:** Databento (GLBX.MDP3)
- **Instrument:** NQ.v.0 (continuous contract, front month)
- **Timeframe:** 1-minute OHLCV
- **Session:** RTH (09:30 - 16:00 NY)

**Coût Databento:** ~$10-25/mois par 1 mois de données 1m (dépend de la bande passante)

---

## 📌 Différences Clés vs Système Original

| Aspect | Système Original | Code Actuel | Écart |
|--------|------------------|------------|-------|
| **ORB** | 09:30-09:45, close dehors | ✅ Identique | ✅ |
| **Imbalance** | FVG 3-candle | ✅ Identique | ✅ |
| **Entry** | Close de la 3e bougie | ✅ Identique | ✅ |
| **TP** | 1R ou 2R (règle 30pts) | ✅ Identique | ✅ |
| **Gap Confirmation** | Filter important | ❌ Absent | ⚠️ |
| **Clean Break** | Filter important | ❌ Absent | ⚠️ |
| **Session Bias** | Filter discrétionnaire | ❌ Absent | ⚠️ |
| **Structure Clears (BE)** | Basé sur structure | ⚠️ Distance only | ⚠️ |

---

## 🔧 TODO

- [ ] Ajouter gap confirmation filter
- [ ] Ajouter clean break filter
- [ ] Ajouter session bias tracking
- [ ] Ajouter structure clears detection
- [ ] Graphiques (equity curve, drawdown)
- [ ] Breakdown par jour de semaine / mois
- [ ] Backtest MNQ
- [ ] CLI args (--start, --end, --symbol)

---

## 📄 License

MIT

---

**Note:** C'est un MVP de test. Le système original sur Reddit contient probablement plus de nuance discrétionnaire. Tester avant de trader avec de l'argent réel.
