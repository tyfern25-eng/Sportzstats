# Sportzstats

Personal NFL stats database and fantasy football (PPR) start/sit tool.

## What's here

| Path | What it is |
|---|---|
| `index.html`, `assets/` | The website (My Teams, Players, Defenses, QB vs Blitz, How it works) |
| `config/leagues.json` | Your leagues: name, lineup slots, roster (player names) |
| `pipeline/build_data.py` | Downloads NFL data and builds the stats + projections |
| `data/*.json` | Output the website reads (rebuilt automatically) |
| `.github/workflows/update-data.yml` | Reruns the pipeline several times a week |

## Updating your rosters

Either edit `config/leagues.json` on GitHub (permanent), or click **Edit roster** on the site (saved in that browser only).

## Running it yourself

```
pip install -r pipeline/requirements.txt
python pipeline/build_data.py            # add --backtest to score past weeks
python -m http.server                    # then open http://localhost:8000
```

To refresh on demand: GitHub → **Actions** → **Update stats and projections** → **Run workflow**.

Data: [nflverse](https://github.com/nflverse).
