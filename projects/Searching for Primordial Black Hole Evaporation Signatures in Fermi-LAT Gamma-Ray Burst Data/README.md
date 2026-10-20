# Searching for Primordial Black Hole Evaporation Signatures in Fermi-LAT Gamma-Ray Burst Data

# Can a Tiny Black Hole Show Up in Fermi's Gamma-Ray Burst Data?

I used real NASA data to check whether any flashes of gamma rays in space could be a tiny black hole running out of life.

**I did not find a black hole.** The goal was never to prove they exist. I wanted to rank the bursts that look the most like what theory predicts, and be honest about what that does and does not mean.

---

## The idea (simple version)

- Stephen Hawking showed that black holes slowly leak energy and shrink. This is called **Hawking radiation**.
- The smaller a black hole gets, the hotter it is and the faster it leaks, a bit like an ice cube that melts faster as it gets smaller.
- Some tiny black holes (**primordial black holes**, or PBHs) might have formed right after the Big Bang.
- A PBH with a mass of about **5 × 10^11 kg** would be finishing its life around the present age of the Universe, ending in a burst of gamma rays.
- NASA's **Fermi-LAT** telescope has seen hundreds of gamma-ray bursts (GRBs). So I asked: do any of them look like that final flash?

## The data

- **Source:** NASA HEASARC, Fermi-LAT GRB catalog (table `fermilgrb`), loaded with `astroquery`
- **Size:** 231 gamma-ray bursts, 111 columns
- **Columns I kept:** burst duration `gbm_cat_t90`, fluence `gbm_cat_fluence`, flux `like_best_flux`, spectral index `like_best_grbindex`, detection strength `like_lat_ts`, highest photon energy `ext_emission_max_ene`, and `redshift`
- **Missing data:** 186 of 231 bursts have no redshift (only 45 do). 12 are missing fluence. Some photon energies are fake placeholder values (-670), which I changed to "missing".

## What I did, step by step

1. Loaded the catalog and turned it into a pandas table.
2. Checked what data was missing and kept only the useful columns.
3. Calculated the Hawking temperature of a 5 × 10^11 kg black hole with `T = ħc³ / (8πGMk_B)`. Result: **21.15 MeV**.
4. Made plots: highest photon energy, burst duration, spectral index, and a correlation heatmap.
5. Picked the **short bursts** (T90 under 2 seconds), because a black hole's final flash should be quick. That gave **21 bursts**, about 9% of the sample.
6. Ranked those 21 bursts with **two scores** (below).
7. Wrote up which bursts stand out and why none of them is proof.

## The two scores

**Score 1: a quick first score (`PBH_score`)**

`3 × (1 / duration) + 2 × log(1 + TS) + (minus the spectral index) + 1 if no redshift + 1 if the photon energy is valid`

Top 5: GRB170127067 (35.6), GRB200415367 (31.4), GRB160702516 (24.4), GRB190606080 (22.7), GRB090510016 (22.1)

**Score 2: the PBH Consistency Index (the main one)**

Each number is squeezed to a 0 to 1 scale (min-max scaling), then added up with weights:

| What | Weight | Direction in the code |
|---|---|---|
| Burst duration (T90) | 40% | shorter scores higher (uses 1 / T90) |
| Highest photon energy | 30% | higher scores higher |
| Detection strength (TS) | 20% | higher scores higher |
| Spectral index | 10% | the code uses minus the index |

Only bursts with complete data get scored. Two short bursts (GRB090227772 and GRB110529034) had the placeholder energy, so they were dropped, which means **19 bursts** were actually scored, even though the notebook text says 21.

The score is a **ranking**. It is not a probability.

## Results

Top of the PBH Consistency Index:

| Rank | Burst | Score | Duration (s) | TS | Max photon energy (MeV) | Redshift |
|---|---|---|---|---|---|---|
| 1 | GRB090510016 | 0.568 | 0.960 | 2923.6 | 30000 | 0.903 |
| 2 | GRB170127067 | 0.506 | 0.128 | 35.9 | 510 | none |
| 3 | GRB200415367 | 0.391 | 0.144 | 30.6 | 1700 | none |
| 4 | GRB190606080 | 0.308 | 0.224 | 9.4 | 1600 | none |
| 5 | GRB160702516 | 0.297 | 0.200 | 19.3 | 4800 | none |

Other numbers from the notebook:

- Hawking temperature: **21.15 MeV**
- Burst duration: shortest 0.128 s, median 31.23 s, mean 54.91 s, longest 478.03 s. So most bursts are long.
- Highest photon energy: median 2100 MeV, mean 5972 MeV, maximum 94000 MeV
- Spectral index: mean -2.19, median -2.09
- Correlation: the strongest link is between detection strength (TS) and highest photon energy (r = 0.47), then TS and flux (r = 0.33). Everything else is weak.

**What I concluded:** GRB090510016 scored highest, but it has a redshift of 0.903, so it comes from far across the Universe and is not a PBH. After setting it aside, the strongest candidates are **GRB170127067, GRB200415367, GRB190606080 and GRB160702516**. They are very short, clearly detected, have energetic photons, and have no measured distance. But that is **not proof** of a primordial black hole.

## Notes from my notebook

- **Photon energies are higher than 21 MeV.** Hawking radiation gives a broad range of energies, not just one, and secondary particles can make higher-energy photons. Fermi-LAT is also built to see photons above about 20 MeV, so the catalog leans toward high-energy events.
- **Short does not mean black hole.** Neutron-star mergers and magnetar activity also make short bursts.
- **A spectral index around -2 is normal for ordinary GRBs**, so it can't pick out a PBH by itself.
- **Having no redshift does not rule a burst in or out.** Most bursts simply don't have one.
- **How GRB090510016 was handled:** the code does *not* delete it. It stays in the ranking, and I set it aside in my conclusion because of its redshift.
- **A detail in the code:** the spectral index is negative, and the code uses `-index` with the comment "harder spectra score higher". Because of the minus sign, softer spectra actually get the higher score. It only has a 10% weight, but it can change the order a little.

## Extra notes (added while double-checking, not in the notebook)

- **GRB200415367 looks like the famous burst GRB 200415A** (April 15, 2020, 0.144 s long). Published studies (Svinkin et al., *Nature* 2021, and Fermi-GBM papers) say it was a giant flare from a magnetar in the galaxy NGC 253, about 3.5 million parsecs away, with an energy of roughly 10^44 to 10^46 erg. A 5 × 10^11 kg black hole only holds about 4.5 × 10^35 erg in total. I matched the name by date and duration, so please confirm it in the catalog.
- **21 MeV is the temperature today.** A black hole heats up as it shrinks, so its last moments would be much hotter.
- **The duration (T90) comes from Fermi's GBM instrument**, not from LAT.

## Limitations

- I picked the weights by hand. They are not fitted to data or to a real black-hole model.
- The sample is small (19 scored bursts), so the ranking can shuffle easily.
- Min-max scaling lets one extreme burst (GRB090510016) squash everyone else's scores.
- I did not compare real spectra or light curves with a full Hawking-radiation model.
- I did not check where each burst sits in the sky (for example, inside a nearby galaxy).
- The catalog only has bursts already flagged as GRBs. A real PBH flash might not be in it.

## What I would do next

- Fix the spectral-index sign and re-run the ranking
- Try many different weights to see which bursts stay near the top
- Use a scaling method that one extreme burst can't distort
- Check each burst's energy against how far away it could be
- Compare burst positions with nearby galaxies
- Compare real spectra with a simulated Hawking spectrum
- Look at light curves (a dying black hole should get brighter and harder at the end)
- Use a bigger catalog, like the full Fermi-GBM list
- Try machine learning to separate known mergers and magnetar flares from the rest

## Tools

Python, astroquery, astropy, pandas, numpy, matplotlib, seaborn, scikit-learn (Google Colab)

## How to run

1. Open the notebook in Google Colab.
2. Run the first cell (`!pip -q install astroquery astropy pandas`).
3. Run the rest of the cells in order. You need internet because the data downloads live from NASA.

## Author

Amna
