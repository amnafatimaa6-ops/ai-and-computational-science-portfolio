# ai-and-computational-science-portfolio

# Space & AI Research 

**Amna Fatima** · BS student · interested in astrophysics, computational science and AI for space research

This repository brings six of my projects together in one place. Most of them use real data from NASA, ESA/Sentinel-2, Fermi and LIGO. Each project sits in its own folder under [`projects/`](projects) with its own README, code and notebook.

I try to be honest about what each project can and cannot show, so every project README has a **Limitations** section.

---

## Projects at a glance

| # | Project | Question it asks | Main tools |
|---|---------|------------------|------------|
| 1 | [Kepler TCE clustering & anomaly analysis](projects/01-kepler-tce-clustering-anomaly) | Do Kepler threshold-crossing events form natural groups, and which ones look unusual? | Python, scikit-learn (KMeans, DBSCAN, Isolation Forest, PCA), Streamlit |
| 2 | [ORBITAL: reinforcement learning for Earth observation](projects/02-orbital-rl-earth-observation) | Can an RL agent pick more informative satellite targets than random choice under a limited budget? | Python, Sentinel-2 (EuroSAT MSI), Q-learning, SARSA |
| 3 | [NeuroForge: a neural network from scratch in C++17](projects/03-neuroforge-cpp-neural-network) | What really happens inside a neural network? Can I build one without a framework? | C++17, NASA Exoplanet Archive, Streamlit |
| 4 | [GW150914: listening to two black holes collide](projects/04-gw150914-gravitational-waves) | Can I find the first gravitational-wave signal in real LIGO data myself? | Python, GWpy, PyCBC, matched filtering |
| 5 | [Primordial black hole evaporation search](projects/05-pbh-evaporation-fermi-lat) | Do any short Fermi-LAT gamma-ray bursts look like an evaporating primordial black hole? | Python, Astroquery, Astropy, Plotly, Streamlit |
| 6 | [Species population change & ML](projects/06-species-population-risk-ml) | How do species population trends group into risk categories, and where does the analysis break down? | Python, scikit-learn, GBIF API, Streamlit |

Projects 1 to 5 are about space and astrophysics. Project 6 is a general data-science project on biodiversity; I kept it because it shows careful thinking about data leakage and bias.

---

## What each project found

**1. Kepler TCE clustering and anomalies**
Applied KMeans (k=4), DBSCAN and Isolation Forest to 34,032 Kepler threshold-crossing events. KMeans gave a silhouette score of about 0.56, and Isolation Forest flagged 1,702 events (about 5%, which is the contamination setting I chose). These are unsupervised patterns in the data, not confirmed planets. The Streamlit dashboard uses synthetic data, so it is separate from the notebook's findings.

**2. ORBITAL**
Built a simulated observation-planning environment from 600 EuroSAT Sentinel-2 patches (19 spectral features, 8 spectral-cluster targets). In the second simulated environment, Q-learning scored a mean reward of 0.536, a greedy baseline 0.524, and random selection -0.328. The gap between Q-learning and greedy is small, and greedy actually gave higher pairwise spectral diversity, so I describe the advantage as modest. Cloud and weather conditions are synthetic.

**3. NeuroForge**
A 9-8-1 neural network (89 parameters) written in C++17 with no ML library, trained on NASA Exoplanet Archive data to classify whether a planet was found by the transit method. One recorded run gave 89.45% accuracy and 87.41% balanced accuracy. It classifies discovery methods of already-listed planets; it does not discover planets.

**4. GW150914**
Downloaded the real Hanford and Livingston strain data, filtered it, found the chirp, compared it to a theoretical waveform from PyCBC, and ran a matched filter. The single-detector matched-filter signal-to-noise ratio peaked at about 16.8.

**5. Primordial black hole search**
Searched 231 Fermi-LAT GRBs from the HEASARC catalogue, selected 21 short bursts (T90 below 2 s), computed a Hawking temperature of about 21 MeV for a 5e11 kg black hole, and built a heuristic "PBH Consistency Index" to rank candidates. It is a prioritisation tool, not a detection. No definitive evidence for primordial black holes was found.

**6. Species population change**
Random Forest and Gradient Boosting reached about 95.8% accuracy on a 24-sample test set, but the labels were computed from the same population values used as inputs, so this is target leakage and the score does not measure real forecasting ability. The README explains this openly.

---

## Repository layout

```
space-ai-research-portfolio/
├── README.md
├── LICENSE
├── .gitignore
└── projects/
    ├── 01-kepler-tce-clustering-anomaly/
    ├── 02-orbital-rl-earth-observation/
    ├── 03-neuroforge-cpp-neural-network/
    ├── 04-gw150914-gravitational-waves/
    ├── 05-pbh-evaporation-fermi-lat/
    └── 06-species-population-risk-ml/
```

Each project folder has its own `README.md`, and `requirements.txt` where the project needs one. Install and run each project from inside its own folder, because the dependencies differ between projects.

Example:

```bash
cd projects/01-kepler-tce-clustering-anomaly
pip install -r requirements.txt
streamlit run app.py
```

---

## Original repositories

These projects were first published as separate repositories. The originals stay online and still host the live Streamlit dashboards.

- [kepler-tce-clustering-anomaly-analysis](https://github.com/amnafatimaa6-ops/kepler-tce-clustering-anomaly-analysis)
- [Orbital](https://github.com/amnafatimaa6-ops/Orbital)
- [Neuro_Forge](https://github.com/amnafatimaa6-ops/Neuro_Forge)
- [Testing_Black_Theory_With_GW150914](https://github.com/amnafatimaa6-ops/Testing_Black_Theory_With_GW150914)
- [Primordial-Black-Hole-Evaporation-Search-Using-Fermi-LAT-Gamma-Ray-Burst-Data](https://github.com/amnafatimaa6-ops/Primordial-Black-Hole-Evaporation-Search-Using-Fermi-LAT-Gamma-Ray-Burst-Data)
- [Predicting-Global-Species-Extinction-Risk-Using-Machine-Learning](https://github.com/amnafatimaa6-ops/Predicting-Global-Species-Extinction-Risk-Using-Machine-Learning)

---

## Skills shown across these projects

- **Astrophysics data:** Kepler, Fermi-LAT, LIGO strain data, NASA Exoplanet Archive, Sentinel-2
- **Machine learning:** clustering, anomaly detection, classification, reinforcement learning, neural networks from first principles
- **Scientific computing:** signal filtering, whitening, spectrograms, matched filtering, Hawking radiation calculations
- **Software:** Python, C++17, Streamlit dashboards, reproducible notebooks
- **Scientific honesty:** each project states its limits, including leakage, synthetic assumptions and heuristic scores

## Contact

GitHub: [@amnafatimaa6-ops](https://github.com/amnafatimaa6-ops)

## License

MIT. See [LICENSE](LICENSE).
