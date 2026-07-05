# Related Work — Clickable Reference Map

Curated links for the paper's Related Work section. Three tiers:

1. **Tier 0 — Team's downloaded papers.** The 6 PDFs already in the
   team's `Documents/Microgrid papers/` folder. Titles, authors, DOIs
   and one-paragraph summaries extracted directly from each PDF.
2. **Tier 1 — Verified core citations.** Foundational papers directly
   relevant to the CSE and EEE sides. Every DOI / arXiv link has been
   cross-checked.
3. **Tier 2 — Search links.** For everything else, the entries below
   are search-URL shortcuts, not fabricated titles. Click, pick a
   real paper, add it to your reference manager.

**Honesty note:** do not cite anything without first opening the DOI /
arXiv link and verifying the paper. This document is a research aid,
not a pre-approved citation list.

---

## 📁 TIER 0 — Papers the team has already downloaded

All six live in `C:\Users\User\Documents\Microgrid papers\`.

### T0-1. AI-Based Forecasting in Renewable-Rich Microgrids

**"AI-Based Forecasting in Renewable-Rich Microgrids: Challenges and
Comparative Insights"**
Osifeko, M., & Munda, J. L.
*IEEE Access* **13** (2025)

- Local file: `AI-Based_Forecasting_in_Renewable-Rich_Microgrids_Challenges_and_Comparative_Insights.pdf`
- DOI: <https://doi.org/10.1109/ACCESS.2025.3591091>
- IEEE Xplore: <https://ieeexplore.ieee.org/document/11087494>

Benchmarks **12 ML/DL models** (Bi-LSTM, GRU, LSTM, Transformer,
CNN-LSTM, XGBoost, HGB, LR, etc.) on 5-year South African national
grid data for load, PV, and wind forecasting. Bi-LSTM wins for
demand and PV (RMSE 303.95 MW / R² 0.9877 for demand; 48.04 MW /
0.996 for PV). XGBoost competitive for PV/wind. Confirms with
Friedman + Nemenyi tests. Key finding: classical ML rivals DL with
good feature engineering.

**Relevance to us:** direct baseline for our forecasting section.
Compare our LSTM/GRU/MLP results against Bi-LSTM here. Same
statistical rigor (Friedman ≈ our McNemar approach).

### T0-2. AI-Based Power Management System for a DC Micro-Grid

**"Artificial Intelligence-Based Power Management System for a DC
Micro-Grid"**
Prakash, G., Dharmaprakash, R., Sujatha, M. S., Parvez S., S., &
Reddy, J. K.
*SSRG Int. J. of Electrical and Electronics Engineering* **12** (10),
110–122 (October 2025)

- Local file: `IJEEE-V12I10P109.pdf`
- DOI: <https://doi.org/10.14445/23488379/IJEEE-V12I10P109>

Solar PV + wind + battery DC microgrid controlled by an ANN-based
Radial Basis Neural Network (RBN). Compares against a baseline
Fuzzy + FO-PID controller. Shows RBN gives smoother voltage under
disturbances. Full MATLAB/Simulink validation.

**Relevance to us:** the ANN-RBN control concept is a direct
analogue to our "AI-Optimized Control" bridge scenario. Cite it as
prior work on ML-in-the-loop microgrid control, then position our
tau-robustness metric as what the ML side of that literature is
missing.

### T0-3. Power Quality Improvement in Microgrids using AI: A Review

**"Power quality improvement in microgrids using artificial
intelligence techniques: A review"**
Shaji John, S., Sony, H. A., Lavanya, V., & Meera, P. S.
*Science and Technology for Energy Transition* **81**, 14 (2026)

- Local file: `Power_quality_improvement_in_microgrids_using_arti.pdf`
- DOI: <https://doi.org/10.2516/stet/2026015>

**Review paper** covering 100+ swarm-based and hybrid AI methods for
PQ improvement in AC microgrids (ANFIS, PSO-ANN, ACO, GWO, GA, PSO,
etc.). Compares by convergence speed, harmonic mitigation, and
transient response. Concludes hybrid ANFIS and PSO-ANN dominate.

**Relevance to us:** a comprehensive survey to cite in our
introduction ("prior work extensively covers X but does not address
Y = robustness margin"). Also gives us a curated list of ~100 further
references to sample from.

### T0-4. Power Quality Analysis of a Microgrid on IEEE 14-bus

**"Power Quality Analysis of a Microgrid-Based on Renewable Energy
Sources: A Simulation-Based Approach"**
Hernández-Mayoral, E., Jiménez-Román, C. R., Enriquez-Santiago, J. A.,
López-López, A., González-Domínguez, R. A., Ramírez-Torres, J. A.,
Rodríguez-Romero, J. D., & Jaramillo, O. A.
*Computation* **12**, 226 (MDPI, November 2024)

- Local file: `computation-12-00226-v2.pdf`
- DOI: <https://doi.org/10.3390/computation12110226>

Builds a **microgrid on the IEEE 14-bus distribution system** in
MATLAB-Simulink. Analyses PQ at every bus and compares against
**IEEE 519** compatibility levels. Quantitatively evaluates PQ
capacity when the MG runs in parallel with the conventional grid.

**Relevance to us:** ⭐ **the closest match to our EEE-side work.**
Same IEEE 14-bus + same IEEE 519 verdict framework. Cite this as
prior work that our EEE teammates directly build on and extend with
the tau-margin CSE bridge. Very likely to appear in our Related Work
section as a primary reference.

### T0-5. Machine Learning-Based Power Quality Prediction for Community MGs

**"Machine Learning-Based Power Quality Prediction in a Microgrid
for Community Energy Systems"**
Jahan, I., Dinh, K. N. D., Blažek, V., Snášel, V., Mišák, S., Pergl,
I., Mohamed, F., & Mechali, A.
*Preprints.org*, 24 February 2026 (VSB-Technical University of
Ostrava + Libyan Authority for Scientific Research)

- Local file: `preprints202602.1410.v1.pdf`
- DOI: <https://doi.org/10.20944/preprints202602.1410.v1>

Compares **9 predictive architectures** (LSTM, GRU, DNN,
CNN1D-LSTM, BiLSTM, attention, Decision Tree, SVM, XGBoost) for
predicting Power Quality Parameters (voltage U, V-THD, I-THD, and
short-term flicker Pst). Data from an experimental off-grid setup
at VSB-TUO, Czech Republic.

**Relevance to us:** ⭐ **most directly comparable to our forecasting
work.** Same predicted variables (THD_u, THD_i), same modelling
lineage (LSTM, GRU, XGBoost). Use as primary quantitative comparison
in our forecasting section. Note: this is a preprint, so track its
peer-reviewed final version once published.

### T0-6. Dual-Optimization PQ + EMS for Hybrid Microgrids

**"Power quality improvement and energy management in hybrid
microgrids using a dual-optimization approach"**
Daniel, S. J., Karpagam, M., Flah, A., & Ben Chaabane, S.
*Scientific Reports* (Nature) (2025)

- Local file: `s41598-025-20001-0.pdf`
- DOI: <https://doi.org/10.1038/s41598-025-20001-0>

Proposes **ALA-TKAN** — Artificial Lemming Algorithm + Temporal
Kolmogorov-Arnold Network — for coordinated PQ and EMS in hybrid
microgrids. Compares against PDO-MACNN, BWO, PSO, ANN, MRA-FLC.
Reports 2.9 MW power loss, 99.2 % efficiency, 0.8 $/Wh energy cost,
**1.4 % THD**.

**Relevance to us:** ⭐ **Q1 venue benchmark.** Scientific Reports
(Nature) is exactly the impact tier we want to match. Their reported
1.4 % THD is a state-of-the-art number our EEE-side APF result
(2.07 % THD) can be positioned against. Cite as recent SOTA in the
Related Work and Results comparison sections.

---

## ✅ TIER 1 — Verified core papers (foundational citations)

### 1. Schafer et al. 2016  -  the paper behind the UCI dataset

**"Taming instabilities in power grid networks by decentralized
reactive power control"**
Schafer, Grabow, Auer, Kurths, Witthaut, Timme
*European Physical Journal Special Topics* **225**, 569-582 (2016)

- Springer: <https://link.springer.com/article/10.1140/epjst/e2015-50136-y>
- DOI: <https://doi.org/10.1140/epjst/e2015-50136-y>

### 2. Arzamasov, Böhm, Jochem 2018  -  ML-ready UCI dataset paper

**"Towards concise models of grid stability"**
*IEEE SmartGridComm 2018*

- IEEE Xplore: <https://ieeexplore.ieee.org/document/8587498>
- DOI: <https://doi.org/10.1109/SmartGridComm.2018.8587498>

### 3. Goodfellow, Shlens, Szegedy 2015  -  the FGSM adversarial paper

**"Explaining and Harnessing Adversarial Examples"**
*ICLR 2015*

- arXiv: <https://arxiv.org/abs/1412.6572>

### 4. Madry, Makelov, Schmidt, Tsipras, Vladu 2018  -  the PGD paper

**"Towards Deep Learning Models Resistant to Adversarial Attacks"**
*ICLR 2018*

- arXiv: <https://arxiv.org/abs/1706.06083>

### 5. Cohen, Rosenfeld, Kolter 2019  -  certified robustness

**"Certified Adversarial Robustness via Randomized Smoothing"**
*ICML 2019*

- arXiv: <https://arxiv.org/abs/1902.02918>

### 6. Akagi, Kanazawa, Nabae 1984  -  foundational SRF / p-q theory

**"Instantaneous reactive power compensators comprising switching
devices without energy storage components"**
*IEEE Transactions on Industry Applications*

- IEEE Xplore: <https://ieeexplore.ieee.org/document/4504460>
- DOI: <https://doi.org/10.1109/TIA.1984.4504460>

### 7. IEEE 519-2014 Standard

**"IEEE Recommended Practice and Requirements for Harmonic Control
in Electric Power Systems"**

- IEEE Xplore: <https://ieeexplore.ieee.org/document/6826459>
- DOI: <https://doi.org/10.1109/IEEESTD.2014.6826459>

---

## 🔍 TIER 2 — Search links (find the specific papers yourself)

### ML on UCI Grid Stability benchmark

- Google Scholar: [UCI grid stability prediction](https://scholar.google.com/scholar?q=UCI+grid+stability+prediction+machine+learning)
- Google Scholar: [grid stability LSTM CNN](https://scholar.google.com/scholar?q=%22grid+stability%22+LSTM+CNN+prediction)
- IEEE Xplore: [decentral smart grid control](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=%22decentral+smart+grid+control%22)

### Adversarial / perturbation robustness for grid ML

- Google Scholar: [adversarial attacks power grid](https://scholar.google.com/scholar?q=adversarial+attacks+power+grid+neural+network)
- Google Scholar: [robust machine learning smart grid](https://scholar.google.com/scholar?q=robust+machine+learning+smart+grid)

### Multi-horizon time-series forecasting for power quality

- Google Scholar: [THD forecasting LSTM](https://scholar.google.com/scholar?q=THD+forecasting+LSTM)
- Google Scholar: [power quality forecasting deep learning](https://scholar.google.com/scholar?q=power+quality+forecasting+deep+learning)
- IEEE Xplore: [voltage THD prediction](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=voltage+THD+prediction+deep+learning)

### Active Power Filter with SRF control

- Google Scholar: [synchronous reference frame active power filter](https://scholar.google.com/scholar?q=%22synchronous+reference+frame%22+active+power+filter)
- IEEE Xplore: [SRF APF harmonic compensation](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=SRF+active+power+filter+harmonic)

### Microgrid digital twins

- Google Scholar: [microgrid digital twin](https://scholar.google.com/scholar?q=%22microgrid+digital+twin%22&as_ylo=2022)
- IEEE Xplore: [microgrid digital twin](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=microgrid+digital+twin)
- Semantic Scholar: [cyber-physical microgrid twin](https://www.semanticscholar.org/search?q=cyber-physical%20microgrid%20twin)

### Bangladesh microgrid and rural electrification

- Google Scholar: [Bangladesh solar microgrid](https://scholar.google.com/scholar?q=Bangladesh+solar+microgrid)
- Google Scholar: [IDCOL solar home system](https://scholar.google.com/scholar?q=IDCOL+solar+home+system+Bangladesh)
- Google Scholar: [BRAC rural electrification microgrid](https://scholar.google.com/scholar?q=BRAC+rural+electrification+microgrid+Bangladesh)
- IEEE Xplore: [Bangladesh power quality](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=Bangladesh+power+quality+microgrid)

### Battery ageing and SoH modeling for microgrids

- Google Scholar: [battery state of health microgrid](https://scholar.google.com/scholar?q=battery+state+of+health+degradation+microgrid)
- IEEE Xplore: [lithium-ion battery ageing power quality](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=lithium+battery+ageing+power+quality)

---

## 📖 Target-venue browsers

- [IEEE Transactions on Smart Grid — latest issue](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5165411)
- [Applied Energy — latest articles](https://www.sciencedirect.com/journal/applied-energy)
- [Energy and AI](https://www.sciencedirect.com/journal/energy-and-ai)
- [IEEE Transactions on Industrial Informatics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9424)
- [Scientific Reports (Nature)](https://www.nature.com/srep/)  ⭐ same venue as T0-6

---

## 🛠️ How to use these in the Related Work section

Suggested five sub-sections in your Related Work:

1. **UCI Grid Stability benchmark and its ML history**
   Cite: Schafer 2016 (T1-1), Arzamasov 2018 (T1-2), plus 3-5 Tier 2
   search results for recent papers using the benchmark.

2. **Adversarial and perturbation-margin analysis in ML**
   Cite: Goodfellow 2015 (T1-3), Madry 2018 (T1-4), Cohen 2019 (T1-5).
   Position the tau-margin as a physical analogue of adversarial
   robustness.

3. **Time-series forecasting for power quality**
   Cite: T0-1 (Osifeko), T0-5 (Jahan), plus 2-3 Tier 2 search results.
   Compare our LSTM / GRU / MLP results directly against theirs.

4. **Active Power Filters and SRF control**
   Cite: Akagi 1984 (T1-6), T0-3 (Shaji John review), plus recent APF
   applications from search results.

5. **Microgrid digital twins + AI control**
   Cite: T0-2 (Prakash), T0-4 (Hernández-Mayoral IEEE 14-bus), T0-6
   (Daniel Nature). This last one is the strongest comparison for
   our Results section too.

**Total target: 20-30 references for a Q1 submission.**

---

## Workload split for the team

| Section | CSE or EEE | Papers to review |
|---|---|---|
| UCI benchmark ML history | CSE | T1-1, T1-2, +3 Tier 2 |
| Adversarial margin analysis | CSE | T1-3, T1-4, T1-5 |
| Forecasting | CSE | T0-1, T0-5, +2 Tier 2 |
| APF and SRF control | EEE | T1-6, T0-3, +2 Tier 2 |
| Digital twin + IEEE 14-bus | EEE | T0-2, T0-4, T0-6, +1 Tier 2 |
