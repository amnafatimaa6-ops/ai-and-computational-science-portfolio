# Testing_Black_Theory_With_GW150914
# Exploring General Relativity Through GW150914

*A Python notebook exploring real gravitational-wave data, black-hole mergers and theoretical waveform models.*

## 1. About This Project

What if we could study two black holes merging billions of light-years away using data collected on Earth?

That is the idea behind this project.

I use real gravitational-wave data from GW150914 to explore how scientists study black-hole mergers with programming, mathematics and physics. The notebook includes signal filtering, waveform visualisation, frequency estimation, spectrograms, a theoretical waveform comparison and matched filtering.

This is an exploratory learning project. My goal is to understand how real observations can be compared with predictions from Einstein's theory of general relativity and to build a foundation for more advanced work in computational astrophysics.

## 2. What Are Gravitational Waves?

Imagine dropping a stone into a pond. Ripples spread across the water. Gravitational waves are somewhat similar, except they are ripples in space-time itself.

Einstein's general theory of relativity predicts that certain accelerating massive systems, such as two black holes orbiting each other, produce gravitational waves.

When the black holes move closer together and eventually merge, they release gravitational energy in the form of these waves. By measuring them, scientists can study events that cannot be observed directly in the same way as ordinary stars.

## 3. What Is GW150914?

GW150914 was detected on 14 September 2015. It was the first direct detection of gravitational waves and the first direct observation of a binary black-hole merger through gravitational-wave astronomy.

The name refers to the event's date: GW stands for gravitational wave, and 150914 represents 2015-09-14.

The discovery was important because it provided a new way to observe the universe and investigate predictions made by general relativity.

## 4. What Are LIGO Hanford and Livingston?

LIGO stands for the *Laser Interferometer Gravitational-Wave Observatory*. It uses extremely sensitive instruments to measure tiny changes in distance caused by gravitational waves.

The two detectors used in this project are:

- **Hanford (H1):** Located in Washington State, USA.
- **Livingston (L1):** Located in Louisiana, USA.

Each detector has two perpendicular arms, each four kilometres long. Laser light travels along these arms and reflects off mirrors. A passing gravitational wave changes the relative lengths of the arms, producing a tiny measurable change in the light.

Using two detectors allows scientists to compare observations and investigate whether a signal is consistent with a gravitational-wave event.

## 5. What Does the Notebook Do?

### Step 1: Download real detector data

I use the `GWpy` library to retrieve public strain data for H1 and L1 around GW150914.

The notebook uses a 32-second segment for each detector.

| Data detail | Value |
|---|---|
| Event | GW150914 |
| Detectors | Hanford (H1) and Livingston (L1) |
| Recording duration | 32 seconds |
| Sampling rate | 4,096 Hz |
| Samples per detector | 131,072 |
| Main analysis band | 30–400 Hz |

Strain measures the fractional change in distance detected by an instrument. It is a dimensionless quantity.

### Step 2: Visualise the raw signal

I plot the original strain data from both detectors to examine the measurements before focusing on the short event.

The raw data contains detector noise, so the gravitational-wave pattern is not necessarily easy to identify in the full recording.

### Step 3: Filter and zoom in on the chirp

The notebook applies a 30–400 Hz band-pass filter and examines a shorter section around the event.

A *chirp* is a signal whose frequency rises as the black holes approach merger. The filtered, zoomed-in plots make the waveform easier to inspect.

Filtering helps focus the analysis on a selected frequency range, but it does not remove all noise or independently confirm that a signal is astrophysical.

### Step 4: Calculate the amplitude envelope

Using SciPy's Hilbert transform, I calculate an amplitude envelope for the zoomed Hanford signal.

The envelope helps show how the signal's amplitude changes over time.

The saved notebook output reports a maximum envelope amplitude of approximately \(2.913 \times 10^{-21}\) for the selected segment.

This is a result of the particular processed data, not a measurement of the black holes' masses or a complete description of the event's strength.

### Step 5: Estimate the frequency from waveform peaks

The notebook identifies prominent positive peaks in the Hanford signal and calculates the time between consecutive peaks. It then estimates frequency using the inverse of each measured time interval.

The saved output reports:

- Detected peaks: 51
- Frequency estimates: 50
- Estimated frequency range: approximately 37.24–372.36 Hz

This is a simple exploratory estimate. Noise, peak-selection settings and the short analysis window can affect the result, so it should not be treated as a precision measurement of the gravitational-wave frequency.

### Step 6: Generate a theoretical waveform

Using PyCBC, I generate a theoretical binary-black-hole waveform with the `IMRPhenomD` approximant.

The notebook uses component masses of 36 and 29 solar masses, a sampling rate of 4,096 Hz and a lower frequency limit of 20 Hz.

I then filter and normalise the theoretical waveform and compare its shape with the observed Hanford signal.

For the visual comparison, the two waveform peaks are aligned in time. This makes their shapes easier to inspect, but it does not constitute a full parameter-estimation analysis or a rigorous statistical test of general relativity.

### Step 7: Examine the detector noise spectrum

The notebook estimates the power spectral density using Welch's method and converts it to amplitude spectral density for both detectors.

The resulting plot shows how the measured noise level varies with frequency and highlights the 30–400 Hz analysis band.

This helps explain why gravitational-wave analysis requires careful treatment of detector noise.

One limitation of this implementation is that the spectrum is estimated from the same 32-second segment being analysed, which includes the event. A more rigorous analysis should estimate the background noise using suitable data outside the event and assess how the choice of noise estimate affects the results.

### Step 8: Create spectrograms

I use SciPy to create time-frequency spectrograms for both Hanford and Livingston.

A spectrogram shows how signal power is distributed across frequencies as time passes. It provides a different view of the data from an ordinary waveform plot and helps us examine how a chirp changes over time.

The notebook also whitens the data using an estimated noise spectrum and generates additional spectrograms, including views around the event.

Whitening adjusts the frequency-dependent noise level to make different frequency ranges easier to examine. The resulting plots still require careful interpretation because whitening and filtering affect the appearance of the data.

### Step 9: Explore matched filtering

The notebook uses PyCBC to compare the Hanford strain data with a theoretical binary-black-hole waveform using matched filtering.

Matched filtering measures how strongly data match a selected waveform template, taking an estimated noise spectrum into account.

The saved output reports:

- Maximum single-detector SNR: **16.76**
- Time of maximum SNR: **1126259462.429443 GPS**

The notebook also reports an offset of approximately −0.56 milliseconds relative to its chosen reference time of 1126259462.43 GPS.

These values describe the output of this particular calculation. The reported SNR is not, by itself, a false-alarm probability or a complete measure of detection significance.

## 6. Results and What They Mean

This notebook gave me practical experience with several methods used in gravitational-wave data analysis.

- The H1 and L1 recordings can be examined and compared around the event.
- Filtering and zooming make the short chirp easier to inspect.
- Peak spacing provides a basic estimate of how the frequency changes.
- A theoretical waveform gives me a way to compare an observed signal's shape with a model based on general relativity.
- Noise spectra and spectrograms help me understand the frequency-dependent nature of detector noise.
- Matched filtering produces a peak single-detector SNR of 16.76 near the selected event reference time.

Together, these steps demonstrate how programming can help investigate real astrophysical data.

However, this is an exploratory analysis of one event. It does not independently prove general relativity or reproduce the full scientific detection pipeline used by the LIGO Scientific Collaboration.

## 7. Limitations

**One event:** The notebook focuses on GW150914. The approach would need to be evaluated on other events to understand how well it generalises.

**Simplified frequency estimation:** Frequency is estimated from the spacing between detected peaks. The method can be affected by noise and peak-selection choices.

**Effects of filtering and whitening:** These processing steps change the representation of the data and can affect how the signal appears.

**Visual waveform comparison:** The observed and theoretical waveforms are normalised and aligned for visual inspection. The notebook does not perform a complete statistical comparison or estimate the physical parameters of the source from the data.

**Noise estimation:** The noise spectrum is estimated from the same short segment that contains the event. This is a limitation because the event can affect the estimate. A more rigorous analysis would use suitable surrounding data and test the robustness of the result.

**Limited matched-filter analysis:** The notebook performs a single-detector matched-filter calculation. It does not establish a false-alarm rate, combine the evidence from both detectors in a complete search pipeline, or reproduce the full detection-significance analysis.

**No trained AI model:** The notebook uses conventional signal-processing techniques and a theoretical waveform. It does not train or evaluate a machine-learning model for gravitational-wave detection.

## 8. Future Work

This project is a starting point for my longer-term interest in computational astrophysics.

I would like to investigate whether artificial intelligence can help identify gravitational-wave signals in noisy detector data. Before attempting this, I need to understand the physics of gravitational waves, the mathematics behind their waveforms, and the computer science methods used in signal processing and machine learning.

My future goals are to:

- Study more gravitational-wave events and learn how their signals differ.
- Develop a stronger understanding of general relativity, binary black-hole dynamics and gravitational-wave mathematics.
- Improve the analysis by estimating detector noise from appropriate surrounding data and evaluating the effects of processing choices.
- Investigate existing AI approaches and choose a suitable model based on the scientific problem.
- Prepare labelled training data using simulated gravitational-wave signals and realistic detector noise.
- Train and test a model on separate datasets, including data it has not seen during training.
- Evaluate detection performance, false alarms and missed signals.
- Compare the model's performance with established gravitational-wave search methods.

These are future goals, not results achieved in the current notebook. I want to build on this project as I gain the physics, mathematics and computer science knowledge needed to make informed choices and evaluate the results responsibly.

## 9. Tools and Technologies

The notebook uses:

- **Python** — programming and analysis.
- **Google Colab / Jupyter Notebook** — running code and presenting results.
- **GWpy** — retrieving and processing gravitational-wave strain data.
- **NumPy** — numerical calculations.
- **Matplotlib** — plotting waveforms, spectra and spectrograms.
- **SciPy** — signal processing, peak detection, Hilbert transform and spectrograms.
- **PyCBC** — generating theoretical waveforms and performing matched filtering.

## 10. How to Run

1. Open `notebook (1).ipynb` in Google Colab or a compatible Jupyter environment.
2. Run the cells in order.
3. Allow the notebook to install its required libraries.
4. Ensure an internet connection is available to retrieve public detector data.
5. Review the plots and printed outputs alongside the explanations.

The saved notebook contains example outputs. Re-running it may produce small differences due to software versions or data-processing details.

## 11. Why I Chose This Project

I chose this project because I want to understand how computer science can help us investigate the universe.

I am especially interested in black holes, gravitational waves and the mathematics behind them. Working with real scientific data gives me an opportunity to connect programming with physics and explore how computational methods can help us investigate phenomena far beyond Earth.

I am still learning, and I know that reliable AI for gravitational-wave research requires more than simply choosing a model. It requires understanding the physics, preparing appropriate data, evaluating methods carefully and recognising their limitations.

As I continue studying computer science, artificial intelligence and physics, I hope to develop this project further and eventually explore how machine learning can contribute to gravitational-wave research.

For me, this is one step in a longer journey towards computational astrophysics: learning to use mathematics, programming and scientific reasoning to ask better questions about the universe.

## References

- [LIGO Scientific Collaboration — GW150914](https://ligo.org/detections/gw150914/)
- [LIGO Open Science Center](https://gwosc.org/)
- [GWpy Documentation](https://gwpy.github.io/)
- [PyCBC Documentation](https://pycbc.org/pycbc/latest/html/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)

