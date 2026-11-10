# Neuro_Forge — A Neural Network Built from Scratch in C++17

## 1. About the Project

Neuro_Forge is a machine learning project I built to understand how a neural network works internally, using C++17 instead of relying on a machine learning library.

A **neural network** is a computer model that learns patterns from examples. It contains connected units called neurons, which process input values using adjustable numbers called weights and biases. During training, the network adjusts these values to reduce its prediction errors.

I wanted to understand what happens behind the scenes when a neural network makes a prediction, rather than simply using a ready-made model.

For this project, I built a small neural network and trained it using data from the NASA Exoplanet Archive. I also created a Streamlit dashboard that lets users enter planetary measurements, inspect the network's calculations, and explore how changing an input affects its output.

**An important distinction:** the current model predicts whether a listed planet was discovered using the transit method or another method. It does not discover new planets or determine whether an unconfirmed object is a real exoplanet.

## 2. What I Built

- A neural network implemented in C++17 without a machine learning framework.
- A data-processing and training pipeline for planetary data.
- A model that classifies the discovery method as transit or non-transit.
- A Streamlit dashboard connected to the C++ model.
- Visual tools for inspecting weights, neuron activations, predictions, and changes in the output.
- Model evaluation using measures such as accuracy, precision, recall, and balanced accuracy.

## 3. Project Demo

### Dashboard

![Neuro_Forge dashboard](Screenshot1.png)

![Neuro_Forge model evaluation](Screenshot2.png)

![Neuro_Forge training output](Screenshot3.png)

### Demo animation

![Neuro_Forge demo](demo.gif)

## 4. Dataset

The project uses planetary data from the NASA Exoplanet Archive.

The cleaned dataset included in the repository contains **40,194 rows**:

- 36,100 rows labelled as transit discoveries.
- 4,094 rows labelled as discoveries made using other methods.

The model uses nine numerical features describing a planet and its host star.

| Feature | Meaning |
|---|---|
| `pl_orbper` | Orbital period in days |
| `pl_orbsmax` | Semi-major axis in astronomical units (AU) |
| `pl_orbeccen` | Orbital eccentricity |
| `pl_eqt` | Planetary equilibrium temperature in kelvin |
| `pl_rade` | Planet radius in Earth radii |
| `st_teff` | Stellar effective temperature in kelvin |
| `st_mass` | Stellar mass in solar masses |
| `st_rad` | Stellar radius in solar radii |
| `st_met` | Stellar metallicity |

The target label is based on the recorded discovery method:

- `1` = Transit
- `0` = Any other discovery method

These labels describe how a planet was discovered, not whether a planet is real or potentially habitable.

## 5. How the Neural Network Works

The network has three layers:

**9 input neurons → 8 hidden neurons → 1 output neuron**

The input layer receives the nine measurements. The hidden layer combines these values to learn patterns, and the output layer produces a score between 0 and 1.

A threshold of 0.5 is used to assign the predicted class.

I implemented the main neural network calculations myself, including matrix operations, sigmoid activation, forward propagation, backpropagation, and gradient-based weight updates.

### Training process

1. Read the planetary dataset and assign labels.
2. Fill missing numerical values using the median of each feature.
3. Scale the features to a range between 0 and 1.
4. Split the data randomly into training and testing sets, using an 80/20 split.
5. Train the network for 300 epochs using a learning rate of 0.01.
6. Evaluate the model on the held-out portion of that split.
7. Save the trained model for later predictions.

Because the two classes are imbalanced, the training code also uses class weights to give the less common class more influence on the learning process.

The network has 89 trainable parameters: 72 input-to-hidden weights, 8 hidden-layer biases, 8 hidden-to-output weights, and 1 output bias.

## 6. Results

The repository contains results from more than one evaluation run. These should not be treated as if they came from the same experiment.

### One training run

The results recorded from a run of `main.cpp` were:

| Metric | Result |
|---|---:|
| Accuracy | 89.45% |
| Balanced accuracy | 87.41% |
| Precision | Approximately 0.98 |
| Recall | Approximately 0.90 |
| Final reported training loss | 0.0415183 |

In this run, the test set contained 8,039 rows. The model correctly classified many examples, but it also made mistakes, particularly when classifying the less common non-transit group.

### Dashboard evaluation

The dashboard contains a separate evaluation tool. One recorded run showed:

| Metric | Result |
|---|---:|
| Accuracy | 92.08% |
| Balanced accuracy | 88.39% |
| Precision | 98.06% |
| Recall | 93.02% |
| F1 score | 95.47% |
| Brier score | 0.0812 |

These figures are recorded examples, not guaranteed results for every run. The dashboard evaluation has important limitations because it samples from the full dataset, which overlaps with the data used to train the saved model.

For this reason, neither result should be presented as proof that the network can reliably identify new exoplanets.

## 7. The Interactive Dashboard

The dashboard is built with Streamlit and uses a helper executable to run the C++ neural network. It does not require an external AI API.

### Input panel

Users can enter the nine planetary and stellar measurements and submit them to the model.

### Neural Microscope

This section visualises the network's input layer, hidden neurons, output, weights, and activations. Selecting a hidden neuron allows users to inspect its calculations and connected weights.

### Feature Influence Lab

Users can select one feature and run a sensitivity test. The dashboard changes that feature across nine values while keeping the other eight inputs fixed.

A chart displays how the model's output changes during this test.

This is a local sensitivity experiment. It does not prove that a feature is scientifically important for discovering exoplanets.

### Model Accuracy and Confidence Lab

This section displays evaluation metrics, a confusion matrix, score distributions, a calibration table, and individual predictions. It also allows evaluation results to be downloaded.

The scores should not be interpreted as reliable probabilities because the model has not been properly calibrated.

### Computation Results

Users can inspect the output score, predicted class, weighted sums, activations, biases, and weights returned by the C++ model. The computation trace can also be downloaded as JSON.

These tools help make the calculations inside the network easier to examine.

## 8. Limitations

I want to be clear about what this project can and cannot demonstrate.

1. **It does not discover exoplanets.** It classifies the recorded discovery method of planets already present in a dataset.
2. **The classes are imbalanced.** About 89.8% of the dataset belongs to the transit class. Accuracy alone can therefore give a misleading impression of performance.
3. **The evaluation is not fully independent.** The dashboard samples rows from data that was also used for training. A stronger evaluation needs planets that were never included in training.
4. **Duplicate measurements may cause leakage.** The dataset contains repeated feature combinations, and the cleaned data does not retain the planet identifiers needed to group all records for the same planet together.
5. **Feature scaling needs improvement.** Very large values, especially orbital periods, can compress many ordinary values into a very narrow range. Logarithmic scaling may help.
6. **Missing values can affect learning.** Median imputation is simple, but it can hide differences between measured values and values that were missing.
7. **The output scores are not calibrated probabilities.** A score of 0.8 should not automatically be interpreted as an 80% chance that the discovery method was transit.
8. **The model is small.** It has only nine input features and eight hidden neurons. I have not completed a thorough comparison against simpler models or systematically tuned its settings.
9. **Results can vary between runs.** The initial weights and random data split can change the results.
10. **The dashboard depends on the correct executable.** The bundled executable is built for Windows. Linux deployments need a compatible build of the C++ helper program; simply finding the file does not guarantee that it can run.

## 9. Future Work

My long-term goal is to move beyond classifying discovery methods and learn how neural networks could help identify possible exoplanets.

That is a much larger task than the current project, and I would need to develop my knowledge and improve the model before attempting it.

My next steps would include:

- **Learning more about exoplanet science.** I want to understand how astronomers distinguish planetary signals from noise and false positives.
- **Using more suitable data.** Exoplanet detection often involves light curves: measurements of a star's brightness over time. A planet passing in front of its star can produce a small dip in brightness.
- **Training a more suitable neural network.** I would explore architectures that can learn patterns from light curves rather than relying only on nine summary measurements.
- **Increasing the amount of training data.** A more ambitious model would require a larger, carefully labelled dataset with appropriate examples of planetary signals and false alarms.
- **Improving the training process.** I would use better feature scaling, preserve planet identifiers, prevent data leakage, and test the model on independent data.
- **Comparing models.** I would compare the neural network with simpler methods and use suitable metrics to understand whether the more complex model is genuinely better.
- **Improving the dashboard.** I would add clearer explanations of model uncertainty and make it easier to compare results from different experiments.

I do not yet have an exoplanet discovery model, and I would need more data, more training, and a stronger understanding of the science and machine learning methods involved before attempting one.

## 10. Running the Project

### Requirements

- C++17 compiler
- Python 3.11 or newer
- Streamlit
- Pandas

### Install Python dependencies

Clone the repository and open its folder:

```bash
git clone https://github.com/amnafatimaa6-ops/Neuro_Forge.git
cd Neuro_Forge
python -m pip install -r requirements.txt
```

### Run the dashboard

On Windows, use:

```bash
python -m streamlit run app.py
```

The dashboard should open in your browser at `http://localhost:8501`.

The bundled helper executable is Windows-specific. On Linux, compile the helper from the C++ source files using a compatible compiler before running the dashboard. A hosted deployment also needs the correct executable and file permissions.

### Train the network again

The training program expects the raw NASA dataset at `data/exoplanets.csv`. That raw file is not included in the repository.

Download the relevant data from the NASA Exoplanet Archive, ensure the required columns are present, and save it at the expected path. Then compile and run the training program:

```bash
g++ -std=c++17 main.cpp src/*.cpp -Iinclude -o NeuroForge
./NeuroForge
```

Training creates a new model and cleaned dataset, replacing the existing generated files. The results can differ between runs.

## 11. Technologies

- **C++17:** neural network implementation and training
- **Python:** dashboard and supporting code
- **Streamlit:** interactive web interface
- **Pandas:** working with tabular data
- **NASA Exoplanet Archive:** source of the planetary data

## 12. Acknowledgements

The planetary data comes from the NASA Exoplanet Archive. This is an independent educational project and is not developed by or endorsed by NASA.

## 13. Why I Built This

I chose this project because I wanted to understand how neural networks actually work, rather than treating them as black boxes.

I am interested in computer science, artificial intelligence, and physics, especially the ways computing can help us study space. Neuro_Forge is one step in that journey. I built a small model, explored its calculations, and learned that getting a result is only part of the work; understanding its limitations matters just as much.

As I continue studying computer science, machine learning, and astrophysics, I hope to improve this project and eventually explore more advanced methods for finding patterns in astronomical data. I still have a great deal to learn, but that is exactly why I want to keep building and experimenting.

