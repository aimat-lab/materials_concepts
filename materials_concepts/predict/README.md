# Concept Predictor CLI

This directory contains an interactive command-line interface (CLI) for predicting future material science concepts based on **YOUR** concepts.

## Overview

The script `main.py` allows you to input one or more material science concepts and get a list of predicted related concepts that are likely to emerge in the future. The predictions are saved in a markdown file.

## First-Time Setup

The first time you run the script, it will not find a configuration file and will guide you through an interactive setup process. You will be prompted to enter the file paths for the necessary data, such as:

- Feature embeddings
- Concept embeddings
- Graph data
- Pre-trained models
- Lookup tables

You can either accept the default paths (shown in brackets `[]`) by pressing Enter or provide your own.

After providing the paths, you will be asked if you want to save the configuration. It is highly recommended to do so by typing `y`. This will create a `predictor_config.ini` file in the project root, so you won't have to enter the paths again.

## Usage

To run the predictor, execute the following command from the root of the project:

```bash
python -m materials_concepts.predict.main
```

The script will first load the configuration and then prompt you to enter the concepts you want to get predictions for:

```
Enter concepts to predict, separated by commas: 
```

You can enter a single concept or multiple concepts separated by commas (e.g., `concept1, another concept, concept3`).

The script will then process each concept and append the results to the report file (default: `report.md`).

### Command-Line Arguments

You can customize the prediction behavior using these command-line arguments:

- `--k_concepts <int>`: Specifies the number of top predictions to return for each concept. (Default: `15`)
- `--use_min_depth_of_threshold <bool>`: If set to `True`, the prediction algorithm will only consider candidate concepts that are at least 3 steps away in the knowledge graph. This can help find more distant or novel connections (see our paper). (Default: `False`)

Example:
```bash
python -m materials_concepts.predict.main --k_concepts 20 --use_min_depth_of_threshold True
```

## Configuration File

The script uses a `predictor_config.ini` file to store paths and settings. If you need to change a path after the initial setup, you can either edit this file directly or delete it to trigger the interactive setup again on the next run.

## Output

The prediction results are saved in a markdown file (default: `report.md`). For each concept you input, a new section is appended to this file with a list of predicted concepts and their corresponding scores.

### Example output

#### dna concentration
- laser powder bed fusion: *0.9867*
- additive manufacturing: *0.9852*
- high entropy alloy: *0.9796*
- PBF: *0.9781*
- machine learning: *0.9738*
- density functional theory: *0.9730*
- molecular dynamic simulation: *0.9714*
- 3d printing: *0.9711*
- mechanical performance: *0.9700*
- compressive strength: *0.9691*
- electrical conductivity: *0.9686*
- medium entropy alloy: *0.9678*
- directed energy deposition: *0.9673*
- selective laser melting: *0.9663*
- energy dispersive x ray spectroscopy: *0.9652*