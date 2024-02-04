# ALTeGraD 2023 Data Challenge

This repository contains the work of Team "Queniaric," consisting of Hugo Queniat and Simon Queric, both students from Télécom Paris and the Master MVA organized by ENS Paris Saclay, for the ALTeGraD 2023 Data Challenge. The challenge details can be found [here](https://www.kaggle.com/competitions/altegrad-2023-data-challenge/).

Over the public leaderboard, which accounts for around 50% of the test set, our best model achieved a score of 0.8993. This score was achieved using a soft voting classifier, which combines the predictions of several models trained on the same data. The models used in the soft voting classifier are the following and all follow similar Attentive structures:
   ```python
   model_name= 'distilbert-base-uncased-finetuned-sst-2-english'
   ModelGATPerso(model_name, n_in=300, nout=768, nhid=1024, n_heads=8, dropout=0.6)
   ModelGATwMLP(model_name, nout=768, nhid=768, n_heads=4, n_in=300, dropout=0.75)
   ModelTransformer(model_name, n_in=300, nout=768, nhid=768, n_heads=4, dropout=0.6)
   ModelTransformerv2(model_name, n_in=300, nout=768, nhid=100, n_heads=2, dropout=0.75)
   ```

As required by the competition, the code is entirely reproducible and the models are trained on the provided data, meaning that you can reproduce our experiments and results on your own machine by following the instructions below.

## Installation

### Prerequisites

- Python (version 3.10)
- Pip (package installer for Python)

### Instructions

1. Clone the repository to your local machine

   ```bash
   git clone https://github.com/hugo-qea/ALTeGraD-2023-Data-Challenge.git
   ```

2. Navigate to the project directory

   ```bash
   cd ALTeGraD-2023-Data-Challenge
   ```

3. Install the required dependencies

   ```bash
   pip3 install -r requirements.txt
   ```
   The requirements.txt file contains a list of dependencies and their versions required for this project. The command above installs these dependencies in your Python environment.

4. Download the competition data from [Kaggle](https://www.kaggle.com/competitions/altegrad-2023-data-challenge/data).

5. Extract the downloaded data into the `./data` directory of this project.

   ```bash
   unzip -q altegrad-2023-data-challenge.zip -d ./data
   ```

## Usage
Now that you have installed the dependencies, you can run the any model you have setup using the following command.

   ```bash
   python3 source/main.py
   ```
To follow the training of the model, you can also run the following command :
   ```bash
   tensorboard --logdir=source/runs
   ```
The submissions will be stored in the `./submissions` directory while a summary of the models, its training and the saves of its weights will be stored in the `./saves` directory.
There are also several other files you can run in case you just want to create a submission after setuping the file :
   ```bash
   python3 source/submit.py
   ```

You can also train a soft voting classifier where all classifiers are trained with the exact same Text Encoder and through the same loops : 
   ```bash
   python3 source/voting.py
   ```

Finally, you can also create a submission created from the averaging of the similarities found by distinct models :
   ```bash
   python3 source/submitSoftVoting.py
   ```

