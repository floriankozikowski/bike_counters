# Starting kit on the bike counters dataset

Read the instruction from the [Kaggle challenge](https://www.kaggle.com/competitions/mdsb-2023/overview).

### Download the data

Download the data from Kaggle and put the files into the `data` folder.

Note that your goal is to train your model on `train.parquet` (and eventual external datasets)
and then make predictions on `final_test.parquet`.

### Install the local environment

To run the notebook locally you will need the dependencies listed
in `requirements.txt`. 

It is recommended to create a new virtual environement for this project. For instance, with conda,
```bash
conda create -n bikes-count python=3.10
conda activate bikes-count
```

You can install the dependencies with the following command-line:

```bash
pip install -r requirements.txt -U
```

### The starter notebook

Get started on this challenge with the `bike_counters_starting_kit.ipynb` notebook.
This notebook is just a helper to show you different methods and ideas useful for your
exploratory notebooks and your submission script.

Launch the notebook using:

```bash
jupyter lab bike_counters_starting_kit.ipynb
```

### Submissions

Upload your script file `.py` to Kaggle using the Kaggle interface directly.
The platform will then execute your code to generate your submission csv file,
and compute your score.

Note that your submission .csv file must have the columns "Id" and "bike_log_count",
and be of the same length as `final_test.parquet`.



### Notebooks in the Repository

This repository contains several Jupyter notebooks to assist in the development of your model for the bike counters dataset challenge. Below is a description of each file to help you navigate through the project:

- **`preprocessing_modeling.ipynb`**:  
  This notebook contains feature engineering processes applied to the main dataset (`train.parquet`) as well as the implementation of the main models used for prediction. It is the core file for building and training models based on the primary dataset.

- **`eda.ipynb`**:  
  This notebook is dedicated to the Exploratory Data Analysis (EDA) of the main dataset. It provides insights into the data's structure, distributions, and relationships to identify patterns and inform feature engineering decisions.

- **`weather_eda.ipynb`**:  
  This notebook focuses on the EDA of the weather dataset. It explores how weather variables may relate to bike counts and prepares for their potential integration into the main dataset.

- **`modeling_alternative.ipynb`**:  
  This notebook extends the analysis by merging the main dataset with the weather dataset to create a combined dataframe. It performs feature engineering and explores alternative models leveraging the enriched data, aiming to improve predictive performance.
