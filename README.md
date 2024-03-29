# DSA_case
This repository presents the development, analysis, and conclusions of a *Data Science Analyst* case, using data analysis and *machine learning* techniques to explore a provided dataset. It may be constantly subject to change.

## Case Development

### [1. Exploratory Data Analysis (EDA)](justo_bcase/tasks/EDA/)

Conducting an Exploratory Data Analysis to understand the features and relationships within the dataset.

### [2. Building Predictive Models](justo_bcase/tasks/model)

Developing models to predict relevant metrics, evaluating their performance using specific metrics.

### [3. Hyperparameter Optimization](justo_bcase/tasks/model)

Improving model performance through hyperparameter optimization.

### [4. Clustering Techniques](justo_bcase/tasks/cluster)

Applying clustering to identify patterns or segments in the data.

### [5. Unsupervised Learning](justo_bcase/tasks/report)

Using unsupervised learning techniques to uncover hidden insights in unlabeled data.

**Note 1:** While this report contains relevant information and key conclusions, the folders within this repository contain all outcomes, intermediates, code, and relevant files for a more detailed analysis. Reviewing these supplementary materials is recommended for a deeper understanding of the project.

**Note 2:** For interactive files like EDA-Profiling and 2D/3D Scatter... it might not be possible to view them on GitHub or PDF, so it is recommended to download the HTML files located in `outcomes/EDA` and `outcomes/cluster` respectively to open them in a browser.

## Repository Distribution

The distribution of the repository is as follows:

- [data](justo_bcase/data): Contains the database.

- [intermediates](justo_bcase/intermediates): Contains intermediate files that serve as inputs or references between different tasks.

- [tasks](justo_bcase/tasks): Subdivided into four distinct sections, each directory houses the requisite Python scripts designed to generate specific outcomes:
    - [EDA](justo_bcase/tasks/EDA): `eda.py`
    - [cluster](justo_bcase/tasks/cluster): `clus.py`
    - [model](justo_bcase/tasks/model): `compare_models.py`, `cv_params.py`
    - [report](justo_bcase/tasks/report): `report.ipynb`, `report.pdf` (currently only available in Spanish; an English version will be provided later).

- [outcomes](justo_bcase/outcomes): Contains various folders, charts, HTML files, and .csv files that display all the results for a more detailed analysis of the data and the models than in the report.

(Please note that this repo is still in progress, so there may be some small typos/mistakes throughout the project.)
