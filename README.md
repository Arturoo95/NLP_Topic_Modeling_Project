# NLP Topic Modeling Project

## Project Overview

This project focuses on Natural Language Processing (NLP) and topic modeling using Latent Dirichlet Allocation (LDA). The goal is to extract and analyze topics from a large collection of text documents, such as news articles.

## Project Structure

- **data/**: Contains raw and processed data.
- **notebooks/**: Jupyter notebooks for developing and documenting the project.
- **scripts/**: Python scripts for data preprocessing, LDA modeling, and visualization.
- **models/**: Saved LDA model and vectorizer.
- **results/**: Outputs from the LDA model, including visualizations and topic summaries.

## Setup Instructions

1. Clone the repository:

   ```sh
   git clone https://github.com/Arturoo95/NLP_Topic_Modeling_Project.git
   cd NLP_Topic_Modeling_Project
   ```

2. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

3. Run the data preprocessing script:

   ```sh
   python scripts/data_preprocessing.py
   ```

4. Train the LDA model:

   ```sh
   python scripts/lda_modeling.py
   ```

5. Generate visualizations:
   ```sh
   python scripts/visualization.py
   ```

## Results

- LDA topics are saved as `lda_topics.txt`.
- Word clouds for each topic are saved in the `results/` directory.
- The pyLDAvis interactive visualization is saved as `lda_visualization.html`.

## License

This project is licensed under the MIT License.
