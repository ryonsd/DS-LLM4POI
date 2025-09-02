# DS-LLM4POI

An implementation of "A Comparative Study of Demonstration Selection for Practical Large Language Models-based Next POI Prediction"


# Install

Build the Docker container image from the provided Dockerfile:

```bash
docker build -t your-image-name .
```

After entering the container or in your local environment, install the required Python libraries using Poetry:

```bash
poetry install
```

# Dataset

Foursquare location-based service in the New York City and Tokyo areas, and Gowalla in the California and Nevada areas (NYC, TKY, and CA). 
The datasets were installed and preprocessed in the same way as described in https://github.com/neolifer/LLM4POI.

The dataset is organized in the following directory structure:

```markdown
datasets/
└── nyc/
    ├── raw.zip
    └── preprocessed/
        ├── similarity/
        ├── train_sample.csv
        ├── validate_sample_with_traj.csv
        └── test_sample_with_traj.csv
```

# Preprocess

We compute the similarity between POI visit sequences in the test data and those in the training data using the following methods:

- Dynamic Time Warping (DTW): Measures similarity between sequences by aligning them with minimal temporal distortion.
- Jaccard: Computes the similarity between sets of POIs by comparing the intersection over the union.
- Longest Common Subsequence (LCS): Evaluates the similarity based on the longest sequence of POIs that appear in both sequences in the same order.

```bash
poetry run python src/calc_traj_similarity.py -d {dataset_name} -m {similarity_name} -i {index}
```

The -i option is used to specify the index of the current process when performing parallel computation.

After computing sequence similarity in parallel, merge the output files into a single file.
The merged similarity results should be stored in the following directory: ```datasets/{dataset_name}/preprocessed/similarity/```

# Run


```bash
poetry run python src/run.py --prompting_type {prompting_type} --model_name {model_name} --dataset_name {dataset_name} --data_bank {data_bank} --demo_selection {demo_selection} --similarity {similarity} --num_demo {num_demo}
```

- prompting_type: FS-LLM4POI
- model_name: gpt-4o
- dataset_name nyc, tky, ca
- data_bank: all_users, the_user
- demo_selection: random, date, similarity
- similarity: dtw, jaccard, lcs (when demo_selection is similarity)
- num_demo: 5, 15, 30
- random_seed: 0,1,2,... (when demo_selecition is random)


# License

This project includes code from [xlwang233/LLM-Mob](https://github.com/xlwang233/LLM-Mob), which is licensed under the MIT License.

