# Thesis Project

A data science research project for thesis work.

## Project Structure

```
thesis-project/
├── data/                    # Data directory
│   ├── raw/                # Raw data files
│   ├── interim/            # Intermediate data files
│   └── processed/          # Processed data files
├── notebooks/              # Jupyter notebooks
│   └── sample_preprocess.ipynb
├── reports/                # Generated reports
│   ├── figures/            # Plots and visualizations
│   └── tables/             # Data tables
├── src/                    # Source code
│   ├── config.py          # Configuration settings
│   ├── data/              # Data processing modules
│   │   └── prepare.py
│   └── utils/             # Utility functions
│       └── io_utils.py
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd thesis-project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Processing
The main data processing pipeline is in `src/data/prepare.py`. Configuration settings are in `src/config.py`.

### Jupyter Notebooks
Run Jupyter notebooks for exploratory analysis:
```bash
jupyter notebook notebooks/
```

## Dependencies

- numpy
- pandas
- torch
- transformers
- tqdm
- datasets
- scikit-learn
- matplotlib

## Configuration

Edit `src/config.py` to modify:
- Data directory paths
- Random seed for reproducibility
- Other project settings

## Contributing

1. Create a feature branch
2. Make your changes
3. Test your changes
4. Submit a pull request

## License

[Add your license information here]
