# VC Profit Predictor

## Overview
Welcome to the **VC Profit Predictor**! This application is designed to help venture capitalists gain insights into potential startup profitability based on various spending categories.

My machine learning model has been trained on a dataset of successful startups. By inputting different spending values, you can predict potential profits and understand the impact of various financial allocations.

### Key Features
- **Analyze** how different spending patterns affect profitability.
- **Compare** the impact of R&D, Administration, and Marketing spend on potential profits.
- **Understand** regional variations in profitability (California, New York, Florida).
- **Develop** data-driven investment strategies.
- **Train** custom models using `.csv` or `.json` data files.
- **Use** trained models (`.pkl` files) for profit predictions.

You can also train your own model using `.csv` or `.json` data files, which will generate a `.pkl` model file. This custom model can then be used for making profit predictions, allowing you to tailor the predictions to your specific dataset or industry focus.

> **Note:** This application is created for learning purposes only and should not be used as the sole basis for investment decisions.

## Technologies Used
- **Frontend:** Next.js, Tailwind, Shadcn UI
- **Backend:** Flask
- **Machine Learning:** Scikit-Learn, NumPy, Pandas

## Getting Started

### Prerequisites
- Python 3.x
- Node.js and npm/pnpm/yarn/bun
- Virtual Environment (optional but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/trishan9/VC-Profit-Predictor.git
   cd VC-Profit-Predictor

2. Setup the backend
   ```bash
   cd server
   pip install -r requirements.txt

3. Setup the frontend
   ```bash
   cd frontend
   npm/pnpm/bun/yarn install

### Running the application

1. Start the Flask Server (Backend):
   ```bash
   cd server
   gunicorn -w 4 -k gevent --preload -b 0.0.0.0:8000 api.index:app

2. Setup the Next.js Application (Frontend):
   ```bash
   cd frontend
   npm/pnpm/bun/yarn dev

### API Endpoints
- **POST /api/train**: Train a new model using a dataset.
- **POST /api/predict**: Predict profit based on input features.

### Software Overview
![swappy-20241013_174012](https://github.com/user-attachments/assets/12fa60b0-7ca6-4afa-98eb-dc6a384dcf82)
![swappy-20241013_174119](https://github.com/user-attachments/assets/deec332a-b7b3-41dc-9552-3055bc8e1968)
![swappy-20241013_174150](https://github.com/user-attachments/assets/41966c03-8eef-44f0-bdb5-7914783d0ca9)

### ML Model Performance Evaluation
![375972942-477ad37a-6504-40dd-90cf-ce666ac05d6e](https://github.com/user-attachments/assets/2d674757-31f5-4bb5-836f-9a0912853538)
