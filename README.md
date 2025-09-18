"""
# Seattle House Price Prediction & Analysis Dashboard

This project provides an end-to-end solution for predicting house prices in Seattle, featuring an interactive Streamlit dashboard for data exploration and a FastAPI endpoint for inference.

## 🚀 Quick Start

This entire application stack is containerized using Docker. To get started, ensure you have Docker and Docker Compose installed.

From the root directory of the project, simply run:

```
docker-compose up

```

This will build the necessary images and start all the services.

## 🌐 Accessing the Services

Once the containers are running, you can access the different components of the project at the following local URLs:

* **📊 Interactive Dashboard:** [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)

* **🧠 Prediction API Docs:** [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)

* **📈 MLflow Tracking UI:** [http://localhost:5000](https://www.google.com/search?q=http://localhost:5000)

The inference endpoint is available at `POST http://localhost:8000/predict`.
"""
