# GreenToBuy-ML 🚀

**GreenToBuy-ML** is a machine learning-powered stock analysis and prediction platform that provides real-time insights through an interactive web interface.  
It works smoothly on Windows and Linux (Arch, Ubuntu, Debian).

## ✨ Features

- 📈 Real-time stock price fetching
- 📊 Technical indicators using TA-Lib
- 💬 Sentiment analysis using NLTK VADER
- 🤖 Multiple ML models for prediction
- 🖥️ Interactive Streamlit dashboard
- ⚡ High-performance FastAPI backend

## 🛠 Technology Stack

- Python – Core programming language
- Streamlit – Web application framework
- Scikit-learn – Machine learning algorithms
- Yahoo Finance API (yfinance) – Stock data
- Matplotlib / Seaborn – Data visualization
- FastAPI + Uvicorn – Backend server

## 📦 Installation

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   ```

   Activate it:

   - **Linux / macOS**
     ```bash
     source venv/bin/activate
     ```

   - **Windows**
     ```bash
     venv\Scripts\activate
     ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

- **Start Streamlit UI**
  ```bash
  streamlit run app.py
  ```

- **Start FastAPI Backend**
  ```bash
  uvicorn server:app --reload
  ```

Open in browser:

- Streamlit UI: http://localhost:8501
- API Server: http://127.0.0.1:8000

## 📊 Using the App

1. Enter stock symbol (e.g., AAPL, GOOGL, TSLA, MSFT)
2. Select time period
3. Click "Analyze Stock"
4. View predictions, charts & sentiment analysis

## 🤖 Machine Learning Models

- **Linear Regression**  
  Simple & interpretable  
  Detects linear trends  
  Fast training

- **Random Forest**  
  Ensemble trees  
  Handles non-linear patterns  
  Resistant to overfitting

- **Gradient Boosting**  
  Sequential learning  
  High accuracy  
  Great for complex data

## 📁 Project Structure

```
GreenToBuy-ML/
├── app.py
├── server.py
├── models/
├── utils/
├── requirements.txt
└── README.md
```

## 🛑 Troubleshooting Missing Modules

If you encounter import errors for missing packages:

```bash
pip install <module_name>
```

(Ensure your virtual environment is activated.)

# Data Sources

The application fetches real-time stock data using the **yfinance** API (Yahoo Finance), which provides:
- Historical stock prices
- Volume data
- Market indicators
- Company information

## Disclaimer

This application is for educational and research purposes only. Stock price predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial professionals and conduct thorough research before making investment choices.

**Important**: Past performance does not guarantee future results. Stock markets are volatile and unpredictable.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the documentation
- Review existing issues for similar problems

## Acknowledgments

- **Yahoo Finance (yfinance)** for providing free stock data API
- **TA-Lib** for technical analysis indicators
- **NLTK VADER** for sentiment analysis
- Scikit-learn contributors for machine learning tools
- Streamlit team for the amazing web framework
- FastAPI team for the high-performance backend
