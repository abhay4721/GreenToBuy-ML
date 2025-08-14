# GreenToBuy-ML

A machine learning application that predicts stock prices using advanced algorithms and provides real-time analysis through an interactive web interface.

## Technology Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms -----------
- **Yahoo Finance API**: Real-time stock data source ---------
- **Matplotlib/Seaborn**: Data visualization libraries

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application(frontend, backend):
```bash
streamlit run app.py
streamlit run server.py or uvicorn server:app --reload
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Using the application:
   - Enter a stock symbol (e.g., AAPL, GOOGL, TSLA, MSFT)
   - Select your desired time period for analysis -------------
   - Click "Analyze Stock" to generate predictions
   - View the results including predictions, visualizations, and analysis

## Machine Learning Models

--------------------------------------------------

### Linear Regression
- Simple and interpretable model
- Good for identifying linear trends
- Fast training and prediction times

### Random Forest
- Ensemble method using multiple decision trees
- Handles non-linear relationships well
- Robust against overfitting

### Gradient Boosting
- Sequential ensemble method
- High predictive accuracy
- Excellent for complex pattern recognition

## Project Structure

```
greentobuy-ml/
├── app.py                 # Main Streamlit application
├── server.py                # Machine learning models
├── -----                 # ---
├── -----                # ------
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Data Sources

The application fetches real-time stock data using the ----- API, which provides:
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

- --------- for providing free stock data API
- --------
- Streamlit team 
