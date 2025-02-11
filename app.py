from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

app = Flask(__name__)

def predict_stock(stock_code):
    # 获取历史数据
    stock = yf.Ticker(stock_code)
    hist = stock.history(period="1y")
    
    # 准备数据
    hist['Date'] = hist.index
    hist['Days'] = (hist['Date'] - hist['Date'].min()).dt.days
    X = hist[['Days']]
    y = hist['Close']
    
    # 训练模型
    model = LinearRegression()
    model.fit(X, y)
    
    # 预测未来30天
    future_days = np.array(range(hist['Days'].max() + 1, hist['Days'].max() + 31)).reshape(-1, 1)
    future_prices = model.predict(future_days)
    
    # 生成日期
    last_date = hist['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    
    # 生成建议
    current_price = hist['Close'].iloc[-1]
    predicted_price = future_prices[-1]
    
    if predicted_price > current_price * 1.05:
        advice = "买入"
    elif predicted_price < current_price * 0.95:
        advice = "卖出"
    else:
        advice = "持有"
    
    add_position_advice = "加仓" if predicted_price > current_price else "观望"
    
    return {
        "advice": advice,
        "add_position_advice": add_position_advice,
        "dates": [date.strftime('%Y-%m-%d') for date in future_dates],
        "close_prices": future_prices.tolist(),
        "high_prices": (future_prices * 1.02).tolist(),  # 模拟高价格
        "low_prices": (future_prices * 0.98).tolist(),  # 模拟低价格
        "open_prices": (future_prices * 1.01).tolist()  # 模拟开盘价格
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    stock_code = request.args.get('stockCode')
    prediction = predict_stock(stock_code)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
