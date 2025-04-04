# Gelişmiş Turtle Trading Bot

Bu proje, Turtle Trading stratejisini kullanan ve çeşitli gelişmiş özelliklerle zenginleştirilmiş bir algoritmik ticaret botudur.

## Özellikler

- **Çoklu Zaman Dilimi Analizi**: Farklı zaman dilimlerinde trendleri tespit etme
- **Piramit Pozisyon Açma**: Trendin güçlenmesi durumunda pozisyonu artırma
- **Gelişmiş Çıkış Stratejisi**: Kısmi kar alma hedefleri ve trailing stop
- **Trend Filtreleri**: ADX ve Hareketli Ortalama (MA) filtrelemeleri
- **Akıllı Risk Yönetimi**: Pozisyon başına ve toplam risk limitlemesi

## Strateji Mantığı

Bu bot, şu temel bileşenlere dayalı bir ticaret stratejisi uygular:

1. **Trend Analizi**:
   - 1 günlük grafikte 200 günlük hareketli ortalama ile ana trendler belirlenir
   - ADX göstergesi ile trend gücü ölçülür (25 üzeri değerler güçlü trend)

2. **Giriş Sinyalleri**:
   - Donchian Kanalları trend-takip eden giriş sinyalleri üretir
   - Giriş sinyalleri ana trend yönü ile karşılaştırılarak onaylanır

3. **Piramitleme**:
   - İlk giriş: Planlanmış pozisyon boyutunun %40'ı
   - Ek girişler: Kalan büyüklüğün %30'luk dilimleri

4. **Çıkış Stratejisi**:
   - İlk Hedef: 3 ATR mesafesinde pozisyonun %50'si çıkılır
   - İkinci Hedef: 5 ATR mesafesinde pozisyonun %30'u çıkılır
   - Son dilim için: Trailing stop kullanılır

5. **Kaldıraç Yönetimi**:
   - Trend yönünde işlemlerde: 2-3x kaldıraç
   - Trend tersine işlemlerde: Maksimum 1.5x kaldıraç

## Kurulum

1. Depoyu klonlayın:
```bash
git clone [repo-url]
cd TurtleTrading
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. `.env` dosyasını düzenleyin:
```
# API anahtarlarınızı buraya yazın
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
USE_TESTNET=True  # Gerçek ticarete başlamadan önce test modunda çalıştırın
```

4. Strateji parametrelerini `.env` dosyasında isteğe göre ayarlayın:
```
# Risk parametrelerini kendi tercihlerinize göre ayarlayın
RISK_PER_TRADE=0.02  # Sermayenizin %2'si
STOP_LOSS_ATR_MULTIPLE=1.5  # ATR'nin 1.5 katı stop loss mesafesi
```

## Çalıştırma

Botu başlatmak için:

```bash
python turtle_trading_bot.py
```

## Dikkat Edilmesi Gerekenler

- Gerçek parayla kullanmadan önce testnet üzerinde test edin
- Risk yönetimi parametrelerini kendi risk toleransınıza göre ayarlayın
- Bot, sermayenizin tamamını kaybetme riskiyle çalışır, sorumluluk size aittir

## Özelleştirme

Strateji parametrelerini `.env` dosyasında değiştirerek botun davranışını özelleştirebilirsiniz:

- `USE_MULTI_TIMEFRAME`: Çoklu zaman dilimi analizini açar/kapatır
- `USE_PYRAMIDING`: Piramitleme stratejisini açar/kapatır
- `USE_TRAILING_STOP`: Trailing stop kullanımını açar/kapatır
- `USE_PARTIAL_EXITS`: Kısmi kar alma hedeflerini açar/kapatır
- `USE_ADX_FILTER` ve `USE_MA_FILTER`: Trend filtrelerini açar/kapatır

## Project Structure

```
turtle_trading_bot/
├── turtle_trading_bot.py    # Main entry point
├── .env                     # Configuration (API keys, trading parameters)
├── .env.example             # Example configuration template
├── requirements.txt         # Dependencies
├── logs/                    # Trading logs directory
├── config/                  # Bot state and configuration files
└── bot/                     # Core modules
    ├── __init__.py          # Package initialization
    ├── core.py              # TurtleTradingBot class
    ├── exchange.py          # Binance exchange operations
    ├── indicators.py        # Technical indicators and signal detection
    ├── models.py            # Data models and type definitions
    ├── risk.py              # Risk management and position sizing
    └── utils.py             # Utility functions
```

## Setup and Configuration

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and update with your Binance API credentials:
   ```
   cp .env.example .env
   ```
4. Edit parameters in `.env` as needed (risk, timeframe, symbol, etc.)

## Usage

Run the bot:

```
python turtle_trading_bot.py
```

## Configuration Parameters

Edit these in the `.env` file:

- **API_KEY/API_SECRET**: Your Binance API credentials
- **USE_TESTNET**: Set to True for testing, False for live trading
- **SYMBOL**: Trading pair (e.g., BTCUSDT)
- **TIMEFRAME**: Candlestick interval (e.g., 1h, 4h, 1d)
- **DC_LENGTH_ENTER**: Donchian Channel period for entries
- **DC_LENGTH_EXIT**: Donchian Channel period for exits
- **ATR_LENGTH**: ATR calculation period
- **RISK_PER_TRADE**: Risk percentage per trade (0.02 = 2%)
- **STOP_LOSS_ATR_MULTIPLE**: ATR multiplier for stop loss placement

## Improvements from Original Codebase

- **Object-Oriented Design**: Proper encapsulation of state and behavior
- **Type Hints**: Enhanced code quality and IDE support
- **Modular Structure**: Separate modules for different concerns
- **Improved Error Handling**: Consistent exception handling
- **Better Documentation**: Comprehensive docstrings and code comments
- **Short Position Support**: Added ability to trade in both directions
- **Enhanced Testability**: Easier to write unit tests

## License

[MIT License](LICENSE)

## Disclaimer

Trading cryptocurrencies carries significant risk. This bot is provided for educational purposes only. Use at your own risk.
