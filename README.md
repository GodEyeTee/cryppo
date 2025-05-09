<h1 align="center">
  <br>
<img src="https://github.com/GodEyeTee/cryppo/blob/main/CRYPPO.png" alt="Markdownify" width="200">
  <br>
CRYPtocurrency Position Optimization (CRYPPO)
</h1>



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3129/)

**Cryptocurrency Position Optimization (CRYPPO)** คือระบบจำลองการเทรดที่สมจริงสำหรับการพัฒนาและทดสอบอัลกอริทึม Reinforcement Learning (RL) ขั้นสูง เช่น Prioritized Experience Replay, Regularized Q-Learning, Noisy Networks, Dueling DQN และ Double DQN

## คุณสมบัติหลัก

- **สมจริงสูง**: จำลองค่าธรรมเนียมแบบเรียลไทม์ (0.25% ต่อธุรกรรม), การเกิด liquidation, slippage และปัจจัยอื่นๆ ในตลาดจริง
- **ข้อมูลละเอียด**: รองรับการใช้ข้อมูลไทม์เฟรม 1 นาทีเพื่อสร้างความสมจริงเพิ่มขึ้น (คล้ายกับ `--timeframe-detail 1m` ของ Freqtrade)
- **ประสิทธิภาพสูง**: ออกแบบให้ทำงานได้อย่างมีประสิทธิภาพบนทรัพยากรจำกัด ด้วยการใช้ CUDA/GPU อย่างเหมาะสม
- **ความยืดหยุ่น**: รองรับทั้ง discrete และ continuous action spaces, เข้ากันได้กับมาตรฐาน Gym/Gymnasium

## การเตรียมข้อมูล

CRYPPO ใช้เทคนิค Log Transform + Z-score Normalization เพื่อเตรียมข้อมูล OHLCV (Open, High, Low, Close, Volume) และตัวชี้วัดต่างๆ ให้เหมาะกับโมเดล RL

- **Log Transform**: แก้ปัญหาการกระจายตัวแบบเบ้ (skewness) ของราคาและปริมาณ
- **Z-score Normalization**: ปรับข้อมูลให้มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1
- **Indicator Generation**: รองรับการคำนวณตัวชี้วัดที่หลากหลาย รวมถึง Relative Volume at Time

## ความต้องการของระบบ

- Python 3.12+
- PyTorch 1.9+
- CUDA พร้อมกับ GPU ที่รองรับ (แนะนำ: VRAM 6GB+)
- RAM 16GB+

## การติดตั้ง

```bash
# Clone repository
git clone https://github.com/GodEyeTee/cryppo.git
cd cryppo

# สร้าง virtual environment (แนะนำ)
python -m venv venv
source venv/bin/activate  # สำหรับ Linux/Mac
# หรือ
venv\Scripts\activate  # สำหรับ Windows

# ติดตั้งแพ็คเกจที่จำเป็น
pip install -r requirements.txt
```

## การเริ่มต้นใช้งานอย่างรวดเร็ว

### 1. ดาวน์โหลดข้อมูล

```bash
python data/download_data.py --symbol BTCUSDT --timeframe 1m --start 2024-01-01 --end 2024-12-31
python data/download_data.py --symbol BTCUSDT --timeframe 5m --start 2023-01-01 --end 2023-12-31
```

### 2. ประมวลผลข้อมูล

```bash
python -m src.data.data_processor --input data/raw/BTC-USDT --output data/processed/BTC-USDT
```

### 3. เทรนโมเดล

```bash
python scripts/train_model.py --config configs/model_config.json --model_type double_dqn
```

### 4. ทดสอบย้อนหลัง

```bash
python scripts/backtest.py --model outputs/models/double_dqn_5m --config configs/backtest_config.json
```

## การใช้งานขั้นสูง

### การสร้างสภาพแวดล้อมแบบกำหนดเอง

```python
from src.data.data_manager import MarketDataManager
from src.environment.trading_env import TradingEnv

# สร้าง data manager
data_manager = MarketDataManager(
    file_path="data/processed/BTC-USDT/5m_processed.parquet",
    batch_size=1024,
    window_size=60,
    detail_timeframe='1m',
    base_timeframe='5m'
)

# สร้างสภาพแวดล้อมการเทรด
env = TradingEnv(
    data_manager=data_manager,
    action_type='discrete',  # หรือ 'continuous'
    render_mode='human'      # สำหรับการแสดงผลแบบเรียลไทม์
)

# ใช้กับโมเดล RL
observation, info = env.reset()
for i in range(1000):
    action = your_model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
```

### การปรับแต่งการจำลองการเทรด

```python
from src.environment.trading_simulator import TradingSimulationEnvironment

# สร้างตัวจำลองการเทรดแบบกำหนดเอง
simulator = TradingSimulationEnvironment(
    data_manager=data_manager,
    initial_balance=10000.0,    # เงินเริ่มต้น
    fee_rate=0.0025,            # ค่าธรรมเนียม 0.25%
    leverage=3.0,               # คันเร่ง (leverage)
    liquidation_threshold=0.8,  # เกณฑ์การ liquidation
    stop_loss=0.1,              # Stop Loss 10%
    take_profit=0.2             # Take Profit 20%
)
```

## โมเดลที่รองรับ

CRYPPO ออกแบบมาเพื่อรองรับอัลกอริทึม RL ขั้นสูงหลายประเภท:

- **Double DQN**: ลดการประเมินค่าเกินจริง (overestimation bias) ในการเรียนรู้ Q-value
- **Dueling DQN**: แยกการประเมินค่าของสถานะและการกระทำ
- **Prioritized Experience Replay (PER)**: ให้ความสำคัญกับประสบการณ์ที่มีประโยชน์มากกว่า
- **Noisy Networks**: เพิ่มการสำรวจ (exploration) โดยการเพิ่ม noise ในพารามิเตอร์ของเครือข่าย
- **Regularized Q-Learning**: ป้องกันการ overfit และปรับปรุงความสามารถในการทั่วไปของโมเดล

## โครงสร้างโปรเจค

```
cryppo/                      # โฟลเดอร์หลักของโปรเจค
│
├── data/                             # โฟลเดอร์สำหรับเก็บข้อมูล
│   ├── raw/                          # ข้อมูลดิบจาก Binance
│   │   ├── BTC-USDT/                 # ข้อมูลแยกตามคู่สกุลเงิน
│   │   │   ├── 1m/                   # ข้อมูลไทม์เฟรม 1 นาที
│   │   │   ├── 5m/                   # ข้อมูลไทม์เฟรม 5 นาที
│   │   │   └── ...
│   │   └── ...
│   ├── processed/                    # ข้อมูลที่ผ่านการประมวลผลแล้ว
│   │   ├── BTC-USDT/
│   │   │   ├── 1m_processed.parquet  # ข้อมูลที่เตรียมไว้สำหรับโมเดล
│   │   │   ├── 5m_processed.parquet
│   │   │   └── ...
│   │   └── ...
│   └── indicators/                   # ข้อมูลตัวชี้วัดที่คำนวณไว้ล่วงหน้า
│       └── ...
│
├── src/                              # โค้ดซอร์ส
│   ├── __init__.py                  
│   │
│   ├── data/                         # โมดูลเกี่ยวกับข้อมูล
│   │   ├── __init__.py
│   │   ├── binance_downloader.py     # ดาวน์โหลดข้อมูลจาก Binance API
│   │   ├── data_processor.py         # การประมวลผลข้อมูล (LogTransform+Z-score)
│   │   ├── indicators.py             # คำนวณตัวชี้วัดต่างๆ (indicators)
│   │   └── data_manager.py           # การจัดการข้อมูลสำหรับการเทรนและทดสอบ
│   │
│   ├── environment/                  # โมดูลสภาพแวดล้อมจำลอง
│   │   ├── __init__.py
│   │   ├── trading_simulator.py      # จำลองการเทรดในตลาด
│   │   ├── trading_env.py            # สภาพแวดล้อม Gym/Gymnasium
│   │   └── renderer.py               # การแสดงผลกราฟและข้อมูล
│   │
│   ├── models/                       # โมดูลของโมเดล RL
│   │   ├── __init__.py
│   │   ├── dqn.py                    # Double DQN
│   │   ├── per.py                    # Prioritized Experience Replay
│   │   ├── noisy_nets.py             # Noisy Networks
│   │   ├── dueling_dqn.py            # Dueling DQN
│   │   └── regularized_q.py          # Regularized Q-Learning
│   │
│   ├── utils/                        # โมดูลยูทิลิตี้
│   │   ├── __init__.py
│   │   ├── config.py                 # การตั้งค่าต่างๆ
│   │   ├── metrics.py                # การวัดประสิทธิภาพต่างๆ
│   │   └── visualization.py          # การสร้างกราฟและภาพต่างๆ
│   │
│   └── common/                       # โมดูลที่ใช้ร่วมกัน
│       ├── __init__.py
│       ├── constants.py              # ค่าคงที่ต่างๆ
│       └── logger.py                 # ระบบบันทึก log
│
├── notebooks/                        # Jupyter notebooks สำหรับการวิเคราะห์และทดสอบ
│   ├── 01_data_exploration.ipynb     # การสำรวจข้อมูล
│   ├── 02_indicators_analysis.ipynb  # การวิเคราะห์ตัวชี้วัด
│   ├── 03_model_testing.ipynb        # การทดสอบโมเดล
│   └── ...
│
├── tests/                            # ทดสอบโค้ด
│   ├── __init__.py
│   ├── test_data_processor.py        # ทดสอบการประมวลผลข้อมูล
│   ├── test_trading_simulator.py     # ทดสอบตัวจำลองการเทรด
│   └── ...
│
├── scripts/                          # สคริปต์ต่างๆ
│   ├── download_data.py              # สคริปต์ดาวน์โหลดข้อมูล
│   ├── train_model.py                # เทรนโมเดล
│   ├── backtest.py                   # ทดสอบย้อนหลัง
│   └── evaluate.py                   # ประเมินผลโมเดล
│
├── configs/                          # ไฟล์การตั้งค่า
│   ├── data_config.json              # การตั้งค่าเกี่ยวกับข้อมูล
│   ├── env_config.json               # การตั้งค่าสภาพแวดล้อม
│   ├── model_config.json             # การตั้งค่าโมเดล
│   └── ...
│
├── outputs/                          # ผลลัพธ์จากการเทรนและทดสอบ
│   ├── models/                       # โมเดลที่เทรนแล้ว
│   ├── logs/                         # บันทึกการเทรนและทดสอบ
│   ├── plots/                        # กราฟและแผนภาพ
│   └── backtest_results/             # ผลลัพธ์การทดสอบย้อนหลัง
│
├── .gitignore                        # ไฟล์ที่ต้องการให้ git ไม่สนใจ
├── requirements.txt                  # แพ็คเกจที่จำเป็น
├── setup.py                          # สำหรับติดตั้งแพ็คเกจ
└── README.md                         # เอกสารการใช้งาน
```

## ตัวอย่างผลลัพธ์

(ให้จินตนาการว่ามีภาพกราฟผลลัพธ์การเทรดของโมเดลที่แสดงให้เห็นประสิทธิภาพที่ดี)

## การอ้างอิง

หากคุณใช้ CRYPPO ในงานวิจัยหรือโปรเจคของคุณ กรุณาอ้างอิงตามนี้:

```
@misc{cryppo2025,
  author = {GodEyeTee},
  title = {Cryptocurrency Position Optimization},
  year = {2025},
  url = {https://github.com/GodEyeTee/cryppo}
}
```

## การติดต่อ

หากมีคำถาม ข้อเสนอแนะ หรือพบเจอปัญหา กรุณาเปิด issue หรือติดต่อทาง [Shiroaims@gmail.com](mailto:Shiroaims@gmail.com)

## สัญญาอนุญาต

โปรเจคนี้อยู่ภายใต้สัญญาอนุญาต MIT - ดูรายละเอียดในไฟล์ [LICENSE](https://opensource.org/license/MIT)
