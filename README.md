# Trading Simulation Environment (TSE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Trading Simulation Environment (TSE)** คือระบบจำลองการเทรดที่สมจริงสำหรับการพัฒนาและทดสอบอัลกอริทึม Reinforcement Learning (RL) ขั้นสูง เช่น Prioritized Experience Replay, Regularized Q-Learning, Noisy Networks, Dueling DQN และ Double DQN

## คุณสมบัติหลัก

- **สมจริงสูง**: จำลองค่าธรรมเนียมแบบเรียลไทม์ (0.25% ต่อธุรกรรม), การเกิด liquidation, slippage และปัจจัยอื่นๆ ในตลาดจริง
- **ข้อมูลละเอียด**: รองรับการใช้ข้อมูลไทม์เฟรม 1 นาทีเพื่อสร้างความสมจริงเพิ่มขึ้น (คล้ายกับ `--timeframe-detail 1m` ของ Freqtrade)
- **ประสิทธิภาพสูง**: ออกแบบให้ทำงานได้อย่างมีประสิทธิภาพบนทรัพยากรจำกัด ด้วยการใช้ CUDA/GPU อย่างเหมาะสม
- **ความยืดหยุ่น**: รองรับทั้ง discrete และ continuous action spaces, เข้ากันได้กับมาตรฐาน Gym/Gymnasium

## การเตรียมข้อมูล

TSE ใช้เทคนิค Log Transform + Z-score Normalization เพื่อเตรียมข้อมูล OHLCV (Open, High, Low, Close, Volume) และตัวชี้วัดต่างๆ ให้เหมาะกับโมเดล RL

- **Log Transform**: แก้ปัญหาการกระจายตัวแบบเบ้ (skewness) ของราคาและปริมาณ
- **Z-score Normalization**: ปรับข้อมูลให้มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1
- **Indicator Generation**: รองรับการคำนวณตัวชี้วัดที่หลากหลาย รวมถึง Relative Volume at Time

## ความต้องการของระบบ

- Python 3.8+
- PyTorch 1.9+
- CUDA พร้อมกับ GPU ที่รองรับ (แนะนำ: VRAM 6GB+)
- RAM 16GB+

## การติดตั้ง

```bash
# Clone repository
git clone https://github.com/yourusername/trading-sim-env.git
cd trading-sim-env

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
python scripts/download_data.py --symbol BTCUSDT --timeframe 1m --start 2023-01-01 --end 2023-12-31
python scripts/download_data.py --symbol BTCUSDT --timeframe 5m --start 2023-01-01 --end 2023-12-31
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

TSE ออกแบบมาเพื่อรองรับอัลกอริทึม RL ขั้นสูงหลายประเภท:

- **Double DQN**: ลดการประเมินค่าเกินจริง (overestimation bias) ในการเรียนรู้ Q-value
- **Dueling DQN**: แยกการประเมินค่าของสถานะและการกระทำ
- **Prioritized Experience Replay (PER)**: ให้ความสำคัญกับประสบการณ์ที่มีประโยชน์มากกว่า
- **Noisy Networks**: เพิ่มการสำรวจ (exploration) โดยการเพิ่ม noise ในพารามิเตอร์ของเครือข่าย
- **Regularized Q-Learning**: ป้องกันการ overfit และปรับปรุงความสามารถในการทั่วไปของโมเดล

## โครงสร้างโปรเจค

```
trading_sim_env/                      # โฟลเดอร์หลักของโปรเจค
│
├── data/                             # โฟลเดอร์สำหรับเก็บข้อมูล
│   ├── raw/                          # ข้อมูลดิบจาก Binance
│   ├── processed/                    # ข้อมูลที่ผ่านการประมวลผลแล้ว
│   └── indicators/                   # ข้อมูลตัวชี้วัดที่คำนวณไว้ล่วงหน้า
│
├── src/                              # โค้ดซอร์ส
│   ├── data/                         # โมดูลเกี่ยวกับข้อมูล
│   ├── environment/                  # โมดูลสภาพแวดล้อมจำลอง
│   ├── models/                       # โมดูลของโมเดล RL
│   ├── utils/                        # โมดูลยูทิลิตี้
│   └── common/                       # โมดูลที่ใช้ร่วมกัน
│
├── notebooks/                        # Jupyter notebooks
├── tests/                            # ทดสอบโค้ด
├── scripts/                          # สคริปต์ต่างๆ
├── configs/                          # ไฟล์การตั้งค่า
└── outputs/                          # ผลลัพธ์จากการเทรนและทดสอบ
```

## ตัวอย่างผลลัพธ์

(ให้จินตนาการว่ามีภาพกราฟผลลัพธ์การเทรดของโมเดลที่แสดงให้เห็นประสิทธิภาพที่ดี)

## การอ้างอิง

หากคุณใช้ TSE ในงานวิจัยหรือโปรเจคของคุณ กรุณาอ้างอิงตามนี้:

```
@misc{trading-sim-env2024,
  author = {Your Name},
  title = {Trading Simulation Environment for Advanced Reinforcement Learning},
  year = {2024},
  url = {https://github.com/yourusername/trading-sim-env}
}
```

## การติดต่อ

หากมีคำถาม ข้อเสนอแนะ หรือพบเจอปัญหา กรุณาเปิด issue หรือติดต่อทาง [your.email@example.com](mailto:your.email@example.com)

## สัญญาอนุญาต

โปรเจคนี้อยู่ภายใต้สัญญาอนุญาต MIT - ดูรายละเอียดในไฟล์ [LICENSE](https://opensource.org/license/MIT)
