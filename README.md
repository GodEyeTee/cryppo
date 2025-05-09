<h1 align="center">
  <br>
<img src="https://github.com/GodEyeTee/cryppo/blob/main/CRYPPO.png" alt="Markdownify" width="200">
  <br>
CRYPtocurrency Position Optimization (CRYPPO)
</h1>



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3129/)

**CRYPPO** (Cryptocurrency Position Optimization) คือระบบจำลองการเทรดที่สมจริง ออกแบบมาสำหรับการพัฒนาและทดสอบอัลกอริทึม Reinforcement Learning (RL) บนข้อมูลตลาดคริปโตความละเอียดสูง พร้อมการจัดการทรัพยากรที่เหมาะสมกับเครื่องสเปกจำกัด

---

## 📋 สารบัญ

1. [คุณสมบัติหลัก](#คุณสมบัติหลัก)
2. [ความต้องการของระบบ](#ความต้องการของระบบ)
3. [การติดตั้ง](#การติดตั้ง)
4. [การเตรียมข้อมูล](#การเตรียมข้อมูล)
5. [การจัดการข้อมูลในหน่วยความจำ](#การจัดการข้อมูลในหน่วยความจำ)
6. [การปรับแต่งสถิติ (Normalization & Transform)](#การปรับแต่งสถิติ-normalization--transform)
7. [การจัดการสถานะการเทรด (Action & Position)](#การจัดการสถานะการเทรด-action--position)
8. [การกำหนดค่า (Configuration)](#การกำหนดค่า-configuration)
9. [การเทรนและทดสอบ](#การเทรนและทดสอบ)
10. [การจัดการทรัพยากร (Resource Optimization)](#การจัดการทรัพยากร-resource-optimization)
11. [โครงสร้างโปรเจค](#โครงสร้างโปรเจค)
12. [ตัวอย่างผลลัพธ์](#ตัวอย่างผลลัพธ์)
13. [อ้างอิง](#อ้างอิง)
14. [ติดต่อ](#ติดต่อ)
15. [สัญญาอนุญาต](#สัญญาอนุญาต)

---

## 🚀 คุณสมบัติหลัก

* จำลองตลาดคริปโตอย่างสมจริง พร้อมค่าธรรมเนียม, Slippage และสถานะ Liquidation
* รองรับ Timeframe ระดับ 1 นาทีขึ้นไป
* ใช้ Parquet + PyTorch Tensor + Memory Mapping เพื่อจัดการข้อมูลขนาดใหญ่
* เข้ากันได้กับ Gym API และ RL libraries ชั้นนำ
* รองรับการทำงานกับ CPU/GPU และสามารถใช้กับเครื่องทรัพยากรจำกัดได้

---

## 💻 ความต้องการของระบบ

* **CPU**: Intel Core i5-9300H หรือเทียบเท่า
* **RAM**: 16 GB
* **GPU**: NVIDIA RTX 2060 (6GB GDDR6, Bus 192-bit)
* **OS**: Windows / Linux / macOS
* **Python**: 3.12+
* **PyTorch**: 1.9+ (แนะนำใช้กับ CUDA)

---

## ⚙️ การติดตั้ง

```bash
# 1. Clone โปรเจค
$ git clone https://github.com/GodEyeTee/cryppo.git
$ cd cryppo

# 2. สร้าง Virtual Environment
$ python -m venv venv
$ source venv/bin/activate      # สำหรับ Linux/macOS
# หรือ
$ venv\Scripts\activate        # สำหรับ Windows

# 3. ติดตั้ง dependencies
$ pip install -r requirements.txt
```

---

## 📊 การเตรียมข้อมูล

### รูปแบบข้อมูล: Parquet

* บีบอัดประสิทธิภาพสูง (ดีกว่า HDF5)
* อ่านเฉพาะคอลัมน์ได้ (columnar)
* โหลดเร็วกว่า memory-mapped files สำหรับข้อมูลขนาดใหญ่
* รองรับดีใน pandas / PyArrow

### ดาวน์โหลดข้อมูล

```bash
python scripts/download_data.py \
  --symbol BTCUSDT \
  --timeframe 1m \
  --start 2023-01-01 \
  --end 2023-12-31
```

### ประมวลผลข้อมูล

```bash
python -m src.data.data_processor \
  --input data/raw/BTCUSDT/1m \
  --output data/processed/BTCUSDT/1m
```

---

## 🧠 การจัดการข้อมูลในหน่วยความจำ

ใช้ PyTorch Tensor สำหรับจัดการข้อมูล:

* Data type: `float32`
* รองรับ batching, slicing, memory map
* เหมาะกับ GPU และ CPU
* ใช้กับ Autograd ได้ทันที

```python
from src.data.data_manager import MarketDataManager

data_manager = MarketDataManager(
    file_path="data/processed/BTCUSDT/5m_processed.parquet",
    batch_size=512,
    window_size=60,
    detail_timeframe='1m',
    base_timeframe='5m',
    memory_map=True
)
```

---

## 🔄 การปรับแต่งสถิติ (Normalization & Transform)

1. **Log Transform**: แก้ skewness ของราคาและ volume
2. **Z-score Normalization**: ปรับข้อมูลให้อยู่ในรูป mean=0, std=1

> ทำแบบ Online ได้เพื่อลดการใช้หน่วยความจำ

---

## 📈 การจัดการสถานะการเทรด (Action & Position)

| Action | ค่าที่ใช้ |
| ------ | --------- |
| NONE   | 0         |
| LONG   | 1         |
| SHORT  | 2         |
| EXIT   | 3         |

* **Maximum Open Positions**: จำกัด 1 ตำแหน่งในแต่ละครั้ง
* **การนับว่ากำไร**: จะถือว่า "ได้กำไร" ต่อเมื่อกำไรมากกว่า **หรือเท่ากับ 0.5%** จากราคาเข้าซื้อ
* **Liquidation Condition**: ถ้าขาดทุนเกิน **5%** จากราคาเข้าซื้อ จะถูกบังคับปิด (ไม่ใช่ stop loss ปกติ)

---

## 🛠️ การกำหนดค่า (Configuration)

จัดเก็บไว้ใน `configs/`:

```json
{
  "initial_balance": 10000,
  "fee_rate": 0.0025,
  "leverage": 3.0,
  "liquidation_threshold": 0.8,
  "stop_loss": 0.05,
  "take_profit": 0.005
}
```

---

## 🎯 การเทรนและทดสอบ

### เทรนโมเดล:

```bash
python scripts/train_model.py \
  --config configs/model_config.json \
  --model_type double_dqn
```

### Backtest:

```bash
python scripts/backtest.py \
  --model outputs/models/double_dqn_5m \
  --config configs/backtest_config.json
```

---

## 🧩 การจัดการทรัพยากร (Resource Optimization)

ปรับแต่งเพื่อใช้กับเครื่องจำกัดทรัพยากร:

* ใช้ `batch_size` ขนาดเล็ก เช่น 32–128
* `memory_map=True` ใน DataManager
* `pin_memory=True` และ `num_workers=2` ใน DataLoader
* ใช้ `with torch.no_grad()` เมื่อต้องการ inference
* ใช้ Mixed Precision (`torch.cuda.amp`) ลดการใช้ VRAM

---

## 🗂️ โครงสร้างโปรเจค

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

---

## 📊 ตัวอย่างผลลัพธ์

![Trade Result](assets/examples/trade_result.png)

---

## 📚 อ้างอิง

```
@misc{cryppo2025,
  author = {GodEyeTee},
  title = {Cryptocurrency Position Optimization},
  year = {2025},
  url = {https://github.com/GodEyeTee/cryppo}
}
```

---

## 📫 ติดต่อ

* GitHub Issue
* Email: [Shiroaims@gmail.com](mailto:Shiroaims@gmail.com)

---

## 📄 สัญญาอนุญาต

โปรเจคนี้อยู่ภายใต้ MIT License - ดูรายละเอียดในไฟล์ [LICENSE](https://opensource.org/license/MIT)

