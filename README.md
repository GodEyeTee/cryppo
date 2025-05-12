<h1 align="center">
  <br>
<img src="https://github.com/GodEyeTee/cryppo/blob/main/CRYPPO.png" alt="Markdownify" width="200">
  <br>
CRYPtocurrency Position Optimization (CRYPPO)
</h1>



[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/release/python-3129/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/GodEyeTee/cryppo)

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
python -m src.cli.main data download --symbol BTCUSDT --timeframe 1m --start 2023-01-01 --end 2023-12-31
```

### ประมวลผลข้อมูล

```bash
python -m src.cli.main data process --input data/raw/BTCUSDT/1m --output data/processed/BTCUSDT/1m --file-pattern "*.parquet"
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
python -m src.cli.main train model --input data/processed/BTCUSDT/1m/btcusdt_1m_combined.parquet --output models/btcusdt --model-type double_dqn --window-size 60 --batch-size 128 --epochs 5 --learning-rate 0.0001 --use-gpu
```

### Backtest:

```bash
python -m src.cli.main backtest run --model models/btcusdt/double_dqn_20250511_141809/model.pt --input data/processed/BTCUSDT/1m/btcusdt_1m_combined.parquet --output results/backtest --initial-balance 10000 --leverage 3.0 --fee-rate 0.0025 --stop-loss 5.0 --take-profit 0.5 --plot
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
cryppo/                           # โฟลเดอร์หลักของโปรเจค
│
├── data/                             # โฟลเดอร์สำหรับเก็บข้อมูล
│   ├── raw/                          # ข้อมูลดิบจาก Binance
│   ├── processed/                    # ข้อมูลที่ผ่านการประมวลผลแล้ว
│   └── indicators/                   # ข้อมูลตัวชี้วัดที่คำนวณไว้ล่วงหน้า
│
├── src/                              # โค้ดซอร์ส
│   ├── __init__.py                   # ทำให้เป็น package
│   │
│   ├── data/                         # โมดูลเกี่ยวกับข้อมูล
│   │   ├── __init__.py               # ทำให้เป็น package
│   │   ├── api/                      # API สำหรับดึงข้อมูล
│   │   │   ├── __init__.py
│   │   │   └── binance_api.py        # เฉพาะการติดต่อกับ Binance API
│   │   │
│   │   ├── downloaders/              # โมดูลสำหรับการดาวน์โหลด
│   │   │   ├── __init__.py
│   │   │   ├── base_downloader.py    # คลาสพื้นฐานสำหรับ downloader
│   │   │   └── binance_downloader.py # การดาวน์โหลดข้อมูลจาก Binance
│   │   │
│   │   ├── transforms/               # โมดูลสำหรับการแปลงข้อมูล
│   │   │   ├── __init__.py
│   │   │   ├── data_transforms.py    # การแปลงข้อมูล (Log, Z-score, etc.)
│   │   │   └── data_cleaning.py      # การทำความสะอาดข้อมูล
│   │   │
│   │   ├── indicators/               # โมดูลสำหรับตัวชี้วัดทางเทคนิค
│   │   │   ├── __init__.py
│   │   │   ├── basic_indicators.py   # ตัวชี้วัดพื้นฐาน (RSI, MACD, etc.)
│   │   │   ├── advanced_indicators.py # ตัวชี้วัดขั้นสูง
│   │   │   └── indicator_registry.py # ระบบทะเบียนตัวชี้วัด
│   │   │
│   │   ├── managers/                 # โมดูลสำหรับการจัดการข้อมูล
│   │   │   ├── __init__.py
│   │   │   └── data_manager.py       # การจัดการข้อมูลในหน่วยความจำ
│   │   │
│   │   ├── processors/               # โมดูลสำหรับการประมวลผลข้อมูล
│   │   │   ├── __init__.py
│   │   │   └── data_processor.py     # การประมวลผลข้อมูล
│   │   │
│   │   └── utils/                    # ยูติลิตี้สำหรับการจัดการข้อมูล
│   │       ├── __init__.py
│   │       ├── time_utils.py         # ยูติลิตี้เกี่ยวกับเวลา
│   │       └── file_utils.py         # ยูติลิตี้เกี่ยวกับไฟล์
│   │
│   ├── environment/                  # โมดูลสภาพแวดล้อมจำลอง
│   │   ├── __init__.py
│   │   ├── base_env.py               # คลาสพื้นฐานสำหรับสภาพแวดล้อม
│   │   ├── trading_env.py            # สภาพแวดล้อมการเทรด
│   │   ├── simulators/               # โมดูลจำลองการทำงาน
│   │   │   ├── __init__.py
│   │   │   └── trading_simulator.py  # จำลองการเทรดในตลาด
│   │   └── renderers/                # โมดูลการแสดงผล
│   │       ├── __init__.py
│   │       └── renderer.py           # การแสดงผลกราฟและข้อมูล
│   │
│   ├── models/                       # โมดูลของโมเดล RL
│   │   ├── __init__.py
│   │   ├── base_model.py             # คลาสพื้นฐานสำหรับโมเดล
│   │   ├── components/               # ส่วนประกอบของโมเดล
│   │   │   ├── __init__.py
│   │   │   ├── networks.py           # เครือข่ายประสาทเทียม
│   │   │   ├── memories.py           # หน่วยความจำสำหรับ RL
│   │   │   └── policies.py           # นโยบายการตัดสินใจ
│   │   │
│   │   ├── dqn/                      # โมดูลสำหรับ DQN
│   │   │   ├── __init__.py
│   │   │   ├── dqn.py                # Basic DQN
│   │   │   ├── double_dqn.py         # Double DQN
│   │   │   └── dueling_dqn.py        # Dueling DQN
│   │   │
│   │   └── utils/                    # ยูติลิตี้สำหรับโมเดล
│   │       ├── __init__.py
│   │       ├── exploration.py        # กลยุทธ์การสำรวจ
│   │       └── loss_functions.py     # ฟังก์ชันการสูญเสีย
│   │
│   ├── utils/                        # โมดูลยูทิลิตี้
│   │   ├── __init__.py
│   │   ├── config_manager.py         # การจัดการการตั้งค่า
│   │   ├── metrics.py                # การวัดประสิทธิภาพต่างๆ
│   │   ├── visualization.py          # การสร้างกราฟและภาพต่างๆ
│   │   └── loggers.py                # ระบบบันทึก log
│   │
│   └── cli/                          # โมดูลสำหรับ Command Line Interface
│       ├── __init__.py
│       ├── main.py                   # จุดเริ่มต้นของคำสั่ง cryppo
│       └── commands/                 # โมดูลสำหรับคำสั่งต่างๆ
│           ├── __init__.py
│           ├── data_commands.py      # คำสั่งเกี่ยวกับข้อมูล
│           ├── train_commands.py     # คำสั่งเกี่ยวกับการเทรน
│           └── backtest_commands.py  # คำสั่งเกี่ยวกับการทดสอบย้อนหลัง
│
├── notebooks/                        # Jupyter notebooks สำหรับการวิเคราะห์
│
├── tests/                            # ทดสอบโค้ด
│   ├── __init__.py
│   ├── data/                         # การทดสอบโมดูลข้อมูล
│   ├── environment/                  # การทดสอบโมดูลสภาพแวดล้อม
│   └── models/                       # การทดสอบโมดูลโมเดล
│
├── configs/                          # ไฟล์การตั้งค่า
│   ├── default_config.yaml           # การตั้งค่าเริ่มต้น
│   └── example_config.yaml           # ตัวอย่างการตั้งค่า
│
├── outputs/                          # ผลลัพธ์จากการเทรนและทดสอบ
│   ├── models/                       # โมเดลที่เทรนแล้ว
│   ├── logs/                         # บันทึกการเทรนและทดสอบ
│   └── backtest_results/             # ผลลัพธ์การทดสอบย้อนหลัง
│
├── setup.py                          # สำหรับติดตั้งแพ็คเกจ
├── pyproject.toml                    # ข้อมูลเกี่ยวกับโปรเจค
├── requirements.txt                  # แพ็คเกจที่จำเป็น
├── .gitignore                        # ไฟล์ที่ต้องการให้ git ไม่สนใจ
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

---

## 🧑‍💻 ผู้พัฒนา

โปรเจกต์นี้พัฒนาโดย [GodEyeTee](https://github.com/GodEyeTee)

อีเมล: Shiroaims@gmail.com

