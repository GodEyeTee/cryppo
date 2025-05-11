import os
import logging
from datetime import datetime
import json
import pandas as pd

from src.data.managers.data_manager import MarketDataManager
from src.environment.trading_env import TradingEnv
from src.models.model_factory import ModelFactory
from src.utils.config_manager import get_config
from src.utils.metrics import PerformanceTracker

logger = logging.getLogger('cli.backtest')

def setup_run_parser(parser):
    """
    ตั้งค่า parser สำหรับคำสั่งทดสอบย้อนหลัง
    """
    parser.add_argument("--model", type=str, required=True, help="ไฟล์โมเดลที่ต้องการทดสอบ")
    parser.add_argument("--input", type=str, required=True, help="ไฟล์ข้อมูลที่ใช้ทดสอบ")
    parser.add_argument("--output", type=str, required=True, help="ไดเรกทอรีสำหรับบันทึกผลการทดสอบ")
    parser.add_argument("--start-date", type=str, default=None, help="วันที่เริ่มต้นทดสอบ (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="วันที่สิ้นสุดการทดสอบ (YYYY-MM-DD)")
    parser.add_argument("--initial-balance", type=float, default=None, help="เงินทุนเริ่มต้น")
    parser.add_argument("--leverage", type=float, default=None, help="คูณทุน")
    parser.add_argument("--fee-rate", type=float, default=None, help="อัตราค่าธรรมเนียม")
    parser.add_argument("--stop-loss", type=float, default=None, help="การจำกัดการขาดทุน (เปอร์เซ็นต์)")
    parser.add_argument("--take-profit", type=float, default=None, help="การทำกำไร (เปอร์เซ็นต์)")
    parser.add_argument("--window-size", type=int, default=None, help="ขนาดของหน้าต่างข้อมูล")
    parser.add_argument("--batch-size", type=int, default=None, help="ขนาดของแต่ละ batch")
    parser.add_argument("--use-gpu", action="store_true", default=None, help="ใช้ GPU ในการทดสอบ")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="ไม่ใช้ GPU ในการทดสอบ")
    parser.add_argument("--plot", action="store_true", help="สร้างกราฟผลการทดสอบ")
    parser.add_argument("--verbose", action="store_true", help="แสดงรายละเอียดการทดสอบ")
    parser.add_argument("--config", type=str, default=None, help="ไฟล์การตั้งค่าเฉพาะสำหรับการทดสอบ")

def setup_analyze_parser(parser):
    """
    ตั้งค่า parser สำหรับคำสั่งวิเคราะห์ผลการทดสอบ
    """
    parser.add_argument("--input", type=str, required=True, help="ไฟล์ผลการทดสอบที่ต้องการวิเคราะห์")
    parser.add_argument("--output", type=str, default=None, help="ไฟล์สำหรับบันทึกผลการวิเคราะห์")
    parser.add_argument("--metrics", type=str, default="all", 
                      help="รายการ metrics ที่ต้องการวิเคราะห์ (คั่นด้วยเครื่องหมายจุลภาค)")
    parser.add_argument("--benchmark", type=str, default=None, 
                      help="ไฟล์ผลการทดสอบที่ใช้เป็น benchmark (คั่นด้วยเครื่องหมายจุลภาค)")
    parser.add_argument("--plot", choices=["trades", "equity", "returns", "drawdown", "all"],
                      default="all", help="กราฟที่ต้องการแสดง")
    parser.add_argument("--period", choices=["daily", "weekly", "monthly"], 
                      default="daily", help="ช่วงเวลาสำหรับการวิเคราะห์")

def calculate_basic_metrics(trades_df, initial_equity, periods_per_year=252):
    """คำนวณเมตริกพื้นฐาน"""
    tracker = PerformanceTracker(initial_equity=initial_equity, periods_per_year=periods_per_year)
    
    # เพิ่มข้อมูลทีละช่วงเวลา
    for i in range(1, len(trades_df)):
        equity = trades_df['portfolio_value'].iloc[i]
        position = trades_df['position'].iloc[i] if 'position' in trades_df.columns else 0
        timestamp = None
        if 'timestamp' in trades_df.columns:
            timestamp = pd.to_datetime(trades_df['timestamp'].iloc[i])
        
        tracker.update(equity=equity, timestamp=timestamp, position=position)
    
    return tracker.calculate_metrics(risk_free_rate=0.0)

def calculate_trade_analysis_metrics(trades_df):
    """วิเคราะห์การเทรดและคำนวณเมตริกที่เกี่ยวข้อง"""
    metrics = {}
    
    if 'action' in trades_df.columns and 'position' in trades_df.columns:
        positions = trades_df['position'].values
        
        # นับการเปลี่ยนแปลงตำแหน่ง
        position_changes = []
        for i in range(1, len(positions)):
            if positions[i] != positions[i-1]:
                position_changes.append(i)
        
        # วิเคราะห์การเทรด
        trade_count = 0
        winning_trades = 0
        losing_trades = 0
        
        for i in range(0, len(position_changes) - 1, 2):
            if i + 1 < len(position_changes):
                trade_count += 1
                
                start_idx = position_changes[i]
                end_idx = position_changes[i + 1]
                
                start_value = trades_df['portfolio_value'].iloc[start_idx]
                end_value = trades_df['portfolio_value'].iloc[end_idx]
                
                trade_return = (end_value / start_value) - 1
                
                if trade_return > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
        
        metrics['total_trades'] = trade_count
        metrics['winning_trades'] = winning_trades
        metrics['losing_trades'] = losing_trades
        
        if trade_count > 0:
            metrics['win_rate'] = (winning_trades / trade_count) * 100
    
    return metrics

def calculate_trading_metrics(trades_df, raw_data=None, risk_free_rate=0.0, periods_per_year=252):
    """
    คำนวณเมตริกสำหรับการประเมินผลการเทรด
    """
    if trades_df.empty:
        logger.warning("ไม่มีข้อมูลการเทรดสำหรับการคำนวณเมตริก")
        return {}
    
    # คำนวณเมตริกพื้นฐาน
    initial_equity = trades_df['portfolio_value'].iloc[0] if 'portfolio_value' in trades_df.columns else 10000.0
    metrics = calculate_basic_metrics(trades_df, initial_equity, periods_per_year)
    
    # เพิ่มเมตริกเกี่ยวกับช่วงเวลา
    if raw_data is not None and 'timestamp' in trades_df.columns:
        # คำนวณจำนวนวันทดสอบ
        first_date = pd.to_datetime(trades_df['timestamp'].iloc[0]).date()
        last_date = pd.to_datetime(trades_df['timestamp'].iloc[-1]).date()
        days_diff = (last_date - first_date).days
        metrics['trading_days'] = days_diff if days_diff > 0 else 1
        
        # คำนวณผลตอบแทนของ Buy & Hold
        if hasattr(raw_data, 'columns') and 'close' in raw_data.columns and len(raw_data) > 1:
            try:
                if hasattr(raw_data.index, 'date'):
                    mask = (raw_data.index.date >= first_date) & (raw_data.index.date <= last_date)
                    raw_data_in_range = raw_data.loc[mask]
                else:
                    raw_data_in_range = raw_data
                
                if len(raw_data_in_range) > 1:
                    first_price = raw_data_in_range['close'].iloc[0]
                    last_price = raw_data_in_range['close'].iloc[-1]
                    metrics['buy_hold_return'] = ((last_price / first_price) - 1) * 100
            except Exception as e:
                logger.warning(f"ไม่สามารถคำนวณผลตอบแทนของ Buy & Hold: {e}")
    
    # วิเคราะห์การเทรด
    trade_metrics = calculate_trade_analysis_metrics(trades_df)
    metrics.update(trade_metrics)
    
    return metrics

def handle_run(args):
    """
    จัดการคำสั่งทดสอบย้อนหลัง
    """
    # โหลดการตั้งค่า
    config = get_config()
    
    # ถ้ามีไฟล์การตั้งค่าเฉพาะ ให้โหลดเพิ่มเติม
    if args.config and os.path.exists(args.config):
        config.load_config(args.config)
    
    # ตรวจสอบไฟล์
    if not os.path.exists(args.model):
        logger.error(f"ไม่พบไฟล์โมเดล: {args.model}")
        return
    
    if not os.path.exists(args.input):
        logger.error(f"ไม่พบไฟล์ข้อมูล: {args.input}")
        return
    
    # โหลดการตั้งค่าจากโมเดล
    model_dir = os.path.dirname(args.model)
    config_path = os.path.join(model_dir, "config.json")
    
    # ถ้ามีไฟล์การตั้งค่าของโมเดล ให้โหลดก่อน
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_config = json.load(f)
            config.update_from_dict(model_config)
    
    # ปรับการตั้งค่าตามอาร์กิวเมนต์
    backtest_config = config.extract_subconfig("backtest")
    
    if args.window_size:
        config.set("data.window_size", args.window_size)
    
    if args.batch_size:
        config.set("model.batch_size", args.batch_size)
    
    if args.start_date:
        backtest_config["start_date"] = args.start_date
    
    if args.end_date:
        backtest_config["end_date"] = args.end_date
    
    if args.initial_balance:
        backtest_config["initial_balance"] = args.initial_balance
    
    if args.leverage:
        backtest_config["leverage"] = args.leverage
    
    if args.fee_rate:
        backtest_config["fee_rate"] = args.fee_rate
    
    if args.stop_loss:
        backtest_config["stop_loss"] = args.stop_loss / 100.0  # แปลงเป็นทศนิยม
    
    if args.take_profit:
        backtest_config["take_profit"] = args.take_profit / 100.0  # แปลงเป็นทศนิยม
    
    if args.use_gpu is not None:
        config.set("cuda.use_cuda", args.use_gpu)
    
    backtest_config["plot_results"] = args.plot
    backtest_config["verbose"] = args.verbose
    
    # สร้างโฟลเดอร์สำหรับบันทึกผลการทดสอบ
    os.makedirs(args.output, exist_ok=True)
    
    # โหลดข้อมูล
    data_manager = MarketDataManager(
        file_path=args.input,
        window_size=config.get("data.window_size"),
        batch_size=config.get("model.batch_size")
    )
    
    if not data_manager.data_loaded:
        logger.error(f"ไม่สามารถโหลดข้อมูลจาก {args.input} ได้")
        return
    
    # กรองข้อมูลตามช่วงวันที่
    if backtest_config.get("start_date") or backtest_config.get("end_date"):
        data_manager.filter_data(
            start_date=backtest_config.get("start_date"),
            end_date=backtest_config.get("end_date")
        )
    
    # โหลดสถิติ
    stats_path = os.path.join(model_dir, "data_stats.json")
    if os.path.exists(stats_path):
        data_manager.load_stats(stats_path)
    
    # สร้างสภาพแวดล้อมการเทรด
    env = TradingEnv(
        data_manager=data_manager,
        config=config
    )
    
    # โหลดโมเดล
    model_type = config.get("model.model_type")
    input_size = 25  # หรือค่าที่ถูกต้องตามโมเดลของคุณ

    model = ModelFactory.create_model(
        model_type=model_type,
        input_size=input_size,
        config=config
    )
    
    model.load(args.model)
    
    # ตั้งค่าการบันทึกผล
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.output, f"backtest_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # ทดสอบย้อนหลัง
    logger.info("กำลังทดสอบย้อนหลัง...")
    
    # รีเซ็ตสภาพแวดล้อม
    observation = env.reset()
    done = False
    total_reward = 0
    actions = []
    rewards = []
    dones = []
    positions = []
    portfolio_values = []
    timestamps = []
    
    # ดำเนินการทดสอบ
    while not done:
        # ตัดสินใจด้วยโมเดล
        action = model.predict(observation)
        
        # ดำเนินการตามการตัดสินใจ
        next_observation, reward, done, info = env.step(action)
        
        # บันทึกผล
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        positions.append(info.get('position', 0))
        portfolio_values.append(info.get('portfolio_value', backtest_config.get("initial_balance")))
        timestamps.append(info.get('timestamp', None))
        
        # อัพเดตสถานะ
        observation = next_observation
        total_reward += reward
        
        # แสดงความคืบหน้า
        if args.verbose and (len(rewards) % 100 == 0):
            logger.info(f"Step: {len(rewards)}, Reward: {reward:.4f}, Total: {total_reward:.4f}, Portfolio: {portfolio_values[-1]:.2f}")
    
    logger.info(f"ทดสอบย้อนหลังเสร็จสิ้น! Total Reward: {total_reward:.4f}, Final Portfolio: {portfolio_values[-1]:.2f}")
    
    # สร้าง DataFrame ของการเทรด
    trades_df = pd.DataFrame({
        'timestamp': timestamps,
        'action': actions,
        'reward': rewards,
        'position': positions,
        'portfolio_value': portfolio_values,
        'done': dones
    })
    
    # บันทึกผลการทดสอบ
    trades_path = os.path.join(result_dir, "trades.csv")
    trades_df.to_csv(trades_path, index=False)
    
    # บันทึกการตั้งค่า
    config_path = os.path.join(result_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # คำนวณ metrics
    metrics = calculate_trading_metrics(trades_df, data_manager.raw_data)
    
    # บันทึก metrics
    metrics_path = os.path.join(result_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # แสดงสรุปผล
    print("\nสรุปผลการทดสอบย้อนหลัง:")
    print(f"จำนวนวันทดสอบ: {metrics.get('trading_days', 'N/A')} วัน")
    print(f"เงินทุนเริ่มต้น: {backtest_config.get('initial_balance', trades_df['portfolio_value'].iloc[0]):.2f}")
    print(f"มูลค่าพอร์ตสุดท้าย: {portfolio_values[-1]:.2f}")
    print(f"กำไร/ขาดทุนรวม: {portfolio_values[-1] - trades_df['portfolio_value'].iloc[0]:.2f} ({metrics.get('total_return', 0):.2f}%)")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
    print(f"จำนวนการเทรดทั้งหมด: {metrics.get('total_trades', 0)}")
    print(f"จำนวนการเทรดที่ทำกำไร: {metrics.get('winning_trades', 0)}")
    print(f"จำนวนการเทรดที่ขาดทุน: {metrics.get('losing_trades', 0)}")
    
    # สร้างกราฟผลการทดสอบ
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            from src.utils.visualization import plot_backtest_results
            
            # สร้างกราฟ
            fig = plot_backtest_results(trades_df, data_manager.raw_data, metrics)
            
            # บันทึกกราฟ
            plot_path = os.path.join(result_dir, "backtest_results.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"บันทึกกราฟที่: {plot_path}")
            
            # แสดงกราฟ
            plt.show()
        
        except ImportError:
            logger.error("ไม่สามารถสร้างกราฟได้: ไม่พบโมดูล matplotlib")
    
    return result_dir, trades_df, metrics

def handle_analyze(args):
    """
    จัดการคำสั่งวิเคราะห์ผลการทดสอบ
    """
    # ตรวจสอบไฟล์
    if not os.path.exists(args.input):
        logger.error(f"ไม่พบไฟล์ผลการทดสอบ: {args.input}")
        return
    
    try:
        # โหลดข้อมูลผลการทดสอบ
        backtest_dir = args.input
        
        # ตรวจสอบว่าเป็นไฟล์หรือไดเรกทอรี
        if os.path.isfile(backtest_dir):
            # ถ้าเป็นไฟล์ ให้ใช้ไดเรกทอรีที่ไฟล์นั้นอยู่
            backtest_dir = os.path.dirname(backtest_dir)
        
        # ตรวจสอบไฟล์ที่จำเป็น
        trades_path = os.path.join(backtest_dir, "trades.csv")
        config_path = os.path.join(backtest_dir, "config.json")
        metrics_path = os.path.join(backtest_dir, "metrics.json")
        
        if not os.path.exists(trades_path):
            logger.error(f"ไม่พบไฟล์ข้อมูลการเทรด: {trades_path}")
            return
        
        if not os.path.exists(config_path):
            logger.error(f"ไม่พบไฟล์การตั้งค่า: {config_path}")
            return
        
        # โหลดข้อมูล
        trades_df = pd.read_csv(trades_path)
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # โหลดหรือคำนวณ metrics
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            # ค้นหาข้อมูลราคา
            data_path = None
            if "data" in config_data and "file_path" in config_data["data"]:
                data_path = config_data["data"]["file_path"]
            
            if data_path and os.path.exists(data_path):
                data_manager = MarketDataManager(file_path=data_path)
                if data_manager.data_loaded:
                    metrics = calculate_trading_metrics(trades_df, data_manager.raw_data)
                else:
                    logger.warning(f"ไม่สามารถโหลดข้อมูลราคาจาก {data_path} ได้")
                    metrics = calculate_trading_metrics(trades_df)
            else:
                logger.warning("ไม่พบข้อมูลราคา ใช้เฉพาะข้อมูลการเทรดในการคำนวณ metrics")
                metrics = calculate_trading_metrics(trades_df)
        
        # กำหนด metrics ที่ต้องการวิเคราะห์
        if args.metrics != "all":
            metrics_list = [m.strip() for m in args.metrics.split(',')]
            filtered_metrics = {k: v for k, v in metrics.items() if k in metrics_list}
        else:
            filtered_metrics = metrics
        
        # แสดงผลการวิเคราะห์
        print(f"\nผลการวิเคราะห์การทดสอบย้อนหลัง: {os.path.basename(backtest_dir)}")
        
        for metric_name, metric_value in filtered_metrics.items():
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")
        
        # เปรียบเทียบกับ benchmark
        if args.benchmark:
            benchmark_paths = [p.strip() for p in args.benchmark.split(',')]
            
            for benchmark_path in benchmark_paths:
                if not os.path.exists(benchmark_path):
                    logger.warning(f"ไม่พบไฟล์ benchmark: {benchmark_path}")
                    continue
                
                # โหลดข้อมูล benchmark
                if os.path.isfile(benchmark_path):
                    benchmark_dir = os.path.dirname(benchmark_path)
                else:
                    benchmark_dir = benchmark_path
                
                benchmark_trades_path = os.path.join(benchmark_dir, "trades.csv")
                benchmark_metrics_path = os.path.join(benchmark_dir, "metrics.json")
                
                if not os.path.exists(benchmark_trades_path):
                    logger.warning(f"ไม่พบไฟล์ข้อมูลการเทรดของ benchmark: {benchmark_trades_path}")
                    continue
                
                if os.path.exists(benchmark_metrics_path):
                    with open(benchmark_metrics_path, 'r') as f:
                        benchmark_metrics = json.load(f)
                else:
                    benchmark_trades_df = pd.read_csv(benchmark_trades_path)
                    benchmark_metrics = calculate_trading_metrics(benchmark_trades_df)
                
                # เปรียบเทียบ metrics
                print(f"\nเปรียบเทียบกับ benchmark: {os.path.basename(benchmark_dir)}")
                
                for metric_name in filtered_metrics.keys():
                    if metric_name in benchmark_metrics:
                        model_value = filtered_metrics[metric_name]
                        benchmark_value = benchmark_metrics[metric_name]
                        
                        if isinstance(model_value, float) and isinstance(benchmark_value, float):
                            diff = model_value - benchmark_value
                            diff_percent = diff / abs(benchmark_value) * 100 if benchmark_value != 0 else float('inf')
                            
                            print(f"  {metric_name}: {model_value:.4f} vs {benchmark_value:.4f} ({diff:.4f}, {diff_percent:.2f}%)")
                        else:
                            print(f"  {metric_name}: {model_value} vs {benchmark_value}")
        
        # สร้างกราฟ
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                from src.utils.visualization import plot_backtest_analysis
                
                # โหลดข้อมูลราคา
                data_manager = None
                if "data" in config_data and "file_path" in config_data["data"]:
                    data_path = config_data["data"]["file_path"]
                    
                    if os.path.exists(data_path):
                        data_manager = MarketDataManager(file_path=data_path)
                
                # สร้างกราฟ
                fig = plot_backtest_analysis(
                    trades_df, 
                    metrics, 
                    raw_data=data_manager.raw_data if data_manager and data_manager.data_loaded else None,
                    plot_type=args.plot,
                    period=args.period
                )
                
                # บันทึกกราฟ
                if args.output:
                    fig.savefig(args.output, dpi=300, bbox_inches='tight')
                    logger.info(f"บันทึกกราฟที่: {args.output}")
                
                # แสดงกราฟ
                plt.show()
            
            except ImportError:
                logger.error("ไม่สามารถสร้างกราฟได้: ไม่พบโมดูล matplotlib")
        
        # บันทึกผลการวิเคราะห์
        if args.output and not args.plot:
            # กำหนดนามสกุลไฟล์ตามความเหมาะสม
            if not (args.output.endswith('.json') or args.output.endswith('.csv')):
                output_path = args.output + '.json'
            else:
                output_path = args.output
            
            # สร้างโฟลเดอร์หากไม่มี
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # บันทึกไฟล์
            if output_path.endswith('.json'):
                with open(output_path, 'w') as f:
                    json.dump(filtered_metrics, f, indent=2)
            else:
                # แปลง dict เป็น DataFrame
                metrics_df = pd.DataFrame([filtered_metrics])
                metrics_df.to_csv(output_path, index=False)
            
            logger.info(f"บันทึกผลการวิเคราะห์ที่: {output_path}")
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ผลการทดสอบ: {e}")