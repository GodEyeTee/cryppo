import os
import torch
import logging
from datetime import datetime
import json
import pandas as pd

from src.data.managers.data_manager import MarketDataManager
from src.environment.trading_env import TradingEnv
from src.models.model_factory import ModelFactory
from src.utils.config_manager import get_config
from src.utils.metrics import PerformanceTracker
import matplotlib.pyplot as plt
from src.utils.visualization import plot_backtest_results

logger = logging.getLogger('cli.backtest')

def setup_run_parser(parser):
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--initial-balance", type=float, default=None)
    parser.add_argument("--leverage", type=float, default=None)
    parser.add_argument("--fee-rate", type=float, default=None)
    parser.add_argument("--stop-loss", type=float, default=None)
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--use-gpu", action="store_true", default=None)
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config", type=str, default=None)

def setup_analyze_parser(parser):
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--metrics", type=str, default="all")
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--plot", choices=["trades", "equity", "returns", "drawdown", "all"], default="all")
    parser.add_argument("--period", choices=["daily", "weekly", "monthly"], default="daily")

def calculate_trading_metrics(trades_df, raw_data=None, risk_free_rate=0.0, periods_per_year=252):
    if trades_df.empty:
        logger.warning("ไม่มีข้อมูลการเทรดสำหรับการคำนวณเมตริก")
        return {}
    
    metrics = {}
    initial_equity = trades_df['portfolio_value'].iloc[0] if 'portfolio_value' in trades_df.columns else 10000.0
    
    tracker = PerformanceTracker(initial_equity=initial_equity, periods_per_year=periods_per_year)
    for i in range(1, len(trades_df)):
        equity = trades_df['portfolio_value'].iloc[i]
        position = trades_df['position'].iloc[i] if 'position' in trades_df.columns else 0
        timestamp = None
        if 'timestamp' in trades_df.columns:
            timestamp = pd.to_datetime(trades_df['timestamp'].iloc[i])
        tracker.update(equity=equity, timestamp=timestamp, position=position)
    
    metrics.update(tracker.calculate_metrics(risk_free_rate=risk_free_rate))

    if raw_data is not None and 'timestamp' in trades_df.columns:
        first_date = pd.to_datetime(trades_df['timestamp'].iloc[0]).date()
        last_date = pd.to_datetime(trades_df['timestamp'].iloc[-1]).date()
        days_diff = (last_date - first_date).days
        metrics['trading_days'] = days_diff if days_diff > 0 else 1
        
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
    
    if 'action' in trades_df.columns and 'position' in trades_df.columns:
        positions = trades_df['position'].values
        
        position_changes = [i for i in range(1, len(positions)) if positions[i] != positions[i-1]]
        
        trade_count = 0
        winning_trades = 0
        losing_trades = 0
        
        for i in range(0, len(position_changes) - 1, 2):
            if i + 1 < len(position_changes):
                start_idx = position_changes[i]
                end_idx = position_changes[i + 1]

                start_value = trades_df['portfolio_value'].iloc[start_idx]
                end_value = trades_df['portfolio_value'].iloc[end_idx]

                if start_value <= 1e-9:
                    continue

                trade_count += 1
                trade_return = (end_value / start_value) - 1

                if trade_return > 0:
                    winning_trades += 1
                elif trade_return < 0:
                    losing_trades += 1

        metrics['total_trades'] = trade_count
        metrics['trades_count'] = trade_count
        metrics['winning_trades'] = winning_trades
        metrics['losing_trades'] = losing_trades

        if trade_count > 0:
            metrics['win_rate'] = winning_trades / trade_count
    
    if 'win_rate' not in metrics:
        metrics['win_rate'] = 0.0

    return metrics

def handle_run(args):
    config = get_config()

    if args.config and os.path.exists(args.config):
        config.load_config(args.config)
    
    if not os.path.exists(args.model):
        logger.error(f"ไม่พบไฟล์โมเดล: {args.model}")
        return
    
    if not os.path.exists(args.input):
        logger.error(f"ไม่พบไฟล์ข้อมูล: {args.input}")
        return

    model_dir = os.path.dirname(args.model)
    config_path = os.path.join(model_dir, "config.json")

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
            config.update_from_dict(model_config)
    
    backtest_config = config.extract_subconfig("backtest")
    
    if args.window_size:
        config.set("data.window_size", args.window_size)
    
    if args.batch_size:
        config.set("model.batch_size", args.batch_size)
    
    if args.start_date:
        backtest_config["start_date"] = args.start_date
        config.set('environment.start_date', args.start_date)
    
    if args.end_date:
        backtest_config["end_date"] = args.end_date
        config.set('environment.end_date', args.end_date)
    
    if args.initial_balance:
        backtest_config["initial_balance"] = args.initial_balance
        config.set('environment.initial_balance', args.initial_balance)
    
    if args.leverage:
        backtest_config["leverage"] = args.leverage
        config.set('environment.leverage', args.leverage)
    
    if args.fee_rate:
        backtest_config["fee_rate"] = args.fee_rate
        config.set('environment.fee_rate', args.fee_rate)
    
    if args.stop_loss:
        backtest_config["stop_loss"] = args.stop_loss / 100.0
        config.set('environment.stop_loss', args.stop_loss / 100.0)
    
    if args.take_profit:
        backtest_config["take_profit"] = args.take_profit / 100.0
        config.set('environment.take_profit', args.take_profit / 100.0)
    
    if args.use_gpu is not None:
        config.set("cuda.use_cuda", args.use_gpu)
    
    backtest_config["plot_results"] = args.plot
    backtest_config["verbose"] = args.verbose

    os.makedirs(args.output, exist_ok=True)
    
    # Initialize data manager
    data_manager = MarketDataManager(
        file_path=args.input,
        window_size=config.get("data.window_size"),
        batch_size=config.get("model.batch_size")
    )
    
    if not data_manager.data_loaded:
        logger.error(f"ไม่สามารถโหลดข้อมูลจาก {args.input} ได้")
        return

    if backtest_config.get("start_date") or backtest_config.get("end_date"):
        data_manager.filter_data(
            start_date=backtest_config.get("start_date"),
            end_date=backtest_config.get("end_date")
        )

    stats_path = os.path.join(model_dir, "data_stats.json")
    if os.path.exists(stats_path):
        data_manager.load_stats(stats_path)

    # Initialize environment with file path (TradingEnv expects file_path, not data_manager)
    env = TradingEnv(
        file_path=args.input,
        window_size=config.get("data.window_size"),
        initial_balance=backtest_config.get("initial_balance", 10000.0),
        transaction_fee=backtest_config.get("fee_rate", 0.0025),
        use_position_info=False,
        config=config
    )

    if args.leverage and hasattr(env, 'simulator'):
        env.simulator.leverage = float(args.leverage)

    model_type = config.get("model.model_type")

    input_size = None
    action_dim = None
    model_config_path = os.path.join(model_dir, "model_config.json")
    if os.path.exists(model_config_path):
        try:
            with open(model_config_path, 'r', encoding='utf-8') as f:
                stored_model_cfg = json.load(f)
            input_size = stored_model_cfg.get('state_dim')
            action_dim = stored_model_cfg.get('action_dim')
        except Exception as e:
            logger.warning(f"ไม่สามารถโหลดการตั้งค่าโมเดลจาก {model_config_path}: {e}")

    if input_size is None:
        try:
            checkpoint = torch.load(args.model, map_location='cpu')
            if isinstance(checkpoint, dict):
                for key in ("state_dim", "input_size"):
                    if key in checkpoint and checkpoint[key]:
                        input_size = int(checkpoint[key])
                        break
                if input_size is None:
                    for weight_key in ("network.0.weight", "0.weight"):
                        if weight_key in checkpoint.get('policy_net', {}):
                            input_size = checkpoint['policy_net'][weight_key].shape[1]
                            break
        except Exception as e:
            logger.warning(f"ไม่สามารถอ่านขนาดอินพุตจากโมเดล: {e}")

    if input_size is None:
        numeric_columns = data_manager.data.select_dtypes(include=['number']).columns.tolist()
        if 'timestamp' in numeric_columns and not pd.api.types.is_numeric_dtype(data_manager.data['timestamp']):
            numeric_columns.remove('timestamp')
        input_size = len(numeric_columns)
        logger.warning(f"ใช้จำนวนฟีเจอร์จากข้อมูล ({input_size}) เป็นขนาดอินพุตเนื่องจากไม่พบในโมเดล")

    if action_dim is not None:
        config.set('environment.action_dim', action_dim)

    model = ModelFactory.create_model(
        model_type=model_type,
        input_size=input_size,
        config=config
    )

    model.load(args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.output, f"backtest_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    logger.info("กำลังทดสอบย้อนหลัง...")
    
    observation, _ = env.reset()
    done = False
    total_reward = 0
    actions = []
    rewards = []
    dones = []
    positions = []
    portfolio_values = []
    timestamps = []
    
    while not done:
        # Greedy action from policy network (use last timestep)
        with torch.no_grad():
            st = torch.tensor(observation, dtype=torch.float32, device=model.device)
            q = model.policy_net(st)
            if q.dim() == 3:
                q = q[-1]
            if q.dim() == 2:
                q = q[-1]
            action = int(torch.argmax(q).item())
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)
        
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        positions.append(info.get('position', 0))
        portfolio_values.append(info.get('equity', backtest_config.get("initial_balance")))
        timestamps.append(None)
        
        observation = next_observation
        total_reward += reward
        
        if args.verbose and (len(rewards) % 100 == 0):
            logger.info(f"Step: {len(rewards)}, Reward: {reward:.4f}, Total: {total_reward:.4f}, Portfolio: {portfolio_values[-1]:.2f}")
    
    logger.info(f"ทดสอบย้อนหลังเสร็จสิ้น! Total Reward: {total_reward:.4f}, Final Portfolio: {portfolio_values[-1]:.2f}")

    trades_df = pd.DataFrame({
        'action': actions,
        'reward': rewards,
        'position': positions,
        'portfolio_value': portfolio_values,
        'done': dones
    })
    
    trades_path = os.path.join(result_dir, "trades.csv")
    trades_df.to_csv(trades_path, index=False)
    
    config_path = os.path.join(result_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    metrics = calculate_trading_metrics(trades_df, data_manager.raw_data)
    
    # Save metrics
    metrics_path = os.path.join(result_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    total_return_pct = metrics.get('total_return', 0) * 100
    max_drawdown_pct = metrics.get('max_drawdown', 0) * 100
    win_rate_pct = metrics.get('win_rate', 0) * 100

    print("\nสรุปผลการทดสอบย้อนหลัง:")
    print(f"จำนวนวันทดสอบ: {metrics.get('trading_days', 'N/A')} วัน")
    print(f"เงินทุนเริ่มต้น: {backtest_config.get('initial_balance', trades_df['portfolio_value'].iloc[0]):.2f}")
    print(f"มูลค่าพอร์ตสุดท้าย: {portfolio_values[-1]:.2f}")
    print(f"กำไร/ขาดทุนรวม: {portfolio_values[-1] - trades_df['portfolio_value'].iloc[0]:.2f} ({total_return_pct:.2f}%)")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"Max Drawdown: {max_drawdown_pct:.2f}%")
    print(f"Win Rate: {win_rate_pct:.2f}%")
    print(f"จำนวนการเทรดทั้งหมด: {metrics.get('total_trades', 0)}")
    print(f"จำนวนการเทรดที่ทำกำไร: {metrics.get('winning_trades', 0)}")
    print(f"จำนวนการเทรดที่ขาดทุน: {metrics.get('losing_trades', 0)}")

    # Plot if requested
    if args.plot:
        try:
            fig = plot_backtest_results(trades_df, data_manager.raw_data, metrics)
            plot_path = os.path.join(result_dir, "backtest_results.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"บันทึกกราฟที่: {plot_path}")
            plt.show()
        except ImportError:
            logger.error("ไม่สามารถสร้างกราฟได้: ไม่พบโมดูล matplotlib")
    
    return result_dir, trades_df, metrics

def handle_analyze(args):
    if not os.path.exists(args.input):
        logger.error(f"ไม่พบไฟล์ผลการทดสอบ: {args.input}")
        return
    
    try:
        backtest_dir = args.input
        if os.path.isfile(backtest_dir):
            backtest_dir = os.path.dirname(backtest_dir)
        
        trades_path = os.path.join(backtest_dir, "trades.csv")
        config_path = os.path.join(backtest_dir, "config.json")
        metrics_path = os.path.join(backtest_dir, "metrics.json")
        
        if not os.path.exists(trades_path):
            logger.error(f"ไม่พบไฟล์ข้อมูลการเทรด: {trades_path}")
            return
        
        if not os.path.exists(config_path):
            logger.error(f"ไม่พบไฟล์การตั้งค่า: {config_path}")
            return

        trades_df = pd.read_csv(trades_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Get metrics
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        else:
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
        
        # Filter metrics if specified
        if args.metrics != "all":
            metrics_list = [m.strip() for m in args.metrics.split(',')]
            filtered_metrics = {k: v for k, v in metrics.items() if k in metrics_list}
        else:
            filtered_metrics = metrics
        
        # Print metrics
        print(f"\nผลการวิเคราะห์การทดสอบย้อนหลัง: {os.path.basename(backtest_dir)}")
        for metric_name, metric_value in filtered_metrics.items():
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")
        
        # Handle benchmark comparison
        if args.benchmark:
            benchmark_paths = [p.strip() for p in args.benchmark.split(',')]
            
            for benchmark_path in benchmark_paths:
                if not os.path.exists(benchmark_path):
                    logger.warning(f"ไม่พบไฟล์ benchmark: {benchmark_path}")
                    continue
                
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

        # Handle plotting
        if args.plot:
            try:
                data_manager = None
                if "data" in config_data and "file_path" in config_data["data"]:
                    data_path = config_data["data"]["file_path"]
                    
                    if os.path.exists(data_path):
                        data_manager = MarketDataManager(file_path=data_path)
                
                from src.utils.visualization import plot_backtest_analysis
                fig = plot_backtest_analysis(
                    trades_df, 
                    metrics, 
                    raw_data=data_manager.raw_data if data_manager and data_manager.data_loaded else None,
                    plot_type=args.plot,
                    period=args.period
                )
                
                if args.output:
                    fig.savefig(args.output, dpi=300, bbox_inches='tight')
                    logger.info(f"บันทึกกราฟที่: {args.output}")
                plt.show()
            
            except ImportError:
                logger.error("ไม่สามารถสร้างกราฟได้: ไม่พบโมดูล matplotlib")
        
        # Save results if requested
        if args.output and not args.plot:
            if not (args.output.endswith('.json') or args.output.endswith('.csv')):
                output_path = args.output + '.json'
            else:
                output_path = args.output
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if output_path.endswith('.json'):
                with open(output_path, 'w') as f:
                    json.dump(filtered_metrics, f, indent=2)
            else:
                metrics_df = pd.DataFrame([filtered_metrics])
                metrics_df.to_csv(output_path, index=False)
            
            logger.info(f"บันทึกผลการวิเคราะห์ที่: {output_path}")
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ผลการทดสอบ: {e}")
