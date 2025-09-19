import os
import logging
from datetime import datetime
import json
import random
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from src.data.managers.data_manager import MarketDataManager
from src.models.model_factory import ModelFactory
from src.environment.trading_env import TradingEnv
from src.utils.config_manager import get_config

logger = logging.getLogger('cli.train')


def update_config_from_args(config, args, param_mapping):
    for arg_name, config_path in param_mapping.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            config.set(config_path, arg_value)


def setup_model_parser(parser):
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="double_dqn",
                      choices=["dqn", "double_dqn", "dueling_dqn", "per_dqn"])
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--discount-factor", type=float, default=None)
    parser.add_argument("--target-update", type=int, default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--use-gpu", action="store_true", default=None)
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)


def setup_evaluate_parser(parser):
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--metrics", type=str, default="all")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--use-gpu", action="store_true", default=None)
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")


def _numeric_feature_count(data_manager: MarketDataManager) -> int:
    if data_manager.data is None or data_manager.data.empty:
        raise ValueError("ไม่พบข้อมูลที่โหลดมา")

    numeric_columns = data_manager.data.select_dtypes(include=['number']).columns.tolist()
    if 'timestamp' in numeric_columns and not pd.api.types.is_numeric_dtype(data_manager.data['timestamp']):
        numeric_columns.remove('timestamp')
    if not numeric_columns:
        raise ValueError("ไม่มีฟีเจอร์เชิงตัวเลขหลังจากคัดกรอง")
    return len(numeric_columns)


def _build_environment(config, input_path: str) -> TradingEnv:
    backtest_cfg = config.extract_subconfig("backtest")
    env = TradingEnv(
        file_path=input_path,
        window_size=config.get("data.window_size"),
        initial_balance=backtest_cfg.get("initial_balance", 10000.0),
        transaction_fee=backtest_cfg.get("fee_rate", 0.0025),
        use_position_info=False,
        config=config
    )

    leverage = float(backtest_cfg.get("leverage", 3.0))
    if hasattr(env, 'simulator'):
        env.simulator.leverage = leverage

    return env




def _coerce_action(action: int, env: TradingEnv) -> int:
    short_id = TradingEnv.ACTIONS.get('SHORT') if hasattr(TradingEnv, 'ACTIONS') else 2
    none_id = TradingEnv.ACTIONS.get('NONE') if hasattr(TradingEnv, 'ACTIONS') else 0

    leverage = getattr(getattr(env, 'simulator', None), 'leverage', 1.0)
    if leverage <= 1.0 and short_id is not None and action == short_id:
        return none_id
    return action

def train_with_environment(model, env: TradingEnv, config, model_dir: str) -> Dict[str, Any]:
    total_timesteps = int(config.get("training.total_timesteps", 200000))
    max_episode_steps = int(config.get("training.max_episode_steps", 1000) or 1000)
    seed = config.get("general.random_seed")
    rng = random.Random(seed)

    timesteps = 0
    episode = 0
    history = []

    os.makedirs(model_dir, exist_ok=True)

    # Data manager helper values for sampling different starting points
    data_len = len(env.data_manager.data) if env.data_manager and env.data_manager.data is not None else 0
    if data_len:
        total_timesteps = min(total_timesteps, max(data_len, 50000))
    max_start = max(0, data_len - env.window_size - 2)

    while timesteps < total_timesteps:
        options = {}
        if max_start > 0:
            options['start_index'] = rng.randint(0, max_start)
        state, _ = env.reset(options=options)

        episode_reward = 0.0
        steps = 0
        losses = []
        done = False

        while not done and timesteps < total_timesteps:
            action = model.select_action(state)
            action = _coerce_action(action, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            model.store_experience(state, action, reward, next_state, done)
            loss = model.update()
            if loss is not None:
                losses.append(loss)

            state = next_state
            episode_reward += reward
            steps += 1
            timesteps += 1

            if steps >= max_episode_steps:
                break

        avg_loss = float(np.mean(losses)) if losses else None
        history.append({
            "episode": episode + 1,
            "steps": steps,
            "timesteps": timesteps,
            "reward": episode_reward,
            "avg_loss": avg_loss,
            "epsilon": model.epsilon
        })

        episode += 1

    history_path = os.path.join(model_dir, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return {
        "episodes": episode,
        "timesteps": timesteps,
        "history": history
    }


def evaluate_with_environment(model, env: TradingEnv) -> Dict[str, Any]:
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    action_counts = {}
    portfolio_values = []

    model.policy_net.eval()

    while not done:
        action = model.select_action(state, evaluation=True)
        action = _coerce_action(action, env)
        action_counts[action] = action_counts.get(action, 0) + 1

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1
        state = next_state

        equity = info.get('equity') or env.simulator.get_equity(info.get('price', 0))
        portfolio_values.append(float(equity))

    model.policy_net.train()

    total_actions = sum(action_counts.values()) or 1
    action_distribution = {
        str(action): count / total_actions for action, count in sorted(action_counts.items())
    }

    return {
        "total_reward": total_reward,
        "steps": steps,
        "action_distribution": action_distribution,
        "portfolio": portfolio_values
    }


def handle_model(args):
    config = get_config()

    if args.config and os.path.exists(args.config):
        config.load_config(args.config)

    param_mapping = {
        'model_type': 'model.model_type',
        'window_size': 'data.window_size',
        'batch_size': 'model.batch_size',
        'learning_rate': 'model.learning_rate',
        'discount_factor': 'model.discount_factor',
        'target_update': 'model.target_update_frequency',
        'epochs': 'training.total_timesteps',
        'use_gpu': 'cuda.use_cuda',
        'tensorboard': 'training.use_tensorboard',
        'seed': 'general.random_seed'
    }

    update_config_from_args(config, args, param_mapping)

    if not os.path.exists(args.input):
        logger.error(f"ไม่พบไฟล์: {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)

    data_manager = MarketDataManager(
        file_path=args.input,
        window_size=config.get("data.window_size"),
        batch_size=config.get("model.batch_size")
    )

    if not data_manager.data_loaded:
        logger.error(f"ไม่สามารถโหลดข้อมูลจาก {args.input} ได้")
        return

    try:
        feature_size = _numeric_feature_count(data_manager)
    except ValueError as exc:
        logger.error(str(exc))
        return

    logger.info(f"ข้อมูลมีคอลัมน์: {data_manager.data.columns.tolist()}")
    logger.info(f"รูปร่างของข้อมูล: {data_manager.data.shape}")

    model = ModelFactory.create_model(
        model_type=config.get("model.model_type"),
        input_size=feature_size,
        config=config
    )

    env = _build_environment(config, args.input)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config.get('model.model_type')}_{timestamp}"
    model_dir = os.path.join(args.output, model_name)
    os.makedirs(model_dir, exist_ok=True)

    train_summary = train_with_environment(model, env, config, model_dir)
    model.is_trained = True
    logger.info(f"การเทรนเสร็จสิ้น: episodes={train_summary["episodes"]}, timesteps={train_summary["timesteps"]}")

    model_path = os.path.join(model_dir, "model.pt")
    model.save(model_path)
    logger.info(f"บันทึกโมเดลที่: {model_path}")

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)

    stats_path = os.path.join(model_dir, "data_stats.json")
    data_manager.save_stats(stats_path)

    try:
        eval_env = _build_environment(config, args.input)
        metrics = evaluate_with_environment(model, eval_env)

        print()
        print("ผลการประเมินโมเดลกับชุดข้อมูล test:")
        print(f"  total_reward: {metrics['total_reward']:.6f}")
        print(f"  steps: {metrics['steps']}")
        print(f"  action_distribution: {metrics['action_distribution']}")

        metrics_path = os.path.join(model_dir, "test_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"เกิดข้อผิดพลาดในการประเมินโมเดล: {exc}")


def handle_evaluate(args):
    config = get_config()

    if not os.path.exists(args.model):
        logger.error(f"ไม่พบไฟล์โมเดล: {args.model}")
        return

    if not os.path.exists(args.input):
        logger.error(f"ไม่พบไฟล์ข้อมูล: {args.input}")
        return

    try:
        model_dir = os.path.dirname(args.model)
        config_path = os.path.join(model_dir, "config.json")

        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
                config.update_from_dict(model_config)

        if args.batch_size:
            config.set("model.batch_size", args.batch_size)

        if args.window_size:
            config.set("data.window_size", args.window_size)

        if args.use_gpu is not None:
            config.set("cuda.use_cuda", args.use_gpu)

        data_manager = MarketDataManager(
            file_path=args.input,
            window_size=config.get("data.window_size"),
            batch_size=config.get("model.batch_size")
        )

        if not data_manager.data_loaded:
            logger.error(f"ไม่สามารถโหลดข้อมูลจาก {args.input} ได้")
            return

        stats_path = os.path.join(model_dir, "data_stats.json")
        if os.path.exists(stats_path):
            data_manager.load_stats(stats_path)

        feature_size = _numeric_feature_count(data_manager)

        model = ModelFactory.create_model(
            model_type=config.get("model.model_type"),
            input_size=feature_size,
            config=config
        )
        model.load(args.model)

        env = _build_environment(config, args.input)
        metrics = evaluate_with_environment(model, env)

        print()
        print(f"ผลการประเมินโมเดล {os.path.basename(args.model)} กับชุดข้อมูล {os.path.basename(args.input)}:")
        for metric_name, metric_value in metrics.items():
            if metric_name == 'portfolio':
                continue
            print(f"  {metric_name}: {metric_value}")

        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            logger.info(f"บันทึกผลการประเมินที่: {args.output}")

    except Exception as exc:
        logger.error(f"เกิดข้อผิดพลาดในการประเมินโมเดล: {exc}")
