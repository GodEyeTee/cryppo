import os
import logging
from datetime import datetime
import json
import pandas as pd

from src.data.managers.data_manager import MarketDataManager
from src.models.model_factory import ModelFactory
from src.utils.config_manager import get_config
from src.utils.metrics import PerformanceTracker

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

def prepare_training_data(input_path, config, validation_ratio, test_ratio):
    data_manager = MarketDataManager(
        file_path=input_path,
        window_size=config.get("data.window_size"),
        batch_size=config.get("model.batch_size")
    )
    
    if not data_manager.data_loaded:
        logger.error(f"ไม่สามารถโหลดข้อมูลจาก {input_path} ได้")
        return None

    training_data = data_manager.create_training_data(
        validation_ratio=validation_ratio,
        test_ratio=test_ratio
    )
    
    logger.info(f"ข้อมูลมีคอลัมน์: {data_manager.data.columns.tolist()}")
    logger.info(f"รูปร่างของข้อมูล: {data_manager.data.shape}")
    
    return data_manager, training_data

def train_and_save_model(model, training_data, config, model_dir):
    history = model.train(
        train_loader=training_data["train_loader"],
        val_loader=training_data["val_loader"],
        epochs=config.get("training.total_timesteps"),
        log_dir=model_dir if config.get("training.use_tensorboard", False) else None
    )
    
    history_path = os.path.join(model_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    model_path = os.path.join(model_dir, "model.pt")
    model.save(model_path)
    
    logger.info(f"บันทึกโมเดลที่: {model_path}")
    
    return history

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

    data_result = prepare_training_data(args.input, config, args.validation_ratio, args.test_ratio)
    if data_result is None:
        return
    
    data_manager, training_data = data_result
    
    input_size = training_data["feature_size"]
    model = ModelFactory.create_model(
        model_type=config.get("model.model_type"),
        input_size=input_size,
        config=config
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config.get('model.model_type')}_{timestamp}"
    model_dir = os.path.join(args.output, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    history = train_and_save_model(model, training_data, config, model_dir)
    
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    stats_path = os.path.join(model_dir, "data_stats.json")
    data_manager.save_stats(stats_path)
    
    if "test_loader" in training_data:
        try:
            metrics = model.evaluate(training_data["test_loader"])
            
            print("\nผลการประเมินโมเดลกับชุดข้อมูล test:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value}")
            
            metrics_path = os.path.join(model_dir, "test_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการประเมินโมเดล: {e}")

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
            with open(config_path, 'r') as f:
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
        
        data = data_manager.create_training_data(
            validation_ratio=0,
            test_ratio=0
        )
        
        model_type = config.get("model.model_type")
        model = ModelFactory.create_model(
            model_type=model_type,
            input_size=data["feature_size"],
            config=config
        )
        
        model.load(args.model)
        
        metrics_list = None
        if args.metrics != "all":
            metrics_list = [m.strip() for m in args.metrics.split(',')]
        
        metrics = model.evaluate(data["train_loader"], metrics_list)
        
        print(f"\nผลการประเมินโมเดล {os.path.basename(args.model)} กับชุดข้อมูล {os.path.basename(args.input)}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")
        
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"บันทึกผลการประเมินที่: {args.output}")
        
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                from src.utils.visualization import plot_evaluation_results
                
                plt.figure(figsize=(12, 8))
                plot_evaluation_results(model, data_manager.raw_data, metrics)
                plt.tight_layout()
                plt.show()
                
                if args.output:
                    plot_path = os.path.splitext(args.output)[0] + ".png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    logger.info(f"บันทึกกราฟที่: {plot_path}")
            
            except ImportError:
                logger.error("ไม่สามารถสร้างกราฟได้: ไม่พบโมดูล matplotlib")
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการประเมินโมเดล: {e}")
