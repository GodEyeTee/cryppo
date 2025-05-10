import os
import logging
from datetime import datetime
import json

from src.data.managers.data_manager import MarketDataManager
from src.models.model_factory import ModelFactory
from src.utils.config_manager import get_config
from src.utils.metrics import PerformanceTracker

logger = logging.getLogger('cli.train')

def setup_model_parser(parser):
    """
    ตั้งค่า parser สำหรับคำสั่งเทรนโมเดล
    """
    parser.add_argument("--input", type=str, required=True, help="ไฟล์ข้อมูลที่ใช้เทรน")
    parser.add_argument("--output", type=str, required=True, help="ไดเรกทอรีสำหรับบันทึกโมเดล")
    parser.add_argument("--model-type", type=str, default="double_dqn", 
                      choices=["dqn", "double_dqn", "dueling_dqn", "per_dqn"],
                      help="ประเภทของโมเดล")
    parser.add_argument("--window-size", type=int, default=None, help="ขนาดของหน้าต่างข้อมูล")
    parser.add_argument("--batch-size", type=int, default=None, help="ขนาดของแต่ละ batch")
    parser.add_argument("--epochs", type=int, default=None, help="จำนวนรอบการเทรน")
    parser.add_argument("--learning-rate", type=float, default=None, help="อัตราการเรียนรู้")
    parser.add_argument("--discount-factor", type=float, default=None, help="discount factor")
    parser.add_argument("--target-update", type=int, default=None, 
                      help="ความถี่ในการอัพเดต target network")
    parser.add_argument("--validation-ratio", type=float, default=0.1, 
                      help="สัดส่วนข้อมูลสำหรับ validation")
    parser.add_argument("--test-ratio", type=float, default=0.1, 
                      help="สัดส่วนข้อมูลสำหรับ test")
    parser.add_argument("--use-gpu", action="store_true", default=None, help="ใช้ GPU ในการเทรน")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="ไม่ใช้ GPU ในการเทรน")
    parser.add_argument("--tensorboard", action="store_true", help="บันทึก log สำหรับ TensorBoard")
    parser.add_argument("--seed", type=int, default=None, help="เลข seed สำหรับความสามารถในการทำซ้ำ")
    parser.add_argument("--config", type=str, default=None, 
                      help="ไฟล์การตั้งค่าเฉพาะสำหรับการเทรน")

def setup_evaluate_parser(parser):
    """
    ตั้งค่า parser สำหรับคำสั่งประเมินโมเดล
    """
    parser.add_argument("--model", type=str, required=True, help="ไฟล์โมเดลที่ต้องการประเมิน")
    parser.add_argument("--input", type=str, required=True, help="ไฟล์ข้อมูลที่ใช้ประเมิน")
    parser.add_argument("--output", type=str, default=None, help="ไฟล์สำหรับบันทึกผลการประเมิน")
    parser.add_argument("--batch-size", type=int, default=None, help="ขนาดของแต่ละ batch")
    parser.add_argument("--window-size", type=int, default=None, help="ขนาดของหน้าต่างข้อมูล")
    parser.add_argument("--metrics", type=str, default="all", 
                      help="รายการ metrics ที่ต้องการประเมิน (คั่นด้วยเครื่องหมายจุลภาค)")
    parser.add_argument("--plot", action="store_true", help="แสดงกราฟผลการประเมิน")
    parser.add_argument("--use-gpu", action="store_true", default=None, help="ใช้ GPU ในการประเมิน")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="ไม่ใช้ GPU ในการประเมิน")

def handle_model(args):
    """
    จัดการคำสั่งเทรนโมเดล
    """
    # โหลดการตั้งค่า
    config = get_config()
    
    # ถ้ามีไฟล์การตั้งค่าเฉพาะ ให้โหลดเพิ่มเติม
    if args.config and os.path.exists(args.config):
        config.load_config(args.config)
    
    # ปรับการตั้งค่าตามอาร์กิวเมนต์
    model_config = config.extract_subconfig("model")
    training_config = config.extract_subconfig("training")
    
    if args.model_type:
        model_config["model_type"] = args.model_type
    
    if args.window_size:
        config.set("data.window_size", args.window_size)
    
    if args.batch_size:
        model_config["batch_size"] = args.batch_size
    
    if args.learning_rate:
        model_config["learning_rate"] = args.learning_rate
    
    if args.discount_factor:
        model_config["discount_factor"] = args.discount_factor
    
    if args.target_update:
        model_config["target_update_frequency"] = args.target_update
    
    if args.epochs:
        training_config["total_timesteps"] = args.epochs
    
    if args.use_gpu is not None:
        config.set("cuda.use_cuda", args.use_gpu)
    
    if args.tensorboard:
        training_config["use_tensorboard"] = True
    
    if args.seed:
        config.set("general.random_seed", args.seed)
    
    # ตรวจสอบไฟล์นำเข้า
    if not os.path.exists(args.input):
        logger.error(f"ไม่พบไฟล์: {args.input}")
        return
    
    # สร้างโฟลเดอร์สำหรับบันทึกโมเดล
    os.makedirs(args.output, exist_ok=True)
    
    # โหลดข้อมูล
    data_manager = MarketDataManager(
        file_path=args.input,
        window_size=config.get("data.window_size"),
        batch_size=model_config.get("batch_size")
    )
    
    if not data_manager.data_loaded:
        logger.error(f"ไม่สามารถโหลดข้อมูลจาก {args.input} ได้")
        return
    
    # แบ่งข้อมูลสำหรับการเทรน
    training_data = data_manager.create_training_data(
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio
    )
    
    # สร้างโมเดล
    model = ModelFactory.create_model(
        model_type=model_config.get("model_type"),
        input_size=data_manager.data.shape[1] if 'timestamp' not in data_manager.data.columns else data_manager.data.shape[1] - 1,
        config=config
    )
    
    # ตั้งค่าทิศทางการบันทึก
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{model_config.get('model_type')}_{timestamp}"
    model_dir = os.path.join(args.output, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # บันทึกการตั้งค่า
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # บันทึกสถิติข้อมูล
    stats_path = os.path.join(model_dir, "data_stats.json")
    data_manager.save_stats(stats_path)
    
    # เทรนโมเดล
    history = model.train(
        train_loader=training_data["train_loader"],
        val_loader=training_data["val_loader"],
        epochs=training_config.get("total_timesteps"),
        log_dir=model_dir if training_config.get("use_tensorboard", False) else None
    )
    
    # บันทึกประวัติการเทรน
    history_path = os.path.join(model_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # บันทึกโมเดล
    model_path = os.path.join(model_dir, "model.pt")
    model.save(model_path)
    
    logger.info(f"บันทึกโมเดลที่: {model_path}")
    
    # ประเมินโมเดลกับชุดข้อมูล test
    metrics = model.evaluate(training_data["test_loader"])
    
    print("\nผลการประเมินโมเดลกับชุดข้อมูล test:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")
    
    # บันทึกผลการประเมิน
    metrics_path = os.path.join(model_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def handle_evaluate(args):
    """
    จัดการคำสั่งประเมินโมเดล
    """
    # โหลดการตั้งค่า
    config = get_config()
    
    # ตรวจสอบไฟล์
    if not os.path.exists(args.model):
        logger.error(f"ไม่พบไฟล์โมเดล: {args.model}")
        return
    
    if not os.path.exists(args.input):
        logger.error(f"ไม่พบไฟล์ข้อมูล: {args.input}")
        return
    
    # โหลดโมเดล
    try:
        model_dir = os.path.dirname(args.model)
        config_path = os.path.join(model_dir, "config.json")
        
        # ถ้ามีไฟล์การตั้งค่าของโมเดล ให้โหลดก่อน
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
                config.update_from_dict(model_config)
        
        # ปรับการตั้งค่าตามอาร์กิวเมนต์
        if args.batch_size:
            config.set("model.batch_size", args.batch_size)
        
        if args.window_size:
            config.set("data.window_size", args.window_size)
        
        if args.use_gpu is not None:
            config.set("cuda.use_cuda", args.use_gpu)
        
        # โหลดข้อมูล
        data_manager = MarketDataManager(
            file_path=args.input,
            window_size=config.get("data.window_size"),
            batch_size=config.get("model.batch_size")
        )
        
        if not data_manager.data_loaded:
            logger.error(f"ไม่สามารถโหลดข้อมูลจาก {args.input} ได้")
            return
        
        # โหลดสถิติ
        stats_path = os.path.join(model_dir, "data_stats.json")
        if os.path.exists(stats_path):
            data_manager.load_stats(stats_path)
        
        # แบ่งข้อมูลสำหรับการประเมิน
        data = data_manager.create_training_data(
            validation_ratio=0,
            test_ratio=0
        )
        
        # โหลดโมเดล
        model_type = config.get("model.model_type")
        model = ModelFactory.create_model(
            model_type=model_type,
            input_size=data["feature_size"],
            config=config
        )
        
        model.load(args.model)
        
        # กำหนด metrics ที่ต้องการประเมิน
        if args.metrics != "all":
            metrics_list = [m.strip() for m in args.metrics.split(',')]
        else:
            metrics_list = None
        
        # ประเมินโมเดล
        metrics = model.evaluate(data["train_loader"], metrics_list)
        
        print(f"\nผลการประเมินโมเดล {os.path.basename(args.model)} กับชุดข้อมูล {os.path.basename(args.input)}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")
        
        # บันทึกผลการประเมิน
        if args.output:
            # สร้างโฟลเดอร์หากไม่มี
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"บันทึกผลการประเมินที่: {args.output}")
        
        # แสดงกราฟผลการประเมิน
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                from src.utils.visualization import plot_evaluation_results
                
                plt.figure(figsize=(12, 8))
                plot_evaluation_results(model, data_manager.raw_data, metrics)
                plt.tight_layout()
                plt.show()
                
                # บันทึกกราฟถ้ามีการระบุ output
                if args.output:
                    plot_path = os.path.splitext(args.output)[0] + ".png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    logger.info(f"บันทึกกราฟที่: {plot_path}")
            
            except ImportError:
                logger.error("ไม่สามารถสร้างกราฟได้: ไม่พบโมดูล matplotlib")
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการประเมินโมเดล: {e}")