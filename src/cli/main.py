import argparse
import logging
from pathlib import Path

from src.utils.loggers import setup_logger
from src.cli.commands import data_commands, train_commands, backtest_commands
from src.utils.config_manager import get_config, set_cuda_env

# ตั้งค่า logger
logger = setup_logger('cli')

def setup_global_args(parser):
    """เพิ่มอาร์กิวเมนต์ทั่วไปสำหรับทุกคำสั่ง"""
    parser.add_argument('--verbose', '-v', action='count', default=0, 
                        help="เพิ่มระดับความละเอียดของ log (สามารถใช้ -vv หรือ -vvv ได้)")
    parser.add_argument('--quiet', '-q', action='store_true', 
                        help="ลดการแสดงผลให้เหลือเฉพาะข้อผิดพลาดเท่านั้น")
    parser.add_argument('--config', '-c', type=str, default=None,
                        help="ไฟล์การตั้งค่าที่ต้องการใช้")
    parser.add_argument('--cuda', action='store_true', 
                        help="กำหนดให้ใช้ CUDA")

def handle_command(args):
    """ดำเนินการตามคำสั่งที่กำหนด"""
    # จัดการคำสั่งโดยใช้ dict แทน if-elif chains
    command_handlers = {
        'data': {
            'download': data_commands.handle_download,
            'update': data_commands.handle_update,
            'process': data_commands.handle_process,
            'analyze': data_commands.handle_analyze
        },
        'train': {
            'model': train_commands.handle_model,
            'evaluate': train_commands.handle_evaluate
        },
        'backtest': {
            'run': backtest_commands.handle_run,
            'analyze': backtest_commands.handle_analyze
        }
    }
    
    if args.command not in command_handlers:
        logger.error(f"ไม่รู้จักคำสั่ง: {args.command}")
        return
    
    sub_command = getattr(args, f"{args.command}_command", None)
    if sub_command not in command_handlers[args.command]:
        logger.error(f"ไม่รู้จักคำสั่งย่อย: {sub_command} สำหรับคำสั่ง {args.command}")
        return
    
    # เรียกฟังก์ชันที่จะจัดการคำสั่ง
    handler = command_handlers[args.command][sub_command]
    handler(args)

def main():
    """ฟังก์ชันหลักของ Command Line Interface"""
    parser = argparse.ArgumentParser(
        description="CRYPPO - CRYPtocurrency Position Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    setup_global_args(parser)

    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='คำสั่งที่ต้องการใช้')
    
    # ตั้งค่า parsers ต่างๆ (โค้ดส่วนนี้ยังคงเหมือนเดิม)
    
    args = parser.parse_args()
    
    # ตั้งค่าระดับของ log
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose >= 3:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 2:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.WARNING)
    
    # ตั้งค่า CUDA ถ้าจำเป็น
    if hasattr(args, 'cuda') and args.cuda:
        set_cuda_env()
    
    # ตรวจสอบคำสั่ง
    if args.command is None:
        parser.print_help()
        return
    
    handle_command(args)

if __name__ == '__main__':
    main()