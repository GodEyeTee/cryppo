import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import argparse
import logging
from pathlib import Path

# เพิ่มโฟลเดอร์หลักเข้าใน Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.loggers import setup_logger
from src.cli.commands import data_commands, train_commands, backtest_commands

# ตั้งค่า logger
logger = setup_logger('cli')

def main():
    """
    ฟังก์ชันหลักของ Command Line Interface
    """
    parser = argparse.ArgumentParser(
        description="CRYPPO - CRYPtocurrency Position Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='count', default=0, 
                        help="เพิ่มระดับความละเอียดของ log (สามารถใช้ -vv หรือ -vvv ได้)")
    parser.add_argument('--quiet', '-q', action='store_true', 
                        help="ลดการแสดงผลให้เหลือเฉพาะข้อผิดพลาดเท่านั้น")
    parser.add_argument('--config', '-c', type=str, default=None,
                        help="ไฟล์การตั้งค่าที่ต้องการใช้")

    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='คำสั่งที่ต้องการใช้')
    
    # Data commands
    data_parser = subparsers.add_parser('data', help='คำสั่งเกี่ยวกับข้อมูล')
    data_subparsers = data_parser.add_subparsers(dest='data_command')
    
    # สร้าง subparser สำหรับการดาวน์โหลดข้อมูล
    download_parser = data_subparsers.add_parser('download', help='ดาวน์โหลดข้อมูลประวัติราคา')
    data_commands.setup_download_parser(download_parser)
    
    # สร้าง subparser สำหรับการอัพเดตข้อมูล
    update_parser = data_subparsers.add_parser('update', help='อัพเดตข้อมูลให้เป็นปัจจุบัน')
    data_commands.setup_update_parser(update_parser)
    
    # สร้าง subparser สำหรับการประมวลผลข้อมูล
    process_parser = data_subparsers.add_parser('process', help='ประมวลผลข้อมูลดิบให้เป็น format ที่ใช้งานได้')
    data_commands.setup_process_parser(process_parser)
    
    # สร้าง subparser สำหรับการวิเคราะห์ข้อมูล
    analyze_parser = data_subparsers.add_parser('analyze', help='วิเคราะห์ข้อมูล')
    data_commands.setup_analyze_parser(analyze_parser)
    
    # Train commands
    train_parser = subparsers.add_parser('train', help='คำสั่งเกี่ยวกับการเทรนโมเดล')
    train_subparsers = train_parser.add_subparsers(dest='train_command')
    
    # สร้าง subparser สำหรับการเทรนโมเดล
    model_parser = train_subparsers.add_parser('model', help='เทรนโมเดล')
    train_commands.setup_model_parser(model_parser)
    
    # สร้าง subparser สำหรับการประเมินโมเดล
    evaluate_parser = train_subparsers.add_parser('evaluate', help='ประเมินประสิทธิภาพโมเดล')
    train_commands.setup_evaluate_parser(evaluate_parser)
    
    # Backtest commands
    backtest_parser = subparsers.add_parser('backtest', help='คำสั่งเกี่ยวกับการทดสอบย้อนหลัง')
    backtest_subparsers = backtest_parser.add_subparsers(dest='backtest_command')
    
    # สร้าง subparser สำหรับการทดสอบย้อนหลัง
    run_parser = backtest_subparsers.add_parser('run', help='ทดสอบโมเดลย้อนหลัง')
    backtest_commands.setup_run_parser(run_parser)
    
    # สร้าง subparser สำหรับการวิเคราะห์ผลการทดสอบ
    analyze_parser = backtest_subparsers.add_parser('analyze', help='วิเคราะห์ผลการทดสอบย้อนหลัง')
    backtest_commands.setup_analyze_parser(analyze_parser)
    
    # Parse arguments
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
    
    # ตรวจสอบคำสั่ง
    if args.command is None:
        parser.print_help()
        return
    
    # ดำเนินการตามคำสั่ง
    if args.command == 'data':
        if args.data_command == 'download':
            data_commands.handle_download(args)
        elif args.data_command == 'update':
            data_commands.handle_update(args)
        elif args.data_command == 'process':
            data_commands.handle_process(args)
        elif args.data_command == 'analyze':
            data_commands.handle_analyze(args)
        else:
            data_parser.print_help()
    
    elif args.command == 'train':
        if args.train_command == 'model':
            train_commands.handle_model(args)
        elif args.train_command == 'evaluate':
            train_commands.handle_evaluate(args)
        else:
            train_parser.print_help()
    
    elif args.command == 'backtest':
        if args.backtest_command == 'run':
            backtest_commands.handle_run(args)
        elif args.backtest_command == 'analyze':
            backtest_commands.handle_analyze(args)
        else:
            backtest_parser.print_help()

if __name__ == '__main__':
    main()