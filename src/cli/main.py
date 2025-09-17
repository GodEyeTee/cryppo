import argparse
import logging
import sys

from src.cli.commands import data_commands, train_commands, backtest_commands
from src.utils.loggers import setup_logger
from src.utils.config_manager import get_config, set_cuda_env

logger = setup_logger('cli')


def _create_subparsers(parser, **kwargs):
    """Create subparsers with optional ``required`` argument support."""
    if sys.version_info < (3, 7) and 'required' in kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key != 'required'}
    return parser.add_subparsers(**kwargs)

def setup_global_args(parser):
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--config', '-c', type=str, default=None)
    parser.add_argument('--cuda', action='store_true')

def handle_command(args):
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
    
    handler = command_handlers[args.command][sub_command]
    handler(args)

def main():
    parser = argparse.ArgumentParser(
        description="CRYPPO - CRYPtocurrency Position Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    setup_global_args(parser)

    subparsers = _create_subparsers(
        parser,
        dest='command',
        help='คำสั่งที่ต้องการใช้'
    )

    data_parser = subparsers.add_parser('data', help='Data utilities')
    data_subparsers = _create_subparsers(
        data_parser,
        dest='data_command',
        required=sys.version_info >= (3, 7),
        help='คำสั่งย่อยสำหรับการจัดการข้อมูล'
    )

    download_parser = data_subparsers.add_parser('download', help='ดาวน์โหลดข้อมูลตลาด')
    data_commands.setup_download_parser(download_parser)
    download_parser.set_defaults(command='data', data_command='download')

    update_parser = data_subparsers.add_parser('update', help='อัปเดตข้อมูลที่มีอยู่')
    data_commands.setup_update_parser(update_parser)
    update_parser.set_defaults(command='data', data_command='update')

    process_parser = data_subparsers.add_parser('process', help='ประมวลผลชุดข้อมูล')
    data_commands.setup_process_parser(process_parser)
    process_parser.set_defaults(command='data', data_command='process')

    analyze_parser = data_subparsers.add_parser('analyze', help='วิเคราะห์ข้อมูล')
    data_commands.setup_analyze_parser(analyze_parser)
    analyze_parser.set_defaults(command='data', data_command='analyze')

    train_parser = subparsers.add_parser('train', help='การฝึกโมเดล')
    train_subparsers = _create_subparsers(
        train_parser,
        dest='train_command',
        required=sys.version_info >= (3, 7),
        help='คำสั่งย่อยสำหรับการฝึกและประเมินโมเดล'
    )

    model_parser = train_subparsers.add_parser('model', help='ฝึกโมเดลใหม่')
    train_commands.setup_model_parser(model_parser)
    model_parser.set_defaults(command='train', train_command='model')

    evaluate_parser = train_subparsers.add_parser('evaluate', help='ประเมินโมเดลที่ฝึกไว้')
    train_commands.setup_evaluate_parser(evaluate_parser)
    evaluate_parser.set_defaults(command='train', train_command='evaluate')

    backtest_parser = subparsers.add_parser('backtest', help='การทดสอบย้อนหลัง')
    backtest_subparsers = _create_subparsers(
        backtest_parser,
        dest='backtest_command',
        required=sys.version_info >= (3, 7),
        help='คำสั่งย่อยสำหรับการทดสอบย้อนหลัง'
    )

    run_parser = backtest_subparsers.add_parser('run', help='รันการทดสอบย้อนหลัง')
    backtest_commands.setup_run_parser(run_parser)
    run_parser.set_defaults(command='backtest', backtest_command='run')

    analyze_backtest_parser = backtest_subparsers.add_parser('analyze', help='วิเคราะห์ผลการทดสอบย้อนหลัง')
    backtest_commands.setup_analyze_parser(analyze_backtest_parser)
    analyze_backtest_parser.set_defaults(command='backtest', backtest_command='analyze')

    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose >= 3:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 2:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.WARNING)

    if hasattr(args, 'cuda') and args.cuda:
        set_cuda_env()
    
    if args.command is None:
        parser.print_help()
        return
    
    handle_command(args)

if __name__ == '__main__':
    main()
