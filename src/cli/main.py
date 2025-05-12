import argparse
import logging
from src.cli.commands import data_commands, train_commands, backtest_commands
from src.utils.loggers import setup_logger
from src.utils.config_manager import get_config, set_cuda_env

logger = setup_logger('cli')

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
    parser.add_subparsers(dest='command', help='คำสั่งที่ต้องการใช้')
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
