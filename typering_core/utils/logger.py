import time
from colorama import Fore
from typing import Any


__all__ = ['system', 'info', 'warning', 'error']


def system(msg:Any):
    tag = f'[SYSTEM {time.perf_counter():.3f} s]'
    print(f'{Fore.GREEN}{tag} {str(msg)}{Fore.RESET}')
    

def info(msg:Any):
    tag = f'[INFO {time.perf_counter():.3f} s]'
    print(f'{Fore.WHITE}{tag} {str(msg)}{Fore.RESET}')
    
    
def warning(msg:Any):
    tag = f'[WARNING {time.perf_counter():.3f} s]'
    print(f'{Fore.YELLOW}{tag} {str(msg)}{Fore.RESET}')
    

def error(msg:Any):
    tag = f'[ERROR {time.perf_counter():.3f} s]'
    print(f'{Fore.RED}{tag} {str(msg)}{Fore.RESET}')
    
    
if __name__ == '__main__':
    system(f'log system-level messages')
    info(f'log user-level messages')
    warning(f'log warning messages')
    error(f'log error messages')
    