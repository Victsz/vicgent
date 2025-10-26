from contextlib import contextmanager
import time
import datetime as dt
import logging
log_method = lambda x: logging.info(x) if len(logging.getLogger().handlers) > 0 else print(x)
@contextmanager
def stopwatch(description="操作"):
    log_method(f"{dt.datetime.now()} - {description} 开始...")
    start_time = time.time()
    yield
    end_time = time.time()
    log_method(f"{dt.datetime.now()} - {description} 结束...")
    log_method(f"{description} - 耗时: {end_time - start_time:.4f} 秒")

