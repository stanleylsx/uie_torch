import logging
import functools
import colorlog
import contextlib
import threading
import time
import datetime


class Logger(object):
    """
    Default logger in UIE
    Args:
        name(str) : Logger name, default is 'UIE'
    """
    def __init__(self, name, log_dir, mode):

        log_config = {
            'DEBUG': {'level': 10, 'color': 'purple'},
            'INFO': {'level': 20, 'color': 'green'},
            'TRAIN': {'level': 21, 'color': 'cyan'},
            'EVAL': {'level': 22, 'color': 'blue'},
            'WARNING': {'level': 30, 'color': 'yellow'},
            'ERROR': {'level': 40, 'color': 'red'},
            'CRITICAL': {'level': 50, 'color': 'bold_red'}}

        name = 'UIE' if not name else name
        self.logger = logging.getLogger(name)
        log_file = log_dir + '/' + (datetime.datetime.now().strftime('%Y%m%d%H%M%S-' + mode + '.log'))
        for key, conf in log_config.items():
            logging.addLevelName(conf['level'], key)
            self.__dict__[key] = functools.partial(
                self.__call__, conf['level'])
            self.__dict__[key.lower()] = functools.partial(
                self.__call__, conf['level'])

        self.format = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)-15s] [%(levelname)8s]%(reset)s - %(message)s',
            log_colors={key: conf['color']
                        for key, conf in log_config.items()})

        self.handler = logging.FileHandler(log_file)
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logLevel = 'DEBUG'
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self._is_enable = True

    def disable(self):
        self._is_enable = False

    def enable(self):
        self._is_enable = True

    @property
    def is_enable(self) -> bool:
        return self._is_enable

    def __call__(self, log_level: str, msg: str):
        if not self.is_enable:
            return

        self.logger.log(log_level, msg)

    @contextlib.contextmanager
    def use_terminator(self, terminator: str):
        old_terminator = self.handler.terminator
        self.handler.terminator = terminator
        yield
        self.handler.terminator = old_terminator

    @contextlib.contextmanager
    def processing(self, msg: str, interval: float = 0.1):
        """
        Continuously print a progress bar with rotating special effects.

        Args:
            msg(str): Message to be printed.
            interval(float): Rotation interval. Default to 0.1.
        """
        end = False

        def _printer():
            index = 0
            flags = ['\\', '|', '/', '-']
            while not end:
                flag = flags[index % len(flags)]
                with self.use_terminator('\r'):
                    self.logger.info('{}: {}'.format(msg, flag))
                time.sleep(interval)
                index += 1

        t = threading.Thread(target=_printer)
        t.start()
        yield
        end = True