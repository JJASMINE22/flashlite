# -*- coding: UTF-8 -*-
import codecs
import datetime
import os
from logging.handlers import BaseRotatingHandler


class LogFileHandler(BaseRotatingHandler):
    def __init__(self, filename):
        self.suffix = "%Y-%m-%d"
        self.date = datetime.datetime.now()
        super(BaseRotatingHandler, self).__init__(filename, mode='a', encoding='UTF-8', delay=False)

    def shouldRollover(self, record):
        return self.date != datetime.datetime.now()

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        self.date = datetime.datetime.now()

    def _open(self):
        filename = '%s-%s.log' % (self.baseFilename, self.date.strftime(self.suffix))
        if self.encoding is None:
            stream = open(filename, self.mode)
        else:
            stream = codecs.open(filename, self.mode, self.encoding)
        if os.path.exists(self.baseFilename):
            try:
                os.remove(self.baseFilename)
            except OSError:
                pass
        try:
            os.symlink(filename, self.baseFilename)
        except OSError:
            pass
        return stream
