import os, sys, configparser, platform
if platform.architecture()[0] == '64bit':
    sys.path.insert(0, '../apps/python/IS_AddShortcutKey/dll64')
else:
    sys.path.insert(0, '../apps/python/IS_AddShortcutKey/dll')
from ctypes import *
from ctypes.wintypes import MAX_PATH
import time, socket

def getACConfig(filename):
    buf = create_unicode_buffer(MAX_PATH + 1)
    if windll.shell32.SHGetSpecialFolderPathW(None, buf, 5, False):
        docPath = buf.value
    else:
        docPath = os.path.expanduser('~/Documents')
    config = configparser.SafeConfigParser()
    config.read(docPath + '/Assetto Corsa' + filename)
    return config


def getKeyState(code):
    return bool(windll.user32.GetAsyncKeyState(code) & 32768 != 0)


addr = ('localhost', 9666)
cs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def sendCMD(pack):
    global addr
    global cs
    cs.sendto(pack.to_bytes(4, byteorder='little'), addr)
    time.sleep(0.5)