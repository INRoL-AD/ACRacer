##############################################################
# In-game Keyboard Shortcuts Extension
# ver. 1.2.1 / 23.Aug.2017
# Innovare Studio Inc. / D.Tsukamoto
#############################################################

import os
import sys
import platform
if platform.architecture()[0] == "64bit":
    sys.path.insert(0, "apps/python/IS_AddShortcutKey/dll64")
else:
    sys.path.insert(0, "apps/python/IS_AddShortcutKey/dll")
os.environ['PATH'] = os.environ['PATH'] + ";."

import ac
import acsys
import threading
import configparser
from IS_ACUtil import *

isAutoStarted = False
runing_keyhook = False
shutdown = False

isEnableAutoStart = False
isVR = False

def acShutdown():
	global shutdown
	
	shutdown = True

def acMain(acVersion):
	global isEnableAutoStart, isVR, runing_keyhook
	global KEY_COMB, KEY_PIT, KEY_SETUP, KEY_START, KEY_RESTART, KEY_EXIT
	
	config = configparser.ConfigParser()
	config.read("apps/python/IS_AddShortCutKey/config.ini")
	AUTOSTART = config.get("CONFIG", "AUTOSTART")
	isEnableAutoStart = AUTOSTART == "ALWAYS"
	
	KEY_COMB = config.getint("KEY", "COMB")
	KEY_PIT = config.getint("KEY", "PIT")
	KEY_SETUP = config.getint("KEY", "SETUP")
	KEY_START = config.getint("KEY", "START")
	KEY_RESTART = config.getint("KEY", "RESTART")
	KEY_EXIT = config.getint("KEY", "EXIT")
	
	acconfig = getACConfig("/cfg/video.ini")
	if acconfig.get("CAMERA", "MODE") in ["OCULUS", "OPENVR"]:
		isVR = True
	
	if AUTOSTART == "VR" and isVR:
		isEnableAutoStart = True
	
	runing_keyhook = True
	t_kh = threading.Thread(target=keyhook)
	t_kh.start()
	
	return "AddShortcutKey"

elapsedT = 0

def acUpdate(deltaT):
	global elapsedT, isAutoStarted, isEnableAutoStart, isVR
	
	if isEnableAutoStart and not isAutoStarted:
		elapsedT += deltaT
		if isVR and elapsedT > 0.1:
			sendCMD(69)
			isAutoStarted = True
		elif ac.getCameraMode(0) == acsys.CM.Start:
			sendCMD(69)
		else:
			isAutoStarted = True

def keyhook():
	global runing_keyhook, shutdown
	global KEY_COMB, KEY_PIT, KEY_SETUP, KEY_START, KEY_RESTART, KEY_EXIT
	
	while True:
		if getKeyState(KEY_COMB):
			if getKeyState(KEY_PIT):
				sendCMD(75)
			elif getKeyState(KEY_SETUP):
				sendCMD(76)
			elif getKeyState(KEY_START):
				sendCMD(69)
			elif getKeyState(KEY_RESTART):
				sendCMD(68)
			elif getKeyState(KEY_EXIT):
				sendCMD(66)
		
		if shutdown:
			runing_keyhook = False
			break
