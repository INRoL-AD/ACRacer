import vgamepad
import subprocess
from utils.IS_ACUtil import *

PREFIX_autohotkey_exe = 'C:/Program Files/AutoHotkey/v1.1.37.01/AutoHotkeyU64.exe'
PREFIX_autohotkey_files = './utils/'

class ACController:
    """
    A virtual controller for Assetto Corsa.
    This class uses the vgamepad library to send inputs to the game.
    """
    def __init__(self, steer_scale=[-360, 360]):
        """
        Initialize the virtual controller.
        """
        self.steer_scale = steer_scale
        self.gamepad = vgamepad.VX360Gamepad()
        

    def perform(self, throttle, brake, steer):
        """
        Perform the actions in the game.
        """
        throttle = max(0.0, throttle)
        brake = max(0.0, brake)
        self.gamepad.left_trigger_float(value_float=brake)
        self.gamepad.right_trigger_float(value_float=throttle)
        self.gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        self.gamepad.update()


    def reset_controller(self):
        """
        Reset the controller.
        """
        self.perform(0,1,0)
        self.gamepad.reset()
        self.gamepad.update()


    def reset_car(self):
        """
        Reset the environment.
        """
        self.perform(0,1,0)
        sendCMD(68)
        sendCMD(69)
        self.perform(0,1,0)
        self.gamepad.reset()
        self.gamepad.update()


    ### Control the environment with AutoHotkey scripts ###
    def pause(self):
        subprocess.run([PREFIX_autohotkey_exe, PREFIX_autohotkey_files + 'AC_pause.ahk'])
     
    def resume(self):
        subprocess.run([PREFIX_autohotkey_exe, PREFIX_autohotkey_files + 'AC_resume.ahk'])
     
    def ai_on(self):
        subprocess.run([PREFIX_autohotkey_exe, PREFIX_autohotkey_files + 'AC_ai.ahk'])
    
    def ai_off(self):
        subprocess.run([PREFIX_autohotkey_exe, PREFIX_autohotkey_files + 'AC_ai.ahk'])