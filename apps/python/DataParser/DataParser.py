APP_NAME = 'DataParser' # Name of the app
PATH_PREFIX = 'C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/python/DataParser/data_bucket/'  # Path to save the data
SAMPLING_FREQ = 200     # Hz (Deprecated... we ran as fast as possible)
IS_VIS = False          # Visualize by AC App
IS_SAVE = False         # Save all the past data as json file

## Add the third party libraries to the path ##
try:
    import sys
    import os
    import traceback
    import platform
    import json
    import time
    import shutil

    import ac_api.car_info as ci
    import ac_api.car_stats as css
    import ac_api.input_info as ii
    import ac_api.lap_info as li
    import ac_api.session_info as si
    import ac_api.tyre_info as ti
    import ac

    if platform.architecture()[0] == "64bit":
        sysdir = "stdlib64"
    else:
        sysdir = "stdlib"
    sys.path.insert(
        len(sys.path), 'apps/python/{}/third_party'.format(APP_NAME))
    os.environ['PATH'] += ";."
    sys.path.insert(len(sys.path), os.path.join(
        'apps/python/{}/third_party'.format(APP_NAME), sysdir))
    os.environ['PATH'] += ";."

    from IS_ACUtil import *

except Exception as e:
    ac.log("[DataParser] Error in importing libraries: {}".format(traceback.format_exc()))
    ac.console("[DataParser] Error importing libraries: %s" % e)


class StateBucket:
    def __init__(self):
        self.time_at_record_start = time.time() * 1000              # Time when the record(=session) starts (Float, ms)

        self.dict_for_json = {}
        self.dict_for_json['ego'] = {}
        self.dict_label = {}

        ## Basic info.
        self.dict_for_json['time_at_sampling'] = 0                  # Time when sampling starts (Float, ms)
        self.dict_for_json['time_to_collect'] = 0                   # Elapsed time for data collection (Float, ms)

        ## Car info.
        self.dict_for_json['ego']['car_name'] = None                # Speed (Float, kph)
        self.dict_for_json['ego']['speed'] = None                   # Speed (Float, kph)
        self.dict_for_json['ego']['velocity_local'] = None          # Local velocity (Float array, [x,y,z] m/s)
        self.dict_for_json['ego']['angular_velocity_local'] = []    # Local angular velocity (Float array, [x,y,z])
        self.dict_for_json['ego']['track_progress'] = None          # Lap progress (Float, [0,1])
        self.dict_for_json['ego']['position_global'] = []           # Position in global coord. ([x,y,z])
        self.dict_for_json['ego']['orientation_global'] = []        # Heading in global coord. ([roll, pitch, yaw])

        # self.dict_for_json['ego']['camber_angle'] = []              # Camber angle (Float array (FL,FR,RL,RR), each of [0,360] deg)
        self.dict_for_json['ego']['slip_angle'] = []                # Slip angle (Float array (FL,FR,RL,RR), each of [0,360] deg)
        self.dict_for_json['ego']['slip_ratio'] = []                # Slip ratio (Float array (FL,FR,RL,RR), each of [0,1])
        # self.dict_for_json['ego']['tyre_load'] = []                 # Tyre load (Float array (FL,FR,RL,RR))

        # self.dict_for_json['ego']['rank'] = None                    # Rank (Int, 0: lead car)
        # self.dict_for_json['ego']['is_drs_avail'] = None            # DRS avail (w.r.t track)? (Int, 0 or 1)
        # self.dict_for_json['ego']['is_drs_enabled'] = None          # DRS enabled? (Bool)
        self.dict_for_json['ego']['gear_status'] = None             # Gear (0=R, 1=N, 2=1, 3=2, 4=3, 5=4, 6=5, 7=6, 8=7, etc.)
        self.dict_for_json['ego']['rpm'] = None                     # RPM (Float)
        # self.dict_for_json['ego']['fuel_amount'] = None             # Fuel weight (Float, kg)

        self.dict_for_json['ego']['num_tyres_off_track'] = None     # Num of tyres off-track (Int)
        self.dict_for_json['ego']['collision_impulse'] = []         # Accumulated damage (Float array, (Front, Rear, Left, Right, Center), Slight tap ~10)
        # self.dict_for_json['ego']['cg_height'] = None               # Height of center of gravity of the car from ground (Float. maybe m)
        # self.dict_for_json['ego']['speed_drivetrain'] = None        # Speed delivered to wheels... why different w/ actual speed? (Float)
        self.dict_for_json['ego']['velocity_global'] = []           # Velocity in global coord. ([x,y,z] m/s)
        self.dict_for_json['ego']['acceleration_local'] = []        # Acceleration in local coord. = GForce ([x,y,z])
        # self.dict_for_json['ego']['tc_limit'] = None                # Slip ratio limit for TC (Float)
        # self.dict_for_json['ego']['brake_bias_amount'] = None       # Brake bias btw front/rear (Float, [0,1])
        # self.dict_for_json['ego']['engine_brake_amount'] = None     # Engine brake (Float)

        ## Car stat.    
        # self.dict_for_json['ego']['is_drs_exist'] = None            # Whether DRS is available (w.r.t car) (Int, 0 or 1)
        # self.dict_for_json['ego']['is_ers_exist'] = None            # Whether ERS is available (w.r.t car) (Int, 0 or 1)
        # self.dict_for_json['ego']['is_kers_exist'] = None           # Whether KERS is available (w.r.t car) (Int, 0 or 1)
        # self.dict_for_json['ego']['abs_amount'] = None              # ABS active level (Float, [0,1])
        # self.dict_for_json['ego']['rpm_max'] = None                 # Max rpm value (Int, [])
        # self.dict_for_json['ego']['fuel_max'] = None                # Max fuel amount (Int, kg)

        ## Input info.  
        self.dict_for_json['ego']['throttle_report'] = None         # (Float, [0,1])
        self.dict_for_json['ego']['brake_report'] = None            # (Float, [0,1])
        self.dict_for_json['ego']['steer_report'] = None            # (Float, deg])
        self.dict_for_json['ego']['clutch_report'] = None           # (Float, [0,1]) (1 is fully deployed, 0 is not deployed)
        # self.dict_for_json['ego']['last_ff_report'] = None          # Last force feedback signal sent to wheel (Float)

        ## Lap info.    
        self.dict_for_json['ego']['current_lap_time'] = None        # Lap time (Int, ms)
        self.dict_for_json['ego']['last_lap_time'] = None           # Lap time (Int, ms)
        self.dict_for_json['ego']['best_lap_time'] = None           # Lap time (Int, ms)
        self.dict_for_json['ego']['is_invalid'] = None              # Whehter current lap is invalid (Bool) (0: Valid, 1: Invalid) (PythonAPI decision "or" num_off_tyres>2)
        self.dict_for_json['ego']['lap_count'] = None               # (Int) (Start = (lap = 1 default))
        self.dict_for_json['ego']['lap_count_max'] = None           # Total possible laps in a race (Int)

        ## Session info.    
        self.dict_for_json['cars_count'] = None                     # Number of cars in session (Int)
        self.dict_for_json['track_name'] = None                     # Track name (Str)
        
        # self.dict_for_json['ego']['ballast'] = None                 # Get ballast (Int, kg)
        # self.dict_for_json['ego']['caster_angle'] = None            # Get caster angle (Int, rad)
        # self.dict_for_json['ego']['tyre_radius'] = []               # Tyre radius (Int array, (FL,FR,RL,RR), maybe m)

        ## Specify the interested data to visualize in AC
        if IS_VIS:
            self.interested = ['track_progress', 'num_tyres_off_track', 'steer_report']
            for key in self.interested:
                self.dict_label[key] = None

    
def acMain(ac_version):
    """
    The main function of the app, called on app start.
    :param ac_version: The version of Assetto Corsa as a string.
    """
    try:
        global SB, sampling_time_pre
        SB = StateBucket()
        sampling_time_pre = SB.time_at_record_start

        # Create the app window
        if IS_VIS:
            APP_WINDOW = ac.newApp(APP_NAME)

            ac.setSize(APP_WINDOW, 300, 100)
            ac.setTitle(APP_WINDOW, APP_NAME)
            ac.setBackgroundOpacity(APP_WINDOW, 0.4)

            for idx, key in enumerate(SB.interested):
                SB.dict_label[key] = ac.addLabel(APP_WINDOW, key + " :\t")  # Init label in AC
                ac.setPosition(SB.dict_label[key], 3, 20*(idx+2))         # Set position of label
                ac.setFontSize(SB.dict_label[key], 12)
                
        return APP_NAME
    
    except Exception as e:
        ac.log("[DataParser] Error in main: {}".format(traceback.format_exc()))
        ac.console("[DataParser] Error in main: %s" % e)


def acUpdate(deltaT):
    """
    The update function of the app, called every frame.
    :param deltaT: The time since the last frame as a float.
    """
    try:
        global SB, sampling_time_pre
        sampling_time = time.time() * 1000      # ms

        # if sampling_time >= (SB.time_at_record_start + (1/SAMPLING_FREQ * 1000) * (((sampling_time_pre - SB.time_at_record_start) // (1/SAMPLING_FREQ * 1000)) + 1)): # Deprecated
            
        sampling_time_pre = sampling_time

        ## Car info.
        SB.dict_for_json['ego']['car_name'] = si.get_car_name(car=0)
        SB.dict_for_json['ego']['speed'] = ci.get_speed(car=0, unit="kmh")
        SB.dict_for_json['ego']['track_progress'] = ci.get_location(car=0)
        
        velocity_local = ci.get_local_velocity()
        angular_velocity_local = ci.get_local_angular_velocity()
        position_global = ci.get_world_location(car=0)
        orientation_global = ci.get_world_orientation()
        SB.dict_for_json['ego']['velocity_local'] = []
        SB.dict_for_json['ego']['angular_velocity_local'] = []
        SB.dict_for_json['ego']['position_global'] = []
        SB.dict_for_json['ego']['orientation_global'] = []
        for i in range(len(position_global)):
            SB.dict_for_json['ego']['velocity_local'].append(velocity_local[i])
            SB.dict_for_json['ego']['angular_velocity_local'].append(angular_velocity_local[i])
            SB.dict_for_json['ego']['position_global'].append(position_global[i])
            SB.dict_for_json['ego']['orientation_global'].append(orientation_global[i])

        # camber_angle = ci.get_camber_angle(car=0)
        slip_angle = ci.get_slip_angle(car=0)
        slip_ratio = ci.get_slip_ratio(car=0)
        # tyre_load = ci.get_tyre_load(car=0)
        # SB.dict_for_json['ego']['camber_angle'] = []
        SB.dict_for_json['ego']['slip_angle'] = []
        SB.dict_for_json['ego']['slip_ratio'] = []
        # SB.dict_for_json['ego']['tyre_load'] = []
        for i in range(len(slip_ratio)):
            # SB.dict_for_json['ego']['camber_angle'].append(camber_angle[i])
            SB.dict_for_json['ego']['slip_angle'].append(slip_angle[i])
            SB.dict_for_json['ego']['slip_ratio'].append(slip_ratio[i])
            # SB.dict_for_json['ego']['tyre_load'].append(tyre_load[i])
        
        # SB.dict_for_json['ego']['rank'] = ci.get_position(car=0)
        # SB.dict_for_json['ego']['is_drs_avail'] = ci.get_drs_available()
        # SB.dict_for_json['ego']['is_drs_enabled'] = ci.get_drs_enabled()
        SB.dict_for_json['ego']['gear_status'] = ci.get_gear(car=0, formatted=False)
        SB.dict_for_json['ego']['rpm'] = ci.get_rpm(car=0)
        SB.dict_for_json['ego']['num_tyres_off_track'] = ci.get_tyres_off_track()
        # SB.dict_for_json['ego']['fuel_amount'] = ci.get_fuel()
        
        collision_impulse = ci.get_total_damage()[0:4]
        SB.dict_for_json['ego']['collision_impulse'] = []
        for i in range(len(collision_impulse)):
            SB.dict_for_json['ego']['collision_impulse'].append(collision_impulse[i])
        
        # SB.dict_for_json['ego']['cg_height'] = ci.get_cg_height(car=0)
        # SB.dict_for_json['ego']['speed_drivetrain'] = ci.get_drive_train_speed(car=0)
        
        velocity_global = ci.get_velocity()
        acceleration_local = ci.get_acceleration()
        SB.dict_for_json['ego']['velocity_global'] = []
        SB.dict_for_json['ego']['acceleration_local'] = []
        for i in range(len(velocity_global)):
            SB.dict_for_json['ego']['velocity_global'].append(velocity_global[i])
            SB.dict_for_json['ego']['acceleration_local'].append(acceleration_local[i])

        # SB.dict_for_json['ego']['tc_limit'] = ci.get_tc_in_action()
        # SB.dict_for_json['ego']['brake_bias_amount'] = ci.get_brake_bias()
        # SB.dict_for_json['ego']['engine_brake_amount'] = ci.get_engine_brake()
        
        ## Car stats.
        # SB.dict_for_json['ego']['is_drs_exist'] = css.get_has_drs()
        # SB.dict_for_json['ego']['is_ers_exist'] = css.get_has_ers()
        # SB.dict_for_json['ego']['is_kers_exist'] = css.get_has_kers()
        # SB.dict_for_json['ego']['abs_amount'] = css.abs_level()

        ## Input info.
        SB.dict_for_json['ego']['throttle_report'] = ii.get_gas_input(car=0)
        SB.dict_for_json['ego']['brake_report'] = ii.get_brake_input(car=0)
        SB.dict_for_json['ego']['steer_report'] = ii.get_steer_input(car=0)
        SB.dict_for_json['ego']['clutch_report'] = ii.get_clutch(car=0)
        # SB.dict_for_json['ego']['last_ff_report'] = ii.get_last_ff(car=0)

        ## Lap info.
        SB.dict_for_json['ego']['current_lap_time'] = li.get_current_lap_time(car=0)
        SB.dict_for_json['ego']['last_lap_time'] = li.get_last_lap_time(car=0)
        SB.dict_for_json['ego']['best_lap_time'] = li.get_best_lap_time(car=0)
        SB.dict_for_json['ego']['is_invalid'] = li.get_invalid(car=0)
        SB.dict_for_json['ego']['lap_count'] = li.get_lap_count(car=0)
        SB.dict_for_json['ego']['lap_count_max'] = li.get_laps()

        ## Session info.
        SB.dict_for_json['cars_count'] = si.get_cars_count()
        SB.dict_for_json['track_name'] = si.get_track_name() + " " + si.get_track_config()
        
        # SB.dict_for_json['ego']['ballast'] = si.get_car_ballast(car=0)
        # SB.dict_for_json['ego']['caster_angle'] = si.get_car_ballast(car=0)

        # tyre_radius = si.get_radius(car=0)
        # SB.dict_for_json['ego']['tyre_radius'] = []
        # for i in range(len(tyre_radius)):
        #     SB.dict_for_json['ego']['tyre_radius'].append(tyre_radius[i])


        ## Surrounding vehicles info.
        for i in range(1, SB.dict_for_json['cars_count']):
            SB.dict_for_json['surr_' + str(i)] = {}
            
            SB.dict_for_json['surr_' + str(i)]['car_name'] = si.get_car_name(car=i)
            SB.dict_for_json['surr_' + str(i)]['position_global'] = []
            SB.dict_for_json['surr_' + str(i)]['velocity_global'] = []
            SB.dict_for_json['surr_' + str(i)]['velocity_local'] = []
            SB.dict_for_json['surr_' + str(i)]['angular_velocity_local'] = []

            position_global = ci.get_world_location(car=i)
            velocity_global = ci.get_velocity_pythonAPI(car=i)
            velocity_local = ci.get_local_velocity_pythonAPI(car=i)
            angular_velocity_local = ci.get_local_angular_velocity_pythonAPI(car=i)
            
            for j in range(len(position_global)):
                SB.dict_for_json['surr_' + str(i)]['position_global'].append(position_global[j])
                SB.dict_for_json['surr_' + str(i)]['velocity_global'].append(velocity_global[j])
                SB.dict_for_json['surr_' + str(i)]['velocity_local'].append(velocity_local[j])
                SB.dict_for_json['surr_' + str(i)]['angular_velocity_local'].append(angular_velocity_local[j])
        
        SB.dict_for_json['time_at_sampling'] = sampling_time
        SB.dict_for_json['time_to_collect'] = time.time() * 1000 - sampling_time     # ms

        folder_path = PATH_PREFIX + 'exp_' + str(int(SB.time_at_record_start)) + '_' + str(SAMPLING_FREQ) + 'hz'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(folder_path + '/tmp.json', 'w') as f:
            json.dump(SB.dict_for_json, f)
        shutil.move(folder_path + '/tmp.json', folder_path + '/exp.json')
        
        if IS_SAVE:
            with open(folder_path + '/' + str(sampling_time) + '.json', 'w') as f:
                json.dump(SB.dict_for_json, f)
        
        if IS_VIS:
            for key in SB.interested:
                ac.setText(SB.dict_label[key], key + " :\t" + "{}".format(SB.dict_for_json['ego'][key]))

    except Exception as e:
        ac.log("[DataParser] Error in update: {}".format(traceback.format_exc()))
        ac.console("[DataParser] Error in update: %s" %e)


def acShutdown():
    try:
        pass
    
    except Exception as e:
        ac.log("[DataParser] Error in shutdown: {}".format(traceback.format_exc()))
        ac.console("[DataParser] Error in shutdown: %s" %e)