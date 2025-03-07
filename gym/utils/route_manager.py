import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import LineString, Point, Polygon, MultiPoint


class Calculator:
    """
    From the map data, calculate offline data of curvature, slope, and bank angle.
    Additonally, 2D rangefinder is adopted and functions to find nearest index are adopted.
    """
    def __init__(self, name, data):
        self.track_name = name
        self.track_data = data                              # (N, 9) np array. Each column is : [x_c, y_c, z_c, x_lb, y_lb, z_lb, x_rb, y_rb, z_rb] 
        self.num_points = len(self.track_data)
        self.curvature_offline = np.zeros(self.num_points)  # 1/R on zx-plane
        self.slope_offline = np.zeros(self.num_points)      # dy/ds (ds^2 = dx^2 + dz^2)
        self.bank_offline = np.zeros(self.num_points)       # dy/dw (w^2 = (x_lb - x_rb)^2 + (y_lb - y_rb)^2)
        self.init_flag = True
    
    
    def calc_curvature_offline(self, n_forward, n_backward, pfit_degree, is_smooth=False, filter_sigma=1, is_visualize=False):
        """
        Calculate offline curvature data from the track data.
        
        Args:
            n_forward: Number of forward sample points to consider
            n_backward: Number of backward sample points to consider
            pfit_degree: Degree of polynomial fitting
            is_smooth: If True, apply Gaussian filter to the data
            filter_sigma: Sigma value for Gaussian filter
            is_visualize: If True, visualize the data for debugging.
        """
        x_c = self.track_data[:,0]
        z_c = self.track_data[:,2]
        s_c = np.cumsum(np.sqrt(np.diff(x_c)**2 + np.diff(z_c)**2))
        s_c = np.insert(s_c, 0, 0)
        curvature_offline = np.zeros(self.num_points)

        for i in range(self.num_points):
            start = max(0, i - n_backward)
            end = min(self.num_points, i + n_forward + 1)
            
            x_c_local = x_c[start:end]
            z_c_local = z_c[start:end]
            s_c_local = s_c[start:end]

            ### Function parameterization
            coeffs_fx = np.polyfit(s_c_local, x_c_local, pfit_degree)
            coeffs_dfx = np.polyder(coeffs_fx)          # First derivative
            coeffs_d2fx = np.polyder(coeffs_dfx)        # Second derivative

            coeffs_fz = np.polyfit(s_c_local, z_c_local, pfit_degree)
            coeffs_dfz = np.polyder(coeffs_fz)          # First derivative
            coeffs_d2fz = np.polyder(coeffs_dfz)        # Second derivative

            ### Calculate curvature
            dfx = np.polyval(coeffs_dfx, s_c[i])
            d2fx = np.polyval(coeffs_d2fx, s_c[i])
            dfz = np.polyval(coeffs_dfz, s_c[i])
            d2fz = np.polyval(coeffs_d2fz, s_c[i])
            curvature = - (dfx * d2fz - dfz * d2fx) / ((dfx**2 + dfz**2) ** 1.5)    # (+): left, (-): right
            curvature_offline[i] = curvature

        if is_smooth:
            self.curvature_offline = gaussian_filter1d(curvature_offline, sigma=filter_sigma)
        else:
            self.curvature_offline = curvature_offline

        if is_visualize:
            plt.figure(figsize=(10,6))
            plt.plot(x_c, z_c, c='black')
            plt.scatter(x_c, z_c, c=self.curvature_offline, cmap='seismic', vmin=-max(np.abs(min(self.curvature_offline)), np.abs(max(self.curvature_offline))), vmax=max(np.abs(min(self.curvature_offline)), np.abs(max(self.curvature_offline))))
            plt.colorbar(label='curvature')
            plt.title('Curvature visualization of track : ' + self.track_name)
            plt.xlabel('x [m]')
            plt.ylabel('z [m]')
            plt.gca().invert_xaxis()
            plt.axis('equal')

            plt.figure(figsize=(10,6))
            plt.plot(s_c, curvature_offline, c='k', label='Original data')
            if is_smooth:
                plt.plot(s_c, self.curvature_offline, c='r', linestyle='--', label='Smoothed data')
            plt.title('Plot of curvature data')
            plt.xlabel('s [m]')
            plt.ylabel('curvature')
            plt.legend()
            plt.show()


    def calc_slope_offline(self, n_forward, n_backward, pfit_degree, is_smooth=False, filter_sigma=1, is_visualize=False):
        """
        Calculate offline slope data from the track data.
        """
        x_c = self.track_data[:,0]
        y_c = self.track_data[:,1]
        z_c = self.track_data[:,2]
        s_c = np.cumsum(np.sqrt(np.diff(x_c)**2 + np.diff(z_c)**2))
        s_c = np.insert(s_c, 0, 0)
        slope_offline = np.zeros(self.num_points)

        for i in range(self.num_points):
            start = max(0, i - n_backward)
            end = min(self.num_points, i + n_forward + 1)
            y_c_local = y_c[start:end]
            s_c_local = s_c[start:end]
            coeffs_fy = np.polyfit(s_c_local, y_c_local, pfit_degree)
            coeffs_dfy = np.polyder(coeffs_fy)        # First derivative
            dfy = np.polyval(coeffs_dfy, s_c[i])
            slope_offline[i] = dfy

        if is_smooth:
            self.slope_offline = gaussian_filter1d(slope_offline, sigma=filter_sigma)
        else:
            self.slope_offline = slope_offline
        
        if is_visualize:
            plt.figure(figsize=(10,6))
            plt.plot(x_c, z_c, c='black')
            plt.scatter(x_c, z_c, c=self.slope_offline, cmap='seismic', vmin=-max(np.abs(min(self.slope_offline)), np.abs(max(self.slope_offline))), vmax=max(np.abs(min(self.slope_offline)), np.abs(max(self.slope_offline))))
            plt.colorbar(label='slope')
            plt.title('Slope visualization of track : ' + self.track_name)
            plt.xlabel('x [m]')
            plt.ylabel('z [m]')
            plt.gca().invert_xaxis()
            plt.axis('equal')

            plt.figure(figsize=(10,6))
            plt.plot(s_c, slope_offline, c='k', label='Original data')
            if is_smooth:
                plt.plot(s_c, self.slope_offline, c='r', linestyle='--', label='Smoothed data')
            plt.title('Plot of slope data')
            plt.xlabel('s [m]')
            plt.ylabel('slope')
            plt.legend()
            plt.show()


    def calc_bank_offline(self, is_smooth=False, filter_sigma=1, is_visualize=False):
        """
        Calculate offline bank data from the track data.
        """
        x_c = self.track_data[:,0]
        z_c = self.track_data[:,2]
        s_c = np.cumsum(np.sqrt(np.diff(x_c)**2 + np.diff(z_c)**2))
        s_c = np.insert(s_c, 0, 0)
        x_lb = self.track_data[:,3]
        y_lb = self.track_data[:,4]
        z_lb = self.track_data[:,5]
        x_rb = self.track_data[:,6]
        y_rb = self.track_data[:,7]
        z_rb = self.track_data[:,8]
        w = np.sqrt((x_lb - x_rb)**2 + (z_lb - z_rb)**2)
        bank_offline = (y_lb - y_rb) / w

        if is_smooth:
            self.bank_offline = gaussian_filter1d(bank_offline, sigma=filter_sigma)
        else:
            self.bank_offline = bank_offline
        
        if is_visualize:
            plt.figure(figsize=(10,6))
            plt.plot(x_c, z_c, c='black')
            plt.scatter(x_c, z_c, c=self.bank_offline, cmap='seismic', vmin=-max(np.abs(min(self.bank_offline)), np.abs(max(self.bank_offline))), vmax=max(np.abs(min(self.bank_offline)), np.abs(max(self.bank_offline))))
            plt.colorbar(label='bank')
            plt.title('Bank visualization of track : ' + self.track_name)
            plt.xlabel('x [m]')
            plt.ylabel('z [m]')
            plt.gca().invert_xaxis()
            plt.axis('equal')

            plt.figure(figsize=(10,6))
            plt.plot(s_c, bank_offline, c='k', label='Original data')
            if is_smooth:
                plt.plot(s_c, self.bank_offline, c='r', linestyle='--', label='Smoothed data')
            plt.title('Plot of bank data')
            plt.xlabel('s [m]')
            plt.ylabel('bank')
            plt.legend()
            plt.show()


    def target_track(self, axis, which_lane='center'):
        """
        Simple mapping function to get target track data.
        """
        if which_lane == 'center':
            lane = 0
        elif which_lane == 'left':
            lane = 1
        elif which_lane == 'right':
            lane = 2        
        if axis == 'x':
            return self.track_data[:, 3*lane]
        elif axis == 'z':
            return self.track_data[:, 3*lane+2]


    def find_nearest_idx_at_first(self, pos, which_lane='center'):
        """
        Find the nearest index of the track data from the current position.
        For the first time, this function is used for the global scope.
        """
        target_x = self.target_track('x', which_lane)
        target_z = self.target_track('z', which_lane)
        idx_ret = -1
        dist_ret = -1

        for i in range(len(target_x)):
            dist = np.sqrt((target_x[i] - pos[0])**2 + (target_z[i] - pos[2])**2)
            if idx_ret == -1 or (dist < dist_ret):
                idx_ret = i
                dist_ret = dist

        return idx_ret


    def find_nearest_idx(self, pos, prev_idx, search_window_size=20, which_lane='center'):
        """
        Find the nearest index of the track data from the current position.
        After the first time, this function is used only for the local scope.
        """
        target_x = self.target_track('x', which_lane)
        target_z = self.target_track('z', which_lane)
        idx_ret = -1
        dist_ret = -1
        search_idx = [i%len(target_x) for i in range(prev_idx-search_window_size, prev_idx+search_window_size)]

        for i in search_idx:
            dist = np.sqrt((target_x[i] - pos[0])**2 + (target_z[i] - pos[2])**2)
            if idx_ret == -1 or (dist < dist_ret):
                idx_ret = i
                dist_ret = dist
        if dist_ret > search_window_size:
            idx_ret = self.find_nearest_idx_at_first(pos, which_lane)

        a = target_z[idx_ret] - target_z[(idx_ret+1)%len(target_x)]
        b = target_x[(idx_ret+1)%len(target_x)] - target_x[idx_ret]
        c = target_x[idx_ret] * target_z[(idx_ret+1)%len(target_x)] - target_x[(idx_ret+1)%len(target_x)] * target_z[idx_ret]
        if np.sqrt((a**2 + b**2)) == 0:
            dist_ret = 0
        else:
            dist_ret = (a*pos[0] + b*pos[2] + c) / (np.sqrt(a**2 + b**2))

        return idx_ret, dist_ret
    
    
    def find_result_idx(self, idx_ret, num_sample, num_sample_prev=0, sample_interval=10, vel=200, factor=1, mode='static', which_lane='center'):
        """
        Return the result index.
        """
        target_x = self.target_track('x', which_lane)
        target_z = self.target_track('z', which_lane)
        if mode == 'static':        # Fixed preview distance
            pass
        elif mode == 'dynamic':     # Dynamic preview distance (depends on the speed)
            sample_interval = int(max(1, (vel*factor)//num_sample))     # When speed=200kmh; If factor=1, you see 200m preview. If factor=0.5, you see 100m preview.

        result_idx = [i%len(target_x) for i in range(idx_ret - num_sample_prev * sample_interval, idx_ret + (num_sample - num_sample_prev) * sample_interval, sample_interval)]
        return result_idx, target_x[result_idx], target_z[result_idx]
    

    def find_preview(self, result_idx):
        return self.curvature_offline[result_idx], self.slope_offline[result_idx], self.bank_offline[result_idx]


    def find_distance(self, pos, track_x, track_z):
        return np.sqrt((track_x - pos[0])**2 + (track_z - pos[2])**2)
    
    
    def global_to_local(self, pos, orientation, track_x, track_z):
        """
        Transform the global coordinate to the local coordinate.
        """
        rotation_matrix = np.array([[np.cos(orientation[2]), -np.sin(orientation[2])], 
                                    [np.sin(orientation[2]), np.cos(orientation[2])]])
        track_x = np.array(track_x)
        track_z = np.array(track_z)
        local_pos = np.dot(rotation_matrix, np.array([track_z - pos[2], track_x - pos[0]]))

        return local_pos[1], local_pos[0]
    
    
    def lidar_2d(self, lb_x_local, lb_z_local, rb_x_local, rb_z_local, num_rays=11, distance_max=100, roi_deg_min=-100, roi_deg_max=100, is_add_flag=True):
        """
        Virtual 2D rangefinder.
        In the vehicle’s body frame, the sensor is positioned on the xyplane,
        and its Region of Interest (ROI) is defined by angles measured with respect to the z-axis.
        A total of 21 rays are uniformly distributed within this range.
        """
        start_time = time.time()
        lb_local = list(zip(lb_x_local, lb_z_local))
        rb_local = list(zip(rb_x_local, rb_z_local))
        lb_boundary = LineString(lb_local)
        rb_boundary = LineString(rb_local)
        closed_boundary = lb_local + rb_local[::-1]
        polygon = Polygon(closed_boundary)
        ray_origin = Point(0, 0)    # Origin = Center of the vehicle (local coord.)

        ### First, check where the car is relatively located.
        ### Left outside: [1 0 0], Inside: [0 1 0], Right outside: [0 0 1], Error: [0 0 0]
        if is_add_flag:
            flag = None
            
            # Check inside.
            inside = polygon.contains(ray_origin)
            if inside:
                flag = [0, 1, 0]
            
            # Check outside - determine left/right.
            else:
                '''
                Select any point B on the right boundary.
                This point must be "right side" of the left boundary.
                Form a linestring object with the given point A and point B.
                If the number of intersected point is odd, then point A is on the "left side".
                Otherwise (if even), then point A is on the "right side".
                '''
                line = LineString([(0, 0), rb_local[len(rb_local) // 2]])
                num = lb_boundary.intersection(line)

                if num.is_empty or (isinstance(num, MultiPoint) and (len(num.geoms)%2 == 0)):               # Even
                    flag = [0, 0, 1]
                elif isinstance(num, Point) or (isinstance(num, MultiPoint) and (len(num.geoms)%2 == 1)):   # Odd
                    flag = [1, 0, 0]
                else:
                    flag = [0, 0, 0]

        ### Next, get pointcloud data.
        angles_deg = np.linspace(roi_deg_min, roi_deg_max, num_rays)
        pcl_distance = []

        for angle_deg in angles_deg:
            ray_direction = np.array([np.cos(np.radians(angle_deg + 90)), np.sin(np.radians(angle_deg + 90))])
            ray_end = Point(ray_origin.x + ray_direction[0] * distance_max, ray_origin.y + ray_direction[1] * distance_max)
            ray_line = LineString([ray_origin, ray_end])
            left_intersection = ray_line.intersection(lb_boundary)
            right_intersection = ray_line.intersection(rb_boundary)
            if left_intersection.is_empty and right_intersection.is_empty:
                pcl_distance.append(distance_max)
                continue

            distances = []
            if not left_intersection.is_empty:
                if isinstance(left_intersection, Point):
                    distances.append(ray_origin.distance(left_intersection))
                elif isinstance(left_intersection, MultiPoint) or isinstance(left_intersection, LineString):
                    for pt in left_intersection.geoms:
                        distances.append(ray_origin.distance(pt))

            if not right_intersection.is_empty:
                if isinstance(right_intersection, Point):
                    distances.append(ray_origin.distance(right_intersection))
                elif isinstance(right_intersection, MultiPoint) or isinstance(right_intersection, LineString):
                    for pt in right_intersection.geoms:  # 수정된 부분
                        distances.append(ray_origin.distance(pt))

            pcl_distance.append(min(distances) if distances else distance_max)

        if is_add_flag and ((flag == [1, 0, 0]) or (flag == [0, 0, 1])):
            pcl_distance = [0 if dist >= 100 else dist for dist in pcl_distance]
            
        calc_time = (time.time() - start_time) * 1000   # ms
        if is_add_flag:
            return flag, pcl_distance, calc_time
        else:
            return None, pcl_distance, calc_time


    def plot_lidar_2d(self, lb_x_local, lb_z_local, rb_x_local, rb_z_local, num_rays=41, distance_max=150, roi_deg_min=-100, roi_deg_max=100):
        """
        Plot the 2D LiDAR data. (Only for debugging)
        """
        flag, pcl_distance, calc_time = self.lidar_2d(lb_x_local, lb_z_local, rb_x_local, rb_z_local, num_rays, distance_max, roi_deg_min, roi_deg_max)
        angles_deg = np.linspace(roi_deg_min, roi_deg_max, num_rays)
        ray_origin = Point(0, 0)

        if self.init_flag:
            self.fig4, self.axs4 = plt.subplots(1, 1)
            self.axs4.scatter([0], [0], c='red', s=120)
            self.rays = [Line2D([], [], color='green', linewidth=1) for _ in range(num_rays)]
            for ray in self.rays:
                self.axs4.add_line(ray)
            self.texts = [self.axs4.text(0, 0, '', color='black') for _ in range(num_rays)]

            # Map background
            self.ln22 = self.axs4.plot(lb_x_local, lb_z_local, color='blue', linestyle='-', linewidth=1, label='Left Boundary')
            self.ln23 = self.axs4.plot(rb_x_local, rb_z_local, color='red', linestyle='-', linewidth=1, label='Right Boundary')
            self.axs4.set_title('2D LiDAR')
            self.axs4.set_xlabel('x [m]')
            self.axs4.set_ylabel('z [m]')
            self.axs4.invert_xaxis()
            self.axs4.axis('equal')
            # self.axs4.set_xlim([-20, 20])
            # self.axs4.set_ylim([-10, 100])
            self.init_flag=False
        
        for angle_deg, distance, ray, text in zip(angles_deg, pcl_distance, self.rays, self.texts):
            ray_direction = np.array([np.cos(np.radians(angle_deg + 90)), np.sin(np.radians(angle_deg + 90))])
            ray_end = ray_origin.x + ray_direction[0] * distance, ray_origin.y + ray_direction[1] * distance
            self.ln22[0].set_data(lb_x_local, lb_z_local)
            self.ln23[0].set_data(rb_x_local, rb_z_local)
            ray.set_data([0, ray_end[0]], [0, ray_end[1]])
            text.set_position(ray_end)
            text.set_text(f'{distance:.2f}m')

        self.fig4.canvas.draw_idle()
        self.axs4.legend()
        plt.pause(0.01)
        print(flag)


    def plot_animation(self, pos_ego, nearest_idx):
        """
        Plot the animation of the vehicle on the track. (Only for debugging)
        """
        x_c = self.track_data[:,0]
        z_c = self.track_data[:,2]

        plt.figure(figsize=(10,6))
        plt.plot(self.track_data[:,0], self.track_data[:,2], linestyle='--')    # Centerline of track
        plt.plot(self.track_data[:,3], self.track_data[:,5], linestyle='--')    # Left boundary of track
        plt.plot(self.track_data[:,6], self.track_data[:,8], linestyle='--')    # Right boundary of track
        plt.scatter(x_c, z_c, c=self.bank_offline, cmap='seismic', vmin=-max(np.abs(min(self.bank_offline)), np.abs(max(self.bank_offline))), vmax=max(np.abs(min(self.bank_offline)), np.abs(max(self.bank_offline))))
        plt.colorbar(label='slope')
        plt.title('Bank visualization of track : ' + self.track_name)
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.gca().invert_xaxis()
        plt.axis('equal')