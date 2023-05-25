import numpy as np

DEBUG = False

def debug(on=False, *args):
    if on:
        print("[DEBUG] " + " ".join(map(str,args)))

def transform_point(X, RT):
    X_hom = np.append(X, 1)

    debug(DEBUG, "X_hom: ", X_hom)
    debug(DEBUG, "RT: ", RT)
    
    return np.dot(RT, X_hom)

def project_point(X, fx, fy, cx, cy):
    # Construct the camera matrix K
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # Multiply the camera matrix K with the 3D point
    uvw = K @ X
    
    # Convert to regular 2D coordinates
    u, v, w = uvw
    x = u / w
    y = v / w
    
    return x, y

def project_point_distort(X, Y, Z, fx, fy, cx, cy, omega):
    # Normalize the 3D point
    X_normalized = X / Z
    Y_normalized = Y / Z

    # Compute the undistorted radius
    r_undistorted = np.sqrt(X_normalized**2 + Y_normalized**2)

    # Compute the distorted radius
    if omega == 0:
        r_distorted = r_undistorted
    else:
        r_distorted = 1/omega * np.arctan(2 * r_undistorted * np.tan(omega/2))

    # Distort the normalized point
    X_distorted = (r_distorted / r_undistorted * X_normalized) if r_undistorted != 0 else X_normalized
    Y_distorted = (r_distorted / r_undistorted * Y_normalized) if r_undistorted != 0 else Y_normalized

    # Apply the pinhole camera model
    u_distorted = fx * X_distorted + cx
    v_distorted = fy * Y_distorted + cy

    return u_distorted, v_distorted

def back_project_point(u, v, d, fx, fy, cx, cy):
    # Construct the camera matrix K
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # Convert the 2D point to homogeneous coordinates
    uv_homogeneous = np.array([u, v, 1])
    
    # Multiply the inverse of the camera matrix K with the 2D point
    XYZ_prime = np.linalg.inv(K) @ uv_homogeneous
    
    # Normalize the vector
    XYZ_prime_normalized = XYZ_prime / np.linalg.norm(XYZ_prime)
    
    # Scale by depth
    XYZ = d * XYZ_prime_normalized
    
    return XYZ

def back_project_distorted_point(u, v, d, fx, fy, cx, cy, omega):
    # Construct the camera matrix K
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    
    # Compute the distorted radius
    r_distorted = np.sqrt(x_n**2 + y_n**2)
    
    # Handle the case where r is 0 to avoid division by zero
    if r_distorted == 0:
        r_undistorted = 0
        x_distorted = 0
        y_distorted = 0
    else:
        # Apply the inverse distortion function to compute the undistorted radius
        r_undistorted = np.tan(r_distorted * omega) / (2 * np.tan(omega / 2))
    
        # Compute the undistorted coordinates
        x_distorted = x_n / r_distorted
        y_distorted = y_n / r_distorted
    
    x_undistorted = x_distorted * r_undistorted
    y_undistorted = y_distorted * r_undistorted
    
    # Convert the 2D point to camera coordinates
    xyz_camera = np.array([x_undistorted, y_undistorted, 1])
    
    # Normalize the vector
    xyz_camera_normalized = xyz_camera / np.linalg.norm(xyz_camera)
    
    # Scale by depth
    P = d * xyz_camera_normalized
    
    return P

def parse_camera(parts):
    model = parts[0]
    width, height = map(int, parts[1:3])
    
    if model == "fov":
        fx, fy, cx, cy, omega = map(float, parts[3:])
        return width, height, fx, fy, cx, cy, omega
    else:
        fx, fy, cx, cy = map(float, parts[3:])
        return width, height, fx, fy, cx, cy, None
    
def main():
    first_camera_input = input().split()
    first_camera_model = (first_camera_input[0].strip()=="pinhole")
    first_camera_params = parse_camera(first_camera_input)

    second_camera_input = input().split()
    second_camera_model = (second_camera_input[0].strip()=="pinhole")
    second_camera_params = parse_camera(second_camera_input)

    transformation = np.array([np.array(list(map(float, input().split()))) for _ in range(3)])
    transformation = np.vstack([transformation, [0, 0, 0, 1]])  # Add the fourth row

    points = np.array([np.array(list(map(float, input().split()))) for _ in range(9)])

    debug(DEBUG, "[1ST CAMERA]")
    debug(DEBUG, first_camera_input[0].strip(), first_camera_model)
    debug(DEBUG, first_camera_params)

    debug(DEBUG, "[2ND CAMERA]")
    debug(DEBUG, second_camera_input[0].strip(), second_camera_model)
    debug(DEBUG, second_camera_params)

    debug(DEBUG, '\ntransformation:')
    debug(DEBUG, transformation)
    # debug(DEBUG, '\npoints:')
    # debug(DEBUG, points)

    debug(DEBUG, '\n================================================================\n')

    # Project each point
    for u, v, d in points:
        if first_camera_params[-1] is None:  
            X = back_project_point(u, v, d, *first_camera_params[2:-1])
        else: # FOV model
            X = back_project_distorted_point(u, v, d, *first_camera_params[2:])
            
        debug(DEBUG, 'back projected: ', X)
        X_ = transform_point(X, transformation)
        debug(DEBUG, 'transformation RT: ', X_)
        
        if second_camera_params[-1] is not None:  # FOV model
            u, v = project_point_distort(*X, *second_camera_params[2:])
        else:
            u, v = project_point(X_[:3], *second_camera_params[2:-1])
            debug(DEBUG, 'generic projection: ', u, v)
            
        debug(DEBUG, "re-projection result: ", u, v)
        debug(DEBUG, "\n")
        if 0 <= u < second_camera_params[0] and 0 <= v < second_camera_params[1]:  # within image bounds
            print(round(u, 3), round(v, 3))
        else:
            print("OB")

if __name__ == "__main__":
    main()