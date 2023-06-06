import numpy as np

DEBUG = False

def get_matrix_q_i(point_3d, points_2d):
    Xi, Yi, Zi = point_3d
    ui, vi = points_2d

    return np.array([[Xi, Yi, Zi, 1,  0,  0,  0,  0, -ui*Xi, -ui*Yi, -ui*Zi, -ui],
                        [ 0,  0,  0,  0, Xi, Yi, Zi, 1, -vi*Xi, -vi*Yi, -vi*Zi, -vi]])

def solve_projection_matrix(Q):
    # Perform Singular Value Decomposition
    U, s, V = np.linalg.svd(Q)
    
    # The solution is the last column of V (or last row of V.T)
    M = V[-1]
    
    # Reshape M into a 3x4 matrix
    P = M.reshape(3, 4)
    
    return P

def decompose_projection_matrix_qr(M):
    debug(DEBUG, 'M:')
    debug(DEBUG, M)
    
    # Separate the last column of M (the translation vector t)
    M_t = M[:, 3]
    K_R = M[:, :3]

    P = np.fliplr(np.eye(K_R.shape[0]))

    A_hat = np.dot(P, K_R)

    # Perform QR decomposition on A_hat transpose to get Q_hat and R_hat
    Q_hat, R_hat = np.linalg.qr(A_hat.T)

    R = np.dot(P, Q_hat.T)
    K = np.dot(np.dot(P, R_hat.T), P)

    # Normalize K so that K[2, 2] is 1
    scale = 1 / K[2, 2]
    K = K * scale 
    
    # Ensure that the diagonal elements of K are all positive
    T = np.diag(np.sign(np.diag(K)))
    
    # Correct K and RT
    K = np.dot(K, T)
    R = -1*np.dot(T, R)

    # Compute t from M_t and K
    t = np.linalg.inv(K) @ M_t
    t = -1 * t * scale
    t = t.reshape(-1, 1)
    
    return K, R, t

def debug(on=False, *args):
    if on:
        print("[DEBUG] " + " ".join(map(str,args)))

def main():
    point_3d = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1]])
    
    points_2d = np.array([np.array(list(map(float, input().split()))) for _ in range(8)])
    debug(DEBUG, 'point_3d:')
    debug(DEBUG, point_3d)

    Q = np.empty((0,12), float)
    for i in range(points_2d.shape[0]):
        Q = np.append(Q, get_matrix_q_i(point_3d[i], points_2d[i]), axis=0)
    
    P = solve_projection_matrix(Q)
    K, R, t = decompose_projection_matrix_qr(P)
    
    for r in K:
        print("{}\t{}\t{}".format(round(r[0]), round(r[1]), round(r[2])))
    print("")
    transformation = np.hstack([R, t])
    
    if transformation[2,3] < 0:
        transformation *= -1
    
    for r in transformation:
        print("{}\t{}\t{}\t{}".format(round(r[0], 7), round(r[1], 7), round(r[2], 7), round(r[3], 7)))

if __name__ == "__main__":
    main()