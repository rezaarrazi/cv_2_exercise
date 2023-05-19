import numpy as np

def affine_subspaces_intersection(points1, points2):
    # Calculate the reference points p and q
    p = points1[0]
    q = points2[0]

    # Calculate the direction vectors for both subspaces (A and B matrices)
    A = np.array(points1[1:]) - p
    B = np.array(points2[1:]) - q

    # Create the matrix C
    C = np.vstack((A, -B))
    C_transpose = C.T

    # Solve the linear system C * z = q - p
    try:
        z, residuals, rank, s = np.linalg.lstsq(C_transpose, q - p, rcond=None)
    except ValueError:
        return 'N', None
    
    # Check if the system has a unique solution and if the residuals are close to zero
    if rank == np.linalg.matrix_rank(np.hstack((C.T, (q - p).reshape(-1, 1)))) and np.all(np.isclose(residuals, 0)):
        # Separate t and u from z
        t = z[:A.shape[0]]
        u = z[A.shape[0]:]

        # Calculate the intersection point x
        x = p + A.T @ t
        return 'Y', x
    else:
        return 'N', None

def main():
    m = int(input())
    
    n_A = int(input())
    person_A_points = np.array([np.array(list(map(float, input().split()[1:]))) for _ in range(n_A)])
    
    n_B = int(input())
    person_B_points = np.array([np.array(list(map(float, input().split()[1:]))) for _ in range(n_B)])
    
    answer, point = affine_subspaces_intersection(person_A_points, person_B_points)
    
    if answer == 'Y':
        rounded_point = [round(x, 6) for x in point]
        print(answer, *rounded_point)
    else:
        print(answer)

if __name__ == "__main__":
    main()
