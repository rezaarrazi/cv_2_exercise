def swap_rows(matrix, row1, row2):
    matrix[row1], matrix[row2] = matrix[row2], matrix[row1]

def multiply_row(matrix, row, scalar):
    matrix[row] = [scalar * x for x in matrix[row]]

def add_rows(matrix, row1, row2, scalar):
    matrix[row1] = [x + scalar * y for x, y in zip(matrix[row1], matrix[row2])]

def find_inverse(matrix, n):
    inverse = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    operations = []
    for i in range(n):
        if matrix[i][i] == 0:
            for j in range(i + 1, n):
                if matrix[j][i] != 0:
                    swap_rows(matrix, i, j)
                    swap_rows(inverse, i, j)
                    operations.append(f"S {i} {j}")
                    break
            else:
                return operations, None

        pivot = matrix[i][i]
        if pivot != 1:
            scalar = 1 / pivot
            multiply_row(matrix, i, scalar)
            multiply_row(inverse, i, scalar)
            operations.append(f"M {i} {scalar}")

        for j in range(n):
            if j != i:
                scalar = -matrix[j][i]
                add_rows(matrix, j, i, scalar)
                add_rows(inverse, j, i, scalar)
                if scalar != 0:
                    operations.append(f"A {j} {i} {scalar}")

    return operations, inverse

def format_output(matrix):
    return "\n".join(" ".join(f"{x:.16f}" for x in row) for row in matrix)

def main():
    n = int(input())
    matrix = [list(map(float, input().split())) for _ in range(n)]

    operations, inverse = find_inverse(matrix, n)

    print("\n".join(operations))

    if inverse is None:
        print("DEGENERATE")
    else:
        print("SOLUTION")
        print(format_output(inverse))

if __name__ == "__main__":
    main()
