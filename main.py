import numpy as np


class MethodError(Exception):
    def __init__(self, comment="", message="The method is not applicable!"):
        if comment != "":
            self.message = "\n" + comment + "\n" + message
        else:
            self.message = message
        super().__init__(self.message)

def simplex_method(C, A, b, accuracy):
    """
        C: Coefficients for the objective function as a 1D numpy array.
        A: Coefficients for the inequalities as a 2D numpy array.
        b: Right-hand side numbers as a 1D numpy array.
        accuracy: Approximation accuracy as a float.

        Returns:
        - A string "The method is not applicable!" if the method is not applicable.
        - A tuple containing the solution vector x and the maximum value
          of the objective function, otherwise.
    """

    # Preliminary checks
    m, n = A.shape
    new_matrix = A

    if m != len(b) or n != len(C):
        raise MethodError("Shape of matrix with coefficients for the inequalities is not valid")
    if accuracy <= 0:
        raise MethodError("Accuracy value is not valid")

    superplus_vars = []
    for i in range(m):
        if b[i] < 0:
            new_matrix[i] *= -1
            new_column = np.zeros(m)
            new_column[i] = -1
            b[i] *= -1
            new_matrix = np.column_stack((new_matrix, new_column))
            superplus_vars.append(i)

    # Basis to keep track of basic variables
    basis = []

    for i in range(len(new_matrix)):
        if i in superplus_vars:
            for j in range(len(new_matrix[i])):
                if new_matrix[i][j] == 1 and all((new_matrix[p][j] == 0 or p == i) for p in range(len(new_matrix))):
                    basis.append(j)
                    break
            else:
                raise MethodError("Simplex method can not be used to solve this task")
        else:
            new_column = np.zeros(m)
            new_column[i] = 1
            new_matrix = np.column_stack((new_matrix, new_column))
            basis.append(len(new_matrix[0])-1)

    if len(basis) < m:
        raise MethodError("Simplex method can not be used to solve this task")
    # Create initial tableau
    tableau = np.zeros((m + 1, len(new_matrix[0]) + 1))
    tableau[:-1, :len(new_matrix[0])] = new_matrix
    tableau[:-1, -1] = b
    tableau[-1, :n] = -C



    # Main loop
    while True:
        # Choose entering variable
        entering = np.argmin(tableau[-1, :-1])
        if tableau[-1, entering] >= -accuracy:
            break

        # Choose leaving variable
        min_ratio = np.inf
        leaving = -1
        for i in range(m):
            if tableau[i, entering] > 0:
                ratio = tableau[i, -1] / tableau[i, entering]
                if ratio < min_ratio:
                    min_ratio = ratio
                    leaving = i

        if leaving == -1:
            raise MethodError("Leaving variable can not be defined")

        # Pivot
        tableau[leaving, :] /= tableau[leaving, entering]
        for i in range(m + 1):
            if i != leaving:
                tableau[i, :] -= tableau[i, entering] * tableau[leaving, :]

        # Update basis
        basis[leaving] = entering

    # Extract solution
    x = np.zeros(n)
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = tableau[i, -1]

    return x, tableau[-1, -1]


# Example usage:
"""
C = np.array([3, 4])
A = np.array([[1, 1],
              [2, 3]])
b = np.array([55, 120])
accuracy = 1e-6
"""

# Read input variables
print("Coefficients for the objective function as a 1D numpy array: (space - separator)")
C = np.array(list(map(float, input().split(" "))))
print("Number of inequalities:")
n = int(input())
A = []
for i in range(n):
    print(f"Inequality #{i + 1}:")
    A.append(list(map(float, input().split())))
A = np.array(A)
print("Right-hand side numbers as a 1D numpy array: (space - separator):")
b = np.array(list(map(float, input().split(" "))))
print("Accuracy:")
accuracy = float(input())

print("\nResults:\n")

result = simplex_method(C, A, b, accuracy)



if result == "The method is not applicable!":
    print(result)
else:
    x_star, obj_value = result
    print(f"Solution vector x*: {x_star}\n")
    print(f"Objective function value: {obj_value}")
