{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Linear Algebra Homework: Basis and Orthogonal Projection\n",
    "## Objective\n",
    "Implement functions to:\n",
    "1. Extract a basis from a set of spanning vectors\n",
    "2. Compute the orthogonal projection matrix onto a subspace"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import numpy as np\n",
    "import sympy as sp\n",
    "\n",
    "def compute_basis(vectors):\n",
    "    \"\"\"\n",
    "    Compute a basis for the subspace spanned by the input vectors using RREF method.\n",
    "    \n",
    "    Parameters:\n",
    "    vectors (list): List of vectors (as numpy arrays or SymPy matrices)\n",
    "    \n",
    "    Returns:\n",
    "    list: Basis vectors\n",
    "    \"\"\"\n",
    "    # Convert vectors to SymPy matrix\n",
    "    matrix = sp.Matrix(vectors).T\n",
    "    \n",
    "    # Compute RREF\n",
    "    rref_matrix, pivot_columns = matrix.rref()\n",
    "    \n",
    "    # Extract basis vectors\n",
    "    basis_vectors = []\n",
    "    for col in pivot_columns:\n",
    "        basis_vectors.append(rref_matrix[:, col])\n",
    "    \n",
    "    return basis_vectors\n",
    "\n",
    "def compute_projection_matrix(basis):\n",
    "    \"\"\"\n",
    "    Compute the orthogonal projection matrix onto the subspace defined by the basis.\n",
    "    \n",
    "    Parameters:\n",
    "    basis (list): List of basis vectors\n",
    "    \n",
    "    Returns:\n",
    "    numpy.ndarray: Projection matrix P\n",
    "    \"\"\"\n",
    "    # Convert basis to matrix B\n",
    "    B = sp.Matrix(basis).T\n",
    "    \n",
    "    # Compute B^T * B\n",
    "    BT_B = B.T * B\n",
    "    \n",
    "    # Check if B^T * B is invertible\n",
    "    try:\n",
    "        BT_B_inv = BT_B**-1\n",
    "    except Exception as e:\n",
    "        print(\"Error: B^T * B is not invertible.\")\n",
    "        raise e\n",
    "    \n",
    "    # Compute projection matrix P = B * (B^T * B)^-1 * B^T\n",
    "    P = B * BT_B_inv * B.T\n",
    "    \n",
    "    return np.array(P).astype(float)\n",
    "\n",
    "def project_vector(v, P):\n",
    "    \"\"\"\n",
    "    Project a vector onto the subspace defined by projection matrix P.\n",
    "    \n",
    "    Parameters:\n",
    "    v (numpy.ndarray): Vector to project\n",
    "    P (numpy.ndarray): Projection matrix\n",
    "    \n",
    "    Returns:\n",
    "    numpy.ndarray: Projected vector\n",
    "    \"\"\"\n",
    "    return P @ v"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Example Usage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Example vectors\n",
    "vectors = [\n",
    "    [1, 2, 3],\n",
    "    [2, 4, 6],\n",
    "    [3, 6, 9]\n",
    "]\n",
    "\n",
    "# Compute Basis\n",
    "basis = compute_basis(vectors)\n",
    "print(\"Basis Vectors:\")\n",
    "for vec in basis:\n",
    "    print(vec)\n",
    "\n",
    "# Compute Projection Matrix\n",
    "P = compute_projection_matrix(basis)\n",
    "print(\"\\nProjection Matrix P:\")\n",
    "print(P)\n",
    "\n",
    "# Project a vector\n",
    "v = np.array([4, 5, 6])\n",
    "projected_v = project_vector(v, P)\n",
    "print(\"\\nOriginal Vector:\", v)\n",
    "print(\"Projected Vector:\", projected_v)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Verification and Validation\n",
    "\n",
    "### Checking Linear Independence of Basis Vectors\n",
    "- The RREF method ensures that the basis vectors are linearly independent\n",
    "- Basis vectors form a minimal spanning set for the subspace"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
