{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cap. 2, Sec. 0 #\n",
    "### Introduction to NumPy ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'1.16.4'"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "numpy.__version__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cap. 2, Sec. 1 #\n",
    "### Understanding Data Types in Python ###"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Ejemplo en C\n",
    "\n",
    "int r=0;\n",
    "\n",
    "for (int i=0; i<100; i++}\n",
    "\n",
    "{ r+=i; }"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Ejemplo en python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "r=0\n",
    "for i in range(100):\n",
    "    r+=i"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Tipos de variable inferidos dinámicamente "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "x=3\n",
    "x=\"tres\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Listas y asignación de los tipos de valiable de sus elementos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "L=list(range(10))\n",
    "L"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "int"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(L[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Una lista de \"strings\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "L2=[str(c) for c in L]\n",
    "L2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "str"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(L2[3])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Una lista con una combinación heterogenea de tipos de variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[bool, float, int, str]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "L3=[True, 2.0 , 3, \"4\"]\n",
    "[type(i) for i in L3]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Usando \"array\" para crear arreglos de un tipo de variable fijo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array('i', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import array\n",
    "L=list(range(10))\n",
    "A=array.array(\"i\", L)\n",
    "A"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Creando arreglos de numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.array([range(10)])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Los arreglos de numpy tienen un tipo de variable fijo a fin de reducir el uso de memoria. Por ello al tener solo un float todos los elementos del arreglo pasan a serlo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1. , 2. , 3. , 4.5])"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.array([1, 2, 3, 4.5])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Se puede elejir de forma explícita el tipo de variable del arreglo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 2., 3., 4.], dtype=float32)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.array([1,2,3,4],dtype=\"float32\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2, 3, 4],\n",
       "       [4, 5, 6],\n",
       "       [6, 7, 8]])"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.array([range(i,i+3) for i in [2,4,6]])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Creando arreglos de tamaño dado llenos con un valor y un tipo de variable específicos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 0., 0., 0., 0., 0., 0., 0.])"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.zeros(8, dtype=int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1.]])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.ones((3, 5), dtype=float)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Llenando él arreglo con un valor personalizado"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[4.11, 4.11, 4.11, 4.11],\n",
       "       [4.11, 4.11, 4.11, 4.11]])"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.full((2, 4), 4.11)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* O llenandolo con una secuencia lineal de intervalo personalizable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  3,  6,  9, 12, 15])"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.arange(0, 16, 3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* O una cantidad determinada de elementos distribuidos en un intervalo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.        , 0.14285714, 0.28571429, 0.42857143, 0.57142857,\n",
       "       0.71428571, 0.85714286, 1.        ])"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.linspace(0, 1, 8)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Arreglo relleno con valores aleatorios entre 0 y 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.99139927, 0.74532733, 0.81392335],\n",
       "       [0.63946842, 0.81039526, 0.29964512],\n",
       "       [0.06006319, 0.85503271, 0.58237934]])"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.random.random((3,3))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Similar pero con una media y desviación estándar dadas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.24711923,  0.48083848, -0.51505104, -4.21875917],\n",
       "       [ 0.38555193,  3.90091675, -0.94865566, -2.38022398],\n",
       "       [ 0.9393923 , -1.53448923,  0.21636696,  1.79404948],\n",
       "       [-0.36565325, -2.51801476, -3.02571011,  2.14707402]])"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.random.normal(0,2, (4,4))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Arreglo de enteros aleatorios"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[9, 6, 5, 7],\n",
       "       [6, 4, 2, 7]])"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.random.randint(0,10,(2,4))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Una matriz identidad"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 0., 0., 0.],\n",
       "       [0., 1., 0., 0.],\n",
       "       [0., 0., 1., 0.],\n",
       "       [0., 0., 0., 1.]])"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.eye(4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Un arreglo sin inicializar"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([3.31033942e-33, 6.97666413e-76, 3.45618036e-86])"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.empty(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cap. 2, Sec.2 #\n",
    "### The Basics of NumPy Arrays ###"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Creando 3 arreglos aleatorios de diferente dimensión y usando la instrucción \"seed\" para asegurar la reproducibilidad"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "np.random.seed(0)\n",
    "x1=np.random.randint(10, size=6)\n",
    "x2 = np.random.randint(10, size=(3, 4))\n",
    "x3 = np.random.randint(10, size=(3, 4, 5)) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Cada arreglo tiene los atributos de número de dimensiones, tamaño de las dimensiones y tamaño total"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x3 ndim:  3\n",
      "x3 shape: (3, 4, 5)\n",
      "x3 size:  60\n"
     ]
    }
   ],
   "source": [
    "print(\"x3 ndim: \", x3.ndim)\n",
    "print(\"x3 shape:\", x3.shape)\n",
    "print(\"x3 size: \", x3.size)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Se puede leer también el tipo de variable de un arreglo determinado"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tipo de variable de x3: int64\n"
     ]
    }
   ],
   "source": [
    "print(\"Tipo de variable de x3:\", x3.dtype)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* O el tamaño en Bytes de cada elemento o del areeglo completo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "itemsize: 8 bytes\n",
      "nbytes: 480 bytes\n"
     ]
    }
   ],
   "source": [
    "print(\"itemsize:\", x3.itemsize, \"bytes\")\n",
    "print(\"nbytes:\", x3.nbytes, \"bytes\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Se pueden leer los valores de un elemento específico de un arreglo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 0, 3, 3, 7, 9])"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x1[3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x1[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Es posible identificar los elementos en sentido inverso con índices negativos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "9"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x1[-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x1[-6]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* En areeglos de dimensión mayor deben indicarse ambos índices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[3, 5, 2, 4],\n",
       "       [7, 6, 8, 8],\n",
       "       [1, 6, 7, 7]])"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x2[0,0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x2[2,3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x2[-1,3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "x2[0,1]=4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[3, 4, 2, 4],\n",
       "       [7, 6, 8, 8],\n",
       "       [1, 6, 7, 7]])"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* No se piede insertar un tipo de variable que no sea el correspondiente al arreglo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "x2[0,0]=7.1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x2[0,0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Se puede acceder a sub-arreglos dentro de un arreglo específico"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.arange(10)\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4])"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x[5:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([4, 5, 6])"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x[4:7]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 2, 4, 6, 8])"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x[::2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 3, 5, 7, 9])"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x[1::2]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Los valores de intervalo negativo recorren en sentido opuesto el arreglo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x[::-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 3, 1])"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x[5::-2]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* En los sub-arreglos multi-dimensionales se separa cada dimensión por una coma"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[7, 4, 2, 4],\n",
       "       [7, 6, 8, 8],\n",
       "       [1, 6, 7, 7]])"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[7, 4, 2],\n",
       "       [7, 6, 8]])"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x2[:2, :3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[7, 2],\n",
       "       [7, 8],\n",
       "       [1, 7]])"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x2[:3, ::2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[7, 7, 6, 1],\n",
       "       [8, 8, 6, 7],\n",
       "       [4, 2, 4, 7]])"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x2[::-1, ::-1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Se pueden aislar filas o columnas específicas de un arreglo multi-dimensional"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[7 7 1]\n"
     ]
    }
   ],
   "source": [
    "print(x2[:, 0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[7 4 2 4]\n"
     ]
    }
   ],
   "source": [
    "print(x2[0, :])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Suprimeindo una fila o columna de forma más compacta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[7 4 2 4]\n"
     ]
    }
   ],
   "source": [
    "print(x2[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Modificar los arreglos definidos como sub-arreglos de otros modifica al original"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[7 4 2 4]\n",
      " [7 6 8 8]\n",
      " [1 6 7 7]]\n"
     ]
    }
   ],
   "source": [
    "print(x2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[7 4]\n",
      " [7 6]]\n"
     ]
    }
   ],
   "source": [
    "x2_sub = x2[:2, :2]\n",
    "print(x2_sub)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[99  4]\n",
      " [ 7  6]]\n"
     ]
    }
   ],
   "source": [
    "x2_sub[0, 0] = 99\n",
    "print(x2_sub)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[99  4  2  4]\n",
      " [ 7  6  8  8]\n",
      " [ 1  6  7  7]]\n"
     ]
    }
   ],
   "source": [
    "print(x2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Con el comando \"copy\" el arreglo copia no modifica el original"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[99  4]\n",
      " [ 7  6]]\n"
     ]
    }
   ],
   "source": [
    "x2_sub_copy = x2[:2, :2].copy()\n",
    "print(x2_sub_copy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[42  4]\n",
      " [ 7  6]]\n"
     ]
    }
   ],
   "source": [
    "x2_sub_copy[0, 0] = 42\n",
    "print(x2_sub_copy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[99  4  2  4]\n",
      " [ 7  6  8  8]\n",
      " [ 1  6  7  7]]\n"
     ]
    }
   ],
   "source": [
    "print(x2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Con la instucción \"reshape\" se puede modificar la forma de un arreglo, incluyendo su dimensión"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1 2 3]\n",
      " [4 5 6]\n",
      " [7 8 9]]\n"
     ]
    }
   ],
   "source": [
    "grid = np.arange(1, 10).reshape((3, 3))\n",
    "print(grid)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Usando \"newaxis\" se convierte un arreglo uni-dimensional a uno bi-dimensional columna o fila"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3]])"
      ]
     },
     "execution_count": 96,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.array([1, 2, 3])\n",
    "x.reshape((1, 3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3]])"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x[np.newaxis, :]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1],\n",
       "       [2],\n",
       "       [3]])"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.reshape((3, 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1],\n",
       "       [2],\n",
       "       [3]])"
      ]
     },
     "execution_count": 99,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x[:, np.newaxis]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* El comando \"concatenate\" justamente concatena, o combina, dos arreglos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 3, 2, 1])"
      ]
     },
     "execution_count": 100,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.array([1, 2, 3])\n",
    "y = np.array([3, 2, 1])\n",
    "np.concatenate([x, y])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 1  2  3  3  2  1 99 99 99]\n"
     ]
    }
   ],
   "source": [
    "z = [99, 99, 99]\n",
    "print(np.concatenate([x, y, z]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [],
   "source": [
    "grid = np.array([[1, 2, 3],\n",
    "                 [4, 5, 6]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3],\n",
       "       [4, 5, 6],\n",
       "       [1, 2, 3],\n",
       "       [4, 5, 6]])"
      ]
     },
     "execution_count": 103,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.concatenate([grid, grid])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3, 1, 2, 3],\n",
       "       [4, 5, 6, 4, 5, 6]])"
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.concatenate([grid, grid], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Para arreglos multi-dimensionales puede ser mas claro usar las instrucciones \"vstack\" y \"hstack\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3],\n",
       "       [9, 8, 7],\n",
       "       [6, 5, 4]])"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.array([1, 2, 3])\n",
    "grid = np.array([[9, 8, 7],\n",
    "                 [6, 5, 4]])\n",
    "np.vstack([x, grid])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 9,  8,  7, 99],\n",
       "       [ 6,  5,  4, 99]])"
      ]
     },
     "execution_count": 106,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = np.array([[99],\n",
    "              [99]])\n",
    "np.hstack([grid, y])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* De forma análoga se pueden separar arreglos con los comandos \"split\" \"hsplit\" y \"vsplit\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 2 3] [99 99] [3 2 1]\n"
     ]
    }
   ],
   "source": [
    "x = [1, 2, 3, 99, 99, 3, 2, 1]\n",
    "x1, x2, x3 = np.split(x, [3, 5])\n",
    "print(x1, x2, x3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0,  1,  2,  3],\n",
       "       [ 4,  5,  6,  7],\n",
       "       [ 8,  9, 10, 11],\n",
       "       [12, 13, 14, 15]])"
      ]
     },
     "execution_count": 108,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid = np.arange(16).reshape((4, 4))\n",
    "grid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2 3]\n",
      " [4 5 6 7]]\n",
      "[[ 8  9 10 11]\n",
      " [12 13 14 15]]\n"
     ]
    }
   ],
   "source": [
    "upper, lower = np.vsplit(grid, [2])\n",
    "print(upper)\n",
    "print(lower)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0  1]\n",
      " [ 4  5]\n",
      " [ 8  9]\n",
      " [12 13]]\n",
      "[[ 2  3]\n",
      " [ 6  7]\n",
      " [10 11]\n",
      " [14 15]]\n"
     ]
    }
   ],
   "source": [
    "left, right = np.hsplit(grid, [2])\n",
    "print(left)\n",
    "print(right)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cap. 2, Sec, 3 #\n",
    "### Computation on NumPy Arrays: Universal Functions ###"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Python resulta relativamente lento al realizar un gran número de pequeñas operaciones debido a la forma en la que construye sus variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.16666667, 1.        , 0.25      , 0.25      , 0.125     ])"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "np.random.seed(0)\n",
    "\n",
    "def compute_reciprocals(values):\n",
    "    output = np.empty(len(values))\n",
    "    for i in range(len(values)):\n",
    "        output[i] = 1.0 / values[i]\n",
    "    return output\n",
    "        \n",
    "values = np.random.randint(1, 10, size=5)\n",
    "compute_reciprocals(values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3.05 s ± 23.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)\n"
     ]
    }
   ],
   "source": [
    "big_array = np.random.randint(1, 100, size=1000000)\n",
    "%timeit compute_reciprocals(big_array)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Usando operaciones vectorizadas el cálculo debería ser más eficiente"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.16666667 1.         0.25       0.25       0.125     ]\n",
      "[0.16666667 1.         0.25       0.25       0.125     ]\n"
     ]
    }
   ],
   "source": [
    "print(compute_reciprocals(values))\n",
    "print(1.0 / values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3.5 ms ± 66.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)\n"
     ]
    }
   ],
   "source": [
    "%timeit (1.0 / big_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.        , 0.5       , 0.66666667, 0.75      , 0.8       ])"
      ]
     },
     "execution_count": 115,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.arange(5) / np.arange(1, 6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  1,   2,   4],\n",
       "       [  8,  16,  32],\n",
       "       [ 64, 128, 256]])"
      ]
     },
     "execution_count": 116,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.arange(9).reshape((3, 3))\n",
    "2 ** x"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Operaciones fundamentales de la aritmética de vectores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x     = [0 1 2 3]\n",
      "x + 5 = [5 6 7 8]\n",
      "x - 5 = [-5 -4 -3 -2]\n",
      "x * 2 = [0 2 4 6]\n",
      "x / 2 = [0.  0.5 1.  1.5]\n",
      "x // 2 = [0 0 1 1]\n"
     ]
    }
   ],
   "source": [
    "x = np.arange(4)\n",
    "print(\"x     =\", x)\n",
    "print(\"x + 5 =\", x + 5)\n",
    "print(\"x - 5 =\", x - 5)\n",
    "print(\"x * 2 =\", x * 2)\n",
    "print(\"x / 2 =\", x / 2)\n",
    "print(\"x // 2 =\", x // 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-x     =  [ 0 -1 -2 -3]\n",
      "x ** 2 =  [0 1 4 9]\n",
      "x % 2  =  [0 1 0 1]\n"
     ]
    }
   ],
   "source": [
    "print(\"-x     = \", -x)\n",
    "print(\"x ** 2 = \", x ** 2)\n",
    "print(\"x % 2  = \", x % 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-1.  , -2.25, -4.  , -6.25])"
      ]
     },
     "execution_count": 119,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "-(0.5*x + 1) ** 2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* El símbolo \"+\" resume la operación \"add\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 3, 4, 5])"
      ]
     },
     "execution_count": 120,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.add(x, 2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* También existe la operación de valor absoluto"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 1, 0, 1, 2])"
      ]
     },
     "execution_count": 121,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.array([-2, -1, 0, 1, 2])\n",
    "abs(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* La función de numpy equivalente es:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 1, 0, 1, 2])"
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.absolute(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 1, 0, 1, 2])"
      ]
     },
     "execution_count": 123,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.abs(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* También puede usarse en números complejos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5., 5., 2., 1.])"
      ]
     },
     "execution_count": 124,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])\n",
    "np.abs(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Numpy también incluye funciones trigonométricas, para usarlas primero hay que definir un arreglo de ángulos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [],
   "source": [
    "theta = np.linspace(0, np.pi, 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "theta      =  [0.         1.57079633 3.14159265]\n",
      "sin(theta) =  [0.0000000e+00 1.0000000e+00 1.2246468e-16]\n",
      "cos(theta) =  [ 1.000000e+00  6.123234e-17 -1.000000e+00]\n",
      "tan(theta) =  [ 0.00000000e+00  1.63312394e+16 -1.22464680e-16]\n"
     ]
    }
   ],
   "source": [
    "print(\"theta      = \", theta)\n",
    "print(\"sin(theta) = \", np.sin(theta))\n",
    "print(\"cos(theta) = \", np.cos(theta))\n",
    "print(\"tan(theta) = \", np.tan(theta))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x         =  [-1, 0, 1]\n",
      "arcsin(x) =  [-1.57079633  0.          1.57079633]\n",
      "arccos(x) =  [3.14159265 1.57079633 0.        ]\n",
      "arctan(x) =  [-0.78539816  0.          0.78539816]\n"
     ]
    }
   ],
   "source": [
    "x = [-1, 0, 1]\n",
    "print(\"x         = \", x)\n",
    "print(\"arcsin(x) = \", np.arcsin(x))\n",
    "print(\"arccos(x) = \", np.arccos(x))\n",
    "print(\"arctan(x) = \", np.arctan(x))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* También existen las funciones exponenciales y logarítmicas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x     = [1, 2, 3]\n",
      "e^x   = [ 2.71828183  7.3890561  20.08553692]\n",
      "2^x   = [2. 4. 8.]\n",
      "3^x   = [ 3  9 27]\n"
     ]
    }
   ],
   "source": [
    "x = [1, 2, 3]\n",
    "print(\"x     =\", x)\n",
    "print(\"e^x   =\", np.exp(x))\n",
    "print(\"2^x   =\", np.exp2(x))\n",
    "print(\"3^x   =\", np.power(3, x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x        = [1, 2, 4, 10]\n",
      "ln(x)    = [0.         0.69314718 1.38629436 2.30258509]\n",
      "log2(x)  = [0.         1.         2.         3.32192809]\n",
      "log10(x) = [0.         0.30103    0.60205999 1.        ]\n"
     ]
    }
   ],
   "source": [
    "x = [1, 2, 4, 10]\n",
    "print(\"x        =\", x)\n",
    "print(\"ln(x)    =\", np.log(x))\n",
    "print(\"log2(x)  =\", np.log2(x))\n",
    "print(\"log10(x) =\", np.log10(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "exp(x) - 1 = [0.         0.0010005  0.01005017 0.10517092]\n",
      "log(1 + x) = [0.         0.0009995  0.00995033 0.09531018]\n"
     ]
    }
   ],
   "source": [
    "x = [0, 0.001, 0.01, 0.1]\n",
    "print(\"exp(x) - 1 =\", np.expm1(x))\n",
    "print(\"log(1 + x) =\", np.log1p(x))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Se pueden realizar productos exteriores entre los arreglos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3,  4,  5],\n",
       "       [ 2,  4,  6,  8, 10],\n",
       "       [ 3,  6,  9, 12, 15],\n",
       "       [ 4,  8, 12, 16, 20],\n",
       "       [ 5, 10, 15, 20, 25]])"
      ]
     },
     "execution_count": 134,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.arange(1, 6)\n",
    "np.multiply.outer(x, x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cap. 2, Sec. 4 #\n",
    "### Aggregations: Min, Max, and Everything In Between ###"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* COmo primera aproximación a la estadística, python puede sumar todos los valores de un arreglo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "50.461758453195614"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "L = np.random.random(100)\n",
    "sum(L)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "50.46175845319564"
      ]
     },
     "execution_count": 137,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.sum(L)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Comparación de los tiempos con y sin numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "135 ms ± 754 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)\n",
      "802 µs ± 23.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)\n"
     ]
    }
   ],
   "source": [
    "big_array = np.random.rand(1000000)\n",
    "%timeit sum(big_array)\n",
    "%timeit np.sum(big_array)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Python y numpy cuentan también con funciones para encontrar máximos y mínimos en un arreglo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7.071203171893359e-07, 0.9999997207656334)"
      ]
     },
     "execution_count": 139,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "min(big_array), max(big_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7.071203171893359e-07, 0.9999997207656334)"
      ]
     },
     "execution_count": 140,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.min(big_array), np.max(big_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "109 ms ± 4.15 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)\n",
      "833 µs ± 73.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)\n"
     ]
    }
   ],
   "source": [
    "%timeit min(big_array)\n",
    "%timeit np.min(big_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "7.071203171893359e-07 0.9999997207656334 500216.8034810001\n"
     ]
    }
   ],
   "source": [
    "print(big_array.min(), big_array.max(), big_array.sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Se pueden analizar de forma similar arreglos multi-dimensionales o a través de filas y columnas específicas del mismo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.79832448 0.44923861 0.95274259 0.03193135]\n",
      " [0.18441813 0.71417358 0.76371195 0.11957117]\n",
      " [0.37578601 0.11936151 0.37497044 0.22944653]]\n"
     ]
    }
   ],
   "source": [
    "M = np.random.random((3, 4))\n",
    "print(M)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5.1136763453287335"
      ]
     },
     "execution_count": 144,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "M.sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.18441813, 0.11936151, 0.37497044, 0.03193135])"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "M.min(axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.95274259, 0.76371195, 0.37578601])"
      ]
     },
     "execution_count": 146,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "M.max(axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
