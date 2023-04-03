### Files

#### Opening Files

```python3
file = open(file_name, mode)

with open (file_name, mode) as file: 
	# work with file here
```
##### Modes for opening a file: 
- r: reading only
- r+: reading and writing
- w: writing only
- w+: writing and reading 
- a: appending 
- a+: appending and reading 

##### Advantage of using 'with'
- file is closed after block finishes

#### Reading Files

##### Read entire file

```python3
file.read()
```

##### Reading line by line

```python3
file.readline() 
```

##### Reading multiple items in one line

- Use the .split(separator) function, which returns a list of strings 

```python3
new_file = open('file.txt', 'r')
lines = new_file.read()
content = []
for line in lines:
	content.append(line.split())
```

#### Closing Files

```python3
file_obj.close()
```

#### Writing FIles

```python3
file_obj.write(content)
```

Note: you can only write strings to a text file, you have to use the str() functino when writing integers. 

##### Another way of writing to a file:

```python3
print(content,file=file_obj)
```

#### Exceptions for Files

```python3 
try:
	f = open('goaway.txt', 'r')
except FileNotFoundError: 
	print('File does not exist.')
```

- Can have multiple exceptions in the same try/except block 

### NumPy

#### Arrays using NumPy

- Array object is named: *ndarry*
- Created using array() function
- Can convert a list, tuple, or any array-like object into an array 
- *ndim*: attribute that represents thhe dimensions of the array

#### 1-D Array

```python3 
arr = np.array([1, 2, 3, 4, 5])
```

#### 2-D Arrays 

- Has 1-D arrays as its elements
- Used to represent matrices 

```python3
arr = np.array([[1,2,3], [1,2,3]])
```

#### 3-D Arrays

- Has two 2-D arrays as its elements 

```python3 
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
```

#### Accessing Array Elements 

- For 1-D arrays, you can access them through its index number
- For 2-D arrays, you have to use comma-separated indices for the row and column

```python3
arr = np.array([1, 2, 3, 4, 5])
print(arr[0])

arr_2 = np.array([[1,2,3], [1,2,3]])
print(arr_2[0,1])
```

#### Iterating through arrays

- 1-D arrays: 

```python3
arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  print(x)
```

- 2-D arrays: 

```python3
arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  for y in x:
    print(y)
```

#### Sorting Arrays

- numpy.sort(arr, axis, kind, order) function - sorts a specified array in place
	- arr: array to be sorted
	- axis: axis to be sorted along, None means the array is flattened, default is -1 (last)
	- kind: quicksort, mergesort, heapsport, stable (default is quicksort)
	- order: str or list of strings 

```python3
arr = np.array([3, 2, 0, 1])
print(np.sort(arr))
```

#### Calculation between two arrays

- can add two arrays together

```python3
x = np.array([1,2,3,4])
y = np.array([4,5,6,7])
```

- adding a number to an array

```python3
x = np.array([1,2,3,4])
print(x + 2) # 3, 4, 5, 6
```

#### Square Root Array 

- np.sqrt(arr) -> performs the square root operation on each of the elements in the array

```python3
x = np.array([1,2,3,4])
np.sqrt(x)
```

#### Element-wise multiplication

```python3
A = np.array([
    [2,3,4],
    [3,1,0],
    [0,1,1]
])

x = np.array([1,3,4])
A*x

# A is now: 
array([[ 2,  9, 16],
       [ 3,  3,  0],
       [ 0,  3,  4]])
```

- Multiply matrix by matrix 

```python3 
A = np.arange(1,5).reshape(2,2)
print(A)

'''
[[1 2]
 [3 4]]
'''

B = np.arange(0,4).reshape(2,2)
print(B)

'''
[[0 1]
 [2 3]]
'''

print(type(A))
print(type(B))

'''
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
'''

A * B 

'''
array([[ 0,  2],
       [ 6, 12]])
'''
```

### Pandas 

#### DataFrame

```python3
football_df = pd.DataFrame(dictionary)
```

##### DataFrame Methods 

- df.head(number): prints the top $number$ entries of the DataFrame
- df\[number:]: allows you to slice the DataFrame (similar to slicing a list in Python)
- df\['column_name']: gets a whole column
	- df\[\['column_one', 'column_two']]: gets two columns
- Adding a new column: df\['new_column'] = data
- df.describe(): runs basic statistics on the columns (count, mean, std, min, first, second, third quartile, max)
	- can also use functions like .mean(), .count(), .min()
- Getting rows where with specific column values: df\['column'] == 'value'