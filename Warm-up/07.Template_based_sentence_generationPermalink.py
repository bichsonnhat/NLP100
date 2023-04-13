x = 12
y = "temperature"
z = 22.4

def func(x, y, z):
	result = "{1} is {2} at {0}"
	return result.format(x, y, z)

print(func(x, y, z))