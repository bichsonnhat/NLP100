a = "paraparaparadise"
b = "paragraph"

def n_gram(n, s):
	L = []
	for i in range(0, len(s) - n + 1):
			L.append(s[i] + s[i + 1])
	return L;

X = n_gram(2, a)
Y = n_gram(2, b)

# print(X)
# print(Y)

def comp(U, V, type):
	ret = []
	hashMap = {}
	for i in U: 
		hashMap[i] = 1
		if (i in hashMap):
			hashMap[i] += 1
		else:
			hashMap[i] = 1

	for j in V: 
		if (j in hashMap): 
			hashMap[j] += 1
		else:
			hashMap[j] = 1

	if (type == 0):
		for i in hashMap:
			ret.append(i)

	if (type == 1):
		for i in hashMap:
			if (i in U and i in V):
				ret.append(i)

	if (type == 2): # U \ V
		for i in hashMap:
			if (i in U and i not in V):
				ret.append(i)

	if (type == 3): # V \ U
		for i in hashMap:
			if (i not in U and i in V):
				ret.append(i)
	return ret

print(*comp(X, Y, 0)) # union
print(*comp(X, Y, 1)) # intersection
print(*comp(X, Y, 2)) # X \ Y
print(*comp(X, Y, 3)) # Y \ X

if ("se" in X and "se" in Y):
	print("\"se\" is included in the sets X and Y")
else:
	print("\"se\" is not included in the sets X and Y")

