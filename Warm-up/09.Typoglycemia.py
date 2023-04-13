import random
txt = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind "
txt = txt.strip()

a = txt.split()

def suffle(s):
    ret = s[0]
    L = []
    for i in range(1, len(s) - 2):
    	L.append(s[i])
    random.shuffle(L)
    for i in L:
    	ret += i
    return ret + s[len(s) - 1]

for i in range(len(a)):
	if (len(a[i]) > 4):
		a[i] = suffle(a[i])

print(*a)
