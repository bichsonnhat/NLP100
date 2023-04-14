txt = "I am an NLPer"

a = txt.split()	
b = txt.replace(" ", "")

def n_gram(n, s):
	L = [list() for i in range(len(s) - n + 1)]
	for i in range(0, len(s) - n + 1):
		for j in range(i, i + n):
			L[i].append(s[j])
	return L;

resultWords = n_gram(2, a)
resultLetters = n_gram(2, b)
print(resultWords)
print(resultLetters)