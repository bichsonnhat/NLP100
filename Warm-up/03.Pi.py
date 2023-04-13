txt = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics"

a = txt.split()

freq = [dict() for _ in range(len(a))]

for i in range(len(a)):
	for j in range(len(a[i])):
		if (a[i][j] in freq[i]):
			freq[i][a[i][j]] += 1
		else:
			freq[i][a[i][j]] = 1

print(*freq)