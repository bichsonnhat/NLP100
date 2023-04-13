txt = "schooled"

pos = [0, 2, 4, 6]

result = ""

for i in range(len(txt)):
	if (i in pos):
		result += txt[i]

print(result)