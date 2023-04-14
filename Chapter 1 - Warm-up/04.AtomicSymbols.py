txt = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can"

a = txt.split()

idx = [1, 5, 6, 7, 8, 9, 15, 16, 19]

pos = {}

for i in range(len(a)):
	if ((i + 1) in idx):
		pos[a[i][0]] = i + 1
	else:
		pos[a[i][0] + a[i][1]] = i + 1

print(*pos)
