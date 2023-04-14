def cipher(txt):
	result = ""
	for c in txt:
		if (c.islower()):
			result += chr(219 - ord(c))
		else:
			result += c	
	return result
