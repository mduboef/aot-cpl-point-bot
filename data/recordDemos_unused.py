import sys
# give graphic interface through which I can record demos using arrow keys

# python3 recordDemos.py 1
	# saves to demos/1 folder

def main():

	# save demo to folder specified in command line
	demoType = sys.argv[1]
	savePath = f'demos/{demoType}'

	return

main()


