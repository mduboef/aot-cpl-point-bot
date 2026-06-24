import os
import re
import shutil
import random

# directories relative to this script
baseDir = os.path.dirname(os.path.abspath(__file__))
allDataDir = os.path.join(baseDir, "allData")
trainingDir = os.path.join(baseDir, "trainingData")
testingDir = os.path.join(baseDir, "testingData")

# demo types to split
demoTypes = ["1", "2", "3", "4", "5"]

# fixed seed so the split is reproducible across runs
randomSeed = 42

# matches the trailing demo index in a filename, e.g. states_12.txt -> 12
indexPattern = re.compile(r"_(\d+)\.[^.]+$")

# returns the trailing demo index for a filename, or None if absent
def getDemoIndex(fileName):
	match = indexPattern.search(fileName)
	if match is None:
		return None
	return int(match.group(1))

# removes a directory's contents then recreates it empty
def resetDir(dirPath):
	if os.path.isdir(dirPath):
		shutil.rmtree(dirPath)
	os.makedirs(dirPath)

# copies every file whose demo index is in keepSet from srcDir to dstDir
def copyDemos(srcDir, dstDir, keepSet):
	for fileName in os.listdir(srcDir):
		index = getDemoIndex(fileName)
		if index is None or index not in keepSet:
			continue
		shutil.copy2(os.path.join(srcDir, fileName), os.path.join(dstDir, fileName))

# splits one type's demos 50/50 into training and testing
def splitType(demoType):
	srcDir = os.path.join(allDataDir, demoType)
	trainDst = os.path.join(trainingDir, demoType)
	testDst = os.path.join(testingDir, demoType)

	# collect the unique demo indices present in this type
	indices = set()
	for fileName in os.listdir(srcDir):
		index = getDemoIndex(fileName)
		if index is not None:
			indices.add(index)
	indices = sorted(indices)

	# shuffle then halve, extra demo on odd counts goes to training
	random.Random(randomSeed).shuffle(indices)
	half = len(indices) // 2
	testIndices = set(indices[:half])
	trainIndices = set(indices[half:])

	# overwrite whatever was in the destination directories
	resetDir(trainDst)
	resetDir(testDst)
	copyDemos(srcDir, trainDst, trainIndices)
	copyDemos(srcDir, testDst, testIndices)

	print(f"type {demoType}: {len(indices)} demos -> {len(trainIndices)} train, {len(testIndices)} test")

# splits every demo type
def main():
	for demoType in demoTypes:
		splitType(demoType)

if __name__ == "__main__":
	main()
