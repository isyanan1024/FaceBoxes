import os
import random
import shutil
import argparse

parser = argparse.ArgumentParser('Select samples!')
parser.add_argument("--imgPath", '-i',type=str, default=None, help="Image Path")
parser.add_argument("--sampleNums", '-s',type=float, default=None, help="Sample Nums,小于1按比例，大于1按个数")
parser.add_argument("--targetPath", '-t',type=str, default=None, help="Target Path")
parser.add_argument("--copy", '-c',type=int, required = True, default=None, help="copy : 0, move : 1")
args = parser.parse_args()

if not os.path.exists(args.targetPath):
	os.makedirs(args.targetPath)

images = [x for x in os.listdir(args.imgPath) if x.endswith('.jpg')]
images_num = len(images)

if args.sampleNums < 1:
	sample_nums = int(images_num * args.sampleNums)
else:
	sample_nums = args.sampleNums

samples = random.sample(images, sample_nums)

print(args.copy)
if args.copy == 0:
	print('Copy!!!')
	[shutil.copy(os.path.join(args.imgPath, x), (os.path.join(args.targetPath, x))) for x in samples] 
else:
	print('Move!!!')
	[shutil.move(os.path.join(args.imgPath, x), (os.path.join(args.targetPath, x))) for x in samples] 