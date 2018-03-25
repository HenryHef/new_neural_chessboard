import os, glob, utils

from config import *
import hldet
from laps import LAPS  # == step 2
from slid import pSLID, SLID, slid_tendency  # == step 1
from utils import ImageObject
import numpy as np
import time


#print("<<< \x1b[5;32;40m neural-chessboard \x1b[0m >>>")
import cv2; load = cv2.imread

save = cv2.imwrite

#NC_SCORE = -1

v=[0,0,0]
################################################################################
def layer():
	lt=time.time()
	global NC_LAYER, NC_IMAGE#, NC_SCORE
	
	#print(utils.ribb("==", sep="="))
	#print(utils.ribb("[%d] LAYER " % NC_LAYER, sep="="))
	#print(utils.ribb("==", sep="="), "\n")

	# --- 1 step --- find all possible lines (that makes sense) ----------------
	print("Starting new round")
	lt=time.time()
	segments = pSLID(NC_IMAGE['main'])
	raw_lines = SLID(NC_IMAGE['main'], segments)
	lines = slid_tendency(raw_lines)

	# --- 2 step --- find interesting intersections (potentially a mesh grid) --
	print(utils.clock(), time.time()-lt,"--- 1 step --- found all lines",len(lines))
	v[0]+=time.time()-lt
	lt=time.time()
	points = LAPS(NC_IMAGE['main'], lines)
	
	print(utils.clock(), time.time()-lt,"--- 2 step --- find all intersections",len(points))
	v[1]+=time.time()-lt
	lt=time.time()
	four_points,mat_pts = hldet.getGridFromPoints(points,padding = 0 if NC_LAYER==2 else .25)
	re=four_points
	oim=NC_IMAGE['main'].copy()
	for pt in mat_pts:
		cv2.circle(oim,(int(pt[0]),int(pt[1])),6,(255,0,0),3)
	
	print(utils.clock(), time.time()-lt,"--- 3 step --- fit grid from points")
	v[2]+=time.time()-lt
	lt=time.time()
	try:
		NC_IMAGE.crop(four_points)
	except:
		utils.warn("Error on crop")
	
	print(utils.clock(), time.time()-lt,"--- 4 step --- post crop")
	return re
	#print("\n")

################################################################################

def detect(args):
	global NC_LAYER, NC_IMAGE, NC_CONFIG

	if (not os.path.isfile(args.input)):
		utils.errn("error: the file \"%s\" does not exits" % args.input)

	NC_IMAGE, NC_LAYER = ImageObject(load(args.input)), 0
	for i in range(NC_CONFIG['layers']):
		NC_LAYER += 1; layer()
	save(args.output, NC_IMAGE['orig'])

	#print("DETECT: %s" % args.input)

def dataset(args):
	pass
	#print("DATASET: use dataset.py") # FIXME

def train(args):
	pass
	#print("TRAIN: use train.py") # FIXME

def test(args):
	files = glob.glob('test/in/*.jpg')

	for iname in files:
		oname = iname.replace('in', 'out')
		args.input = iname; args.output = oname
		detect(args)

	#print("TEST: %d images" % len(files))
	
################################################################################

class Board:
	def __init__(self,transmat,square_pix_size):
		self.transmat=transmat
		self.invmat=np.linalg.inv(transmat)
		self.square_pix_size=square_pix_size
	def transform_boardpix_imagepix(self,pt):
		ptn = self.invmat.dot(np.array([pt[0],pt[1],1.0]))
		ptn = [ptn[0]/ptn[2],ptn[1]/ptn[2]]
		return ptn
	def transform_squarenum_imagepix(self,pt):
		ptn = self.invmat.dot(np.array([self.square_pix_size*pt[0],self.square_pix_size*pt[1],1.0]))
		ptn = [ptn[0]/ptn[2],ptn[1]/ptn[2]]
		return ptn

def detect_img(inp):
	global NC_LAYER, NC_IMAGE, NC_CONFIG

	NC_IMAGE, NC_LAYER = ImageObject(inp), 0
	fpts=[]
	for i in range(NC_CONFIG['layers']):
		NC_LAYER += 1
		fpts=layer()
	board = Board(NC_IMAGE.cum_trans,64)
	return NC_IMAGE['orig'],board

if __name__ == "__main__":
	utils.reset()
	n=20
	for i in range(1,n+1):
		im,b=detect_img(cv2.imread("/home/henry/workspace/neural_chessboard/test/in/"+str(i)+".jpg"))
		cv2.imshow("im",NC_IMAGE['orig'])
		cv2.waitKey(0)
	n=10
	for i in range(1,n+1):
		im,b=detect_img(cv2.imread("/home/henry/workspace/neural_chessboard/test/in/cc"+str(i)+".jpg"))
		cv2.imshow("im",NC_IMAGE['orig'])
		cv2.waitKey(0)
	print([e/2.0/n for e in v])

# 	p = argparse.ArgumentParser(description=\
# 	'Find, crop and create FEN from image.')
# 
# 	p.add_argument('mode', nargs=1, type=str, \
# 			help='detect | dataset | train')
# 	p.add_argument('--input', type=str, \
# 			help='input image (default: input.jpg)')
# 	p.add_argument('--output', type=str, \
# 			help='output path (default: output.jpg)')
# 
# 	#os.system("rm test/steps/*.jpg") # FIXME: to jest bardzo grozne
# 	os.system("rm -rf test/steps; mkdir test/steps")
# 
# 	args = p.parse_args(); mode = str(args.mode[0])
# 	modes = {'detect': detect, 'dataset': dataset, 'train': train, 'test': test}
# 
# 	if mode not in modes.keys():
# 		utils.errn("hey, nie mamy takiej procedury!!! (wybrano: %s)" % mode)
# 
# 	modes[mode](args); #print(utils.clock(), "done")
# 	K.clear_session(); gc.collect() # FIX: tensorflow#3388
	
	