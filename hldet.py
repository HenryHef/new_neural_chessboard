import math

import numpy as np
import perspective_fit

def intersection(p1, p2):
	"""solve intersection"""
	x =  (p2[1] -p1[1])/(p1[0]- p2[0])
	y = p1[0] * x + p1[1]
	return (x,y)

def crossproduct(v1,v2):
	return (v1[0]*v2[1]) - (v1[1]*v2[0])
def dotproduct(v1, v2):
	##print ("v1:",v1,"  v2:",v2)
	return v1[0]*v2[0]+v1[1]*v2[1]

def length(v):
	return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
	inner = dotproduct(v1, v2) / (length(v1) * length(v2))
	if(inner>.9999):
		return 0.0
	if(inner<-.9999):
		return np.pi
	return math.acos(inner)

def get_transform_mat(centered_grid):
	from_pts,to_pts=[],[]
	for to_pt in centered_grid:
		from_pts.append(centered_grid[to_pt][0])
		to_pts.append(to_pt)
	M=perspective_fit.get_transform(from_pts,to_pts)
	return M

def get_four_true_corners(centered_grid,padding=0):
	M=get_transform_mat(centered_grid)
	re_pts=[(-1-padding,-1-padding),
		(7+padding,-1-padding),
		(7+padding,7+padding),
		(-1-padding,7+padding)]
	re = [perspective_fit.inv_proj(M,re_pt) for re_pt in re_pts]
	return re	

#TODO do a second pass at the end, using matrix, to find very clsoe stray points
def getGridFromPoints(pts, parr_tol = 20*np.pi/180.0, perp_tol = 20*np.pi/180.0, same_len_tol = .7,padding=0):
	print("PTS:",pts)
	#first build a table from point to 4 closest nieghbors
	point_opoint_dist_map={}#{pt:[(opt,distSq),...]}
	for pt in pts:
		point_opoint_dist_map[pt]=[]
	for pt in pts:
		for opt in pts:
			if(pt==opt):
				continue
			point_opoint_dist_map[pt].append((opt,(pt[0]-opt[0])**2+(pt[1]-opt[1])**2))
	
	for pt in pts:
		point_opoint_dist_map[pt].sort(key=lambda x:x[1])
		point_opoint_dist_map[pt]=point_opoint_dist_map[pt][:4]
	#print("GRID:",point_opoint_dist_map)
	
	#then check pairwise
	n_map = {}#will be of form pt:[(dispVec,opt)] len<=4
	for pt in pts:
		n_map[pt]=[]
	for pt in pts:
		for niegb,distSq in point_opoint_dist_map[pt]:
			for n_niegb,n_distSq in point_opoint_dist_map[niegb]:
				if(n_niegb==pt):
					#niegb is good
					n_map[pt].append(((niegb[0]-pt[0],niegb[1]-pt[1]),niegb))
	n_map_cleaned = {}
	for pt in n_map:
		if(len(n_map[pt])!=0):
			n_map_cleaned[pt]=n_map[pt]
	n_map=n_map_cleaned
	#print("\n\nN_GRID:")
	for pt in n_map:
		pass
		#print(pt," : ",n_map[pt])
	#print ("\n\n")
			
	#then find those that are 4, close to perp, close same dist
	#looks for a point that has 4 nieghbors, in which, pick 1 and there is 1 close
	#to parrelel of the same lenth, and a second set that is close to perperdicular
	#that are the same length
	start_pt = None
	for pt in n_map:
		if(len(n_map[pt])==4):
			vecs = n_map[pt]
			#then check semiperp
			mainVec = vecs[0][0]
			isGood = True
			perpLen=-1
			perpAng=-1
			for vec,opt in vecs[1:]:
				ang = angle(mainVec,vec)
				if(ang>np.pi-parr_tol):
					lenRat = length(vec)/length(mainVec)
					if(lenRat<same_len_tol or lenRat>1.0/same_len_tol):
						isGood=False
						break
				elif(abs(np.pi/2-ang)<perp_tol):
					if(perpLen==-1):
						perpLen=length(vec)
						perpAng=ang
					else:
						lenRat = length(vec)/perpLen
						if(lenRat<same_len_tol or lenRat>1.0/same_len_tol or np.abs(perpAng-ang)>parr_tol):
							isGood=False
							break
				else:
					isGood=False
					break
			if(isGood):
				start_pt=pt
	#print ("START_PT:",start_pt)
	
	grid = {}#(-7,-7)...(7,7):(pt,east_vec,north_vec)   note:east_vec = (1,0)
	#north_vec = (0,1)#i.e. coordiant axes
	edge_points = []
	closed_points = set()
	sev = n_map[start_pt][0][0]
	snv=None
	
	for v,_pt in n_map[start_pt]:
		#print ("sev:",sev,"  v:",v, "  abs(np.pi/2-angle(sev,v)):",abs(np.pi/2-angle(sev,v)),
			#"dotproduct(sev, v):",corssproduct(sev, v))
		if(abs(np.pi/2-angle(sev,v))<perp_tol and crossproduct(sev, v)>0):
			snv=v
			break
	if(snv==None):
		raise ValueError("ERROR bad coor system")
		return
	
	#print("\nNVEC:",snv,"  EVEC:",sev,"\n")
		
	grid[(0,0)]=(start_pt,sev,snv)
	
	pt_loc_map={}#point to location
	pt_loc_map[start_pt]=(0,0)
	edge_points.append(start_pt)
	
	
	def get_oriented(vec_pt_tup_list,e_vec,n_vec):
		n_vecs_tups=[]
		e_vecs_tups=[]
		for vec,pt in vec_pt_tup_list:
			if(angle(vec,n_vec)<parr_tol or angle(vec,n_vec)>np.pi-parr_tol):
				lenRat = abs(length(vec)/length(n_vec))
				if(lenRat<1.0/same_len_tol and lenRat>same_len_tol):
					#print("IS N VEC:",vec)
					n_vecs_tups.append((vec,pt))
			elif(angle(vec,e_vec)<parr_tol or angle(vec,e_vec)>np.pi-parr_tol):
				lenRat = abs(length(vec)/length(e_vec))
				if(lenRat<1.0/same_len_tol and lenRat>same_len_tol):
					#print("IS E VEC:",vec)
					e_vecs_tups.append((vec,pt))
				else:
					pass
					#print("REJECT EVEC ON LEN:",vec,lenRat,length(vec),length(e_vec),1.0/same_len_tol,same_len_tol,lenRat<1.0/same_len_tol,lenRat>same_len_tol)
			else:
				pass
				#print("REJECT VEC ON ANGLE:",vec,angle(vec,n_vec),angle(vec,e_vec),e_vec,n_vec,np.pi-parr_tol)
		re={}
		for nvec,pt in n_vecs_tups:
			if(dotproduct(nvec, n_vec)>0):
				re[(0,1)]=(nvec,pt)
			else:
				re[(0,-1)]=(nvec,pt)
			
		for evec,pt in e_vecs_tups:
			if(dotproduct(evec, e_vec)>0):
				re[(1,0)]=(evec,pt)
			else:
				re[(-1,0)]=(evec,pt)
		
		re_nvec,re_evec	= None,None
		if (0,1) in re:
			re_nvec=re[(0,1)][0]
		elif (0,-1) in re:
			re_nvec=re[(0,-1)][0]
			re_nvec=(-re_nvec[0],-re_nvec[1])
		else:
			re_nvec=n_vec
			
		if (1,0) in re:
			re_evec=re[(1,0)][0]
		elif (-1,0) in re:
			re_evec=re[(-1,0)][0]
			re_evec=(-re_evec[0],-re_evec[1])
		else:
			re_evec=e_vec
		
		return re,re_evec,re_nvec
			
		
	
	while(len(edge_points)>0):
		cedge_pt = edge_points[0]
		#print("EXPANDED:",cedge_pt," at loc ",pt_loc_map[cedge_pt])
		edge_points=edge_points[1:]
		if(cedge_pt in closed_points):
			continue
		e_vec = grid[pt_loc_map[cedge_pt]][1]
		n_vec = grid[pt_loc_map[cedge_pt]][2]
		#take the edge point, and for each other the other outgoing edges,
		#for each edge check that the lenths of the vectors are consistent with neerby vectors. 
		orientation_vec_pt_tups_map,nnvec,nevec = get_oriented(n_map[cedge_pt],e_vec,n_vec)
		#return orthaginal dir map to pair, missing if not there, plus new axes
		for grid_disp in orientation_vec_pt_tups_map:
			pix_disp,opt = orientation_vec_pt_tups_map[grid_disp]
			opt_loc = (grid_disp[0]+pt_loc_map[cedge_pt][0],grid_disp[1]+pt_loc_map[cedge_pt][1])
			if(opt in pt_loc_map):
				#Then check whether the connection is in the map.
				#IF it is, then check its location is consistent
				if not (pt_loc_map[opt] == opt_loc):
					#print("cpt:",cedge_pt,"  opt:",opt,"  opt local loop predicted loc:",opt_loc, "  global loc:",pt_loc_map[opt])
					#print("cpt loc:",pt_loc_map[cedge_pt])
					#print("grid:",grid)
					#print("pt_loc_map:",pt_loc_map)
					raise ValueError('INCONSISTENT MAP')
					return
			#otherwise add the point as an edge node.
			else:
				#print("ADDED:",opt," at loc ",opt_loc, "  from point ",cedge_pt," at loc:",pt_loc_map[cedge_pt]," with disp:",grid_disp)
				grid[opt_loc]=(opt,nnvec,nevec)
				pt_loc_map[opt]=opt_loc
				edge_points.append(opt)
		closed_points.add(cedge_pt)
		#take a point with the last north vector, and add it to the grid,
		#and to open points. validate
	#print ("FINAL GRID:",grid)
	
	def closest_grid_pt(pt):
		return min((pt for pt in grid), key=lambda x:(pt[0]-x[0])**2+(pt[1]-x[1])**2)
	
	#then fill in missing points using the grid that we have already aligned, with out matrix
	M=get_transform_mat(grid)
	maxErrorRadSqL=.1**2#TODO, ensure that also holds with axes if further 
	maxErrorRadSqH=.3**2#TODO, ensure that also holds with axes if further 
	for pt in pts:
		grid_space_pt=perspective_fit.proj(M,pt)
		grid_space_pt_int = (int(grid_space_pt[0]),int(grid_space_pt[1]))
# 		print(grid_space_pt,grid_space_pt_int)
		errSq=(grid_space_pt[0]-grid_space_pt_int[0])**2+(grid_space_pt[1]-grid_space_pt_int[1])**2
		if(grid_space_pt_int[0]>=-6 and grid_space_pt_int[0]<=6\
				 and grid_space_pt_int[1]>=-6 and grid_space_pt_int[1]<=6 and\
				 (not grid_space_pt_int in grid)):
			e_vec,n_vec = grid[closest_grid_pt(pt)][1:]
			orientation_vec_pt_tups_map,nnvec,nevec = get_oriented(n_map[pt],e_vec,n_vec)
			if(errSq<maxErrorRadSqL):
				grid[grid_space_pt_int]=(pt,nnvec,nevec)
			elif(errSq<maxErrorRadSqH and len(orientation_vec_pt_tups_map)>2):
					#TODO then cosider the points orientation
					print("ADDING",pt," AT LOC ",grid_space_pt_int)
					grid[grid_space_pt_int]=(pt,nevec,nnvec)
			
			
	
	#finally, take the grid, and find the dencest 7x7 region.
	den_list=[]
	for i in range(-6,1):#so from -6 to 0
		for j in range(-6,1):
			ct=0
			for x in range(i,i+7):
				for y in range(j,j+7):
					if((x,y) in grid):
						ct=ct+1
			den_list.append((i,j,ct))
	
	#print(den_list)
	mx=max(den_list,key=lambda  x:x[2])
	#print(mx)
	
	centered_grid={(i,j):grid[(i+mx[0],j+mx[1])] for i in range(-6,7) for j in range(-6,7) if (i+mx[0],j+mx[1]) in grid}
	
	return get_four_true_corners(centered_grid,padding),[perspective_fit.inv_proj(M,(i,j)) for i in range(-6,7) for j in range(-6,7)]
	#TODO implement lines on edges, for missing corners, then, later, matrix regression

	