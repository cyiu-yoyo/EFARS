import numpy as np
from dtaidistance import dtw
from NetworkBasedNormalization import NetworkNorm ###
from numpy import dot
from numpy.linalg import norm

class Score(object):

	NumOfFrames = 8 ##can ?

	def percentage_score(self,score,method="CosSimVec"):   #To be replaced with a better scoring algorithm, if found in the future
		if method = "Euclidean":
			percentage =  100 - (score* 100)
		else:
			percentage = score* 100
		return int(percentage)

	def dtwdis(self,model_points,input_points,i,j,method="CosSimVec"): ##model: standard, input:users CHANGED
		"""
		i: number of frame of model
		j: number of frame of input video
		"""
		distance = None
		path_warp = dtw.warping_paths(model_points, input_points)
		if method == 'CosSimVec':
			##using dtw API to directly calculate the distance
			model_points = model_points.reshape(2*j,) ## important  
			input_points = input_points.reshape(2*i,) ## important
			model_points = model_points/ np.linalg.norm(model_points) #?
			input_points = input_points/np.linalg.norm(input_points) #?
		
			distance = dtw.distance(model_points, input_points)
		else:
			distance = None
			#becuase we are using cos similarity, we have to view each frame as a whole, but cannot separate each point and calculate the score separately.
			#thus we need to compare all keypoints separately, and match to the most matching one. as there are only 8 frames.
			frame_match = []
			#set up a dict
			frame_matching_dict = {keypoint:[] for keypoint in range(0,17)}

			for keypoint in range(0,17):
				path = dtw.warping_path(input_points[:,k], model_points[:,k]) # k means which keypoint this array represent
				for pair in range(0,len(path)):
					frame_matching_dict[pair[0]].append(pair[1])
			for keypoint in range(0,17):
				most_common = max(frame_matching_dict[keypoint], key = frame_matching_dict[keypoint].count)
				frame_match.append(most_common)	

			if method == 'CosSimilarity':
				# intotal there are 16 vectors. 
				# have to find the pairs to construct the vectors.
				# every vector should be given a name
				# this should be saved in a general file as it may be used in other methods.
				# skeleton_edge can be retrieved from Zhuoxuan's code. (have checked, currently in human36m.py)

				 ###   used_joint_labels = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 
                 ###        'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 
                 ###        'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
    			 ###	skeleton_edges = [(1, 0), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5), (7, 0), (8, 7), 
                 ###     (9, 8), (10, 9), (11, 8), (12, 11), (13, 12), (14, 8), (15, 14), (16, 15)]

				 ### in this situation all standard poses' skeleton_edges of each frame should be better first calculated.
				 ### Assume we have the following dictionary retrieved by any means, which has a dimention of (8,(16,1))
				
				skeletonsOfFrames_dict = {frame:[] for frame in range(NumOfFrames)}
				scoreOfFrames = []
				for frame in range(NumOfFrames):
					match_frame = frame_match[frame] #matched frame in model
					for edge in skeleton_edges:
					## calculate vector between input_points(frame,edge[0]) and model_points(match_frame,edge[0])
						vector_input = input_points(frame,edge[0]) - input_points(frame,edge[1])
						vector_model = model_points(match_frame,edge) ###to be confirm the API###

						# now calculate the cosinmilarity
						cosSim = dot(vector_input, vector_model)/(norm(vector_input)*norm(vector_model))

						#skeletonsOfFrames_dict[frame].append(vector_input)
						skeletonsOfFrames_dict[frame].append(CosSim)

					scoreOfFrames.append(sum(skeletonsOfFrames_dict[frame])/16)
				
				distance = sum(scoreOfFrames)/NumOfFrames

				#after calculating the similarity of each frame pair, we should normalize them(add tgt and divide by # of vectors(16)) and then total score will also be normalized and then * 100.





		return self.percentage_score(distance,method) 

##normalization
	def normalizeL2(self,input_test):##L2 normalization
		for k in range(0,17):	
			input_test[:,k] = input_test[:,k]/np.linalg.norm(input_test[:,k])	
		return input_test
	
	def normalizeNW(self,input_test): ##Network Based Normalization
		for k in range(0,17):
			input_test[:,k] = None	 ##double check 
		return input_test

	def visualize(self,norm_ip,norm_model): #model should have been normalized and stored
		

		return None



#comparison
	def compare(self,ip,model,i,j,type='L2',method='CosSimVec'): ### CHANGED
		"""
		i: number of frame of model
		j: number of frame of input video
		"""
		if type == "L2":
			ip = self.normalizeL2(ip) ###
			model = self.normalizeL2(model)
		else if type == "NetworkBased":
			ip = self.normalizeNW(ip) ###
			model = self.normalizeNW(model)

		scores = []
		for k in range(0,17): 
			scores.append(self.dtwdis(ip[:,k],model[:,k],i,j,method))	##double check the sequence method. 一行是一個點,原作者後面改過lookup的方法
		return np.mean(scores),scores

	


