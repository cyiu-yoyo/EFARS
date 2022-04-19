class Human36MMetadata:
    num_classes = 15
    classes = {'Directions': 0, 'Discussion': 1, 'Eating': 2, 'Greeting': 3, 'TakingPhoto': 4, 'Photo':4, 
               'Posing': 5, 'Purchases': 6, 'Smoking': 7, 'Waiting': 8, 'Walking': 9, 'Sitting': 10, 
               'SittingDown': 11, 'Phoning': 12, 'WalkingDog': 13, 'WalkDog': 13, 'WalkTogether': 14}
    mean = np.array([0.44245931, 0.2762126, 0.2607548])
    std = np.array([0.25389833, 0.26563732, 0.24224165])
    pos2d_mean = np.array([531.3589047602578, 401.11892849734477])
    pos2d_std = np.array([116.12519808242102, 110.18267048431545])
    pos3d_mean = np.array([58.695619373856935, 221.5308073087531, 900.0432746404251])
    pos3d_std = np.array([448.8852213564668, 667.2435126476839, 459.2800512506026])
    num_joints = 17
    used_joint_mask = np.array([1,1,1,1,0,0,1,1,
                                1,0,0,0,1,1,1,1,
                                0,1,1,1,0,0,0,0,
                                0,1,1,1,0,0,0,0],dtype=np.bool8)
    used_joint_labels = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 
                         'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 
                         'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
    skeleton_edges = [(1, 0), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5), (7, 0), (8, 7), 
                      (9, 8), (10, 9), (11, 8), (12, 11), (13, 12), (14, 8), (15, 14), (16, 15)]
    camera_parameters_path = '/home/samuel/EFARS/data/human36m_camera_parameters.json'