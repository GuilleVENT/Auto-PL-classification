## general
import os
from os import listdir, path
import pandas as pd
import numpy as np
import numpy.matlib as matlib
from itertools import chain
import json

## coloring
from termcolor import colored

## sklearn
from sklearn.base import TransformerMixin, BaseEstimator 
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

## used for parameter selection
from sklearn import svm,  model_selection # grid search
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

## Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

## ensemble
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

## metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

## Principal Component Analyze
from sklearn.decomposition import PCA

## to save parameters and classifiers
import pickle# as pickle

## Plotting
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import pylab
## for PCA+SVD distance
from scipy.spatial import distance

## debugging
import sys

np.seterr(divide='ignore', invalid='ignore')

home_hpc = "/nas/ei/home/ga59qek/FP/"
home_PC = '/Users/guillermoventuramartinezAIR/Desktop/FP/'
## global directories:

home = home_hpc

pl_data = home+'PL_DATA/'

path_RES = home+'user_PL/'

path_CM = home+'CM_DATA/'

## WARNING:
## experiment_2.py:78: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).

def CM_plotter(CM,pl_list,ID,dataset,_clf_):
	target_names = []
	for pl in pl_list:
		target_names.append(pl[1])

	# Normalise
	cmn = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots(figsize=(5,5))
	sns.heatmap(cmn, annot=True, fmt='.2f', yticklabels=target_names)
	plt.ylabel('Actual')
	plt.xlabel('Predicted')
	#plt.xticks(rotation=45) 
	plt.savefig(path_CM+ID+' '+dataset+" "+_clf_+'.eps',bbox_inches='tight')

def PL_selector():	## selects the playlist out of the info file. 
	

	uno = ('spotify','Peaceful Piano','37i9dQZF1DX4sWSpwq3LiO')
	
	#dos = ('spotify','mint','37i9dQZF1DX4dyzvuaRJ0n')
	
	cinco = ('spotify','Rock Classics','37i9dQZF1DWXRqgorJj26U')
	
	cuatro = ('spotify','¡Viva Latino!','37i9dQZF1DX10zKzsJ2jva')	
	
	#cinco = ('spotify_germany','Pop Remix','37i9dQZF1DXcZDD7cfEKhW')

	dos = ('spotify_germany','Techno Bunker','37i9dQZF1DX6J5NfMJS675')

	#tres = ('spotify_france','Fresh Rap','37i9dQZF1DWU4xkXueiKGW')

	tres = ('spotify','All Out 80s','37i9dQZF1DX4UtSsGT1Sbe')
	

	pl_list = [uno, dos,tres,cuatro,cinco] ##init

	ID = 'EXP_2'
	
	
	print(' - - - - - - - - - - - - - - - - ')
	print(pl_list)
	print(' - - - - - - - - - - - - - - - - ')

	
	## SVM 
	get_X_and_y_SVM(ID, pl_list,PCA=True)
	get_X_and_y_SVM(ID,pl_list, PCA=False)

	## RF 
	get_X_and_y_RF(ID, pl_list,PCA=True)
	get_X_and_y_RF(ID,pl_list, PCA=False)

	## KNN
	get_X_and_y_kNN(ID, pl_list,PCA=True)
	get_X_and_y_kNN(ID, pl_list, PCA=False)
	

	# order of this list: X_ALL, X_ALL_val, y_ALL, y_ALL_val
	MIR_DS_PCA,SPO_DS_PCA,LYR_DS_PCA,ALL_DS_PCA 		= pre_process(ID, pl_list,PCA=True)
	MIR_DS_noPCA,SPO_DS_noPCA,LYR_DS_noPCA,ALL_DS_noPCA = pre_process(ID, pl_list,PCA=False)

	
	## BAGGING:

	## PCA == True	
	## DS = ALL
	best_clf, accuracy, CM , y_true , y_pred = Bagging_clf(ALL_DS_PCA[0],ALL_DS_PCA[1],ALL_DS_PCA[2],ALL_DS_PCA[3],pca=True)
	print('Bagging Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'ALL_PCA','Bagging')
	print_save(ID,pl_list,' ALL_PCA ', 'Bagging', y_true, y_pred, ALL_DS_PCA[1], ALL_DS_PCA[3], best_clf.get_params)

	## DS = MIR 
	best_clf, accuracy, CM , y_true , y_pred = Bagging_clf(MIR_DS_PCA[0],MIR_DS_PCA[1],MIR_DS_PCA[2],MIR_DS_PCA[3],pca=True)
	print('Bagging Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'MIR_PCA','Bagging')
	print_save(ID,pl_list,' MIR_PCA ', 'Bagging', y_true, y_pred, MIR_DS_PCA[1], MIR_DS_PCA[3], best_clf.get_params)

	## DS = SPO
	best_clf, accuracy, CM , y_true , y_pred = Bagging_clf(SPO_DS_PCA[0],SPO_DS_PCA[1],SPO_DS_PCA[2],SPO_DS_PCA[3],pca=True)
	print('Bagging Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'SPO_PCA','Bagging')
	print_save(ID,pl_list,' SPO_PCA ', 'Bagging', y_true, y_pred, SPO_DS_PCA[1], SPO_DS_PCA[3], best_clf.get_params)

	## DS = LYR
	best_clf, accuracy, CM , y_true , y_pred = Bagging_clf(LYR_DS_PCA[0],LYR_DS_PCA[1],LYR_DS_PCA[2],LYR_DS_PCA[3],pca=True)
	print('Bagging Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'LYR_PCA','Bagging')
	print_save(ID,pl_list,' LYR_PCA ', 'Bagging', y_true, y_pred, LYR_DS_PCA[1], LYR_DS_PCA[3], best_clf.get_params)


	## PCA == False
	## DS = ALL
	best_clf, accuracy, CM , y_true , y_pred = Bagging_clf(ALL_DS_noPCA[0],ALL_DS_noPCA[1],ALL_DS_noPCA[2],ALL_DS_noPCA[3],pca=False)
	print('Bagging Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'ALL','Bagging')
	print_save(ID,pl_list,' ALL_noPCA ', 'Bagging', y_true, y_pred, ALL_DS_noPCA[1], ALL_DS_noPCA[3], best_clf.get_params)

	## DS = MIR 
	best_clf, accuracy, CM , y_true , y_pred = Bagging_clf(MIR_DS_noPCA[0],MIR_DS_noPCA[1],MIR_DS_noPCA[2],MIR_DS_noPCA[3],pca=False)
	print('Bagging Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'MIR','Bagging')
	print_save(ID,pl_list,' MIR_noPCA ', 'Bagging', y_true, y_pred, MIR_DS_noPCA[1], MIR_DS_noPCA[3], best_clf.get_params)

	## DS = SPO
	best_clf, accuracy, CM , y_true , y_pred = Bagging_clf(SPO_DS_noPCA[0],SPO_DS_noPCA[1],SPO_DS_noPCA[2],SPO_DS_noPCA[3],pca=False)
	print('Bagging Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'SPO','Bagging')
	print_save(ID,pl_list,' SPO_noPCA ', 'Bagging', y_true, y_pred, SPO_DS_noPCA[1], SPO_DS_noPCA[3], best_clf.get_params)

	## DS = LYR 
	best_clf, accuracy, CM , y_true , y_pred = Bagging_clf(LYR_DS_noPCA[0],LYR_DS_noPCA[1],LYR_DS_noPCA[2],LYR_DS_noPCA[3],pca=False)
	print('Bagging Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'LYR','Bagging')
	print_save(ID,pl_list,' LYR_noPCA ', 'Bagging', y_true, y_pred, LYR_DS_noPCA[1], LYR_DS_noPCA[3], best_clf.get_params)

	
	## AdaBoost:

	## PCA == True	
	## DS = ALL
	best_clf, accuracy, CM , y_true , y_pred = AdaBoost_clf(ALL_DS_PCA[0],ALL_DS_PCA[1],ALL_DS_PCA[2],ALL_DS_PCA[3],pca=True)
	print('AdaBoost Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'ALL_PCA','AdaBoost')
	print_save(ID,pl_list,' ALL_PCA ', 'AdaBoost', y_true, y_pred, ALL_DS_PCA[1], ALL_DS_PCA[3], best_clf.get_params)

	## DS = MIR 
	best_clf, accuracy, CM , y_true , y_pred = AdaBoost_clf(MIR_DS_PCA[0],MIR_DS_PCA[1],MIR_DS_PCA[2],MIR_DS_PCA[3],pca=True)
	print('AdaBoost Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'MIR_PCA','AdaBoost')
	print_save(ID,pl_list,' MIR_PCA ', 'AdaBoost', y_true, y_pred, MIR_DS_PCA[1], MIR_DS_PCA[3], best_clf.get_params)
	
	## DS = SPO
	best_clf, accuracy, CM , y_true , y_pred = AdaBoost_clf(SPO_DS_PCA[0],SPO_DS_PCA[1],SPO_DS_PCA[2],SPO_DS_PCA[3],pca=True)
	print('AdaBoost Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'SPO_PCA','AdaBoost')
	print_save(ID,pl_list,' SPO_PCA ', 'AdaBoost', y_true, y_pred, SPO_DS_PCA[1], SPO_DS_PCA[3], best_clf.get_params)

	## DS = LYR
	best_clf, accuracy, CM , y_true , y_pred = AdaBoost_clf(LYR_DS_PCA[0],LYR_DS_PCA[1],LYR_DS_PCA[2],LYR_DS_PCA[3],pca=True)
	print('AdaBoost Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'LYR_PCA','AdaBoost')
	print_save(ID,pl_list,' LYR_PCA ', 'AdaBoost', y_true, y_pred, LYR_DS_PCA[1], LYR_DS_PCA[3], best_clf.get_params)


	## PCA == False
	## DS = ALL
	best_clf, accuracy, CM , y_true , y_pred = AdaBoost_clf(ALL_DS_noPCA[0],ALL_DS_noPCA[1],ALL_DS_noPCA[2],ALL_DS_noPCA[3],pca=False)
	print('AdaBoost Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'ALL','AdaBoost')
	print_save(ID,pl_list,' ALL_noPCA ', 'AdaBoost', y_true, y_pred, ALL_DS_noPCA[1], ALL_DS_noPCA[3], best_clf.get_params)

	## DS = MIR 
	best_clf, accuracy, CM , y_true , y_pred = AdaBoost_clf(MIR_DS_noPCA[0],MIR_DS_noPCA[1],MIR_DS_noPCA[2],MIR_DS_noPCA[3],pca=False)
	print('AdaBoost Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'MIR','AdaBoost')
	print_save(ID,pl_list,' MIR_noPCA ', 'AdaBoost', y_true, y_pred, MIR_DS_noPCA[1], MIR_DS_noPCA[3], best_clf.get_params)

	## DS = SPO
	best_clf, accuracy, CM , y_true , y_pred = AdaBoost_clf(SPO_DS_noPCA[0],SPO_DS_noPCA[1],SPO_DS_noPCA[2],SPO_DS_noPCA[3],pca=False)
	print('AdaBoost Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'SPO','AdaBoost')
	print_save(ID,pl_list,' SPO_noPCA ', 'AdaBoost', y_true, y_pred, SPO_DS_noPCA[1], SPO_DS_noPCA[3], best_clf.get_params)

	## DS = LYR 
	best_clf, accuracy, CM , y_true , y_pred = AdaBoost_clf(LYR_DS_noPCA[0],LYR_DS_noPCA[1],LYR_DS_noPCA[2],LYR_DS_noPCA[3],pca=False)
	print('AdaBoost Accuracy -> '+"{:.2%}".format(accuracy))
	CM_plotter(CM,pl_list,ID,'LYR','AdaBoost')
	print_save(ID,pl_list,' LYR_noPCA ', 'AdaBoost', y_true, y_pred, LYR_DS_noPCA[1], LYR_DS_noPCA[3], best_clf.get_params)



def get_X_and_y_SVM(ID, pl_list, PCA):
	## init X and y 
	y_MIR = [] 
	y_SPO = []
	y_LYR = []

	y_ALL = []

	X_MIR = []
	X_SPO = []
	X_LYR = []

	X_ALL = []

	for index, pl in enumerate(pl_list):

		## unpack tuples
		user = pl[0]
		pl_name = pl[1]
		pl_id = pl[2]

		## open all feature sets:
		f_mir = pl_data+user+'/'+pl_id+'/MIRaudio_features.tsv'
		f_sp  = pl_data+user+'/'+pl_id+'/Spotify_features.tsv'
		f_ly  = pl_data+user+'/'+pl_id+'/Lyrics_features.tsv'
			
		MIR_df = pd.read_csv(f_mir,sep='\t').set_index('Song_ID')
		Sp_df  = pd.read_csv(f_sp,sep='\t').set_index('Song_ID')
		Lyr_df = pd.read_csv(f_ly,sep='\t').set_index('Song_ID')

		## dropping rows with NaN
		MIR_df = MIR_df.dropna()
		Sp_df  = Sp_df.dropna()
		Lyr_df = Lyr_df.dropna()

		'''
		print(MIR_df)
		print('MIR-shape:')
		print(MIR_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		print(Sp_df)
		print('SPO-shape:')
		print(Sp_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		print(Lyr_df)
		print('LYR-shape')
		print(Lyr_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		'''
		## combine feature sets into one
		ALL_df = create_ALL_df(MIR_df,Sp_df,Lyr_df)

		ALL_df = ALL_df.dropna()
		'''
		print(' - - - - - - - - - - - - - - - - - - - ')
		print(ALL_df)
		print('ALL-shape')
		print(ALL_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		'''

		## DF to NP
		MIR_np = MIR_df.to_numpy()
		Sp_np  = Sp_df.to_numpy()
		Lyr_np = Lyr_df.to_numpy()
		ALL_np = ALL_df.to_numpy()


		## creating X-matrix
		if len(X_MIR)==0: ## init
			X_MIR = MIR_np
			X_SPO = Sp_np
			X_LYR = Lyr_np
			X_ALL = ALL_np

		else:
			X_MIR = np.vstack((X_MIR,MIR_np))
			X_SPO = np.vstack((X_SPO,Sp_np))
			X_LYR = np.vstack((X_LYR,Lyr_np))
			X_ALL = np.vstack((X_ALL,ALL_np))

		## creating y-vector
		## for each feature sets

		## Lyrics-features
		size = Lyr_np.shape
		y_ = [index] * size[0]
		y_LYR.append(y_)

		## Spotify-features
		size = Sp_np.shape
		y_ = [index] * size[0]
		y_SPO.append(y_)

		## MIR-features
		size = MIR_np.shape
		y_ = [index] * size[0]
		y_MIR.append(y_)

		size = ALL_np.shape
		y_ = [index]*size[0]
		y_ALL.append(y_)

	y_MIR = list(chain(*y_MIR))
	y_SPO = list(chain(*y_SPO))
	y_LYR = list(chain(*y_LYR))
	y_ALL = list(chain(*y_ALL))

	y_MIR = np.array([y_MIR]).T
	y_SPO = np.array([y_SPO]).T
	y_LYR = np.array([y_LYR]).T
	y_ALL = np.array([y_ALL]).T


	print("- Shapes MIR: ")
	print(y_MIR.shape)
	print(X_MIR.shape)

	print("- Shapes Spotify: ")
	print(y_SPO.shape)
	print(X_SPO.shape)

	print("- Shapes Lyrics: ")
	print(y_LYR.shape)
	print(X_LYR.shape)

	print("- Shapes Combined Feature Set")
	print(y_ALL.shape)
	print(X_ALL.shape)

	# PCA + SVD 
	if PCA==True:
		print('- reducing dimensions with PCA... ')
		print(' ')
		print(' -> reducing MIR')
		X_MIR = do_pca(X_MIR)
		print("- Shapes MIR: ")
		print(y_MIR.shape)
		print(X_MIR.shape)
		print(' ')
		print(' -> reducing SPO:')
		X_SPO = do_pca(X_SPO)
		print("- Shapes Spotify: ")
		print(y_SPO.shape)
		print(X_SPO.shape)
		print(' ')
		print(' -> reducing LYR:')
		X_LYR = do_pca(X_LYR)
		print("- Shapes Lyrics: ")
		print(y_LYR.shape)
		print(X_LYR.shape)
		print(' ')
		print(' -> reducing ALL:')
		X_ALL = do_pca(X_ALL)
		print("- Shapes Combined Feature Set")
		print(y_ALL.shape)
		print(X_ALL.shape)


		pca = 'PCA'
	else:  ## PCA == False
		pca = ''

	## SPLIT FOR VALIDATION:
	
	X_MIR, X_MIR_val, y_MIR, y_MIR_val = train_test_split(X_MIR, y_MIR, test_size=0.15,random_state=42)
	X_SPO, X_SPO_val, y_SPO, y_SPO_val = train_test_split(X_SPO, y_SPO, test_size=0.15,random_state=42)
	X_LYR, X_LYR_val, y_LYR, y_LYR_val = train_test_split(X_LYR, y_LYR, test_size=0.15,random_state=42)
	X_ALL, X_ALL_val, y_ALL, y_ALL_val = train_test_split(X_ALL, y_ALL, test_size=0.15,random_state=42)
	
	## choose a scaler
	ss = StandardScaler()
	X_MIR,X_MIR_val = scale_this(ss, X_MIR, X_MIR_val)
	X_SPO,X_SPO_val = scale_this(ss, X_SPO, X_SPO_val)
	X_LYR,X_LYR_val = scale_this(ss, X_LYR, X_LYR_val)
	X_ALL,X_ALL_val = scale_this(ss, X_ALL, X_ALL_val)

	save_npy(X_MIR, X_MIR_val, y_MIR, y_MIR_val,'MIR'+pca,'SVM')
	save_npy(X_SPO, X_SPO_val, y_SPO, y_SPO_val,'SPO'+pca,'SVM')
	save_npy(X_LYR, X_LYR_val, y_LYR, y_LYR_val,'LYR'+pca,'SVM')
	save_npy(X_ALL, X_ALL_val, y_ALL, y_ALL_val,'ALL'+pca,'SVM')


	## train different models with 
	## this paer trains the parameteres for all the different models FR SVM kNN


	## Select KFOLD! CV 
	## if user input use these 2 lines:
	'''Kfold_text = input ("Enter the # of K-Fold Crossvalidation: ")
	num_folds = int(Kfold_text)'''
	## else:      
	num_folds = 5  ##  !! KFOLD SELECT 

	param_folder = home+'parameters/'+str(ID)+'/'
	

	if not os.path.exists(param_folder):
		os.makedirs(param_folder)
	if not os.path.exists(param_folder+'MIR'+pca+'/'):
		os.makedirs(param_folder+'MIR'+pca+'/')
	if not os.path.exists(param_folder+'SPO'+pca+'/'):
		os.makedirs(param_folder+'SPO'+pca+'/')
	if not os.path.exists(param_folder+'LYR'+pca+'/'):
		os.makedirs(param_folder+'LYR'+pca+'/')
	if not os.path.exists(param_folder+'ALL'+pca+'/'):
		os.makedirs(param_folder+'ALL'+pca+'/')

	## SVM - MIR 
	param_file = param_folder+'/MIR'+pca+'/'+'SVM.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	SVM - Parameter Models')
		print('	MIR:')
		model_parameters_SVM_MIR = open_parameters(param_file,ID)#,y_MIR,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	SVM - Parameter Models')
		print('	MIR:')
		model_parameters_SVM_MIR = perform_grid_search('SVM',X_MIR,y_MIR,num_folds)
		save_parameters(param_folder,model_parameters_SVM_MIR,'SVM','MIR'+pca,ID)	

	## SVM - SPO
	param_file = param_folder+'/SPO'+pca+'/'+'SVM.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	SVM - Parameter Models')
		print('	SPO:')
		model_parameters_SVM_SPO = open_parameters(param_file,ID)#X_SPO,y_SPO,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	SVM - Parameter Models')
		print('	SPO:')
		model_parameters_SVM_SPO = perform_grid_search('SVM',X_SPO,y_SPO,num_folds)
		save_parameters(param_folder,model_parameters_SVM_SPO,'SVM','SPO'+pca,ID)		

	## SVM - LYR
	param_file = param_folder+'/LYR'+pca+'/'+'SVM.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	SVM - Parameter Models')
		print('	LYR:')
		model_parameters_SVM_LYR = open_parameters(param_file,ID)#X_LYR,y_LYR,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	SVM - Parameter Models')
		print('	LYR:')
		model_parameters_SVM_LYR = perform_grid_search('SVM',X_LYR,y_LYR,num_folds)
		save_parameters(param_folder,model_parameters_SVM_LYR,'SVM','LYR'+pca,ID)			

	## SVM - ALL 
	param_file = param_folder+'/ALL'+pca+'/'+'SVM.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	SVM - Parameter Models')
		print('	ALL:')
		model_parameters_SVM_ALL = open_parameters(param_file,ID)#X_ALL,y_ALL,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	SVM - Parameter Models')
		print('	ALL:')
		model_parameters_SVM_ALL = perform_grid_search('SVM',X_ALL,y_ALL,num_folds)
		save_parameters(param_folder,model_parameters_SVM_ALL,'SVM','ALL'+pca,ID)			
	'''
	else:
		print(' # # # # # # ')
		print('- Loading all parameters saved:')
		model_parameters_SVM_MIR = open_parameters('SVM','MIR',ID)
		model_parameters_SVM_SPO = open_parameters('SVM','SPO',ID)
		model_parameters_SVM_LYR = open_parameters('SVM','LYR',ID)
		model_parameters_SVM_ALL = open_parameters('SVM','ALL',ID)
	'''

	
	print(' # # # # # # # # # # # # # # # # # # ')
	print('')
	print(' USING PARAMETERS FROM GRID_SEARCH ')
	print('')
	print(' # # # # # # # # # # # # # # # # # # ')
	

	SVM_clf_MIR, score = train_SVM(X_MIR, y_MIR, num_folds, model_parameters_SVM_MIR)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(SVM_clf_MIR,'SVM','MIR'+pca,ID)

	SVM_clf_SPO, score = train_SVM(X_SPO,y_SPO, num_folds, model_parameters_SVM_SPO)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(SVM_clf_SPO,'SVM','SPO'+pca,ID)
	
	SVM_clf_LYR ,score = train_SVM(X_LYR,y_LYR, num_folds, model_parameters_SVM_LYR)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(SVM_clf_LYR,'SVM','LYR'+pca,ID)
	
	SVM_clf_ALL, score = train_SVM(X_ALL, y_ALL, num_folds, model_parameters_SVM_ALL)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(SVM_clf_ALL,'SVM','ALL'+pca,ID)

	print(" - VALIDATION: ")
	print('')
	print(' # # # # # # # # # # # # # # # # # # ')
	## 		SPOTIFY DATASET 
	## 		SVM
	y_true, y_pred = y_SPO_val, SVM_clf_SPO.predict(X_SPO_val)
	cm=confusion_matrix(y_true, y_pred)
	CM_plotter(cm,pl_list,'EXP2','SPO','SVM')
	print_save(ID,pl_list,'SPOTIFY DATASET '+pca,' SVM ',y_true, y_pred, X_SPO_val, y_SPO_val,model_parameters_SVM_SPO)

	## 		LYRICS DATASET 
	## 		SVM
	y_true, y_pred = y_LYR_val, SVM_clf_LYR.predict(X_LYR_val)
	cm=confusion_matrix(y_true, y_pred)
	CM_plotter(cm,pl_list,'EXP2','LYR','SVM')
	print_save(ID,pl_list,'LYRICS DATASET '+pca,' SVM ',y_true, y_pred, X_LYR_val, y_LYR_val,model_parameters_SVM_LYR)

	## 		MIR DATASET
	## 		SVM
	y_true, y_pred = y_MIR_val, SVM_clf_MIR.predict(X_MIR_val)
	cm=confusion_matrix(y_true, y_pred)
	CM_plotter(cm,pl_list,'EXP2','MIR','SVM')
	print_save(ID,pl_list,'  MIR  DATASET '+pca,' SVM ',y_true, y_pred, X_MIR_val, y_MIR_val,model_parameters_SVM_MIR)

	## 		ALL DATASET 
	##		SVM
	y_true, y_pred = y_ALL_val, SVM_clf_ALL.predict(X_ALL_val)
	cm=confusion_matrix(y_true, y_pred)
	CM_plotter(cm,pl_list,'EXP2','ALL','SVM')
	print_save(ID,pl_list,'  ALL  DATASET '+pca,' SVM ',y_true, y_pred, X_ALL_val, y_ALL_val,model_parameters_SVM_ALL)

def get_X_and_y_RF(ID, pl_list,PCA):

	## init X and y 
	y_MIR = [] 
	y_SPO = []
	y_LYR = []

	y_ALL = []

	X_MIR = []
	X_SPO = []
	X_LYR = []

	X_ALL = []

	for index, pl in enumerate(pl_list):

		## unpack tuples
		user = pl[0]
		pl_name = pl[1]
		pl_id = pl[2]


		## open all feature sets:
		f_mir = pl_data+user+'/'+pl_id+'/MIRaudio_features.tsv'
		f_sp  = pl_data+user+'/'+pl_id+'/Spotify_features.tsv'
		f_ly  = pl_data+user+'/'+pl_id+'/Lyrics_features.tsv'
			
		MIR_df = pd.read_csv(f_mir,sep='\t').set_index('Song_ID')
		Sp_df  = pd.read_csv(f_sp,sep='\t').set_index('Song_ID')
		Lyr_df = pd.read_csv(f_ly,sep='\t').set_index('Song_ID')

		## dropping rows with NaN
		MIR_df = MIR_df.dropna()
		Sp_df  = Sp_df.dropna()
		Lyr_df = Lyr_df.dropna()
		'''
		print(MIR_df)
		print('MIR-shape:')
		print(MIR_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		print(Sp_df)
		print('SPO-shape:')
		print(Sp_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		print(Lyr_df)
		print('LYR-shape')
		print(Lyr_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		'''
		## combine feature sets into one
		ALL_df = create_ALL_df(MIR_df,Sp_df,Lyr_df)

		ALL_df = ALL_df.dropna()
		'''
		print(' - - - - - - - - - - - - - - - - - - - ')
		print(ALL_df)
		print('ALL-shape')
		print(ALL_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		'''

		## DF to NP
		MIR_np = MIR_df.to_numpy()
		Sp_np  = Sp_df.to_numpy()
		Lyr_np = Lyr_df.to_numpy()
		ALL_np = ALL_df.to_numpy()


		## creating X-matrix
		if len(X_MIR)==0: ## init
			X_MIR = MIR_np
			X_SPO = Sp_np
			X_LYR = Lyr_np
			X_ALL = ALL_np

		else:
			X_MIR = np.vstack((X_MIR,MIR_np))
			X_SPO = np.vstack((X_SPO,Sp_np))
			X_LYR = np.vstack((X_LYR,Lyr_np))
			X_ALL = np.vstack((X_ALL,ALL_np))

		## creating y-vector
		## for each feature sets

		## Lyrics-features
		size = Lyr_np.shape
		y_ = [index] * size[0]
		y_LYR.append(y_)

		## Spotify-features
		size = Sp_np.shape
		y_ = [index] * size[0]
		y_SPO.append(y_)

		## MIR-features
		size = MIR_np.shape
		y_ = [index] * size[0]
		y_MIR.append(y_)

		size = ALL_np.shape
		y_ = [index]*size[0]
		y_ALL.append(y_)

	y_MIR = list(chain(*y_MIR))
	y_SPO = list(chain(*y_SPO))
	y_LYR = list(chain(*y_LYR))
	y_ALL = list(chain(*y_ALL))

	y_MIR = np.array([y_MIR]).T
	y_SPO = np.array([y_SPO]).T
	y_LYR = np.array([y_LYR]).T
	y_ALL = np.array([y_ALL]).T

	print("- Shapes MIR: ")
	print(y_MIR.shape)
	print(X_MIR.shape)

	print("- Shapes Spotify: ")
	print(y_SPO.shape)
	print(X_SPO.shape)

	print("- Shapes Lyrics: ")
	print(y_LYR.shape)
	print(X_LYR.shape)

	print("- Shapes Combined Feature Set")
	print(y_ALL.shape)
	print(X_ALL.shape)

		# PCA + SVD 
	if PCA==True:
		print('- reducing dimensions with PCA... ')
		print(' ')
		print(' -> reducing MIR')
		X_MIR = do_pca(X_MIR)
		print("- Shapes MIR: ")
		print(y_MIR.shape)
		print(X_MIR.shape)
		print(' ')
		print(' -> reducing SPO:')
		X_SPO = do_pca(X_SPO)
		print("- Shapes Spotify: ")
		print(y_SPO.shape)
		print(X_SPO.shape)
		print(' ')
		print(' -> reducing LYR:')
		X_LYR = do_pca(X_LYR)
		print("- Shapes Lyrics: ")
		print(y_LYR.shape)
		print(X_LYR.shape)
		print(' ')
		print(' -> reducing ALL:')
		X_ALL = do_pca(X_ALL)
		print("- Shapes Combined Feature Set")
		print(y_ALL.shape)
		print(X_ALL.shape)


		pca = 'PCA'
	else:  ## PCA == False
		pca = ''

	## SPLIT FOR VALIDATION:
	
	X_MIR, X_MIR_val, y_MIR, y_MIR_val = train_test_split(X_MIR, y_MIR, test_size=0.15,random_state=42)
	X_SPO, X_SPO_val, y_SPO, y_SPO_val = train_test_split(X_SPO, y_SPO, test_size=0.15,random_state=42)
	X_LYR, X_LYR_val, y_LYR, y_LYR_val = train_test_split(X_LYR, y_LYR, test_size=0.15,random_state=42)
	X_ALL, X_ALL_val, y_ALL, y_ALL_val = train_test_split(X_ALL, y_ALL, test_size=0.15,random_state=42)

	## choose a scaler
	ss = StandardScaler()
	X_MIR,X_MIR_val = scale_this(ss, X_MIR, X_MIR_val)
	X_SPO,X_SPO_val = scale_this(ss, X_SPO, X_SPO_val)
	X_LYR,X_LYR_val = scale_this(ss, X_LYR, X_LYR_val)
	X_ALL,X_ALL_val = scale_this(ss, X_ALL, X_ALL_val)
	
	save_npy(X_MIR, X_MIR_val, y_MIR, y_MIR_val,'MIR'+pca,'RF')
	save_npy(X_SPO, X_SPO_val, y_SPO, y_SPO_val,'SPO'+pca,'RF')
	save_npy(X_LYR, X_LYR_val, y_LYR, y_LYR_val,'LYR'+pca,'RF')
	save_npy(X_ALL, X_ALL_val, y_ALL, y_ALL_val,'ALL'+pca,'RF')

	
	## train different models with 
	## this paer trains the parameteres for all the different models FR SVM kNN


	## Select KFOLD! CV 
	## if user input use these 2 lines:
	'''Kfold_text = input ("Enter the # of K-Fold Crossvalidation: ")
	num_folds = int(Kfold_text)'''
	## else:      
	num_folds = 5  ##  !! KFOLD SELECT 

	'''
	param_folder = '/Users/guillermoventuramartinezAIR/Desktop/FP/parameters/PL_Set_'+str(ID)+'/'
	if not os.path.exists(param_folder):
		os.makedirs(param_folder)
	
	read_param_folder = os.listdir(param_folder)
	
	'''
	
	param_folder = home+'parameters/'+str(ID)+'/'


	if not os.path.exists(param_folder):
		os.makedirs(param_folder)
	if not os.path.exists(param_folder+'MIR'+pca+'/'):
		os.makedirs(param_folder+'MIR'+pca+'/')
	if not os.path.exists(param_folder+'SPO'+pca+'/'):
		os.makedirs(param_folder+'SPO'+pca+'/')
	if not os.path.exists(param_folder+'LYR'+pca+'/'):
		os.makedirs(param_folder+'LYR'+pca+'/')
	if not os.path.exists(param_folder+'ALL'+pca+'/'):
		os.makedirs(param_folder+'ALL'+pca+'/')

	#	parameters_trained = parameters_questionmark():
	## have here a switch for the case of the param saved already

	## RF - MIR 
	# from save_parameters: param_folder+clf+'.pkl'
	param_file = param_folder+'/MIR'+pca+'/'+'RF.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	RF - Parameter Models')
		print('	MIR:')
		model_parameters_RF_MIR = open_parameters(param_file,ID)#,y_MIR,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	RF - Parameter Models')
		print('	MIR:')
		model_parameters_RF_MIR = perform_grid_search('RF',X_MIR,y_MIR,num_folds)
		save_parameters(param_folder,model_parameters_RF_MIR,'RF','MIR'+pca,ID)		

	## RF - SPO
	param_file = param_folder+'/SPO'+pca+'/'+'RF.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	RF - Parameter Models')
		print('	SPO:')
		model_parameters_RF_SPO = open_parameters(param_file,ID)#X_SPO,y_SPO,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	RF - Parameter Models')
		print('	SPO:')
		model_parameters_RF_SPO = perform_grid_search('RF',X_SPO,y_SPO,num_folds)
		save_parameters(param_folder,model_parameters_RF_SPO,'RF','SPO'+pca,ID)			

	## RF - LYR
	param_file = param_folder+'/LYR'+pca+'/'+'RF.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	RF - Parameter Models')
		print('	LYR:')
		model_parameters_RF_LYR = open_parameters(param_file,ID)#X_LYR,y_LYR,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	RF - Parameter Models')
		print('	LYR:')
		model_parameters_RF_LYR = perform_grid_search('RF',X_LYR,y_LYR,num_folds)
		save_parameters(param_folder,model_parameters_RF_LYR,'RF','LYR'+pca,ID)			

	## RF - ALL 
	param_file = param_folder+'/ALL'+pca+'/'+'RF.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	RF - Parameter Models')
		print('	ALL:')
		model_parameters_RF_ALL = open_parameters(param_file,ID)#X_ALL,y_ALL,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	RF - Parameter Models')
		print('	ALL:')
		model_parameters_RF_ALL = perform_grid_search('RF',X_ALL,y_ALL,num_folds)
		save_parameters(param_folder,model_parameters_RF_ALL,'RF','ALL'+pca,ID)		



	print(' # # # # # # # # # # # # # # # # # # ')
	print('')
	print(' USING PARAMETERS FROM GRID_SEARCH ')
	print('')
	print(' # # # # # # # # # # # # # # # # # # ')

	RF_clf_MIR, score = train_RF(X_MIR, y_MIR, num_folds, model_parameters_RF_MIR)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(RF_clf_MIR,'RF','MIR'+pca,ID)

	RF_clf_SPO, score = train_RF(X_SPO,y_SPO, num_folds, model_parameters_RF_SPO)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(RF_clf_SPO,'RF','SPO'+pca,ID)
	
	RF_clf_LYR ,score = train_RF(X_LYR,y_LYR, num_folds, model_parameters_RF_LYR)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(RF_clf_LYR,'RF','LYR'+pca,ID)
	
	RF_clf_ALL, score= train_RF(X_ALL, y_ALL, num_folds, model_parameters_RF_ALL)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(RF_clf_ALL,'RF','ALL'+pca,ID)


	print(" - VALIDATION: ")
	print('')
	print(' # # # # # # # # # # # # # # # # # # ')
	## 		SPOTIFY DATASET 
	## 		RF
	y_true, y_pred = y_SPO_val, RF_clf_SPO.predict(X_SPO_val)
	cm=confusion_matrix(y_true, y_pred)
	CM_plotter(cm,pl_list,'EXP2','SPO','RF')
	print_save(ID, pl_list,'SPOTIFY DATASET '+pca,' RF ',y_true, y_pred, X_SPO_val, y_SPO_val, model_parameters_RF_SPO)

	## 		LYRICS DATASET 
	## 		RF
	y_true, y_pred = y_LYR_val, RF_clf_LYR.predict(X_LYR_val)
	cm=confusion_matrix(y_true, y_pred)
	CM_plotter(cm,pl_list,'EXP2','LYR','RF')
	print_save(ID,pl_list,'LYRICS DATASET '+pca,' RF ',y_true, y_pred, X_LYR_val, y_LYR_val, model_parameters_RF_LYR)

	## 		MIR DATASET
	## 		RF
	y_true, y_pred = y_MIR_val, RF_clf_MIR.predict(X_MIR_val)
	cm=confusion_matrix(y_true, y_pred)
	CM_plotter(cm,pl_list,'EXP2','MIR','RF')
	print_save(ID,pl_list,'  MIR  DATASET '+pca,' RF ',y_true, y_pred, X_MIR_val, y_MIR_val, model_parameters_RF_MIR)

	## 		ALL DATASET 
	##		RF
	y_true, y_pred = y_ALL_val, RF_clf_ALL.predict(X_ALL_val)
	cm=confusion_matrix(y_true, y_pred)
	CM_plotter(cm,pl_list,'EXP2','ALL','RF')
	print_save(ID,pl_list,'  ALL  DATASET '+pca,' RF ',y_true, y_pred, X_ALL_val, y_ALL_val, model_parameters_RF_ALL)

def get_X_and_y_kNN(ID, pl_list,PCA):

	## init X and y 
	y_MIR = [] 
	y_SPO = []
	y_LYR = []

	y_ALL = []

	X_MIR = []
	X_SPO = []
	X_LYR = []

	X_ALL = []

	for index, pl in enumerate(pl_list):

		## unpack tuples
		user = pl[0]
		pl_name = pl[1]
		pl_id = pl[2]


		## open all feature sets:
		f_mir = pl_data+user+'/'+pl_id+'/MIRaudio_features.tsv'
		f_sp  = pl_data+user+'/'+pl_id+'/Spotify_features.tsv'
		f_ly  = pl_data+user+'/'+pl_id+'/Lyrics_features.tsv'
			
		MIR_df = pd.read_csv(f_mir,sep='\t').set_index('Song_ID')
		Sp_df  = pd.read_csv(f_sp,sep='\t').set_index('Song_ID')
		Lyr_df = pd.read_csv(f_ly,sep='\t').set_index('Song_ID')

		## dropping rows with NaN
		MIR_df = MIR_df.dropna()
		Sp_df  = Sp_df.dropna()
		Lyr_df = Lyr_df.dropna()

		## combine feature sets into one
		#ALL_df = pd.concat([MIR_df,Sp_df,Lyr_df],axis=1,verify_integrity=True)
		'''
		print(MIR_df)
		print('MIR-shape:')
		print(MIR_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		print(Sp_df)
		print('SPO-shape:')
		print(Sp_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		print(Lyr_df)
		print('LYR-shape')
		print(Lyr_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		'''
		## combine feature sets into one
		ALL_df = create_ALL_df(MIR_df,Sp_df,Lyr_df)

		ALL_df = ALL_df.dropna()
		'''
		print(' - - - - - - - - - - - - - - - - - - - ')
		print(ALL_df)
		print('ALL-shape')
		print(ALL_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		'''
		

		## DF to NP
		MIR_np = MIR_df.to_numpy()
		Sp_np  = Sp_df.to_numpy()
		Lyr_np = Lyr_df.to_numpy()
		ALL_np = ALL_df.to_numpy()


		## creating X-matrix
		if len(X_MIR)==0: ## init
			X_MIR = MIR_np
			X_SPO = Sp_np
			X_LYR = Lyr_np
			X_ALL = ALL_np

		else:
			X_MIR = np.vstack((X_MIR,MIR_np))
			X_SPO = np.vstack((X_SPO,Sp_np))
			X_LYR = np.vstack((X_LYR,Lyr_np))
			X_ALL = np.vstack((X_ALL,ALL_np))

		## creating y-vector
		## for each feature sets

		## Lyrics-features
		size = Lyr_np.shape
		y_ = [index] * size[0]
		y_LYR.append(y_)

		## Spotify-features
		size = Sp_np.shape
		y_ = [index] * size[0]
		y_SPO.append(y_)

		## MIR-features
		size = MIR_np.shape
		y_ = [index] * size[0]
		y_MIR.append(y_)

		size = ALL_np.shape
		y_ = [index]*size[0]
		y_ALL.append(y_)

	y_MIR = list(chain(*y_MIR))
	y_SPO = list(chain(*y_SPO))
	y_LYR = list(chain(*y_LYR))
	y_ALL = list(chain(*y_ALL))

	y_MIR = np.array([y_MIR]).T
	y_SPO = np.array([y_SPO]).T
	y_LYR = np.array([y_LYR]).T
	y_ALL = np.array([y_ALL]).T

	print("- Shapes MIR: ")
	print(y_MIR.shape)
	print(X_MIR.shape)

	print("- Shapes Spotify: ")
	print(y_SPO.shape)
	print(X_SPO.shape)

	print("- Shapes Lyrics: ")
	print(y_LYR.shape)
	print(X_LYR.shape)

	print("- Shapes Combined Feature Set")
	print(y_ALL.shape)
	print(X_ALL.shape)
	# PCA + SVD 
	if PCA==True:
		print('- reducing dimensions with PCA... ')
		print(' ')
		print(' -> reducing MIR')
		X_MIR = do_pca(X_MIR)
		print("- Shapes MIR: ")
		print(y_MIR.shape)
		print(X_MIR.shape)
		print(' ')
		print(' -> reducing SPO:')
		X_SPO = do_pca(X_SPO)
		print("- Shapes Spotify: ")
		print(y_SPO.shape)
		print(X_SPO.shape)
		print(' ')
		print(' -> reducing LYR:')
		X_LYR = do_pca(X_LYR)
		print("- Shapes Lyrics: ")
		print(y_LYR.shape)
		print(X_LYR.shape)
		print(' ')
		print(' -> reducing ALL:')
		X_ALL = do_pca(X_ALL)
		print("- Shapes Combined Feature Set")
		print(y_ALL.shape)
		print(X_ALL.shape)

		pca = 'PCA'
	
	else:  ## PCA == False
		pca = ''

	## SPLIT FOR VALIDATION:
	
	X_MIR, X_MIR_val, y_MIR, y_MIR_val = train_test_split(X_MIR, y_MIR, test_size=0.15,random_state=42)
	X_SPO, X_SPO_val, y_SPO, y_SPO_val = train_test_split(X_SPO, y_SPO, test_size=0.15,random_state=42)
	X_LYR, X_LYR_val, y_LYR, y_LYR_val = train_test_split(X_LYR, y_LYR, test_size=0.15,random_state=42)
	X_ALL, X_ALL_val, y_ALL, y_ALL_val = train_test_split(X_ALL, y_ALL, test_size=0.15,random_state=42)

	## choose a scaler
	ss = StandardScaler()
	X_MIR,X_MIR_val = scale_this(ss, X_MIR, X_MIR_val)
	X_SPO,X_SPO_val = scale_this(ss, X_SPO, X_SPO_val)
	X_LYR,X_LYR_val = scale_this(ss, X_LYR, X_LYR_val)
	X_ALL,X_ALL_val = scale_this(ss, X_ALL, X_ALL_val)
	
	save_npy(X_MIR, X_MIR_val, y_MIR, y_MIR_val,'MIR'+pca,'kNN')
	save_npy(X_SPO, X_SPO_val, y_SPO, y_SPO_val,'SPO'+pca,'kNN')
	save_npy(X_LYR, X_LYR_val, y_LYR, y_LYR_val,'LYR'+pca,'kNN')
	save_npy(X_ALL, X_ALL_val, y_ALL, y_ALL_val,'ALL'+pca,'kNN')

	## PCA + SVD 
	#X_MIR = do_pca(X_MIR)
	#X_ALL = do_pca(X_ALL)
	
	## train different models with 
	## this paer trains the parameteres for all the different models FR SVM kNN


	## Select KFOLD! CV 
	## if user input use these 2 lines:
	'''Kfold_text = input ("Enter the # of K-Fold Crossvalidation: ")
	num_folds = int(Kfold_text)'''
	## else:      
	num_folds = 5  ##  !! KFOLD SELECT 

	param_folder = home+'parameters/'+str(ID)+'/'

	if not os.path.exists(param_folder):
		os.makedirs(param_folder)
	if not os.path.exists(param_folder+'MIR'+pca+'/'):
		os.makedirs(param_folder+'MIR'+pca+'/')
	if not os.path.exists(param_folder+'SPO'+pca+'/'):
		os.makedirs(param_folder+'SPO'+pca+'/')
	if not os.path.exists(param_folder+'LYR'+pca+'/'):
		os.makedirs(param_folder+'LYR'+pca+'/')
	if not os.path.exists(param_folder+'ALL'+pca+'/'):
		os.makedirs(param_folder+'ALL'+pca+'/')

	#LYR,SPO,MIR,ALL
	## kNN - MIR 
	param_file = param_folder+'/MIR'+pca+'/'+'kNN.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	MIR:')
		model_parameters_kNN_MIR = open_parameters(param_file,ID)#,y_MIR,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	MIR:')
		model_parameters_kNN_MIR = perform_grid_search('kNN',X_MIR,y_MIR,num_folds)
		save_parameters(param_folder,model_parameters_kNN_MIR,'kNN','MIR'+pca,ID)	

	## kNN - SPO
	param_file = param_folder+'/SPO'+pca+'/'+'kNN.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	SPO:')
		model_parameters_kNN_SPO = open_parameters(param_file,ID)#X_SPO,y_SPO,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	SPO:')
		model_parameters_kNN_SPO = perform_grid_search('kNN',X_SPO,y_SPO,num_folds)
		save_parameters(param_folder,model_parameters_kNN_SPO,'kNN','SPO'+pca,ID)	

	## kNN - LYR
	param_file = param_folder+'/LYR'+pca+'/'+'kNN.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	LYR:')
		model_parameters_kNN_LYR = open_parameters(param_file,ID)#X_LYR,y_LYR,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	LYR:')
		model_parameters_kNN_LYR = perform_grid_search('kNN',X_LYR,y_LYR,num_folds)
		save_parameters(param_folder,model_parameters_kNN_LYR,'kNN','LYR'+pca,ID)		

	## kNN - ALL 
	param_file = param_folder+'/ALL'+pca+'/'+'kNN.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	ALL:')
		model_parameters_kNN_ALL = open_parameters(param_file,ID)#X_ALL,y_ALL,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	ALL:')
		model_parameters_kNN_ALL = perform_grid_search('kNN',X_ALL,y_ALL,num_folds)
		save_parameters(param_folder,model_parameters_kNN_ALL,'kNN','ALL'+pca,ID)		

	'''
	else:
		print(' # # # # # # ')
		print('- Loading all parameters saved:')
		model_parameters_kNN_MIR = open_parameters('kNN','MIR',ID)
		model_parameters_kNN_SPO = open_parameters('kNN','SPO',ID)
		model_parameters_kNN_LYR = open_parameters('kNN','LYR',ID)
		model_parameters_kNN_ALL = open_parameters('kNN','ALL',ID)
	'''

	
	print(' # # # # # # # # # # # # # # # # # # ')
	print('')
	print(' USING PARAMETERS FROM GRID_SEARCH ')
	print('')
	print(' # # # # # # # # # # # # # # # # # # ')
	

	kNN_clf_MIR, score = train_kNN(X_MIR, y_MIR, num_folds, model_parameters_kNN_MIR)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(kNN_clf_MIR,'kNN','MIR'+pca, ID)

	kNN_clf_SPO, score = train_kNN(X_SPO,y_SPO, num_folds, model_parameters_kNN_SPO)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(kNN_clf_SPO,'kNN','SPO'+pca, ID)
	
	kNN_clf_LYR ,score = train_kNN(X_LYR,y_LYR, num_folds, model_parameters_kNN_LYR)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(kNN_clf_LYR,'kNN','LYR'+pca, ID)
	
	kNN_clf_ALL, score = train_kNN(X_ALL, y_ALL, num_folds, model_parameters_kNN_ALL)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(kNN_clf_ALL,'kNN','ALL'+pca, ID)

	print(" - VALIDATION: ")
	print('')
	print(' # # # # # # # # # # # # # # # # # # ')
	## 		SPOTIFY DATASET 
	## 		kNN
	y_true, y_pred = y_SPO_val, kNN_clf_SPO.predict(X_SPO_val)
	cm=confusion_matrix(y_true, y_pred)
	CM_plotter(cm,pl_list,'EXP2','SPO','kNN')
	print_save(ID,pl_list,'SPOTIFY DATASET '+pca,' kNN ',y_true, y_pred, X_SPO_val, y_SPO_val, model_parameters_kNN_SPO)

	## 		LYRICS 	DATASET 
	## 		kNN
	y_true, y_pred = y_LYR_val, kNN_clf_LYR.predict(X_LYR_val)
	cm=confusion_matrix(y_true, y_pred)
	CM_plotter(cm,pl_list,'EXP2','LYR','kNN')
	print_save(ID,pl_list,'LYRICS DATASET '+pca,' kNN ',y_true, y_pred, X_LYR_val, y_LYR_val, model_parameters_kNN_LYR)

	## 		MIR DATASET
	## 		kNN
	y_true, y_pred = y_MIR_val, kNN_clf_MIR.predict(X_MIR_val)
	cm=confusion_matrix(y_true, y_pred)
	CM_plotter(cm,pl_list,'EXP2','MIR','kNN')
	print_save(ID,pl_list,'  MIR  DATASET '+pca,' kNN ',y_true, y_pred, X_MIR_val, y_MIR_val, model_parameters_kNN_MIR)

	## 		ALL DATASET 
	##		kNN
	y_true, y_pred = y_ALL_val, kNN_clf_ALL.predict(X_ALL_val)
	cm=confusion_matrix(y_true, y_pred)
	CM_plotter(cm,pl_list,'EXP2','ALL','KNN')
	print_save(ID,pl_list,'  ALL  DATASET '+pca,' kNN ',y_true, y_pred, X_ALL_val, y_ALL_val, model_parameters_kNN_ALL)

def pre_process(ID, pl_list, PCA):
	## init X and y 
	y_MIR = [] 
	y_SPO = []
	y_LYR = []

	y_ALL = []

	X_MIR = []
	X_SPO = []
	X_LYR = []

	X_ALL = []

	for index, pl in enumerate(pl_list):

		## unpack tuples
		user = pl[0]
		pl_name = pl[1]
		pl_id = pl[2]


		## open all feature sets:
		f_mir = pl_data+user+'/'+pl_id+'/MIRaudio_features.tsv'
		f_sp  = pl_data+user+'/'+pl_id+'/Spotify_features.tsv'
		f_ly  = pl_data+user+'/'+pl_id+'/Lyrics_features.tsv'
			
		MIR_df = pd.read_csv(f_mir,sep='\t').set_index('Song_ID')
		Sp_df  = pd.read_csv(f_sp,sep='\t').set_index('Song_ID')
		Lyr_df = pd.read_csv(f_ly,sep='\t').set_index('Song_ID')

		## combine feature sets into one
		#ALL_df = pd.concat([MIR_df,Sp_df,Lyr_df],axis=1,sort=False)
		ALL_df = MIR_df.join([Sp_df,Lyr_df])

		## dropping rows with NaN
		MIR_df = MIR_df.dropna()
		Sp_df  = Sp_df.dropna()
		Lyr_df = Lyr_df.dropna()

		ALL_df = ALL_df.dropna()

		#print(MIR_df)
		#print(Sp_df)
		#print(Lyr_df)
		#print(ALL_df)

		## DF to NP
		MIR_np = MIR_df.to_numpy()
		Sp_np  = Sp_df.to_numpy()
		Lyr_np = Lyr_df.to_numpy()
		ALL_np = ALL_df.to_numpy()


		## creating X-matrix
		if len(X_MIR)==0: ## init
			X_MIR = MIR_np
			X_SPO = Sp_np
			X_LYR = Lyr_np
			X_ALL = ALL_np

		else:
			X_MIR = np.vstack((X_MIR,MIR_np))
			X_SPO = np.vstack((X_SPO,Sp_np))
			X_LYR = np.vstack((X_LYR,Lyr_np))
			X_ALL = np.vstack((X_ALL,ALL_np))

		## creating y-vector
		## for each feature sets

		## Lyrics-features
		size = Lyr_np.shape
		y_ = [index] * size[0]
		y_LYR.append(y_)

		## Spotify-features
		size = Sp_np.shape
		y_ = [index] * size[0]
		y_SPO.append(y_)

		## MIR-features
		size = MIR_np.shape
		y_ = [index] * size[0]
		y_MIR.append(y_)

		size = ALL_np.shape
		y_ = [index]*size[0]
		y_ALL.append(y_)

	y_MIR = list(chain(*y_MIR))
	y_SPO = list(chain(*y_SPO))
	y_LYR = list(chain(*y_LYR))
	y_ALL = list(chain(*y_ALL))

	y_MIR = np.array([y_MIR]).T
	y_SPO = np.array([y_SPO]).T
	y_LYR = np.array([y_LYR]).T
	y_ALL = np.array([y_ALL]).T


	print("- Shapes MIR: ")
	print(y_MIR.shape)
	print(X_MIR.shape)

	print("- Shapes Spotify: ")
	print(y_SPO.shape)
	print(X_SPO.shape)

	print("- Shapes Lyrics: ")
	print(y_LYR.shape)
	print(X_LYR.shape)

	print("- Shapes Combined Feature Set")
	print(y_ALL.shape)
	print(X_ALL.shape)

	# PCA + SVD 
	if PCA==True:
		print('- reducing dimensions with PCA... ')
		print(' ')
		print(' -> reducing MIR')
		X_MIR = do_pca(X_MIR)
		print("- Shapes MIR: ")
		print(y_MIR.shape)
		print(X_MIR.shape)
		print(' ')
		print(' -> reducing SPO:')
		X_SPO = do_pca(X_SPO)
		print("- Shapes Spotify: ")
		print(y_SPO.shape)
		print(X_SPO.shape)
		print(' ')
		print(' -> reducing LYR:')
		X_LYR = do_pca(X_LYR)
		print("- Shapes Lyrics: ")
		print(y_LYR.shape)
		print(X_LYR.shape)
		print(' ')
		print(' -> reducing ALL:')
		X_ALL = do_pca(X_ALL)
		print("- Shapes Combined Feature Set")
		print(y_ALL.shape)
		print(X_ALL.shape)


		pca = 'PCA'
	else:  ## PCA == False
		pca = ''

	## SPLIT FOR VALIDATION:
	
	X_MIR, X_MIR_val, y_MIR, y_MIR_val = train_test_split(X_MIR, y_MIR, test_size=0.15,random_state=42)
	X_SPO, X_SPO_val, y_SPO, y_SPO_val = train_test_split(X_SPO, y_SPO, test_size=0.15,random_state=42)
	X_LYR, X_LYR_val, y_LYR, y_LYR_val = train_test_split(X_LYR, y_LYR, test_size=0.15,random_state=42)
	X_ALL, X_ALL_val, y_ALL, y_ALL_val = train_test_split(X_ALL, y_ALL, test_size=0.15,random_state=42)

	## choose a scaler
	ss = StandardScaler()
	X_MIR,X_MIR_val = scale_this(ss, X_MIR, X_MIR_val)
	X_SPO,X_SPO_val = scale_this(ss, X_SPO, X_SPO_val)
	X_LYR,X_LYR_val = scale_this(ss, X_LYR, X_LYR_val)
	X_ALL,X_ALL_val = scale_this(ss, X_ALL, X_ALL_val)

	## packing for reading:

	MIR_DS = [X_MIR, X_MIR_val, y_MIR, y_MIR_val]
	SPO_DS = [X_SPO, X_SPO_val, y_SPO, y_SPO_val]
	LYR_DS = [X_LYR, X_LYR_val, y_LYR, y_LYR_val]
	ALL_DS = [X_ALL, X_ALL_val, y_ALL, y_ALL_val]


	return MIR_DS,SPO_DS,LYR_DS,ALL_DS

def scale_this(scaler, X_train, X_test):
	X_train = scaler.fit_transform(X_train)
	X_test  = scaler.transform(X_test)
	return X_train, X_test

def Bagging_clf(X_train,X_test,y_train,y_test,pca):
	y_train = y_train.ravel()
	y_test = y_test.ravel()

	## bagging classifier:
	## from: https://www.kaggle.com/dillonkoch18/bagging-classifier-titanic-submission 
	# create the classifier
	
	grid_params = [{'base_estimator__kernel': ['rbf'],
					'base_estimator__gamma': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100,1000,10000],
					'base_estimator__C': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1, 10,100,1000,10000],
					'n_estimators': [10,50,100,200,300],
					'bootstrap_features':[True, False],
					}]

	if pca==True:
		param_file = home+'parameters/'+'EXP2'+'/ALLPCA/'+'SVM.pkl'
	else:
		param_file = home+'parameters/'+'EXP2'+'/ALL/'+'SVM.pkl'
	param = open_parameters(param_file,'ID')
	
	# Grid Search to determine best parameters
	svm = SVC(param)
	
	bagging_clf = BaggingClassifier(base_estimator=svm, oob_score=False,warm_start=True, random_state=1) #n_estimators=70, random_state=1)
	
	bc_grid = GridSearchCV(estimator=bagging_clf, param_grid=grid_params, cv=5, n_jobs=-1)

	bc_grid.fit(X_train, y_train)
	
	#	bc = BaggingClassifier(base_estimator=base_model)#, oob_score=True, random_state=1) #n_estimators=70, random_state=1)
	
	
	best_params = bc_grid.best_params_
	print(best_params)

	Bagging_clf = bc_grid.best_estimator_
	
	y_true, y_pred = y_test, Bagging_clf.predict(X_test)
	CM = confusion_matrix(y_test,y_pred)
	print(CM)
	print(' --> Validation: ')
	print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	#score.append(accuracy_score(y_true, y_pred))

	## score of best clf
	#best_score, indx = max((val, idx) for (idx, val) in enumerate(score))	

	best_score = bc_grid.best_score_

	return Bagging_clf,best_score, CM , y_true , y_pred

def AdaBoost_clf(X_train,X_test,y_train,y_test,pca):
	y_train = y_train.ravel()
	y_test = y_test.ravel()

	
	
	grid_params = [{'base_estimator__kernel': ['rbf'],
					'base_estimator__gamma': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100,1000,10000],
					'base_estimator__C': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1, 10,100,1000,10000],
					'n_estimators': [10,50,100,200,300],
					'algorithm': ['SAMME']}]#,'SAMME.R']}]

	if pca==True:
		param_file = home+'parameters/'+'EXP2'+'/ALLPCA/'+'SVM.pkl'
	else:
		param_file = home+'parameters/'+'EXP2'+'/ALL/'+'SVM.pkl'
	param = open_parameters(param_file,'ID')
	
	svm = SVC(param)
	
	AdaBoost_clf_ = AdaBoostClassifier(base_estimator=svm)#, oob_score=False,warm_start=True, random_state=1) #n_estimators=70, random_state=1)
	
	ada_grid = GridSearchCV(estimator=AdaBoost_clf_, param_grid=grid_params, cv=5, n_jobs=-1)

	ada_grid.fit(X_train, y_train)
	
	#	bc = BaggingClassifier(base_estimator=base_model)#, oob_score=True, random_state=1) #n_estimators=70, random_state=1)
	
	best_params = ada_grid.best_params_
	print(" - AdaBoost Parameters:")
	print(best_params)

	AdaBoost_clf = ada_grid.best_estimator_

	best_score = ada_grid.best_score_	
	print("- Training score:")
	print(best_score)

	y_true, y_pred = y_test, AdaBoost_clf.predict(X_test)
	CM = confusion_matrix(y_test,y_pred)
	print(CM)
	print(' --> Validation: ')
	print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	
	'''
	print(' -- -- -- -- -- -- -- -- -- -- -- ')
	print('')
	print("Detailed classification report:")
	print("	- Grid scores on development set:")
	print()
	means = grid_search_ABC.cv_results_['mean_test_score']
	stds = grid_search_ABC.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, grid_search_ABC.cv_results_['params']):
		print("- mean: %0.3f \n - std: (+/-%0.03f) \n - parameters: %r"% (mean, std * 2, params))
		print(' * * * * * * * * * * * * * * *')
	print('')
	print(' -- -- -- -- -- -- -- -- -- -- -- ')
	print('')
	'''

	return AdaBoost_clf, best_score, CM , y_true , y_pred
	
#									 MODEL = SVM/RF/kNN
def save_npy(X,X_val,y,y_val,DataSet,MODEL):
	print('- saving Training/Validation Matrixes:')
	#print(ID)
	print(DataSet)
	print(MODEL)
	path__ = home+'/SETS/EXP2/'
	if not os.path.exists(path__):
		os.makedirs(path__)

	path__ = path__+MODEL+'/'
	if not os.path.exists(path__):
		os.makedirs(path__)

	path__DS = path__+DataSet+'/'
	if not os.path.exists(path__DS):
		os.makedirs(path__DS)
	
	path__T = path__DS+'Training/'
	path__V = path__DS+'Validation/'
	
	if not os.path.exists(path__T):
		os.makedirs(path__T)
	
	if not os.path.exists(path__V):
		os.makedirs(path__V)


	## THINGS TO SAVE: X,X_val,y,y_val
	## save Training 
	file = path__T+'X_train'
	np.save(file, X, allow_pickle=True)

	file = path__T+'y_train'
	np.save(file, y, allow_pickle=True)

	file = path__V+'X_val'
	np.save(file, X_val, allow_pickle=True)

	file = path__V+'y_val'
	np.save(file, y_val, allow_pickle=True)

def train_kNN(X,y,num_folds,params):
	clfs = []  ## for best selection

	cms = []	## multiple confussion matrixes
	
	scores = [] 	## for the results
	train_errors = []
	test_errors = []
	training_scores = []
	testing_scores = []


	## for later analysing:
	X_test_l = []
	y_test_l = []

	y = y.ravel()

	scaler = StandardScaler()
	#using 10-fold
	kf = KFold(n_splits= num_folds ,shuffle=True)

	for train_index, test_index in kf.split(X):
		#print("TRAIN:\n", train_index)
		#print("TEST:\n", test_index)
		#print(type(train_index))
		#print(type(test_index))
		print('Training kFold')
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)

		clf = KNeighborsClassifier()

		clf.fit(X_train, y_train)

		train_score = clf.score(X_train, y_train)
		test_score = clf.score(X_test, y_test)

		scores.append(test_score)

		training_scores.append(train_score)
		testing_scores.append(test_score)

		train_errors.append(1 - train_score)
		test_errors.append(1 - test_score)

		X_test_l.append(X_test)
		y_test_l.append(y_test)

		y_pred = clf.predict(X_test)

		cm = confusion_matrix(y_test,y_pred)

		clfs.append(clf)

		cms.append(cm)

	## score of best clf
	best_score, indx = max((val, idx) for (idx, val) in enumerate(scores))
	
	best_clf = clfs[indx]

	#best_clf.get_params()

	
	print(" - - - - - - - - - - - ")
	print("CONFUSSION MATRIX:")
	print(np.asarray(cms))
	print(" - - - - - - - - - - - ")
	print(" - - - - - - - - - - - ")
	print(" BEST CLASSIFICATION SCORE:")
	print("	"+str(best_score*100)+"%")
	print(" - - - - - - - - - - - ")
	print("")
	print("SCORES IN TRAINING: " )
	print("mean ", np.mean(training_scores))
	print("std  ", np.std(training_scores))
	print(training_scores)
	print(" - - - - - - - - - - - " )
	print("Min Test Errors: ",min(test_errors))
	print(" - - - - - - - - - - - ")
	print("SCORES IN TEST: "  )
	print("mean ",np.mean(testing_scores))
	print("std  ",np.std(testing_scores))
	print(testing_scores)
	print(" - - - - - - - - - - - ")
	print("TRAINING Parameters")
	print(" # of folds : ", num_folds)
	print(" PARAMETERS of kNN :")
	print("")
	print(best_clf.get_params())
	print("")
	print(" - - - - - - - - - - - ")

	return best_clf, best_score

def train_RF(X,y,num_folds,params):
	clfs = []  ## for best selection

	cms = []	## multiple confussion matrixes
	
	scores = [] 	## for the results
	train_errors = []
	test_errors = []
	
	training_scores = []
	testing_scores = []

	## for later analysing:
	X_test_l = []
	y_test_l = []

	y = y.ravel()

	parameters 		= params()

	scaler = StandardScaler()

	kf = KFold(n_splits= num_folds ,shuffle=True)

	for train_index, test_index in kf.split(X):
		#print("TRAIN:\n", train_index)
		#print("TEST:\n", test_index)
		#print(type(train_index))
		#print(type(test_index))
		print('Training kFold')

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)

		#clf = RandomForestClassifier(n_estimators=parameters['n_estimators'],bootstrap=parameters['bootstrap'],max_depth=parameters['max_depth'],max_features=parameters['max_features'],min_samples_leaf = parameters['min_samples_leaf'],min_samples_split = parameters['min_samples_split'],random_state=0,criterion=parameters['criterion'])
		clf = RandomForestClassifier(bootstrap=parameters['bootstrap'],n_estimators=parameters['n_estimators'],max_depth=parameters['max_depth'],max_features=parameters['max_features'],min_samples_leaf = parameters['min_samples_leaf'],min_samples_split = parameters['min_samples_split'],random_state=0,criterion=parameters['criterion'])
		clf.fit(X_train, y_train)

		train_score = clf.score(X_train, y_train)
		test_score = clf.score(X_test, y_test)

		scores.append(test_score)

		training_scores.append(train_score)
		testing_scores.append(test_score)

		train_errors.append(1 - train_score)
		test_errors.append(1 - test_score)

		X_test_l.append(X_test)
		y_test_l.append(y_test)

		y_pred = clf.predict(X_test)

		cm = confusion_matrix(y_test,y_pred)

		clfs.append(clf)

		cms.append(cm)

	## score of best clf
	best_score, indx = max((val, idx) for (idx, val) in enumerate(scores))
	
	best_clf = clfs[indx]

	#best_clf.get_params()

	
	print(" - - - - - - - - - - - ")
	print("CONFUSSION MATRIX:")
	print(np.asarray(cms))
	print(" - - - - - - - - - - - ")
	print(" - - - - - - - - - - - ")
	print(" BEST CLASSIFICATION SCORE:")
	print("	"+str(best_score*100)+"%")
	print(" - - - - - - - - - - - ")
	print("")
	print("SCORES IN TRAINING: " )
	print("mean ", np.mean(training_scores))
	print("std  ", np.std(training_scores))
	print(training_scores)
	print(" - - - - - - - - - - - ")
	print("Min Test Errors: ",min(test_errors))
	print(" - - - - - - - - - - - ")	
	print("SCORES IN TEST: "  )
	print("mean ",np.mean(testing_scores))
	print("std  ",np.std(testing_scores))
	print(testing_scores)
	print(" - - - - - - - - - - - ")
	print("TRAINING Parameters")
	print(" # of folds : ", num_folds)
	print(" PARAMETERS of RF :")
	print(best_clf.get_params())
	print(" - - - - - - - - - - - ")
	print(" This clf will be saved with pickle at this point with the input data")
	return best_clf, best_score

def train_SVM(X,y,num_folds,params):
	clfs = []  ## for best selection

	cms = []	## multiple confussion matrixes
	
	scores = [] 	## for the results
	train_errors = []
	test_errors = []

	training_scores = []
	testing_scores = []

	## for later analysing:
	X_test_l = []
	y_test_l = []

	y = y.ravel()

	parameters 	= params()
	
	## leaving scaling out now
	#scaler = StandardScaler()
	
	## OLD ->#C_value, gamma_value,kernel_type = svc_param_selection(X, y, num_folds)
	
	#using 5-fold
	kf = KFold(n_splits=num_folds,shuffle=True)

	for train_index, test_index in kf.split(X):
		#print("TRAIN:\n", train_index)
		#print("TEST:\n", test_index)
		#print(type(train_index))
		#print(type(test_index))
		print('Training kFold')

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#scaler.fit(X_train)
		#X_train = scaler.transform(X_train)
		#X_test = scaler.transform(X_test)

		clf = SVC(C=parameters['C'],kernel=parameters['kernel'],gamma=parameters['gamma'], probability=True)
		
		clf.fit(X_train, y_train)

		train_score = clf.score(X_train, y_train)
		test_score = clf.score(X_test, y_test)

		training_scores.append(train_score)
		testing_scores.append(test_score)

		scores.append(test_score)

		train_errors.append(1 - train_score)
		test_errors.append(1 - test_score)


		X_test_l.append(X_test)
		y_test_l.append(y_test)


		y_pred = clf.predict(X_test)

		cm = confusion_matrix(y_test,y_pred)

		clfs.append(clf)

		cms.append(cm)

	## score of best clf
	best_score, indx = max((val, idx) for (idx, val) in enumerate(scores))
	
	best_clf = clfs[indx]

	#best_clf.get_params()

	print(" - - - - - - - - - - - ")
	print("CONFUSSION MATRIX:")
	print(np.asarray(cms))
	print(" - - - - - - - - - - - ")
	print(" - - - - - - - - - - - ")
	print(" BEST CLASSIFICATION SCORE:")
	print("	"+str(best_score*100)+"%")
	print(" - - - - - - - - - - - ")
	print("")
	print("SCORES IN TRAINING: " )
	print("mean ", np.mean(training_scores))
	print("std  ", np.std(training_scores))
	print(training_scores)
	print(" - - - - - - - - - - - " )
	print("Min. Test Errors: ",min(test_errors))
	print(" - - - - - - - - - - - ")
	print("SCORES IN TEST: "  )
	print("mean ",np.mean(testing_scores))
	print("std  ",np.std(testing_scores))
	print(testing_scores)
	print(" - - - - - - - - - - - ")
	print("TRAINING Parameters")
	print(" # of folds : ", num_folds)
	print(" PARAMETERS of SVM :")
	print("kernel: ", parameters['kernel'] )
	print("C     : ", parameters['C'])
	print("gamma : ", parameters['gamma'])
	print(" - - - - - - - - - - - ")
	#print(" This clf will be saved with pickle at this point with the input data")
	return best_clf	, best_score

def perform_grid_search(model, X, y, num_folds):
	y = np.ravel(y)

	# Split the dataset in two equal parts
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)

	if model == 'SVM':
		## key :
		model = svm.SVC()

		# Set the parameters for cross-validation
		tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100,1000,10000],'C': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1, 10,100,1000,10000]},
							{'kernel': ['sigmoid'],'gamma': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100,1000,10000], 'C': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1, 10,100,1000,10000]}]
		
	
		grid_search_ = GridSearchCV(model, param_grid=tuned_parameters, iid=False, cv=num_folds, n_jobs=-1)
		grid_search_.fit(X_train, y_train)

		print("Best parameters set found on development set:")
		print()
		print(grid_search_.best_params_)
		print("{:.2%}".format(grid_search_.best_score_))
		print()
		
		##'''
		print(' -- -- -- -- -- -- -- -- -- -- -- ')
		print('')
		print("Detailed classification report:")
		print("	- Grid scores on development set:")
		print()
		means = grid_search_.cv_results_['mean_test_score']
		stds = grid_search_.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, grid_search_.cv_results_['params']):
			print("- mean: %0.3f \n - std: (+/-%0.03f) \n - parameters: %r"% (mean, std * 2, params))
			print(' * * * * * * * * * * * * * * *')
		print('')
		print(' -- -- -- -- -- -- -- -- -- -- -- ')
		print('')
		##'''
	
		best_clf = grid_search_.best_estimator_
		y_true, y_pred = y_test, best_clf.predict(X_test)

		print(' --> Validation: ')
		print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
		print(classification_report(y_true, y_pred))
		print(confusion_matrix(y_true,y_pred))
		print()

		res_parameters = best_clf.get_params
		
		print(res_parameters)

		return res_parameters

	if model == "RF":
		model = RandomForestClassifier()

		## TO DO
		## parameters to try out: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
		##
		'''param_grid = {'bootstrap': [True],
					'max_depth': [1,10,50,80, 90, 100],
					'max_features': [2, 3],
					'min_samples_leaf': [3, 4, 5],
					'min_samples_split': [8, 10, 12],
					'n_estimators': [1,10,50,100, 200, 300, 500, 1000],
					}'''
		## for now:
		param_grid = {'bootstrap': [True],
					'max_depth': [1,10,20,50,100,100],#[1,10,100],
					'max_features': [2],
					'min_samples_leaf': [3, 4, 5],
					'min_samples_split': [8, 10, 12],
					#'n_estimators': [1,10] ## leaving this one out for now 
					}
		##
		##


		grid_search_ = GridSearchCV(estimator = model, param_grid = param_grid, cv = num_folds, n_jobs = -1)

		grid_search_.fit(X_train, y_train)

		print("Best parameters set found on development set:")
		print()
		print(grid_search_.best_params_)
		print("{:.2%}".format(grid_search_.best_score_))
		print()
		
		##'''
		print(' -- -- -- -- -- -- -- -- -- -- -- ')
		print('')
		print("Detailed classification report:")
		print("	- Grid scores on development set:")
		print()
		means = grid_search_.cv_results_['mean_test_score']
		stds = grid_search_.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, grid_search_.cv_results_['params']):
			print("- mean: %0.3f \n - std: (+/-%0.03f) \n - parameters: %r"% (mean, std * 2, params))
			print(' * * * * * * * * * * * * * * *')
		print('')
		print(' -- -- -- -- -- -- -- -- -- -- -- ')
		print('')
		##'''
		
		print(' --> Validation: ')

		best_clf = grid_search_.best_estimator_
		y_true, y_pred = y_test, best_clf.predict(X_test)

		print(' VALIDATION SET:')
		print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
		print(classification_report(y_true, y_pred))
		print(confusion_matrix(y_true,y_pred))
		print()

		res_parameters = best_clf.get_params
		print(res_parameters)
		
		return res_parameters

	if model == 'kNN':
		model = KNeighborsClassifier()

		## TO DO
		## parameters to try out: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
		##
		param_grid_top = {'n_neighbors':[4,5,6,7],
              'leaf_size':[1,3,5],
              'algorithm':['auto', 'kd_tree'],
             	}

		param_grid = {'n_neighbors': [3,4,5,6,7,5,10,15,20],
					'weights': ["uniform","distance"],
					'algorithm': ['auto'],#'ball_tree','kd_tree','brute'],
					'leaf_size': [1,5,10,20,30,40,50,100],#,30],
					'p':[1,2],
					'n_jobs':[-1]
					}
		##

		grid_search_ = GridSearchCV(estimator = model, param_grid = param_grid, cv = num_folds, n_jobs = -1)

		#print(type(grid_search_))

		gs_results= grid_search_.fit(X_train, y_train)

		print("Best parameters set found on development set:")
		print()
		print(grid_search_.best_params_)
		print("{:.2%}".format(grid_search_.best_score_))
		print()
		
		##'''
		print(' -- -- -- -- -- -- -- -- -- -- -- ')
		print('')
		print("Detailed classification report:")
		print("	- Grid scores on development set:")
		print()
		means = grid_search_.cv_results_['mean_test_score']
		stds = grid_search_.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, grid_search_.cv_results_['params']):
			print("- mean: %0.3f \n - std: (+/-%0.03f) \n - parameters: %r"% (mean, std * 2, params))
			print(' * * * * * * * * * * * * * * *')
		print('')
		print(' -- -- -- -- -- -- -- -- -- -- -- ')
		print('')
		##'''
		
		print(' --> Validation: ')

		best_clf = grid_search_.best_estimator_
		y_true, y_pred = y_test, best_clf.predict(X_test)

		print(' VALIDATION SET:')
		print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
		print(classification_report(y_true, y_pred))
		print(confusion_matrix(y_true,y_pred))
		print()

		res_parameters = best_clf.get_params
		print(res_parameters)
		
		return res_parameters

def save_parameters(param_folder,params,clf,dataset,ID):
	
	#parameters 	= params()
	#print(parameters)
	#print(type(parameters))

	#	'/Users/guillermoventuramartinezAIR/Desktop/FP/parameters/PL_Set_'+str(ID)+'/'
	#param_folder = home+'parameters/'+str(ID)+'/'
	param_folder_DS = param_folder+dataset+'/'

	if not path.exists(param_folder_DS):
		os.mkdir(param_folder_DS)

	param_file = param_folder_DS+clf+'.pkl'

	with open(param_file,"wb") as file:
		pickle.dump(params, file, pickle.HIGHEST_PROTOCOL)
	
def open_parameters(param_file,ID):
	#	'/Users/guillermoventuramartinezAIR/Desktop/FP/parameters/PL_Set_'+str(ID)+'/'
	#param_file = home+'parameters/'+str(ID)+'/'+dataset+'/'+clf+'.pkl'
	
	file = open(param_file, 'rb')
	parameters = pickle.load(file)
	file.close()
	
	return parameters

def save_clf(clf,clf_type,dataset,ID):

	if not path.exists(home+'classifiers/'+str(ID)+'/'):
		os.mkdir(home+'/classifiers/'+str(ID)+'/')	

	clf_dir = home+'classifiers/'+str(ID)+'/'+dataset+'/'

	if not path.exists(clf_dir):
		os.mkdir(clf_dir)

	clf_file = clf_dir+clf_type+'.pkl'

	with open(clf_file,"wb") as file:
		pickle.dump(clf, file, pickle.HIGHEST_PROTOCOL)

def do_pca(X):
	print("*********** SVD + PCA ***********")

	#print("SVD")
	value = np.linalg.svd(X,compute_uv=False)
	#print(value)
	#print("SORTED")
	value_s = sorted(value)
	#print(value_s)
	#print("INVERT")
	values_si = value_s[::-1]
	#print(values_si)

	sum_ = []
	

	i=0
	while i < len(values_si): 
		#print i
		if i == 0:
			sum_.append(values_si[i])
		
		else:
			sum_.append(sum_[i-1]+values_si[i])

		i += 1

	#print("SUM")
	#print(sum_)

	## gerade
	#gerade = range(int(sum_[0]),int(sum_[-1]),int(sum_[-1]-sum_[0])/len(sum_))
	
	## distance
	curve = sum_
	nPoints = len(curve)
	allCoord = np.vstack((range(nPoints), curve)).T
	np.array([range(nPoints), curve])
	firstPoint = allCoord[0]
	lineVec = allCoord[-1] - allCoord[0]
	lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
	vecFromFirst = allCoord - firstPoint
	scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
	vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
	vecToLine = vecFromFirst - vecFromFirstParallel
	distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
	#global idxOfBestPoint
	idxOfBestPoint = np.argmax(distToLine)
	print('- INDEX OF ELBOW:')
	print(idxOfBestPoint)

	markers_on = [idxOfBestPoint]
	'''
	plt.figure(1)
	plt.plot(range(0,len(sum_)),sum_,color='g',marker='o',markevery=markers_on)
	#plt.plot(range(0,len(gerade)),gerade,color='b')
	plt.xlabel('Feature')
	plt.ylabel('Sum Singular Values')
	plt.title('SVD')
	plt.legend(['Index of bend: %d' % idxOfBestPoint],loc='center right',)
	plt.grid()
	plt.savefig('Figures/SVD-pl-sum.pdf')
	'''

	pca = PCA(n_components=idxOfBestPoint, svd_solver= 'full') 	# Doc:n_components == 'mle' and svd_solver == 'full' Minka's MLE is used to guess the dimension 
	#print(X.shape)
	pca.fit(X)
	X_new = pca.transform(X)

	#print(X_new)
	#print(X_new.shape)
	
	
	return X_new

def print_save(ID,pl_list,data_set, clf, y_true, y_pred, X_val, y_val, params):

	print(" # * # * # * # * # * # * # * #")
	print(" - Results of "+str(ID))
	print(' ')
	## unpack tuples
	print(" - TRAINED WITH:")
	for i,pl in enumerate(pl_list):
		print(' - PL ##'+str(i))
		print(pl)
	print(" -------- RESULTS --------")
	print(" ----- "+data_set+"-----")
	print(' -'+clf+' Classifier ')
	print(' - PARAMETERS:')
	print(params())
	print(' - size of evaluation set')
	print(X_val.shape)
	print(y_val.shape)
	print(' --> Validation: ')
	print(' - Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred))
	print(confusion_matrix(y_true,y_pred))
	print()
	print(" - - - - - - - - - - - ")
	print(" ")
	print(" # * # * # * # * # * # * # * #")

	folder = home+'CLF_results/EXP_2/'
	if not path.exists(folder):
		os.mkdir(folder)

	text_file = open(folder+data_set+" - "+clf+".txt", "a+")
	text_file.write(" # * # * # * # * # * # * # * #\n")
	text_file.write(" - Results of "+str(ID)+"\n")
	text_file.write(" PLAYLIST used:")
	#text_file.write(pl_list)
	text_file.write(" - - - - - - - - - - - - - - -\n")
	for i, pl in enumerate(pl_list):
		text_file.write(' - ############:\n'+str(i)+"\n")
		text_file.write(' - USER:\n')
		text_file.write(str(pl[0])+"\n")
		text_file.write(' - PLAYLIST:')
		text_file.write(str(pl[1])+"\n")
		text_file.write(' - PL ID')
		text_file.write(str(pl[2])+"\n")
	text_file.write(" -------- RESULTS --------\n")
	text_file.write(" ----- "+data_set+"-----\n")
	text_file.write(' -'+clf+' Classifier \n')
	text_file.write(' - PARAMETERS:\n')
	param_str_ = str(params)#, indent=4)
	text_file.write(param_str_)
	#text_file.write(' - size of evaluation set\n')

	#X_val_shape = X_val.shape
	#y_val_shape = y_val.shape
	
	#x_shape = ''.join(X_val_shape)
	#y_shape = ''.join(y_val_shape)
	#text_file.write(x_shape)
	#text_file.write('\n')
	#text_file.write(x_shape)
	text_file.write('\n')
	text_file.write(' --> Validation: ')
	text_file.write('\n')
	text_file.write(' - Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	text_file.write('\n')
	text_file.write(classification_report(y_true, y_pred))
	text_file.write('\n')
	text_file.write(np.array2string(confusion_matrix(y_true,y_pred)))
	text_file.write('\n')
	text_file.write(" - - - - - - - - - - - ")
	text_file.write('\n')
	text_file.write(" ")
	text_file.write('\n')
	text_file.write(" # * # * # * # * # * # * # * #")
	text_file.write('\n')
	text_file.close()

def create_ALL_df(MIR_df,Sp_df,Lyr_df):

	#L = [MIR_df,Sp_df,Lyr_df]
	ALL_df = MIR_df.join([Sp_df,Lyr_df])

	#ALL_df = pd.concat(L,axis=1,sort=False).set_index('Song_ID')
	#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
	#	print(ALL_df)
	
	return ALL_df

PL_selector()

