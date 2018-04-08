import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Classifier(RandomForestClassifier):
    def __init__(self,featuredat='noisy_features.txt',metadata='Data_Batch_TDA3_all.txt',n_estimators=1000,max_features=3,min_samples_split=2):
        ''' Initialiser for classifier class. Class is a subclass of the 
        	RandomForestClassifier class from scikit-learn, with extra 
        	TESS simulation specific preparation and plotting functions.
        
        Parameters
        ----------------
        featuredat: str
            Filepath leading to the features file, output from the FeatCalc class.
            
        metadata: str
            Filepath leading to the metadata file, e.g. Data_Batch1_noisy.txt
            
        n_estimators: int
        	Number fo trees in the forest. See sklearn docs.
        	
        max_features: int
        	Max features per tree. See sklearn docs.
        	
        min_samples_split: int
        	Min samples required to split a node. See sklearn docs.
                    
        Returns
        -----------------
        NA, but will set up the groups array and train the classifier.
        
        Examples
        -----------------
        A typical use of the class/function would follow:
        	B = Classifier()
        	class_probs = B(unclassified_features_array)
        	
        For testing using the training data:
        	B = Classifier()
        	B.crossValidate()   #may take some time - retrains the classifier for each training set member
        	B.makeConfMatrix()
        	B.plotConfMatrix()
        at this point a confusion matrix will be plotted. Cross-validated class 
        probabilities will be available in B.cvprobs
        
        '''        
        RandomForestClassifier.__init__(self, n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split,class_weight='balanced')  #initialise default RF
        
        print('Loading Data')
        self.metadata = np.genfromtxt(metadata,delimiter=',',dtype=None)
        self.metaids = self.metadata['f0'].astype('unicode')
        self.types = self.metadata['f10'].astype('unicode')
        self.features = np.genfromtxt(featuredat)
        print('Loaded')
        
        self.cvprobs = None
        self.cfmatrix = None
        
        print('Setting up groups array')
        self.groups = self.defineGroups()
        touse = self.groups > 0
        self.groups = self.groups[touse]
        self.features = self.features[touse]
        print('Complete')
        
        print('Training Classifier')
        self.trainRF()
        print('Classifier trained')
        
    def __call__(self,inputfeatures):
        '''
		Call returns the class probabilities given feature input.
		See scikit-learn docs for RandomForestClassifier.predict_proba()
		
		Parameters
		-----------------
		inputfeatures: ndarray, size (nobjects, nfeatures)
			Array of features, one row per object to classify. Number of columns must
			match number of features used for training.
		
		Returns
		-----------------
		class probabilities: ndarray, size (nobjects, nclasses)
        '''
        return self.predict_proba(inputfeatures)
            
    def defineGroups(self):
        '''
		Hardwired class definitions. Could use updating to be versatile.
				
		Returns
		-----------------
		groups: ndarray, size (nobjects)
        '''
        ids = self.features[:,0]
        groups = np.zeros(len(ids))
        for i in range(len(ids)):
            idx = np.where(self.metaids=='Star'+str(int(ids[i])))[0][0]
            if self.types[idx].strip(' ') == 'Eclipse':
                groups[i] = 1
            elif 'Cepheid' in self.types[idx].strip(' '):
                groups[i] = 2
            elif 'Solar-like' in self.types[idx].strip(' '):
                groups[i] = 3
            elif 'RR Lyrae; RRab' in self.types[idx].strip(' '):
                groups[i] = 4    
            elif 'RR Lyrae; ab type' in self.types[idx].strip(' '):
                groups[i] = 4 
            elif 'RR Lyrae; RRc' in self.types[idx].strip(' '):
                groups[i] = 5
            elif 'RR Lyrae; c type' in self.types[idx].strip(' '):
                groups[i] = 5
            elif 'RR Lyrae; RRd' in self.types[idx].strip(' '):
                groups[i] = 6
            elif 'bCep+SPB hybrid' in self.types[idx].strip(' '):
                groups[i] = 7            
            elif 'bCep' in self.types[idx].strip(' '):
                groups[i] = 8    
            elif 'SPB' in self.types[idx].strip(' '):
                groups[i] = 9
            elif 'bumpy' in self.types[idx].strip(' '):
                groups[i] = 10                                                         
            elif self.types[idx].strip(' ') == 'LPV;MIRA' or self.types[idx].strip(' ') == 'LPV;SR':
                groups[i] = 11
            elif self.types[idx].strip(' ') == 'roAp':
                groups[i] = 12
            elif self.types[idx].strip(' ') == 'Constant':
                groups[i] = 13
            elif 'dSct+gDor hybrid' in self.types[idx].strip(' '): #early to avoid gDor misgrouping
                groups[i] = 0
            elif 'gDor' in self.types[idx].strip(' '):
                groups[i] = 14 
            elif 'dsct' in self.types[idx].strip(' '):
                groups[i] = 15  
        return groups
   
    def trainRF(self):
        '''
		Trains the random forest.
        '''
        self.oob_score = True
        self.fit(self.features[:,1:],self.groups)  #1: is to ignore id
        print('OOB Score: '+str(self.oob_score_))
    
    def plotConfMatrix(self,cfmatrixfile='cfmatrix_noisyv2.txt'):
        '''
		Plot a confusion matrix. Axes size and labels are hardwired. Could use
		updating to be versatile, along with defineGroups(). Defaults to use 
		self.cfmatrix, will use input file if not populated.
		
		Parameters
		-----------------
		cfmatrixfile: str, optional
			Filepath to a txt file containing a confusion matrix. Format - np.savetxt
			on the output of self.makeConfMatrix()
        '''
        import pylab as p
        p.ion()
        if self.cfmatrix is not None:
            confmatrix = self.cfmatrix.astype('float')
        else:
            confmatrix = np.genfromtxt(cfmatrixfile)
        norms = np.sum(confmatrix,axis=1)
        for i in range(len(confmatrix[:,0])):
            confmatrix[i,:] /= norms[i]
        p.figure()
        p.clf()
        p.imshow(confmatrix,interpolation='nearest',origin='lower',cmap='YlOrRd')

        for x in range(len(confmatrix[:,0])):
             for y in range(len(confmatrix[:,0])):
                 if confmatrix[y,x] > 0:
                     if confmatrix[y,x]>0.7:
                         p.text(x,y,str(np.round(confmatrix[y,x],decimals=2)),va='center',ha='center',color='w')
                     else:
                         p.text(x,y,str(np.round(confmatrix[y,x],decimals=2)),va='center',ha='center')

        for x in [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5]:
            p.plot([x,x],[-0.5,14.5],'k--')
        for y in [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5]:
            p.plot([-0.5,14.5],[y,y],'k--')
        p.xlim(-0.5,14.5)
        p.ylim(-0.5,14.5)
        p.xlabel('Predicted Class')
        p.ylabel('True Class')
        #class labels
        p.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],['Eclipse', 'Cepheid', 'Solar', 'RRab', 'RRc', 'RRd', 'b+S hybrid', 'bCep', 'SPB', 'bumpy', ' LPV', 'roAp', 'Const', 'gDor', 'dSct'],rotation='vertical')
        p.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],['Eclipse', 'Cepheid', 'Solar', 'RRab', 'RRc', 'RRd', 'b+S hybrid', 'bCep', 'SPB', 'bumpy', ' LPV', 'roAp', 'Const', 'gDor', 'dSct'])
    
    def crossValidate(self):
        '''
		Creates cross-validated class probabilities. Splits dataset into groups of 10.
		May take some time.
		
		Returns
		-----------------
		cvprobs: ndarray, size (nobjects, nclasses)
			cross-validated class probabilities one row per object
        '''
        from sklearn.model_selection import KFold
        shuffleidx = np.random.choice(len(self.groups),len(self.groups),replace=False)
        cvfeatures = self.features[shuffleidx,1:]
        cvgroups = self.groups[shuffleidx]
        kf = KFold(n_splits=int(cvfeatures.shape[0]/10))
        probs = []
        self.oob_score = False
        for train_index,test_index in kf.split(cvfeatures,cvgroups):
            print(test_index)
            self.fit(cvfeatures[train_index,:],cvgroups[train_index])
            probs.append(self.predict_proba(cvfeatures[test_index,:]))  
        self.cvprobs = np.vstack(probs)
        unshuffleidx = np.argsort(shuffleidx)
        self.cvprobs = self.cvprobs[unshuffleidx]
        return self.cvprobs
    
    def makeConfMatrix(self,cvprobfile='cvprobs_noisy.txt'):
        '''
		Generates a confusion matrix from a set of class probabilities. Defaults to 
		use self.cvprobs. If not populated, reads from input file.
		
		Parameters
		-----------------
		cvprobfile: str, optional
			Filepath to an array of class probabilities, size (nobjects, nclasses)
		
		Returns
		-----------------
		cfmatrix: ndarray, size (nclasses, nclasses)
			Confusion matrix for the classifier. 
        '''
        if self.cvprobs is None:
            self.cvprobs = np.genfromtxt(cvprobfile)
        from sklearn.metrics import confusion_matrix
        self.class_vals=np.argmax(self.cvprobs,axis=1)+1
        self.cfmatrix = confusion_matrix(self.groups,self.class_vals)
        return self.cfmatrix
            