import numpy as np
import os
import glob

def Init(sample):
    return np.random.uniform(0,1,size=(1,400,64))


class FeatCalc(object):
    def __init__(self, metadata='Data_Batch1_noisy.txt', lclistcall='noisy_shuf_files/Star*.noisy_shuf', freqdir='noisy_shuf_frequencies', LSdir='noisy_shuf_lsspec', somfile='som_noisy_all_1_400_64_300_0.1.txt', nfrequencies=6, plot=False):
        ''' Initialiser for class to bring together and calculate features 
        	for classification
        
        Parameters
        ----------------
        metadata: str
            Filepath leading to the metadata file, e.g. Data_Batch1_noisy.txt
            
        lclistcall: str
            A call to ls to recover all the lightcurve datafiles
            
        freqdir: str
        	Directory containing the frequency files as output by Vicki et al.
        	
        LSdir: str
        	Directory containing the calculated periodograms as output by Vicki et al. 
        	Currently unused, can be ignored.
        	
        somfile: str
        	Filepath to saved SOM Kohonen Layer (must be created separately)
        
        n_frequencies: int
        	Number of frequencies to use, starting with the strongest
        	
        plot: bool
        	Make diagnostic plots?
            
        Returns
        -----------------
        NA
        
        Examples
        -----------------
        A typical use of the class/function would follow:
        	feat = FeatCalc()
        	feat.calc()
        	print feat.features
        	np.savetxt(outputfile,feat.features)
        	
        '''
        #get lightcurve list, input files
        self.metadata = np.genfromtxt(metadata,delimiter=',',dtype=None)
        self.inlist = glob.glob(lclistcall)
        self.freqdir = freqdir
        self.LSdir = LSdir
        self.nfreq = nfrequencies
        self.features = np.zeros([len(self.inlist),self.nfreq+14])
        self.som = self.loadSOM(somfile)
        self.forbiddenfreqs = [13.49/4.] #(4 harmonics will be removed)

        self.ids = self.metadata['f0']
        self.types = self.metadata['f10']
        self.teff = self.metadata['f6']
        self.logg = self.metadata['f8']
        
        self.plot = plot
  
    def calc(self):
        ''' Function to calculate features and populate the features array
        
        Parameters
        ----------------
		NA
		            
        Returns
        -----------------
        No return, but leaves self.features array populated
        
        '''
        for i,infile in enumerate(self.inlist):
            self.activeid = os.path.split(infile)[1][:-10]
            print self.activeid
            
            #load
            self.lc = self.loadLC(infile)
        
            #load relevant frequencies file
            self.freqdat = self.loadFreqs()
            
            #load relevant LS file? Currently unused
            #self.LSdat = self.loadLomb()
            
            #calc features
            self.features[i,0] = int(self.activeid[4:-1])
            self.features[i,1:self.nfreq+1],self.features[i,self.nfreq+1] = self.frequencies(infile)
            self.features[i,self.nfreq+2:self.nfreq+4] = self.freq_ampratios()
            self.features[i,self.nfreq+4:self.nfreq+6] = self.freq_phasediffs()
            self.features[i,1] = self.EBperiod(i)
            self.features[i,self.nfreq+6:self.nfreq+8] = self.SOMloc(i)
            self.features[i,self.nfreq+8:self.nfreq+10] = self.phase_features(i)
            self.features[i,self.nfreq+10:self.nfreq+12] = self.p2p_features()
            self.features[i,self.nfreq+12] = self.get_Teff()
            self.features[i,self.nfreq+13:] = self.guy_features(infile)

            if self.plot:
                print 'Features:'
                print self.features[i,:]
                print 'Period:'
                print self.features[i,1]
                idx = np.where(self.ids==self.activeid.strip('.'))[0]
                print self.metadata['f10'][idx]
                import pylab as p
                p.ion()
                phase = self.phasefold(self.lc['time'],self.features[i,1])
                p.figure(1)
                p.clf()
                p.plot(phase,self.lc['flux'],'r.')
                p.xlabel('Phase')
                p.ylabel('Relative Flux')
                p.figure(2)
                p.clf()
                p.plot(self.lc['time'],self.lc['flux'],'b.')
                p.xlabel('Time')
                p.ylabel('Relative Flux')
                p.pause(1)
                raw_input('Press return to continue')
            
        self.features = self.features[np.argsort(self.features[:,0]),:] #sort by id

    def loadLC(self, infile):
        """
        Loads a TESS lightcurve (currently just the TASC WG0 simulated ones), 
        normalised and with NaNs removed.
        
        Inputs
        -----------------
        infile: str
        	Filepath to one lightcurve file
        	
        Returns
        -----------------
        lc: dict
         	lightcurve as dict, with keys time, flux, error. 
        	error is populated with zeros.
        """
        dat = np.genfromtxt(infile)
        time = dat[:,0]
        flux = dat[:,1]
        err = np.zeros(len(time))
        nancut = np.isnan(time) | np.isnan(flux) | np.isnan(err)
        norm = np.median(flux[~nancut])
        lc = {}
        lc['time'] = time[~nancut]
        lc['flux'] = flux[~nancut]/norm
        lc['error'] = err[~nancut]/norm
        del dat
        return lc
            
    def loadFreqs(self):
        """
        Loads a frequency information file for this target
        """
        freqfile = os.path.join(self.freqdir,self.activeid)+'slscleanlog'
        freqdat = np.genfromtxt(freqfile)
        if freqdat.shape[0] == 0:
            freqdat = np.zeros([2,9]) - 10        
        elif len(freqdat.shape) == 1: #only one significant peak
            temp = np.zeros([2,len(freqdat)])
            temp[0,:] = freqdat
            temp[1,:] -= 10
            freqdat = temp
        return freqdat
    
    def loadLomb(self):
        """
        Loads Lomb-Scargle periodogram for this target
        """
        lombfile = os.path.join(self.LSdir,self.activeid)[:-1]+'_slscleaned.lsspec'
        return np.genfromtxt(lombfile)
        
    def loadSOM(self, somfile):
        """
        Load and set up a previously trained SOM
        """
        import selfsom
        som = selfsom.SimpleSOMMapper((1,400),1,initialization_func=Init,learning_rate=0.1)
        loadk = self.kohonenLoad(somfile)
        som.train(loadk)  #purposeless but tricks the SOM into thinking it's been trained. Don't ask.
        som._K = loadk
        return som
        
    def kohonenLoad(self, infile):
        """
        Loads a saved Kohonen Layer 
        (basically a function to load 3d arrays from a specific type of 2d file)
        """
        with open(infile,'r') as f:
            lines = f.readlines()
        newshape = lines[0].strip('\n').split(',')
        out = np.zeros([int(newshape[0]),int(newshape[1]),int(newshape[2])])
        for i in range(int(newshape[0])):
            for j in range(int(newshape[1])):
                line = lines[1+(i*int(newshape[1]))+j].strip('\n').split(',')
                for k in range(int(newshape[2])):
                    out[i,j,k] = float(line[k])
        return out
            
    def SOMloc(self, i):
        """
        Calculates SOM map location and phase binned amplitude.
        SOM must be loaded, and a period already calculated.
        """
        per = self.features[i,1]
        if per < 0:
            return -10
        SOMarray,range = self.prepFilePhasefold(per,64)
        SOMarray = np.vstack((SOMarray,np.ones(len(SOMarray)))) #tricks som code into thinking we have more than one
        map = self.som(SOMarray)
        map = map[0,1]
        return map, range

    def get_Teff(self):
        """
        Returns effective temperature from the metadata.
        """
        idx = np.where(self.ids==self.activeid.strip('.'))[0]
        return self.teff[idx]
        
    def frequencies(self, infile):
        """
        Returns the desired number of periods, ignoring those longer than
        the dataset duration or near forbidden frequencies defined in __init__
        
        Populates array with -10 when not enough periods present.
        
        Also returns a flag if the strongest period was longer than the dataset
        duration.
        """
        pers = []
        self.usedfreqs = []
        j = 0
        longperflag = 0
        while len(pers)<self.nfreq:
            per = 1./(self.freqdat[j,1]*1e-6)/86400.  #convert to days
            #print freq
            #check to cut bad frequencies
            cut = False
            if (j==0) and (per>np.max(self.lc['time'])-np.min(self.lc['time'])):
                longperflag = 1
            if (per < 0) or (per > np.max(self.lc['time'])-np.min(self.lc['time'])):  #means there weren't even one or two frequencies, or frequency too long
                cut = True
            for freqtocut in self.forbiddenfreqs:
                for k in range(4):  #cuts 4 harmonics of frequency, within +-3% of given frequency
                    if (1./per > (1./((k+1)*freqtocut))*(1-0.01)) & (1./per < (1./((k+1)*freqtocut))*(1+0.01)):
                        cut = True
            if not cut:
                pers.append(per)
                self.usedfreqs.append(j)
            j += 1
            if j >= len(self.freqdat[:,0]):
                break
        #fill in any unfilled frequencies with negative numbers
        gap = self.nfreq - len(pers)
        if gap > 0:
            for k in range(gap):
                pers.append(-10)
        self.usedfreqs = np.array(self.usedfreqs)
        return np.array(pers),longperflag
        
    def freq_ampratios(self):
        """
        Returns 2:1 and 3:1 frequency amplitude ratios.
        If not enough significant frequencies, populates with 0.
        """
        if len(self.usedfreqs) >= 2:
            amp21 = self.freqdat[self.usedfreqs[1],3]/self.freqdat[self.usedfreqs[0],3]
        else:
            amp21 = 0
        if len(self.usedfreqs) >= 3:
            amp31 = self.freqdat[self.usedfreqs[2],3]/self.freqdat[self.usedfreqs[0],3]
        else:
            amp31 = 0
        return amp21,amp31
        
    def freq_phasediffs(self):
        """
        Returns 2:1 and 3:1 frequency phase differences.
        If not enough significant frequencies, populates with -10.
        """
        if len(self.usedfreqs) >= 2:
            phi21 = self.freqdat[self.usedfreqs[1],5] - 2*self.freqdat[self.usedfreqs[0],5]
        else:
            phi21 = -10
        if len(self.usedfreqs) >= 3:
            phi31 = self.freqdat[self.usedfreqs[2],5] - 3*self.freqdat[self.usedfreqs[0],5]
        else:
            phi31 = -10
        return phi21,phi31    
    
    def phase_features(self, i): 
        """
        Returns phase point-to-point 98th percentile and mean
        """
        phase = self.phasefold(self.lc['time'],self.features[i,1])
        p2p = np.abs(np.diff(self.lc['flux'][np.argsort(phase)]))
        return np.percentile(p2p,98),np.mean(p2p)
    
    def p2p_features(self): 
        """
        Returns timeseries point-t0-point 98th percentile and mean
        """
        p2p = np.abs(np.diff(self.lc['flux']))
        return np.percentile(p2p,98),np.mean(p2p)
        
    def EBperiod(self, i, cut_outliers=0):
        """
        Tests for phase variation at double the current prime period,
        to correct EB periods.
        
        Returns
        -----------------
        
        corrected period: float
        	Either initial period or double      
        """
        per = self.features[i,1]
        if per < 0:
            return per
        flux_flat = self.lc['flux'] - np.polyval(np.polyfit(self.lc['time'],self.lc['flux'],1),self.lc['time']) + 1
        
        phaselc2P = np.zeros([len(self.lc['time']),2])
        phaselc2P = self.phasefold(self.lc['time'],per*2)
        idx = np.argsort(phaselc2P)
        phaselc2P = phaselc2P[idx]
        flux = flux_flat[idx]
        binnedlc2P = self.binPhaseLC(phaselc2P,flux,64,cut_outliers=5)

        minima = np.argmin(binnedlc2P[:,1])
        posssecondary = np.mod(np.abs(binnedlc2P[:,0]-np.mod(binnedlc2P[minima,0]+0.5,1.)),1.)
        posssecondary = np.where((posssecondary<0.05) | (posssecondary > 0.95))[0]   #within 0.05 either side of phase 0.5 from minima

        pointsort = np.sort(self.lc['flux'])
        top10points = np.median(pointsort[-30:])
        bottom10points = np.median(pointsort[:30])
        
        periodlim = np.max(self.lc['time'])-np.min(self.lc['time']) #limited to dataset duration
        if np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1] > 0.0025 and np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1] > 0.03*(top10points-bottom10points) and per*2<=periodlim:  
            return 2*per
        else:
            return per

    def get_metric(self, low_f=1.0, high_f=288.0, white_npts=100):
        """
        Metrics for Guy Davies features
        """
        white = np.median(self.ds.power[-white_npts:])
        mean = np.mean(self.ds.power[(self.ds.freq > low_f) \
                                      & (self.ds.freq < high_f)])
        med = np.median(self.ds.power[(self.ds.freq > low_f) \
                                      & (self.ds.freq < high_f)])
        std = np.std(self.ds.power[(self.ds.freq > low_f) \
                                      & (self.ds.freq < high_f)])
        return white, mean, med, std


    def guy_features(self, infile):
        """
        Guy Davies features for identifying giant oscillations
        """
        from TDAdata import Dataset
        self.ds = Dataset(self.activeid, infile)
        vars_dict = {'Flicker': [['w100', 'mean1', 'med1', 'std1'],[0.5, 288.0, 100]]}
        self.ds.read_timeseries(sigma_clip=3.0)
        self.ds.power_spectrum()
        tmp = vars_dict['Flicker'][1]
        w, m, mm, s = self.get_metric(low_f=tmp[0], high_f=tmp[1], white_npts=tmp[2])
        return w, m-w
        
    def binPhaseLC(self, phase, flux, nbins, cut_outliers=0):
        """
        Bin a phase-folded lightcurve.
        
        Parameters
        ----------------
        phase: 1D array, length m
        	phase data for lightcurve
        	
        flux: 1D array, length m
        	flux data for lightcurve
        	
        nbins: int
        	number of bins to create
        	
        cut_outliers: float
        	If > 0, outliers greater than cut_outliers will be ignored
        	when calculating bins. Outlier distance is calculated using flux
        	median and MAD.
        	
        Returns
        ----------------
        binnedlc: ndarray, size (nbins x 2)
        	Binned lightcurve (phases x bin values)
        	
        """
        bin_edges = np.arange(nbins)/float(nbins)
        bin_indices = np.digitize(phase,bin_edges) - 1
        binnedlc = np.zeros([nbins,2])
        binnedlc[:,0] = 1./nbins * 0.5 +bin_edges  #fixes phase of all bins - means ignoring locations of points in bin, but necessary for SOM mapping
        for bin in range(nbins):
            inbin = np.where(bin_indices==bin)[0]
            if cut_outliers:
                mad = np.median(np.abs(flux[inbin]-np.median(flux[inbin])))
                outliers = np.abs((flux[inbin] - np.median(flux[inbin])))/mad <= cut_outliers
                inbin = inbin[outliers]
            if np.sum(bin_indices==bin) > 0:
                binnedlc[bin,1] = np.mean(flux[inbin])  #doesn't make use of sorted phase array, could probably be faster?
            else:
                binnedlc[bin,1] = np.mean(flux)  #bit awkward this, but only alternative is to interpolate?
        return binnedlc

    def phasefold(self,time,per,t0=0):
        """
        Phasefolds lightcurve.
        """
        return np.mod(time-t0,per)/per

    def prepFilePhasefold(self, period, cardinality):
        """
        Prepares a lightcurve for SOM analysis.
        
        Phasefolds, bins, normalises between 0 and 1, 
        offsets to place minimum and phase 0.
        """
        phase = self.phasefold(self.lc['time'],period)
        idx = np.argsort(phase)
        flux = self.lc['flux'][idx]
        phase = phase[idx]
        binnedlc = self.binPhaseLC(phase,flux,cardinality)
        #normalise to between 0 and 1
        minflux = np.min(binnedlc[:,1])
        maxflux = np.max(binnedlc[:,1])
        if maxflux != minflux:
            binnedlc[:,1] = (binnedlc[:,1]-minflux) / (maxflux-minflux)
        else:
            binnedlc[:,1] = np.ones(cardinality)
        #offset so minimum is at phase 0
        binnedlc[:,0] = np.mod(binnedlc[:,0]-binnedlc[np.argmin(binnedlc[:,1]),0],1)
        binnedlc = binnedlc[np.argsort(binnedlc[:,0]),:]
        return binnedlc[:,1],maxflux-minflux
