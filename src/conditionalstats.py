"""@package conditionalstats
Documentation for module conditionalstats.

Class Distribution defines objects containing the statistical distribution of a 
single variable, choosing the bin type (linear, logarithmic, inverse-logarithmic,
...).
"""
import numpy as np
from math import log10,ceil,floor,exp
import time
import sys
from collections import defaultdict

class WrongArgument(Exception):
    pass

class EmptyDataset(Exception):
    pass

class EmptyDistribution:

    """Documentation for class EmptyDistribution

    Parent object. The object will not have the same types of attributes depending 
    on the choice of distribution structure.
    
    """

    ##-- Class constructor
    
    def __init__(self,bintype='linear',nbpd=10,nppb=4,nbins=50,nd=4,fill_last_decade=False):

        """Constructor for class EmptyDistribution.
        Arguments:
        - bintype [linear, log, invlogQ, linQ]: bin structure.
        - nbins: 'number of linear bins' used for all types of statistics. Default is 50.
        - nbpd: number of bins per (log or invlog) decade. Default is 10.
        - nppb: minimum number of data points per bin. Default is 4.
        - nd: (maximum) number of decades for inverse-logarithmic bins. Default is 4.
        - fill_last_decade: boolean to fill up largest percentiles for 'invlog' bin type
        """

        self.bintype = bintype
        self.nbins = nbins
        self.nbpd = nbpd
        self.nppb = nppb
        self.nd = nd
        self.fill_last_decade = fill_last_decade

        # Remove unnecessary attributes
        if self.bintype == 'linear':

            self.nbpd = None
            self.nppb = None
            self.fill_last_decade = None

        elif self.bintype in ['log','invlogQ']:

            # self.nbins = None
            pass

        elif self.bintype == 'linQ':

            self.nlb = None
            self.nbpd = None
            self.nppb = None
            self.fill_last_decade = None

        else:

            raise WrongArgument("ERROR: unknown bintype")
    
    
    ##-- overwrite default class methods for printing 
    
    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< Distribution object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out
    
    def __str__(self):
        """Override string function to print attributes
        """
        # method_names = []
        # str_out = '-- Attributes --'
        str_out = ''
        for a in dir(self):
            if '__' not in a:
                a_str = str(getattr(self,a))
                if 'method' not in a_str:
                    str_out = str_out+("%s : %s\n"%(a,a_str))
        #         else:
        #             method_names.append(a)
        # print('-- Methods --')
        # for m in method_names:
        #     print(m)
        return str_out


class Distribution(EmptyDistribution):
    """Documentation for class Distribution
    
    Inherited class from parent class EmptyDistribution.
    """
    
    ##-- Class constructor

    def __init__(self,name='',bintype='linear',nbpd=10,nppb=4,nbins=50,nd=None,\
        fill_last_decade=False,distribution=None,overwrite=False):
        """Constructor for class Distribution.
        Arguments:
        - name: name of reference variable
        - bintype [linear, log, invlog]: bin structure,
        - nbins: number of bins used for all types of statistics. Default is 50.
        - nbpd: number of bins per log or invlog decade. Default is 10.
        - nppb: minimum number of data points per bin. Default is 4.
        - nd: maximum number of decades in invlogQ bin type. Default is 4
        """

        EmptyDistribution.__init__(self,bintype,nbpd,nppb,nbins,nd,fill_last_decade)
        self.name = name
        self.size = 0
        self.vmin = None
        self.vmax = None
        self.rank_edges = None
        self.ranks = None
        self.percentiles = None
        self.bins = None
        self.density = None
        self.bin_locations_stored = False
        self.overwrite = overwrite

        if distribution is not None: # then copy it in self
            for attr in distribution.__dict__.keys():
                setattr(self,attr,getattr(distribution,attr)) 

    ##-- overwrite default class methods for printing 
    
    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the 
        attribute value when its string fits is small enough."""
        return super().__repr__()

    def __str__(self):
        return super().__str__()

    ##-- class methods
    
    def setSampleSize(self,sample):

        if sample.size == 0:
            raise EmptyDataset("")
        else:
            self.size = sample.size

    def setVminVmax(self,sample=None,vmin=None,vmax=None,minmode='positive',\
        overwrite=False):

        """Compute and set minimum and maximum values
        Arguments:
        - sample: 1D numpy array of data values."""

        # Find minimum value
        if vmin is None:	
            if minmode is None:
                vmin = np.nanmin(sample)
            elif minmode == 'positive':
                vmin = np.nanmin(sample[sample > 0])
        # Find maximum value
        if vmax is None:
            vmax = np.nanmax(sample)
            
        if self.vmin is None or overwrite:
            self.vmin = vmin
        if self.vmax is None or overwrite:
            self.vmax = vmax

    def getInvLogRanks(self):

        """Percentile ranks regularly spaced on an inverse-logarithmic axis (zoom on 
        largest percentiles of the distribution).
        Arguments:
            - fill_last_decade: True (default is False) if want to plot
            up to 99.99 or 99.999, not some weird number in the middle of a decade.
        Sets:
            - ranks: 1D numpy.array of floats"""

        # k indexes bins
        if self.nd is None:
            n_decades = log10(self.size/self.nppb) 		# Number of decades from data size
        else:
            n_decades = self.nd                        # Prescribed number of decades
        dk = 1/self.nbpd
        if self.fill_last_decade:
            k_max = floor(n_decades)				 	# Maximum bin index
        else:
            k_max = int(n_decades*self.nbpd)*dk # Maximum bin index
        scale_invlog = np.arange(0,k_max+dk,dk)
        ranks_invlog = np.subtract(np.ones(scale_invlog.size),
            np.power(10,-scale_invlog))*100

        # store ranks
        self.ranks = ranks_invlog
        # calculate bin edges in rank-space
        self.rank_edges = np.hstack([[0],np.convolve(self.ranks,[0.5,0.5],mode='valid'),[None]])
        # get number of bins
        self.nbins = self.ranks.size # in this case, define nbins from - no no no no noooo, recode this
        
    def getLinRanks(self):

        """Percentile ranks regularly spaced on a linear axis of percentile ranks"""

        self.rank_edges = np.linspace(0,100,self.nbins+1) # need nbins as input
        self.ranks = np.convolve(self.rank_edges,[0.5,0.5],mode='valid') # center of rank 'bins'

    def computePercentilesAndBinsFromRanks(self,sample,crop=False,store=True,output=False):

        """Compute percentiles of the distribution and histogram bins from 
        percentile ranks. 
        Arguments:
            - sample: 1D numpy array of values
            - ranks: 1D array of floats between 0 and 1
        Sets:
            - ranks, cropped by one at beginning and end
            - percentiles (or bin centers)
            - bins (edges)
        """

        sample_no_nan = sample[np.logical_not(np.isnan(sample))]
        if sample_no_nan.size == 0:
            percentiles = np.array([np.nan]*self.nbins)
        else:
            percentiles = np.percentile(sample_no_nan,self.ranks)
            
        # calculate center bins (not minimum edge nor maximum edge)
        bins = np.array([np.nan]*(self.nbins+1))
        bins[1:-1] = np.percentile(sample_no_nan,self.rank_edges[1:-1])

        if not crop:
            bins[0] = self.vmin
            bins[-1] = self.vmax

        if store:
            self.percentiles = percentiles
            self.bins = bins
            
        if output:
            return self.percentiles, self.bins

    def definePercentilesOnInvLogQ(self,sample):

        """Defines percentiles and histogram bins on inverse-logarithmic ranks.
        Arguments:
            - sample: 1D numpy array of values
        Sets:
            - ranks
            - percentiles
            - bins
        """
        
        self.size = sample.size
        # First compute invlog ranks including its edge values
        self.getInvLogRanks()
        # Then compute final stats
        self.computePercentilesAndBinsFromRanks(sample) # keep crop=False to get manually-set bounds

    def definePercentilesOnLinQ(self,sample,vmin=None,vmax=None):

        """Define percentiles and histogram bins on linear ranks.
        Arguments:
            - sample: 1D numpy array of values
        Sets:
            - ranks
            - percentiles
            - bins
        """

        self.setVminVmax(sample=sample,vmin=vmin,vmax=vmax)
        # Compute linear ranks
        self.getLinRanks()
        # Then compute final stats
        self.computePercentilesAndBinsFromRanks(sample)

    def defineLogBins(self,sample,vmin=None,vmax=None,minmode='positive'):

        """Define logarithmic bin centers and edges from sample values.
        Arguments:
            - sample: 1D numpy array of values
            - n_bins_per_decade: number of ranks/bins per logarithmic decade
            - vmin and vmax: extremum values
        Computes:
            - centers (corresponding percentiles, or bin centers)
            - breaks (histogram bin edges)"""

        self.setVminVmax(sample,vmin,vmax,minmode)
        kmin = floor(log10(self.vmin))
        kmax = ceil(log10(self.vmax))
        self.bins = np.logspace(kmin,kmax,(kmax-kmin)*self.nbpd)
        self.percentiles = np.convolve(self.bins,[0.5,0.5],mode='valid')
        self.nbins = self.percentiles.size

    def defineLinearBins(self,sample,vmin=None,vmax=None,minmode='positive'):

        """Define linear bin centers and edges from sample values.
        Arguments:
            - sample: 1D numpy array of values
            - vmin and vmax: extremum values
        Computes:
            - percentiles (or bin centers)
            - bins (edges)
        """

        self.setVminVmax(sample,vmin,vmax,minmode)
        self.bins = np.linspace(self.vmin,self.vmax,self.nbins+1)
        self.percentiles = np.convolve(self.bins,[0.5,0.5],mode='valid')
        
        assert(self.percentiles.size == self.nbins), "wrong number of bins: #(percentiles)=%d and #(bins)=%d"%(self.percentiles.size,self.nbins)

    def computePercentileRanksFromBins(self,sample):

        """Computes percentile ranks corresponding to percentile values.
        Arguments:
            - sample: 1D numpy array of values
        Computes:
            - ranks: 1D numpy.ndarray"""
        
        self.ranks = 100*np.array(list(map(lambda x:(sample < x).sum()/self.size, \
            self.percentiles)))

    def ranksPercentilesAndBins(self,sample,vmin=None,vmax=None,minmode='positive'):

        """Preliminary step to compute probability densities. Define 
        ranks, percentiles, bins from the sample values and binning structure.
        Arguments:
            - sample: 1D numpy array of values
        Computes:
            - ranks, percentiles and bins"""

        self.setSampleSize(sample)
        self.setVminVmax(sample,vmin,vmax,minmode)

        if self.bintype == 'linear':

            self.defineLinearBins(sample,vmin,vmax,minmode)
            self.computePercentileRanksFromBins(sample)

        elif self.bintype == 'log':

            self.defineLogBins(sample,vmin,vmax,minmode)
            self.computePercentileRanksFromBins(sample)

        elif self.bintype == 'invlogQ':

            self.getInvLogRanks()
            self.computePercentilesAndBinsFromRanks(sample)

        elif self.bintype == 'linQ':

            self.definePercentilesOnLinQ(sample)

        else:

            raise WrongArgument("ERROR: unknown bintype")

    def computeDistribution(self,sample,vmin=None,vmax=None,minmode=None):

        """Compute ranks, bins, percentiles and corresponding probability densities.
        Arguments:
            - sample: 1D numpy array of values
        Computes:
            - ranks, percentiles, bins and probability densities"""

        if not self.overwrite:
            pass

        # Compute ranks, bins and percentiles
        self.ranksPercentilesAndBins(sample,vmin,vmax,minmode)
        # Compute probability density
        density, _ = np.histogram(sample,bins=self.bins,density=True)
        self.density = density
        # Number fraction of points below chosen vmin
        self.frac_below_vmin = np.sum(sample < self.vmin)/np.size(sample)
        # Number fraction of points above chosen vmax
        self.frac_above_vmax = np.sum(sample > self.vmax)/np.size(sample)

    def indexOfRank(self,rank):
    
        """Returns the index of the closest rank in numpy.array ranks"""

        dist_to_rank = np.absolute(np.subtract(self.ranks,rank*np.ones(self.ranks.shape)))
        mindist = dist_to_rank.min()
        return np.argmax(dist_to_rank == mindist)

    def rankID(self,rank):

        """Convert rank (float) to rank id (string)
        """

        return "%2.4f"%rank

    def binIndex(self,percentile=None,rank=None):

        """Returns the index of bin corresponding to percentile or rank 
        of interest
        """

        if percentile is not None:
            # Find first bin edge to be above the percentile of interest
            i_perc = np.argmax(self.bins > percentile)
            if i_perc == 0: # Then percentile is outside the range of stored bins
                return None
            return i_perc-1 # Offset by 1

        if rank is not None:
            return self.indexOfRank(rank)
        # raise WrongArgument("no percentile or rank is provided in binIndex")
        return None

    def formatDimensions(self,sample):
        """Reshape the input data, test the validity and returns the dimensions
        and formatted data.

        Arguments:
        - sample: here we assume data is horizontal, formats it in shape (Ncolumns,)
        Controls if it matches the data used for the control distribution. If
        not, aborts.
        """

        # Get shape
        sshape = sample.shape
        # Initialize default output
        sample_out = sample
        # Get dimensions and adjust output shape
        if len(sshape) > 1: # reshape
            sample_out = np.reshape(sample,np.prod(sshape))
        Npoints, = sample_out.shape
        
        # Test if sample size is correct to access sample points
        if Npoints != self.size:
            raise WrongArgument("Error: used different sample size")

        return sample_out

    def storeSamplePoints(self,sample,sizemax=50,verbose=False,method='shuffle_mask'):
        """Find indices of bins in the sample data, to get a mapping or extremes 
        and fetch locations later
        """

        if self.bin_locations_stored and not self.overwrite:
            pass

        if verbose:
            print("Finding bin locations...")
            
        # Check if sample is xarray or np.array         
        type_sample = False if type(sample) == np.ndarray else True

        # print(sample.shape)
        sample = self.formatDimensions(sample)
        # print(sample.shape)

        # Else initalize and find bin locations
        self.bin_locations = [[] for _ in range(self.nbins)]
        self.bin_sample_size = [0 for _ in range(self.nbins)]

        if method == 'random':

            # Here, look at all points, in random order
            
            indices = list(range(self.size))
            np.random.shuffle(indices)

            bins_full = []
            for i_ind in range(len(indices)):

                i = indices[i_ind]

                # Find corresponding bin
                i_bin = self.binIndex(percentile=sample[i])

                # Store only if bin was found
                if i_bin is not None:

                    # Keep count
                    self.bin_sample_size[i_bin] += 1
                    # Store only if there is still room in stored locations list
                    if len(self.bin_locations[i_bin]) < sizemax:
                        self.bin_locations[i_bin].append(i)
                    elif i_bin not in bins_full:
                        bins_full.append(i_bin)
                        bins_full.sort()
                        if verbose:
                            print("%d bins are full (%d iterations)"%(len(bins_full),i_ind))
            
        elif method == 'shuffle_mask':

            if verbose: print('bin #: ',end='')
            # compute mask for each bin, randomize and store 'sizemax' first indices
            for i_bin in range(self.nbins):

                if verbose: print('%d..'%i_bin,end='')

                # compute mask

                # Adapt code for xarray 
                if type_sample:
                    mask = np.logical_and(sample >= self.bins[i_bin], sample < self.bins[i_bin+1])
                
                else:
                    mask = np.logical_and(sample.flatten() >= self.bins[i_bin],
                            sample.flatten() < self.bins[i_bin+1])
                
                # get all indices
                ind_mask = np.where(mask)[0]
                # shuffle
                np.random.seed(int(round(time.time() * 1000)) % 1000)
                np.random.shuffle(ind_mask)

                self.bin_sample_size[i_bin] = ind_mask.size # count all points there
                # select 'sizemax' first elements
                self.bin_locations[i_bin] = ind_mask[:sizemax]


            if verbose: print()

        if verbose:
            print()

        # If reach this point, everything should have worked smoothly, so:
        self.bin_locations_stored = True

    def computeDataOverBins(self,sample, data, label = None, sizemax=50, verbose=False,method = 'iterative reducing'):
        """
        This function doesn't require bin_locations to be stored as it will recomputed everything mask per bin,
        but it will be much faster as it will reduce the size of the arrays after each bin computation.
        
        First output is the data with no treatmeant over bins
        
        the other outputs are optionnal and are computed only if the label array of MCS is provided
        they consist of an analysis of bins>40 so the extremes only
        
        grouped_data is the data aggregated by grouping, quite straightforward
        the 2 following are ordered to correspond to grouped_data 
        values_over_labels is the sum of the values of sample divided by the distinct counts of labels. 
        grouped_counts represent this distinct count of labels
        """

        if verbose:
            print("Finding bin locations...")
            
        sample = sample.flatten()
        data = data.flatten()
        label = label.flatten()

        self.data_over_bins  = [[] for _ in range(self.nbins)]

        XSample_data = []
        XSample_values = []
        XSample_labels = []

        if verbose: print()

        if method == 'iterative reducing':
            # Store a tmp for each array
            first_bin = False
            tmp_sample = sample.copy()
            tmp_data = data.copy()
            tmp_label = label.copy()
            
            for i_bin, bin_size in enumerate(self.bin_sample_size):
                if bin_size == 0 : continue
                elif bin_size !=0:
                    if first_bin :  ## this is the first bin, we need to remove the nan values and then compute a max that only keeps a certain ratio of the points
                        tmp_sample = tmp_sample[~np.isnan(tmp_label)]
                        tmp_data = tmp_data[~np.isnan(tmp_label)]
                        tmp_label = tmp_label[~np.isnan(tmp_label)]
                        first_bin = False
                    
                        print('at bin #: ', i_bin, 'removed ', tmp_label.size/self.labels.size, '% of the arrays according to label == nan')

                        # create a random mask that contains a certain ratio of true elements that correspond to the ration of the next 2 bins 
                        first_excess_amount = bin_size - sizemax
                        mask = np.logical_and(tmp_sample.flatten() >= self.bins[i_bin], tmp_sample.flatten() < self.bins[i_bin+1])
                        # Indices of True values in the mask array
                        true_indices = np.where(mask)[0]
                        
                        # Randomly select indices to change to False
                        false_indices = np.random.choice(true_indices, size=first_excess_amount, replace=False)
                        # Change the selected indices to False
                        mask[false_indices] = False
                            
                        #compute data_over_bin according to mask
                        tmp_data_nan = tmp_data[mask[:]]
                        tmp_data_no_nan = tmp_data_nan[~np.isnan(tmp_data_nan)]
                        self.data_over_bins[i_bin]  = tmp_data_no_nan
                        
                        # reduce tmp array according to ~true_indices
                        tmp_sample = tmp_sample[~true_indices]
                        tmp_data = tmp_data[~true_indices]
                        tmp_label = tmp_label[~true_indices]
                    
                    else : 
                        mask = np.logical_and(tmp_sample.flatten() >= self.bins[i_bin], tmp_sample.flatten() < self.bins[i_bin+1])
                        ## Extremes computations only
                        if i_bin >= 40 and label is not None:
                            # put mask_no_nan to False if Xsample_labels[mask] is nan
                            mask_no_nan = np.logical_and(mask, ~np.isnan(tmp_label)) 
                            ## this must be done before the next line because tmp_sample still contains data of MCS of different durations
                            XSample_data.extend(tmp_data[mask_no_nan])
                            XSample_values.extend(tmp_sample[mask_no_nan])
                            XSample_labels.extend(tmp_label[mask_no_nan])
                            
                        #compute data_over_bin according to mask
                        tmp_data_nan = tmp_data[mask[:]]
                        tmp_data_no_nan = tmp_data_nan[~np.isnan(tmp_data_nan)]
                        self.data_over_bins[i_bin]  = tmp_data_no_nan
                        
                        ## reduce tmp array according to ~mask
                        tmp_sample = tmp_sample[~mask]
                        tmp_data = tmp_data[~mask]
                        tmp_label = tmp_label[~mask]                    

        if label is None : 
            return(self.data_over_bins)
        else:
            ## initialize the dictionnary that creates dictionnary if given a new key (age = x = data)
            data_dict = defaultdict(lambda: {'values': 0, 'distinct_labels': []})
            
            # fill that dict
            for x, value, label in zip(XSample_data, XSample_values, XSample_labels):
                data_dict[x]['values'] += value
                if label not in data_dict[x]['distinct_labels']:
                    data_dict[x]['distinct_labels'].append(label)
                
            ## group by keys and sum the values divided by the number of distinct labels 
            grouped_data = np.sort(list(data_dict.keys()))
            grouped_values = [data_dict[data]['values'] for data in grouped_data]
            grouped_counts = [len(data_dict[data]['distinct_labels']) for data in grouped_data]
            values_over_labels = [value/count for value, count in zip(grouped_values, grouped_counts)] 
            
            ## return data over bins, ages of Xprecip, and Xprecip over this ages 
            return(self.data_over_bins, grouped_data, values_over_labels, grouped_counts)

    def computeAgeAnalysisOverBins(self,sample, MCS_list = None, label = None, sizemax=5000, verbose=False, skip_to_X = False, method = 'iterative reducing'):
        """
        MCS_list and label are built over the TOOCAN files, so they could not be specified and instead simply be read from the file, let's keep it like that for clarity for now
        
        This function doesn't require bin_locations to be stored as it will recomputed everything mask per bin,
        but it will be much faster as it will reduce the size of the arrays after each bin computation.
        
        First output is the data with no treatmeant over bins
        
        the other outputs are optionnal and are computed only if the label array of MCS is provided
        they consist of an analysis of bins>40 so the extremes only
        
        grouped_data is the data aggregated by grouping, quite straightforward
        the 2 following are ordered to correspond to grouped_data 
        values_over_labels is the sum of the values of sample divided by the distinct counts of labels. 
        grouped_counts represent this distinct count of labels
        """
        from myFuncs import createTimeArray, Age_vec

        ## Measure time of init phase 
        t0 = time.time()

        ## Treat MCS_list to extract the labels 
        MCS_labels = [MCS_list[i].label for i in range(len(MCS_list))]
        
        ## Instantiate label_mask then time_array and flatten it as to match label and sample
        mask_label = ~np.isnan(label)
        time_array = createTimeArray(mask_label).flatten()

        ## flatten sample and label
        sample = sample.flatten()
        label = label.flatten()

        ## instantiate data over bins, and data over duration lists
        data_over_bins  = [[] for _ in range(self.nbins)]
        
        ## these actually depends of the durations of MCS studied.. for now 2h to 10h MCS
        MCS_duration_list = np.arange(4, 21, 1).astype(int).tolist() # [4, 5, ..., 20]
        ages_per_duration = [[] for i in range(len(MCS_duration_list))]

        ## instantiate the list that will be use for theAge Analysis
        XSample_data = []
        XSample_values = []
        XSample_labels = []
        XSample_duration = []

        ## instantiate a dictionnary to retrieve the max precip value per MCS at each timestep
        max_precipitations = {} 
        max_ages = {} #Along with the relative age of the mcs at which it occured
        
        ## print time of init phase
        print("init phase : ", time.time()-t0)
        
        if method == 'iterative reducing':
            # Store a tmp for each array
            first_bin = True
            tmp_sample = sample.copy()
            tmp_label = label.copy()
            
            for i_bin, percentile_value in enumerate(self.percentiles):
                ## measure time for each loop
                t0 = time.time()
                if percentile_value == 0 : continue
                elif percentile_value !=0:
                    if first_bin :  ## this is the first bin, we need to remove the nan values and then create a mask that only keeps a certain ratio of the points choosen randomly
                        time_array = time_array[tmp_sample!=0]                    
                        tmp_label = tmp_label[tmp_sample!=0]
                        tmp_sample = tmp_sample[tmp_sample!=0]
                        first_bin = False
                    
                        # create a random mask that contains a certain ratio of true elements that correspond to the ration of the next 2 bins 
                        
                        mask_excess = np.logical_and(tmp_sample.flatten() >= self.bins[i_bin], tmp_sample.flatten() < self.bins[i_bin+1])
                        # Indices of True values in the mask array
                        true_indices = np.where(mask_excess)[0]
                        
                        # Randomly select indices to change to False
                        false_indices = np.random.choice(true_indices, true_indices.size-sizemax, replace=False)

                        # Change the selected indices to False
                        first_mask = mask_excess.copy()
                        first_mask[false_indices] = False

                        first_mask_no_nan = np.logical_and(first_mask, ~np.isnan(tmp_label))
                        
                        first_time_array = time_array[first_mask_no_nan]
                        first_label = tmp_label[first_mask_no_nan]
                        

                        #compute data_over_bin according to label over mask and time_array info crossed by MCS list
                        out = Age_vec(first_label, first_time_array, MCS_list, MCS_labels)
                        ages = out[0]
                        
                        ## store the data over bins
                        data_over_bins[i_bin]  = ages
                        
                        tmp_sample = tmp_sample[~mask_excess]
                        time_array = time_array[~mask_excess]
                        tmp_label = tmp_label[~mask_excess]
                    
                    else : 
                        mask = np.logical_and(tmp_sample.flatten() >= self.bins[i_bin], tmp_sample.flatten() < self.bins[i_bin+1])
                        
                        # put mask_no_nan to False if Xsample_labels is nan
                        mask_no_nan = np.logical_and(mask, ~np.isnan(tmp_label))
                         
                        precips = tmp_sample[mask_no_nan]
                        labels = tmp_label[mask_no_nan]
                        times = time_array[mask_no_nan] 
                                            
                        # compute output with no nan
                        out = Age_vec(labels, times, MCS_list, MCS_labels)
                        ages = out[0]
                        durations = out[1]
                        if i_bin >= 40 and label is not None:
                            # update list for Age Analysis Xtremes only
                            XSample_data.extend(ages)
                            XSample_values.extend(precips)
                            XSample_labels.extend(labels)
                            XSample_duration.extend(durations)
                            
                        ## add data over bins
                        data_over_bins[i_bin]  = ages
                        
                        ## retrieve the max precip grouped by timestep and label
                        
                        # Iterate over the data and update the dictionary with the maximum precipitation values
                        for precip, label, t, age in zip(precips, labels, times, ages):
                            if label not in max_precipitations:
                                # If the label is not in the dictionary, add it with the current precipitation value
                                max_precipitations[label] = {}
                                max_ages[label] = {}
                                
                            if t not in max_precipitations[label]:
                                # If the timestep is not in the label's dictionary, add it with the current precipitation value
                                max_precipitations[label][t] = precip
                                max_ages[label][t] = age
                            else:
                                # If the timestep already exists in the label's dictionary, update the maximum precipitation value
                                if precip > max_precipitations[label][t]:
                                    max_precipitations[label][t] = precip
                                    max_ages[label][t] = age

                        ## reduce tmp array according to ~mask
                        tmp_sample = tmp_sample[~mask]
                        time_array = time_array[~mask]                   
                        tmp_label = tmp_label[~mask] 

                ## print time of each loop
                print("loop ", i_bin, " : ", time.time()-t0)
                
        for age, duration in zip(XSample_data, XSample_duration):
            for i, MCS_duration in enumerate(MCS_duration_list):
                if duration == MCS_duration:
                    ages_per_duration[i].append(age)
                
        ## initialize the dictionnary that creates dictionnary if given a new key (age = x = data)
        mean_Xprecip_data_dict = defaultdict(lambda: {'values': 0, 'distinct_labels': []})
        
        # fill that dict
        for x, value, label in zip(XSample_data, XSample_values, XSample_labels):
            mean_Xprecip_data_dict[x]['values'] += value
            if label not in mean_Xprecip_data_dict[x]['distinct_labels']:
                mean_Xprecip_data_dict[x]['distinct_labels'].append(label)
        
        ## group by keys and sum the values divided by the number of distinct labels 
        meanX_grouped_data = np.sort(list(mean_Xprecip_data_dict.keys()))
        meanX_grouped_values = [mean_Xprecip_data_dict[data]['values'] for data in meanX_grouped_data]
        meanX_grouped_counts = [len(mean_Xprecip_data_dict[data]['distinct_labels']) for data in meanX_grouped_data]
        meanX_values_over_labels = [value/count for value, count in zip(meanX_grouped_values, meanX_grouped_counts)] 
        
        max_precip_data_dict = defaultdict(lambda: {'values': 0, 'distinct_labels': []})
        
        for label, time_max_precip in max_precipitations.items():
            ages_max_precip = max_ages[label]
            for t, max_precip in time_max_precip.items():
                age = ages_max_precip[t]
                max_precip_data_dict[age]['values'] += max_precip
                if label not in max_precip_data_dict[age]['distinct_labels']:
                    max_precip_data_dict[age]['distinct_labels'].append(label)
                    
        max_grouped_data = np.sort(list(max_precip_data_dict.keys()))
        max_grouped_values = [max_precip_data_dict[data]['values'] for data in max_grouped_data]
        max_grouped_counts = [len(max_precip_data_dict[data]['distinct_labels']) for data in max_grouped_data]
        max_values_over_labels = [value/count for value, count in zip(max_grouped_values, max_grouped_counts)]
                
                  
        ## return data over bins, ages of Xprecip, and Xprecip over this ages 
        return((data_over_bins, meanX_grouped_data, meanX_values_over_labels, meanX_grouped_counts, ages_per_duration, max_grouped_data, max_values_over_labels, max_grouped_counts))
    
    def computeIndividualPercentiles(self,sample,ranks,out=False):
        """Computes percentiles of input sample and store in object attribute"""

        if isinstance(ranks,float) or isinstance(ranks,int):
            ranks = [ranks]
        
        result = []

        for r in ranks:
            # calculate percentile
            p = np.percentile(sample,r)
            result.append(p)
            # save
            setattr(self,"perc%2.0f"%r,p)

        if out:
            return result

    def computeInvCDF(self,sample,out=False):
        """Calculate 1-CDF on inverse-logarithmic ranks: fraction of rain mass falling 
        above each percentile"""
        
        self.invCDF = np.ones(self.nbins)*np.nan
        sample_sum = np.nansum(sample)
        for iQ in range(self.nbins):
            rank = self.ranks[iQ]
            perc = self.percentiles[iQ]
            if not np.isnan(perc):
                self.invCDF[iQ] = np.nansum(sample[sample>perc])/sample_sum

        if out:
            return self.invCDF

    def bootstrapPercentiles(self,sample,nd_resample=10,n_bootstrap=50):
        """Perform bootstrapping to evaluate the interquartile range around each
        percentile, for the ranks stored.

        Arguments:
        - sample: np array in Nt,Ny,Nx format
        - nd_resample: number of time indices to randomly select for resampling
        - n_boostrap: number of times to calculate the distribution
        """

        sshape = sample.shape
        d_time = 0

        # calculate and store distribution n_bootstrap times
        perc_list = []
        for i_b in range(n_bootstrap):

            # select days randomly
            indices = list(range(sshape[d_time]))
            np.random.shuffle(indices)
            ind_times = indices[:nd_resample]
            resample = np.take(sample,ind_times,axis=d_time)

            # calculate percentiles on resample
            perc, bins = self.computePercentilesAndBinsFromRanks(resample,
                                            store=False,output=True)

            perc_list.append(perc)

        # combine distributions into statistics and save
        perc_array = np.vstack(perc_list)
        self.percentiles_sigma = np.std(perc_array,axis=0)
        self.percentiles_P5 = np.percentile(perc_array,5,axis=0)
        self.percentiles_Q1 = np.percentile(perc_array,25,axis=0)
        self.percentiles_Q2 = np.percentile(perc_array,50,axis=0)
        self.percentiles_Q3 = np.percentile(perc_array,75,axis=0)
        self.percentiles_P95 = np.percentile(perc_array,95,axis=0)
        
    def getCDF(self):
        """Compute the cumulative density function from the probability density,
        as: fraction of points below vmin + cumulative sum of density*bin_width
        Output is the probability of x < x(bin i), same size as bins (bin edges)"""
        
        # array of bin widths
        bin_width = np.diff(self.bins)
        # CDF from density and bin width
        cdf_base = np.cumsum(bin_width*self.density)
        # readjust  to account for the fraction of points outside the range [vmin,vmax]
        fmin = self.frac_below_vmin
        fmax = self.frac_above_vmax
        cdf = fmin + np.append(0,cdf_base*(1-fmin-fmax))
        
        return cdf


class JointDistribution():
    """Documentation for class JointDistribution
    
    Creates a joint distribution for two variables
    """
    
    def __init__(self,name='',distribution1=None,distribution2=None,overwrite=False):
        """Constructor for class Distribution.
        Arguments:
        - name: name of reference variable
        - distribution1, distribution2: marginal distributions of two reference variables
        - overwrite: option to overwrite stored data in object
        """

        self.name = name
        self.distribution1 = distribution1
        self.distribution2 = distribution2
        self.bins1 = self.distribution1.bins
        self.bins2 = self.distribution2.bins
        self.density = None
        self.bin_locations_stored = False
        self.overwrite = overwrite

        # if distribution is not None: # then copy it in self
        #     for attr in distribution.__dict__.keys():
        #         setattr(self,attr,getattr(distribution,attr)) 
    
    
    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< JointDistribution object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out
    
    
    def __str__(self):
        """Override string function to print attributes
        """
        # method_names = []
        # str_out = '-- Attributes --'
        str_out = ''
        for a in dir(self):
            if '__' not in a:
                a_str = str(getattr(self,a))
                if 'method' not in a_str:
                    str_out = str_out+("%s : %s\n"%(a,a_str))
        #         else:
        #             method_names.append(a)
        # print('-- Methods --')
        # for m in method_names:
        #     print(m)
        return str_out
    
    def formatDimensions(self,sample):
        """Reshape the input data, test the validity and returns the dimensions
        and formatted data.
    
        Arguments:
        - sample: here we assume data is horizontal, formats it in shape (Ncolumns,)
        Controls if it matches the data used for the control distribution. If
        not, aborts.
        """
    
        # Get shape
        sshape = sample.shape
        # Initialize default output
        sample_out = sample
        # Get dimensions and adjust output shape
        if len(sshape) > 1: # reshape
            sample_out = np.reshape(sample,np.prod(sshape))
        Npoints, = sample_out.shape
        
        # Test if sample size is correct to access sample points
        if Npoints != self.size:
            raise WrongArgument("Error: used different sample size")
    
        return sample_out

    def storeSamplePoints(self,sample1,sample2,sizemax=50,verbose=False,method='shuffle_mask'):
        """Find indices of bins in the sample data, to get a mapping or extremes 
        and fetch locations later
        """
    
        if self.bin_locations_stored and not self.overwrite:
            pass

        if verbose:
            print("Finding bin locations...")

        # print(sample.shape)
        sample1 = self.formatDimensions(sample1)
        sample2 = self.formatDimensions(sample2)
        # print(sample.shape)
        
        # Else initalize and find bin locations
        self.bin_locations = [[[] for _ in range(self.distribution2.nbins)] for _ in range(self.distribution1.nbins)]
        self.bin_sample_size = [[0 for _ in range(self.distribution2.nbins)] for _ in range(self.distribution1.nbins)]

        if method == 'shuffle_mask':

            if verbose: print('bin #: ',end='')
            # compute mask for each bin, randomize and store 'sizemax' first indices
            for i_bin in range(self.distribution1.nbins):
                
                for j_bin in range(self.distribution2.nbins):
                    
                    if verbose: print('%d,%d..'%(i_bin,j_bin),end='')
                    
                    # compute mask
                    mask1 = np.logical_and(sample1.flatten() >= self.distribution1.bins[i_bin],
                                sample1.flatten() < self.distribution1.bins[i_bin+1])
                    mask2 = np.logical_and(sample2.flatten() >= self.distribution2.bins[j_bin],
                                sample2.flatten() < self.distribution2.bins[j_bin+1])
                    mask = np.logical_and(mask1,mask2)
                    # get all indices
                    ind_mask = np.where(mask)[0]
                    # shuffle
                    np.random.seed(int(round(time.time() * 1000)) % 1000)
                    np.random.shuffle(ind_mask)
                    # select 'sizemax' first elements
                    self.bin_locations[i_bin][j_bin] = ind_mask[:sizemax]
                    # self.bin_sample_size[i_bin] = min(ind_mask.size,sizemax) # cap at sizemax
                    self.bin_sample_size[i_bin][j_bin] = ind_mask.size # count all points there
                    
                if verbose: print()

        if verbose:
            print()

        # If reach this point, everything should have worked smoothly, so:
        self.bin_locations_stored = True

    def computeDistribution(self,sample1,sample2,vmin=None,vmax=None,minmode=None):

        """Compute ranks, bins, percentiles and corresponding probability densities.
        Arguments:
            - sample1,2: 1D numpy array of values
        Computes:
            - ranks, percentiles, bins and probability densities"""

        if not self.overwrite:
            pass

        # Compute probability density
        self.density, _, _ = np.histogram2d(x=sample1,y=sample2,bins=(self.bins1,self.bins2),density=True)

    def computeNormalizedDensity(self, sample1, sample2, data = None, verbose = False):

        if sample1.shape != sample2.shape : ## here assume sample 1 has bigger 
            sameShape = False
            ratio_t, ratio_y, ratio_x = int(sample1.shape[0]/sample2.shape[0]), int(sample1.shape[1]/sample2.shape[1]), int(sample1.shape[2]/sample2.shape[2])
        else : 
            sameShape = True
            ratio_t, ratio_y, ratio_x = 1,1,1

        l1, l2 = len(self.bins1), len(self.bins2)

        digit1 = np.digitize(sample1, self.bins1, right = True)
        digit2 = np.digitize(sample2, self.bins2, right = True)
        #if verbose : print(digit1, digit2)
        N1 = [np.sum(digit1==i1) for i1 in range(l1)]
        N2 = [np.sum(digit2==i2) for i2 in range(l2)]
        Ntot = sample1.size
        with np.errstate(divide='ignore'):
            Norm = Ntot / np.outer(N1, N2)
        Norm[np.isinf(Norm)] = 0

        self.density = np.zeros(shape = (l1, l2))
        if data is not None : data_over_density = np.zeros(shape=(l1,l2))

        if sameShape :
            for i2 in range(l2): 
                idx = tuple(np.argwhere(digit2==i2).T)
                self.density[:, i2] = np.bincount(digit1[idx], minlength=l1)

                if data is not None: 
                    for i1 in range(l1):
                        data_idx = tuple(np.argwhere((digit1==i1) & (digit2==i2)).T)
                        if len(data_idx)>0 :
                            data_over_density[i1, i2] = np.nanmean(data[data_idx])
                        else : data_over_density[i1, i2] = 0

            self.density *= Norm

        if data is not None : return data_over_density
        else : return self.density/Norm/Ntot, 1/Norm, N1, N2, Ntot


    def computeVariationOverDensity(self, sample1, sample11, bins11, sample2, sample22, bins22, reverse = False, verbose = False):

        l1, l2 = len(self.bins1), len(self.bins2)

        assert l1 == len(bins11) & l1 == len(bins22)

        digit1 = np.digitize(sample1, self.bins1, right = True)
        digit2 = np.digitize(sample2, self.bins2, right = True)

        
        digit11 = np.digitize(sample11, bins11, right = True)
        digit22 = np.digitize(sample22, bins22, right = True)

        N1 = [np.sum(digit1==i1) for i1 in range(l1)]
        N2 = [np.sum(digit2==i2) for i2 in range(l2)]

        with np.errstate(divide='ignore'):
            Norm = 1 / np.sqrt(np.outer(N1, N2))
        Norm[np.isinf(Norm)] = 1

        self.density = np.zeros(shape = (l1, l2))

        data1_over_density = np.zeros(shape=(l1,l2))
        data2_over_density = np.zeros(shape=(l1,l2))


        for i2 in range(l2): 
            idx_i = tuple(np.argwhere(digit2==i2).T)
            self.density[:, i2] = np.bincount(digit1[idx_i], minlength=l1)

            for i1 in range(l1):
                next_idx = tuple(np.argwhere((digit11==i1) & (digit22==i2)).T) #if bins are the same, i2 and i1 correponds to the same bin
                current_idx = tuple(np.argwhere((digit1==i1) & (digit2==i2)).T)

                if (len(next_idx)>0) & (len(current_idx)>0) :
                    if reverse :
                        data1_over_density[i1, i2] =  (- np.mean(sample11[next_idx]) + np.mean(sample1[current_idx])) / (5*np.mean(sample11[next_idx]))
                        data2_over_density[i1, i2] =  (- np.mean(sample22[next_idx]) + np.mean(sample2[current_idx])) / (5*np.mean(sample22[next_idx]))
                        
                    else : 
                        data1_over_density[i1, i2] = (np.mean(sample11[next_idx]) - np.mean(sample1[current_idx])) / (5*np.mean(sample1[current_idx]))
                        data2_over_density[i1, i2] = (np.mean(sample22[next_idx]) - np.mean(sample2[current_idx])) / (5*np.mean(sample2[current_idx]))

                else : 
                    data1_over_density[i1, i2] = 0
                    data2_over_density[i1, i2] = 0

        self.density *= Norm

        return data1_over_density, data2_over_density



class ConditionalDistribution():
    """Documentation for class ConditionalDistribution.

    Stores conditional mean and variance in bins of a reference distribution.
    """

    def __init__(self,name='',is3D=False,isTime=False,on=None):
        """Contructor
        
        Arguments:
        - name
        - is3D: boolean, if variable is defined in the vertical dimension
        - on: Object Distribution containing stats of reference variable
        """

        self.name = name
        self.is3D = is3D
        self.isTime = isTime
        self.on = on
        self.mean = None
        self.cond_mean = None
        self.cond_var = None

    def __repr__(self):
        """Creates a printable version of the ConditionalDistribution object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< ConditionalDistribution object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out

    def computeMean(self,sample):
        """Computes the (full) mean of the input data
        """

        self.mean = np.nanmean(sample)

    def formatDimensions(self,sample):
        """Reshape the input data, test the validity and returns the dimensions
        and formatted data.

        Arguments:
        - sample: if is3D, format it in shape (Nz,Ncolumns), otherwise (Ncolumns,)
        Controls if it matches the data used for the control distribution. If
        not, aborts.
        """

        # Get shape
        sshape = sample.shape
        # Initialize default output
        sample_out = sample
        # Get dimensions and adjust output shape
        if self.is3D:
            # collapse dimensions other than z
            if len(sshape) > 2: # reshape
                # if time dimension (in 2nd dimension), reorder to have z in first dim
                if self.isTime:
                    # nlev = sample_out.shape[0]
                    sample_out = np.swapaxes(sample_out,0,1)
                    sshape = sample_out.shape
                sample_out = np.reshape(sample_out,(sshape[0],np.prod(sshape[1:])))
            Nz,Npoints = sample_out.shape
            # if self.isTime:
            #     Npoints = Npoints/nlev
            #     print(Npoints)
        else:
            if len(sshape) > 1: # reshape
                sample_out = np.reshape(sample,np.prod(sshape))
            Npoints, = sample_out.shape
            Nz = None
        
        # Test if sample size is correct to access sample points
        if Npoints != self.on.size:
            #print(Npoints,self.on.size)
            raise WrongArgument("ABORT: sample size is different than that of the reference variable, so the masks might differ")

        return Nz, sample_out

    def computeConditionalMeanAndVariance(self,sample,verbose=False):
        """Computes mean and variance of input data at each percentile of 
        reference variable (in bins self.on.bins).

        Arguments:
        - sample: if is3D, should be in format (Nz,Ncolumns), otherwise (Ncolumns,)
        If not, it is formatted using method formatDimensions.
        """

        # Abort if sample points for each percentile has not been computed yet
        if self.on is None or not self.on.bin_locations_stored:
            raise EmptyDataset("Abort: must calculate bin locations of reference distribution first")

        # format dataset and test validity of input dataset
        Nz, sample = self.formatDimensions(sample)

        # Initialize storing arrays
        if self.is3D:
            self.cond_mean = np.nan*np.zeros((Nz,self.on.nbins))
            self.cond_var = np.nan*np.zeros((Nz,self.on.nbins))
        else:
            self.cond_mean = np.nan*np.zeros((self.on.nbins,))
            self.cond_var = np.nan*np.zeros((self.on.nbins,))

        # Access sample points to calculate conditional stats
        # automate
        def apply2vector(fun,vector):
            out = np.nan*np.zeros(self.on.nbins)
            for i_b in range(self.on.nbins): # loop over bins
                subsample = np.take(vector,self.on.bin_locations[i_b])
                if subsample.size == 0:
                    if verbose:
                        print('passing bin %d, subsample of size %d'%(i_b,subsample.size))
                    # pass
                else:
                    if verbose:
                        print("bin %d, result:%2.2f"%(i_b,fun(subsample)))
                    out[i_b] = fun(subsample)
            return out
        # compute
        if self.is3D:
            for i_z in range(Nz): # loop over heights
                self.cond_mean[i_z] = apply2vector(np.nanmean,np.squeeze(sample[i_z]))
                self.cond_var[i_z] = apply2vector(np.nanvar,np.squeeze(sample[i_z]))
        else:
            self.cond_mean = apply2vector(np.nanmean,sample)
            self.cond_var = apply2vector(np.nanvar,sample)

        self.cond_std = np.sqrt(self.cond_var)

class DistributionOverTime(Distribution):
    """Time evolution of an object of class Distribution.
    """

    def __init__(self,name='',time_ref=[],width=0,**kwargs):
        """Constructor of class DistributionOverTime

        Arguments:
        - *args: see input parameters of constructor Distribution.__init__
        - time_ref: np.array of time values for reference dataset (in days, usually)
        - time: np.array of time values for calculated statistics
        - width: width of time window used to calculate statistics (same unit as time)
        """
        # for key, value in kwargs.items(): 
        #     print ("%s == %s" %(key, value))
        self.name = name
        self.time_ref = time_ref    
        self.nt = len(self.time_ref)
        self.width = width
        # time step of reference data
        self.dt = 0
        if self.nt > 1:
            self.dt = np.diff(self.time_ref)[0]
        # remove dn first and last points for statistics
        self.dn = 0
        if self.dt != 0:
            self.dn = int(self.width/2./self.dt)
        # destination time values
        self.time = self.time_ref[self.dn:len(self.time_ref)-self.dn]
        # initialize empty distributions
        self.distributions = [Distribution(name,**kwargs) for i in range(self.nt-2*self.dn)]

    def __repr__(self):
        """Creates a printable version of the DistributionOverTime object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< DistributionOverTime object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out

    def iterTime(self):

        return range(self.nt-2*self.dn)

    def iterRefTimeIndices(self):

        ref_inds = range(self.dn,self.nt-self.dn)
        it_slices = [slice(i_t-self.dn,i_t+self.dn+1) for i_t in ref_inds]
        it_stores = [i_t-self.dn for i_t in ref_inds]

        return zip(it_slices,it_stores)

    def testInput(self,sample):
        """Test that time dimension is matching first dimension of imput sample
        """
        sshape = sample.shape
        if self.nt == 1:
            return
        if sshape[0] != self.nt:
            raise WrongArgument('ABORT: input sample does not have the correct'+\
            ' time dimension')

    def computeDistributions(self,sample,**kwargs):
        """Fills up the distribution of timeVar 

        Arguments:
        - sample: np.array of dimensions (nt,...)
        - *args: see input parameters in method Distribution.computeDistribution
        """

        # Test dimensions
        self.testInput(sample)

        # Compute all distributions over time
        for it_slice,it_store in self.iterRefTimeIndices():

            self.distributions[it_store].computeDistribution(sample[it_slice],**kwargs)

    def storeSamplePoints(self,sample,sizemax=50,method='shuffle_mask',verbose=False):
        """Find indices of bins in the sample data, to go back and fetch
        """

        # Test dimensions
        self.testInput(sample)

        # Compute bin locations if not known already
        for it_slice,it_store in self.iterRefTimeIndices():

            print("%d_%d"%(it_slice.start,it_slice.stop),end=' ; ')

            self.distributions[it_store].storeSamplePoints(sample=sample[it_slice],sizemax=sizemax,method=method,verbose=verbose)

        print()

    def computeIndividualPercentiles(self,sample,ranks):
        """Computes percentiles of input sample and store timeseries in object
        attribute. CAREFUL here only do calculation at each time, without using
        iterRefTimeIndices method.
        
        Arguments:
        - sample as above
        - ranks: float, list or np.array"""

        # Test dimensions
        self.testInput(sample)

        if isinstance(ranks,float):
            ranks = [ranks]
        
        for r in ranks:

            vals = np.nan*np.zeros((self.nt,))

            # Compute all distributions over time
            for it_slice,it_store in self.iterRefTimeIndices():

                vals[it_store] = self.distributions[it_store].computeIndividualPercentiles(sample[it_slice],r,out=True)[0]

            # save
            setattr(self,"perc%2.0f"%r,vals)

class ConditionalDistributionOverTime():
    """Time evolution of an object of class ConditionalDistribution.
    """

    def __init__(self,name='',time_ref=[],width=0,is3D=False,isTime=True,on=None):
        """Constructor of class ConditionalDistributionOverTime

        Arguments:
        - *args: see input parameters of constructor ConditionalDistribution.__init__
        """

        self.name = name
        self.time_ref = time_ref
        self.nt = len(self.time_ref)
        self.width = width
        # time step of reference data
        self.dt = 0
        if self.nt > 1:
            self.dt = np.diff(self.time_ref)[0]
        # remove dn first and last points for statistics
        self.dn = 0
        if self.dt != 0:
            self.dn = int(self.width/2./self.dt)
        # destination time values
        self.time = self.time_ref[self.dn:len(self.time_ref)-self.dn]

        self.cond_distributions = []
        # Initializes all reference distributions
        for on_i in on.distributions:
            self.cond_distributions.append(ConditionalDistribution(name,is3D=is3D,isTime=isTime,on=on_i))

    def __repr__(self):
        """Creates a printable version of the ConditionalDistributionOverTime object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< ConditionalDistributionOverTime object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out

    def iterTime(self):

        return range(self.nt-2*self.dn)

    def iterRefTimeIndices(self):

        ref_inds = range(self.dn,self.nt-self.dn)
        it_slices = [slice(i_t-self.dn,i_t+self.dn+1) for i_t in ref_inds]
        it_stores = [i_t-self.dn for i_t in ref_inds]

        return zip(it_slices,it_stores)

    def testInput(self,sample):
        """Test that time dimension is matching first dimension of imput sample
        """
        sshape = sample.shape
        if self.nt == 1:
            return
        if sshape[0] != self.nt:
            raise WrongArgument('ABORT: input sample does not have the correct'+\
            ' time dimension')
            
    def storeSamplePoints(self,sample,sizemax=50,method='shuffle_mask',verbose=False):
        """Find indices of bins in the sample data, to go back and fetch

        Arguments:
        - sample: reference sample! np.array of dimensions (nt,...)
        - sizemax: maximum number of indices stored for each bin
        """

        # Test dimensions
        self.testInput(sample)

        # Compute bin locations if not known already
        for it_slice,it_store in self.iterRefTimeIndices():

            print("%d_%d"%(it_slice.start,it_slice.stop),end=' ; ')

            self.cond_distributions[it_store].on.storeSamplePoints(sample=sample[it_slice],\
                sizemax=sizemax,method=method,verbose=verbose)

        print()

    def computeConditionalStatsOverTime(self,sample,**kwargs):
        """Fills up the distribution of timeVar 

        Arguments:
        - sample: np.array of dimensions (nt,...)
        - **kwargs: see input parameters in method ConditionalDistribution.computeConditionalMeanAndVariance
        """

        # Test dimensions
        self.testInput(sample)

        # Compute all distributions over time
        for it_slice,it_store in self.iterRefTimeIndices():

            self.cond_distributions[it_store].computeConditionalMeanAndVariance(sample[it_slice],**kwargs)
        