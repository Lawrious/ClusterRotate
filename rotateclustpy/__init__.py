# -*- coding: utf-8 -*-
############################# rotateclustpy ####################################
################################################################################
####################### Author: Lawrence E. Bilton #############################

import numpy as np
from progress.bar import IncrementalBar
import time
import scipy.stats as st
import warnings

warnings.filterwarnings('ignore')
rad90 = 90. * (np.pi/180.)

#nme,dirr=pathfinder('/Users/Lawrious/Documents/OneDrive/The University of Hull/Data/SDSS VAC/Sample/ProjectTwo/FITS/')

class detector:

    def __init__(self):
        pass


    def rotate(self,px, py, angle, origin):
        """
        Rotate a point by a given angle around a given origin.

        Negative angles == Clockwise

        Positive Angles == Counter-Clockwise

        The angle should be given in radians.


        """

        angle = angle * (np.pi/180)
        ox, oy = origin
        #px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def clustrotate(self,ra,dec,ra_clu,dec_clu,vlos,radius,r_cut=1.55,origin=(0.,0.)):

        vdiff=[]
        vdifftheta=[]
        vdiffer=[]
        ksap=[]
        pkap=[]


        #Applying radial cut
        rchk = np.where(radius <= r_cut)
        ra = ra[rchk]
        dec = dec[rchk]
        vlos = vlos[rchk]
        radius = radius[rchk]

        # Normalising to the origin (BAX defined cluster galaxy centre)
        ra_origin = ra-ra_clu
        dec_origin = dec-dec_clu

        #Looping through 10 degree increments
        theta = 0
        while theta <= 360:
            # Calculating angles
            tanratios = np.absolute(ra_origin/dec_origin)
            gal_angles = np.arctan(tanratios)#*(180/np.pi)

            # Seperating hemispheres
            hem1 = np.where(ra_origin <= 0.)
            hem2 = np.where(ra_origin > 0.)

            # Calculating modified v_los for each hemispehre for v_diff calculation
            v1 = vlos[hem1]*np.cos(rad90-gal_angles[hem1])
            v1mean = np.nanmean(v1)
            v1std = np.nanstd(v1)

            v2 = vlos[hem2]*np.cos(rad90-gal_angles[hem2])
            v2mean = np.nanmean(v2)
            v2std = np.nanstd(v2)
            # if theta == 0:
            #     vd = 0.0
            # else:
            kd,pk = st.ks_2samp(v1,v2)

            vd = v1mean-v2mean
            vder = np.sqrt(((np.power(v1std,2)/len(v1))+(np.power(v2std,2)/len(v2))))
            vdiff.append(vd)
            vdifftheta.append(theta)
            vdiffer.append(vder)
            ksap.append(kd)
            pkap.append(pk)

            #Rotate cluster by 10 degrees

            ra_origin,dec_origin = self.rotate(px=ra_origin,py=dec_origin,angle=-10.,origin=origin)
            theta = theta + 10

        #kd,pk = st.ks_2samp(np.array(v1mnap),np.array(v2mnap))
        #print 'Calculated vdiff(theta) through to 360 degrees\n'

        self.vdiff = np.array(vdiff)
        self.vdifftheta = np.array(vdifftheta)
        self.vrot = max(vdiff)
        self.vderr = np.array(vdiffer)
        self.ks_stat_theta = np.array(ksap)
        self.p_ks_theta = np.array(pkap)
        self.p_ks_max = self.p_ks_theta[np.argmax(self.vdiff)]

        #Calculating ideal rotation curve

        self.vdiff_ideal,self.vdiffer_ideal,self.vdiff_comp,self.vdiffer_comp = self.rotideal(ra=ra,dec=dec,ra_clu=ra_clu,dec_clu=dec_clu,vlos=vlos,vel_diff=self.vdiff,diff_theta=self.vdifftheta)

        #Calculating random rotation curve

        self.vdiff_rand,self.vdiffer_rand = self.rotrand(ra=ra,dec=dec,ra_clu=ra_clu,dec_clu=dec_clu,vlos=vlos,vel_diff=self.vdiff,diff_theta=self.vdifftheta,resamples=10000)

        #Calculating ideal chi-squared statistic
        self.chisq_ideal = self.chisqu(vdat=self.vdiff_comp,vmod=self.vdiff_ideal,erdat=self.vdiffer_comp,ermod=self.vdiffer_rand)

        #Caluculating random chi-squared statistic

        self.chisq_random = self.chisqu(vdat=self.vdiff_comp,vmod=self.vdiff_rand,erdat=self.vdiffer_comp,ermod=self.vdiffer_rand)
        time.sleep(1)
        print('The chi-sqaured statistic for the ideal curve: %.8f' % self.chisq_ideal)
        print('The chi-sqaured statistic for the random curve: %.8f' % self.chisq_random)

        self.chisq_idedf = self.chisq_ideal/36.0
        self.chisq_randf = self.chisq_random/36.0

        self.chisq_ratio = self.chisq_ideal/self.chisq_random

        # if self.chisq_ideal <= self.chisq_random:
        #     self.chistatcomp = 'True'
        # elif self.chisq_ideal > self.chisq_random:
        #     self.chistatcomp = 'False'

        ## Running Strict Criteria Check for Rotation ##

        if self.chisq_idedf <= 1. and self.chisq_randf > 1.:
            chidf_one = 'True'
        else:
            chidf_one = 'False'

        if self.chisq_ratio <= 0.2:
            self.chistatcomp = 'True'
        elif self.chisq_ratio > 0.2:
            self.chistatcomp = 'False'
        if self.p_ks_max <= 0.01:
            chisq_pm = 'True'
        elif self.p_ks_max > 0.01:
            chisq_pm = 'False'

        # Rewrite this for proper definition of 'strict' and 'loose' criteria
        # if self.chisq_idedf <= 1.0:
        #     chidf_one = 'True'
        # elif self.chisq_idedf > 1.0:
        #     chidf_one = 'False'
        # if self.chisq_randf <= 1.0:
        #     chidf_two = 'True'
        # elif self.chisq_randf > 1.0:
        #     chidf_two = 'False'

        if self.chistatcomp == 'True' and chisq_pm == 'True' and chidf_one == 'True':
            self.chisqstrict = 'True'
        else:
            self.chisqstrict = 'False'

        ## Running Loose Criteria Check ##

        if self.chisq_ratio <= 0.4:
            chisq_ls = 'True'
        elif self.chisq_ratio > 0.4:
            chisq_ls = 'False'

        if chisq_ls == 'True' and chisq_pm == 'True':
            self.chisqloose = 'True'
        else:
            self.chisqloose = 'False'


        time.sleep(1)

        return 1


    def rotideal(self,ra,dec,ra_clu,dec_clu,vlos,vel_diff,diff_theta):

        time.sleep(1)
        print('Intiatiating model ideal curve\n')

        #Producing an ideal curve
        vdiff=[]
        vdiffer=[]

        #Defining the origin of the cluster
        ra_origin = ra-ra_clu
        dec_origin = dec-dec_clu

        #Finding vdiff_theta when v_diff == v_rot of the cluster
        mxdf_theta = diff_theta[np.where(vel_diff == np.nanmax(vel_diff))]
        if np.any(mxdf_theta == 0) == True or np.any(mxdf_theta == 360) == True:
            mxdf_theta = int(0)
        ttcha= '\u03b8'
        time.sleep(1)
        print("vrot = %f kms^{-1} at angle %s = %d\n" % (np.nanmax(vel_diff),ttcha,mxdf_theta))

        ra_origin,dec_origin = self.rotate(px=ra_origin,py=dec_origin,angle=(-mxdf_theta),origin=(0.,0.))


        ### Producing ideal curve ###
        vrot1 = max(vel_diff)/2.
        vrot2 = (-(max(vel_diff)))/2.
        vlos_id=[] #= np.full(len(ra_origin[sem1]),vrot1)
        for i in ra_origin:
            if i <= 0.:
                vlos_id.append(vrot1)
            elif i > 0.:
                vlos_id.append(vrot2)
        vlos_ideal = np.array(vlos_id)

        #vlos2= np.full(len(ra_origin[sem2]),vrot2)
        time.sleep(1)
        print('Rotating cluster for the model ideal curve...\n')
        time.sleep(1)
        theta = 0.
        while theta <= 360:
            # Calculating angles
            tanratios = np.absolute(ra_origin/dec_origin)
            gal_angles = np.arctan(tanratios)#*(180/np.pi)

            # Seperating hemispheres
            hem1 = np.where(ra_origin <= 0.)
            hem2 = np.where(ra_origin > 0.)

            # vrot1 = max(vel_diff)/2
            # vlos1 = np.full(len())

            v1 = vlos_ideal[hem1]*np.cos(rad90-gal_angles[hem1])
            v1mean = np.nanmean(v1)
            v1std = self.sigmonte(data=v1,resamples=500)

            v2 = vlos_ideal[hem2]*np.cos(rad90-gal_angles[hem2])
            v2mean = np.nanmean(v2)
            v2std = self.sigmonte(data=v2,resamples=500)
            # if theta == 0:
            #     vd = 0.0
            # else:

            vd = v1mean-v2mean
            vder = np.sqrt(((np.power(v1std,2))+(np.power(v2std,2))))
            vdiff.append(vd)
            vdiffer.append(vder)

            #Rotate cluster by 10 degrees

            ra_origin,dec_origin = self.rotate(px=ra_origin,py=dec_origin,angle=-10.,origin=(0.,0.))
            theta = theta + 10.

        vdiff_ideal = np.array(vdiff)
        vdiffer_ideal = np.array(vdiffer)


        #Redefining the origin of the cluster again for real data comparison
        ra_origin = ra-ra_clu
        dec_origin = dec-dec_clu

        ### Finding vdiff_theta when v_diff == v_rot of the cluster ###

        ra_origin,dec_origin = self.rotate(px=ra_origin,py=dec_origin,angle=(-mxdf_theta),origin=(0.,0.))

        vdreal=[]
        vdreer=[]

        theta = 0
        while theta <= 360:
            # Calculating angles
            tanratios = np.absolute(ra_origin/dec_origin)
            gal_angles = np.arctan(tanratios)#*(180/np.pi)

            # Seperating hemispheres
            hem1 = np.where(ra_origin <= 0.)
            hem2 = np.where(ra_origin > 0.)

            v1 = vlos[hem1]*np.cos(rad90-gal_angles[hem1])
            v1mean = np.nanmean(v1)
            v1std = np.nanstd(v1)

            v2 = vlos[hem2]*np.cos(rad90-gal_angles[hem2])
            v2mean = np.nanmean(v2)
            v2std = np.nanstd(v2)

            vd = v1mean-v2mean
            vder = np.sqrt(((np.power(v1std,2)/len(v1))+(np.power(v2std,2)/len(v2))))
            vdreal.append(vd)
            vdreer.append(vder)

            ## Rotate cluster by 10 degrees ##

            ra_origin,dec_origin = self.rotate(px=ra_origin,py=dec_origin,angle=-10.,origin=(0.,0.))
            theta = theta + 10

        vdiff_comp = np.array(vdreal)
        vdiff_cmer = np.array(vdreer)

        time.sleep(1)
        print('Ideal curve computed\n')

        return vdiff_ideal,vdiffer_ideal,vdiff_comp,vdiff_cmer

    def sigmonte(self,data,resamples=1000):

        data_app = []

        i = 0

        while i <= resamples:

            np.random.seed()

            monte = np.floor(np.random.rand(len(data))*len(data)).astype(int)

            dat_new=data[monte]

            data_app.append(np.nanmean(dat_new))

            i = i+1

        data_appar =np.array(data_app)

        sigdata = np.nanstd(data_appar)

        return sigdata


    def rotrand(self,ra,dec,ra_clu,dec_clu,vlos,vel_diff,diff_theta,resamples=10000):

        time.sleep(1)

        print('Initialising model random curve\n')

        time.sleep(1)

        vdiff_rdmn=[]

        i = 0

        time.sleep(1)

        print('Iterating through %d resamples\n' % resamples)

        time.sleep(1)

        icbar = IncrementalBar('Running ',suffix='%(percent)d%%',max=resamples)
        while i <= resamples:

            vdap=[]
            ra_origin = ra-ra_clu
            dec_origin = dec-dec_clu

            mxdf_theta = diff_theta[np.where(vel_diff == np.nanmax(vel_diff))]
            if np.any(mxdf_theta == 0) == True or np.any(mxdf_theta == 360) == True:
                mxdf_theta = int(0)

            ra_origin,dec_origin = self.rotate(px=ra_origin,py=dec_origin,angle=(-mxdf_theta),origin=(0.,0.))

            np.random.seed()

            monte = np.floor(np.random.rand(len(vlos))*len(vlos)).astype(int)

            vlos_rand = vlos[monte]


            theta = 0

            while theta <= 360:

                tanratios = np.absolute(ra_origin/dec_origin)
                gal_angles = np.arctan(tanratios)#*(180/np.pi)

                # Seperating hemispheres
                hem1 = np.where(ra_origin <= 0.)
                hem2 = np.where(ra_origin > 0.)

                # Calculating modified v_los for each hemispehre for v_diff calculation
                v1 = vlos_rand[hem1]*np.cos(rad90-gal_angles[hem1])
                v1mean = np.nanmean(v1)

                v2 = vlos_rand[hem2]*np.cos(rad90-gal_angles[hem2])
                v2mean = np.nanmean(v2)

                # if theta == 0:
                #     vd = 0.0
                # else:

                vd = v1mean-v2mean
                vdap.append(vd)


                #Rotate cluster by 10 degrees

                ra_origin,dec_origin = self.rotate(px=ra_origin,py=dec_origin,angle=-10.,origin=(0.,0.))
                theta = theta + 10

            vdiff_rdmn.append(np.array(vdap))

            next(icbar)
            i = i + 1
        icbar.finish()

        print('Computed through %d resamples to produce mean and std values...\n' % resamples)

        vdiffstk = np.column_stack(vdiff_rdmn)

        vdiffr_mn = np.nanmean(vdiffstk,axis=1)

        vdiffr_er = np.nanstd(vdiffstk,axis=1)

        return vdiffr_mn,vdiffr_er

    def chisqu(self,vdat,vmod,erdat,ermod):
        """ Calculated the Chi-Squared values of the Rotating Cluster Methodology

        Outputs: The Chi-Squared Statistic"""


        #Calculating chi-squared

        chisq=0

        #summing = 0

        for i,n in enumerate(vdat):

            chitop = np.power((n-vmod[i]),2)
            chibottom = (np.power(erdat[i],2)+np.power(ermod[i],2))
            prechisq = chitop/chibottom

            chisq = chisq + prechisq

        return chisq
