import os
import re
import glob
import shutil
import random
import numpy as np
import pandas as pd
import nibabel as nib
import dicom2nifti
import matplotlib.pylab as plt


def dicom2Nifti(dicom_dir, nifti_dir, dose, offset=0, log_file='./dicom2nifti_log.csv'):
    log_df = pd.DataFrame()
    for subject in os.listdir(dicom_dir):
        print(subject)
        if int(re.search('_(.*)-', subject).group(1)) > offset:
            print('Converting subject dicom to nifti')
            for dcm in glob.glob(os.path.join(dicom_dir, subject, '*/'+ str(dose))):
                pid = os.path.basename(os.path.dirname(dcm))  # Specify each unique patient folder
                try:
                    # Create placeholder folder for each pid
                    outdir = os.path.join(nifti_dir, pid)
                    if not os.path.isdir(outdir):
                        os.mkdir(outdir)
                    # Convert dicom directory to nii file
                    dicom2nifti.convert_directory(dicom_directory=dcm, 
                                                output_folder=outdir, 
                                                compression=True, reorient=False)
                    nii_generated = [x for x in os.listdir(outdir) if x.endswith('.nii.gz')][0]
                    # Convert pid-based nii file and remove empty placeholder folder
                    shutil.move(os.path.join(outdir, nii_generated),
                            os.path.join(nifti_dir, pid +'.nii.gz'))
                    shutil.rmtree(outdir)   
                except Exception as e:
                    print('Write error pid to file...')
                    log_df = log_df.append({'pid': pid}, ignore_index=True)
    log_df.to_csv(log_file)


def load_comparison_data(dir_original, dir_noisy, dir_baseline, dir_wt, subject=None):
    if not subject:
        subject = random.choice(os.listdir(dir_wt))
    print('Loading data from subject ' + str(subject))
    real = nib.load(os.path.join(dir_original, subject)).get_fdata()
    noisy = nib.load(os.path.join(dir_noisy, subject)).get_fdata()
    baseline_pred = nib.load(os.path.join(dir_baseline, subject)).get_fdata()
    wt_pred = nib.load(os.path.join(dir_wt, subject)).get_fdata()
    return real, noisy, baseline_pred, wt_pred


def display(original, noisy, recn_base, recn_wt):
    """ 
    Randomly display 5 slices from fulldose, lowdose, baseline recn, and wavelet recn 
    :param original: 3D array of fulldose original PET
    :param noisy: 3D array of lowdose noisy PET
    :param recn_base: 3D array of baseline enhanced PET
    :param recn_wt: 3D array of wavelet enhanced PET
    """
    n = 5
    _, _, z = original.shape
    indices = np.random.randint(z-1, size=n) # Select one random slices
    s_original, s_noisy = original[:,:,indices], noisy[:,:,indices]
    s_recn_base, s_recn_wt = recn_base[:,:,indices], recn_wt[:,:,indices]
      
    fig = plt.figure(figsize=(20, 16))
    for i in range(n):
        # Display full-dose slice
        ax = plt.subplot(4, n, i + 1)
        plt.title("original")
        plt.imshow(np.squeeze(s_original[:,:,i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display low-dose slice
        bx = plt.subplot(4, n, i + n + 1)
        plt.title("noisy")
        plt.imshow(np.squeeze(s_noisy[:,:,i]))
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)  

        # Display de-noised slice
        cx = plt.subplot(4, n, i + 2*n + 1)
        plt.title("baseline recn")
        plt.imshow(np.squeeze(s_recn_base[:,:,i]))
        cx.get_xaxis().set_visible(False)
        cx.get_yaxis().set_visible(False)  

        # Display de-noised slice
        dx = plt.subplot(4, n, i + 3*n + 1)
        plt.title("wavelet recn")
        plt.imshow(np.squeeze(s_recn_wt[:,:,i]))
        dx.get_xaxis().set_visible(False)
        dx.get_yaxis().set_visible(False) 

    fig.tight_layout()
    plt.show()