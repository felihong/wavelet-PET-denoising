import os
import h5py
import random 
import tables
import numpy as np
import nibabel as nib
import matplotlib.pylab as plt


class DataLoader:
    def __init__(
        self,
        # Path of full-dose PET images.
        fulldose_dir,
        # List of paths containing low-dose PET images to be included in training dataset.
        lowdose_dirs,
        # Path of H5 file before shuffling.
        h5_file,
        # Path of H5 file after shuffling.
        h5_file_shuffled,
        # Compress level of H5 file, ranging from 0 to 9, by default set to 3.
        compress_level=3,
        # Small patch size, by default set to 64. 
        patch_size=64,
        # Percentage value indicating how many patches along each axis are selected and written to pool.
        # By default set to 0.5, a value of 1 indicates no random selection.
        random_factor=0.5
    ):
        self.fulldose_dir = fulldose_dir
        self.lowdose_dirs = lowdose_dirs
        self.h5_file = h5_file
        self.h5_file_shuffled = h5_file_shuffled
        self.h5_filters = tables.Filters(complib='zlib', shuffle=True, complevel=compress_level) 
        self.patch_size = patch_size
        self.random_factor = random_factor

    def _normalize(self, data):
        """
        Normalize input arrays to (0, 1).
        :param data: Numpy array consisting of a PET image.
        :return: Normalized data.
        """
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def _load_subject(self, lowdose_dir, subject=None, offset=None):
        """
        Load a (fulldose, lowdose) paired array of a chosen subject.
        :param lowdose_dir: Directory containing lowdose nifti subject data.
        :param subject: Subject nifti file name, in forms of PID.nii.gz.
        :param offset: Relative position of current nifti file in a directory, used if subject not specified.
        :return: Subject filename and tuple of fulldose, lowdose arrays.
        """
        while True:
            if not subject:
                subject = os.listdir(self.fulldose_dir)[offset]
            # Load fulldose array and lowdose array
            full_array = nib.load(os.path.join(self.fulldose_dir, subject)).get_fdata()
            low_array = nib.load(os.path.join(lowdose_dir, subject)).get_fdata()
            yield subject, (full_array, low_array)

    def _to_small_patch(self, paired_PET, fulldose_pool, lowdose_pool):
        """
        Write PET small patches to H5 file. 
        Patches are generated along x, y, z axes in a sliding window fashion. 
        :param paired_PET: Paired (fulldose, lowdose) PET images, has shape of (2, x, y, z).
        :param fulldose_pool: H5 e-array for fulldose images.
        :param lowdose_pool: H5 e-array for lowdose images.
        """
        _, i_x, i_y, i_z = paired_PET.shape
        stride = self.patch_size // 4
        x_loop = (i_x - self.patch_size) // stride + 1   
        y_loop = (i_y - self.patch_size) // stride + 1 
        z_loop = (i_z - self.patch_size) // stride + 1  
        # print(str(x_loop*y_loop*z_loop) + ': ' + str(x_loop) + '--' + str(y_loop) + '--' + str(z_loop))
        if self.random_factor == 1:
            x_range = range(x_loop)
            y_range = range(y_loop)
            z_range = range(z_loop)
        else:
            x_range = list(np.random.choice(x_loop, int(self.random_factor * x_loop), replace=False))
            y_range = list(np.random.choice(y_loop, int(self.random_factor * y_loop), replace=False))
            z_range = list(np.random.choice(z_loop, int(self.random_factor * z_loop), replace=False))
        # print(str(len(x_range)*len(y_range)*len(z_range)) + ': ' + str(len(x_range)) + '--' + str(len(y_range)) + '--' + str(len(z_range)))
        for z in z_range:
            for x in x_range:
                for y in y_range:
                    full_patch = paired_PET[0, x*stride:x*stride+self.patch_size, y*stride:y*stride+self.patch_size, z*stride:z*stride+self.patch_size]
                    low_patch = paired_PET[-1, x*stride:x*stride+self.patch_size, y*stride:y*stride+self.patch_size, z*stride:z*stride+self.patch_size]
                    fulldose_pool.append(full_patch[np.newaxis,])
                    lowdose_pool.append(low_patch[np.newaxis,])

    def generate_train_pair(self, drf_dir, train_fulldose_pool, train_lowdose_pool, val_fulldose_pool, val_lowdose_pool, val_index, num_subject, offset=0):
        """
        Write processed subject (fulldose, lowdose) pairs to file.
        For each pair, sequentially apply normalization and small patch generation.
        :param drf_dir: Directory containing lowdose nifti files of a specific DRF.
        :param train_fulldose_pool: H5 e-arrary containing fulldose coefficients for training.
        :param train_lowdose_pool: H5 e-arrary containing lowdose coefficients for training.
        :param val_fulldose_pool: H5 e-arrary containing fulldose coefficients for validation.
        :param val_lowdose_pool: H5 e-arrary containing lowdose coefficients for validation.
        :param val_index: List of indices randomly selected to be included into validation set, it is required
        when using multiple DRFs so that the validation set is fixed for all lowdose DRFs.
        :param num_subject: Total number of subjects.
        :param offset: Relative position of current nifti file in a directory.
        :return: Paired data of the last subject before patch generation, used for testing.
        """
        p_count = 0
        print_msg =True
        while p_count != num_subject:
            # Load subject data
            try:
                subject, (full_array, low_array) = next(self._load_subject(lowdose_dir=drf_dir, offset=offset))
                print('Preparing subject data ' + str(p_count) + ', using ' + str(subject) + ' in DRF ' + str(drf_dir)) 
            except OSError as e:
                print(e)
                p_count += 1
                offset += 1
                continue
            # Normalization
            ds_paired = np.stack((full_array, low_array), axis=0)
            ds_paired = self._normalize(data=ds_paired)
            if print_msg:
                print('Normalization to (0, 1)')
            # Write small patches to H5 file
            if p_count in val_index:
                self._to_small_patch(paired_PET=ds_paired, fulldose_pool=val_fulldose_pool, lowdose_pool=val_lowdose_pool)
                print('Subject ' + str(p_count) + ' is written to validation.')
            else:
                self._to_small_patch(paired_PET=ds_paired, fulldose_pool=train_fulldose_pool, lowdose_pool=train_lowdose_pool)
            if print_msg:
                print('Wrote small patches to h5 file')
            p_count += 1
            offset += 1
            print_msg = False
        return ds_paired

    def create_train_dataset(self, num_subject=10):
        """
        Create baseline model training dataset.
        :param number_subject: Number of subjects included in the training dataset, by default set to 10.
        :return: A preview dataset containing (fulldose, lowdose) pairs for each of the DRF, used for testing.
        """
        target_file = tables.open_file(self.h5_file, mode='w', filters=self.h5_filters)
        # Create volume pools
        train_fulldose_pool = target_file.create_earray(
            target_file.root, 'train_original', tables.Float32Atom(), expectedrows=1000000, 
            shape=(0, self.patch_size, self.patch_size, self.patch_size)
        )
        train_lowdose_pool = target_file.create_earray(
            target_file.root, 'train_noisy', tables.Float32Atom(), expectedrows=1000000, 
            shape=(0, self.patch_size, self.patch_size, self.patch_size)
        )
        val_fulldose_pool = target_file.create_earray(
            target_file.root, 'validation_original', tables.Float32Atom(), expectedrows=200000, 
            shape=(0, self.patch_size, self.patch_size, self.patch_size)
        )
        val_lowdose_pool = target_file.create_earray(
            target_file.root, 'validation_noisy', tables.Float32Atom(), expectedrows=200000,
            shape=(0, self.patch_size, self.patch_size, self.patch_size)
        )

        # Randomly select 20 percent subjects for validation
        val_index = list(np.random.choice(num_subject, int(0.2 * num_subject), replace=False))

        ds_preview = np.empty((0, 2, 440, 440, 644)) 
        for lowdose_dir in self.lowdose_dirs:
            print('-----------------------------------------------')
            drf_preview = self.generate_train_pair(
                drf_dir=lowdose_dir, 
                train_fulldose_pool=train_fulldose_pool, 
                train_lowdose_pool=train_lowdose_pool, 
                val_fulldose_pool=val_fulldose_pool, 
                val_lowdose_pool=val_lowdose_pool, 
                val_index=val_index,
                num_subject=num_subject
            )
            ds_preview = np.append(ds_preview, drf_preview[np.newaxis,], axis=0)
        target_file.close()
        # Return preview dataset for testing
        return ds_preview

    def _shuffle(self, data, h5_file, name='train'):
        """
        Shuffle training dataset stored in H5 file.
        :param data: Data reading from non-shuffled H5 file.
        :param h5_file: H5 file name for shuffled data.
        :param name: Shuffling target, either train or validation. 
        no index shuffling applied if validation, by default set to train.
        """
        N, x, y, z  = data.get(name + '_original').shape
        fulldose_pool = h5_file.create_earray(h5_file.root, name + '_original', tables.Float32Atom(), expectedrows=1000000, shape=(0, x, y, z))
        lowdose_pool = h5_file.create_earray(h5_file.root, name + '_noisy', tables.Float32Atom(), expectedrows=1000000, shape=(0, x, y, z))
        index = np.arange(0, N)
        # Shuffle the index of training data
        if name == 'train':
            np.random.seed(1314)
            np.random.shuffle(index)
        for count, idx, in enumerate(index):
            if count % 2000 == 0:
                print('Finished --- ' + str('{:.0%}'.format(count / N)))
            fulldose_pool.append(np.expand_dims(data.get(name+'_original')[idx,:,:], axis=0))
            lowdose_pool.append(np.expand_dims(data.get(name+'_noisy')[idx,:,:], axis=0))

    def shuffle_train_dataset(self):
        """
        Shuffle train and validation dataset stored in H5 file.
        """
        source_file = h5py.File(self.h5_file, mode='r')
        target_file = tables.open_file(self.h5_file_shuffled, mode='w', filters=self.h5_filters)

        self._shuffle(data=source_file, h5_file=target_file, name='train')
        self._shuffle(data=source_file, h5_file=target_file, name='validation')

        target_file.close()
        source_file.close()

    def check_size(self):
        """Check patch size from H5 file, before and after shuffling."""
        print('--------------------- before shuffling ------------------------------')
        data = h5py.File(self.h5_file, mode='r')
        print('Train original: ', str(data.get('train_original').shape))
        print('Train noisy: ', str(data.get('train_noisy').shape))
        print('Val original: ', str(data.get('validation_original').shape))
        print('Val noisy: ', str(data.get('validation_noisy').shape))

        print('--------------------- after shuffling ------------------------------')
        data = h5py.File(self.h5_file_shuffled, mode='r')
        print('Train original: ', str(data.get('train_original').shape))
        print('Train noisy: ', str(data.get('train_noisy').shape))
        print('Val original: ', str(data.get('validation_original').shape))
        print('Val noisy: ', str(data.get('validation_noisy').shape))

    def display(self, ds_preview):
        """
        Display randomly chosen full-dose and low-dose slices at all DRF levels.
        :param ds_preview: Preview array returned from training dataset creation, with a shape of (num_drf, 2, x, y, z).
        """
        N_drf, _, x, y, z = ds_preview.shape
        s = random.randint(0, z-1)   # Select one random slice
        n = 5
        fig = plt.figure(figsize=(20, 16))
        for i in range(N_drf):
            # Display full-dose slice
            ax = plt.subplot(2, N_drf, i + 1)
            plt.title("original")
            plt.imshow(ds_preview[i,0,:,:,s])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display low-dose slice
            bx = plt.subplot(2, N_drf, i + n + 1)
            plt.title("low dose")
            plt.imshow(ds_preview[i,-1,:,:,s])
            bx.get_xaxis().set_visible(False)
            bx.get_yaxis().set_visible(False)    
        fig.tight_layout()
        plt.show()