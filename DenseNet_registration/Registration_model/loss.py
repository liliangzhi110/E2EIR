import tensorflow as tf


class image_loss_mes(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):

        index=tf.where(y_pred>0.001)
        data_true_image=tf.gather_nd(y_true,index)
        data_pred_image=tf.gather_nd(y_pred,index)

        return NMInformation(data_pred_image-data_true_image)


class matix_loss_mes(tf.keras.losses.Loss):

    def call(self,y_true,y_predict):

        loss = tf.reduce_mean(tf.square(y_predict - y_true))

        return loss
    
class NMInformation(NMI):

    def loss(self, y_true, y_pred):
        return -self.volumes(y_true, y_pred)

class NMI:
    """
    Soft Mutual Information approximation for intensity volumes and probabilistic volumes 
      (e.g. probabilistic segmentaitons)
    More information/citation:
    - Courtney K Guo. 
      Multi-modal image registration with unsupervised deep learning. 
      PhD thesis, Massachusetts Institute of Technology, 2019.
    - M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
      SynthMorph: learning contrast-invariant registration without acquired images
      IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
      https://doi.org/10.1109/TMI.2021.3116879
    # TODO: add local MI by using patches. This is quite memory consuming, though.
    Includes functions that can compute mutual information between volumes, 
      between segmentations, or between a volume and a segmentation map
    mi = MutualInformation()
    mi.volumes      
    mi.segs         
    mi.volume_seg
    mi.channelwise
    mi.maps
    """

    def __init__(self,
                 bin_centers=None,
                 nb_bins=None,
                 soft_bin_alpha=None,
                 min_clip=None,
                 max_clip=None):
        """
        Initialize the mutual information class
        Arguments below are related to soft quantizing of volumes, which is done automatically 
        in functions that comptue MI over volumes (e.g. volumes(), volume_seg(), channelwise()) 
        using these parameters
        Args:
            bin_centers (np.float32, optional): Array or list of bin centers. Defaults to None.
            nb_bins (int, optional):  Number of bins. Defaults to 16 if bin_centers
                is not specified.
            soft_bin_alpha (int, optional): Alpha in RBF of soft quantization. Defaults
                to `1 / 2 * square(sigma)`.
            min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
            max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
        """

        self.bin_centers = None
        if bin_centers is not None:
            self.bin_centers = tf.convert_to_tensor(bin_centers, dtype=tf.float32)
            assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]

        self.nb_bins = nb_bins
        if bin_centers is None and nb_bins is None:
            self.nb_bins = 16

        self.min_clip = min_clip
        if self.min_clip is None:
            self.min_clip = -np.inf

        self.max_clip = max_clip
        if self.max_clip is None:
            self.max_clip = np.inf

        self.soft_bin_alpha = soft_bin_alpha
        if self.soft_bin_alpha is None:
            sigma_ratio = 0.5
            if self.bin_centers is None:
                sigma = sigma_ratio / (self.nb_bins - 1)
            else:
                sigma = sigma_ratio * tf.reduce_mean(tf.experimental.numpy.diff(bin_centers))
            self.soft_bin_alpha = 1 / (2 * tf.square(sigma))
            print(self.soft_bin_alpha)

    def volumes(self, x, y):
        """
        Mutual information for each item in a batch of volumes. 
        Algorithm: 
        - use neurite.utils.soft_quantize() to create a soft quantization (binning) of 
          intensities in each channel
        - channelwise()
        Parameters:
            x and y:  [bs, ..., 1]
        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = K.shape(x)[-1]
        tensor_channels_y = K.shape(y)[-1]
        msg = 'volume_mi requires two single-channel volumes. See channelwise().'
        tf.debugging.assert_equal(tensor_channels_x, 1, msg)
        tf.debugging.assert_equal(tensor_channels_y, 1, msg)

        # volume mi
        return K.flatten(self.channelwise(x, y))

    def segs(self, x, y):
        """
        Mutual information between two probabilistic segmentation maps. 
        Wraps maps()        
        Parameters:
            x and y:  [bs, ..., nb_labels]
        Returns:
            Tensor of size [bs]
        """
        # volume mi
        return self.maps(x, y)

    def volume_seg(self, x, y):
        """
        Mutual information between a volume and a probabilistic segmentation maps. 
        Wraps maps()        
        Parameters:
            x and y: a volume and a probabilistic (soft) segmentation. Either:
              - x: [bs, ..., 1] and y: [bs, ..., nb_labels], Or:
              - x: [bs, ..., nb_labels] and y: [bs, ..., 1]
        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = K.shape(x)[-1]
        tensor_channels_y = K.shape(y)[-1]
        msg = 'volume_seg_mi requires one single-channel volume.'
        tf.debugging.assert_equal(tf.minimum(tensor_channels_x, tensor_channels_y), 1, msg)
        # otherwise we don't know which one is which
        msg = 'volume_seg_mi requires one multi-channel segmentation.'
        tf.debugging.assert_greater(tf.maximum(tensor_channels_x, tensor_channels_y), 1, msg)

        # transform volume to soft-quantized volume
        if tensor_channels_x == 1:
            x = self._soft_sim_map(x[..., 0])                       # [bs, ..., B]
        else:
            y = self._soft_sim_map(y[..., 0])                       # [bs, ..., B]

        return self.maps(x, y)  # [bs]

    def channelwise(self, x, y):
        """
        Mutual information for each channel in x and y. Thus for each item and channel this 
        returns retuns MI(x[...,i], x[...,i]). To do this, we use neurite.utils.soft_quantize() to 
        create a soft quantization (binning) of the intensities in each channel
        Parameters:
            x and y:  [bs, ..., C]
        Returns:
            Tensor of size [bs, C]
        """
        # check shapes
        tensor_shape_x = K.shape(x)
        tensor_shape_y = K.shape(y)
        tf.debugging.assert_equal(tensor_shape_x, tensor_shape_y, 'volume shapes do not match')

        # reshape to [bs, V, C]
        if tensor_shape_x.shape[0] != 3:
            new_shape = K.stack([tensor_shape_x[0], -1, tensor_shape_x[-1]])
            x = tf.reshape(x, new_shape)                             # [bs, V, C]
            y = tf.reshape(y, new_shape)                             # [bs, V, C]

        # move channels to first dimension
        ndims_k = len(x.shape)
        permute = [ndims_k - 1] + list(range(ndims_k - 1))
        cx = tf.transpose(x, permute)                                # [C, bs, V]
        cy = tf.transpose(y, permute)                                # [C, bs, V]

        # soft quantize
        cxq = self._soft_sim_map(cx)                                  # [C, bs, V, B]
        cyq = self._soft_sim_map(cy)                                  # [C, bs, V, B]

        # get mi
        map_fn = lambda x: self.maps(*x)
        cout = tf.map_fn(map_fn, [cxq, cyq], dtype=tf.float32)       # [C, bs]

        # permute back
        return tf.transpose(cout, [1, 0])                            # [bs, C]

    def maps(self, x, y):
        """
        Computes mutual information for each entry in batch, assuming each item contains 
        probability or similarity maps *at each voxel*. These could be e.g. from a softmax output 
        (e.g. when performing segmentaiton) or from soft_quantization of intensity image.
        Note: the MI is computed separate for each itemin the batch, so the joint probabilities 
        might be  different across inputs. In some cases, computing MI actoss the whole batch 
        might be desireable (TODO).
        Parameters:
            x and y are probability maps of size [bs, ..., B], where B is the size of the 
              discrete probability domain grid (e.g. bins/labels). B can be different for x and y.
        Returns:
            Tensor of size [bs]
        """

        # check shapes
        tensor_shape_x = K.shape(x)
        tensor_shape_y = K.shape(y)
        tf.debugging.assert_equal(tensor_shape_x, tensor_shape_y)
        tf.debugging.assert_non_negative(x)
        tf.debugging.assert_non_negative(y)

        eps = K.epsilon()

        # reshape to [bs, V, B]
        if tensor_shape_x.shape[0] != 3:
            new_shape = K.stack([tensor_shape_x[0], -1, tensor_shape_x[-1]])
            x = tf.reshape(x, new_shape)                             # [bs, V, B1]
            y = tf.reshape(y, new_shape)                             # [bs, V, B2]

        # joint probability for each batch entry
        x_trans = tf.transpose(x, (0, 2, 1))                         # [bs, B1, V]
        pxy = K.batch_dot(x_trans, y)                                # [bs, B1, B2]
        pxy = pxy / (K.sum(pxy, axis=[1, 2], keepdims=True) + eps)   # [bs, B1, B2]

        # x probability for each batch entry
        px = K.sum(x, 1, keepdims=True)                              # [bs, 1, B1]
        px = px / (K.sum(px, 2, keepdims=True) + eps)                # [bs, 1, B1]

        # y probability for each batch entry
        py = K.sum(y, 1, keepdims=True)                              # [bs, 1, B2]
        py = py / (K.sum(py, 2, keepdims=True) + eps)                # [bs, 1, B2]

        # independent xy probability
        px_trans = K.permute_dimensions(px, (0, 2, 1))               # [bs, B1, 1]
        pxpy = K.batch_dot(px_trans, py)                             # [bs, B1, B2]
        pxpy_eps = pxpy + eps

        # mutual information
        log_term = K.log(pxy / pxpy_eps + eps)                       # [bs, B1, B2]
        mi = K.sum(pxy * log_term, axis=[1, 2])                      # [bs]
        return mi

    def _soft_log_sim_map(self, x):
        """
        soft quantization of intensities (values) in a given volume
        See neurite.utils.soft_quantize
        Parameters:
            x [bs, ...]: intensity image. 
        Returns:
            volume with one more dimension [bs, ..., B]
        """

        return ne.utils.soft_quantize(x,
                                      alpha=self.soft_bin_alpha,
                                      bin_centers=self.bin_centers,
                                      nb_bins=self.nb_bins,
                                      min_clip=self.min_clip,
                                      max_clip=self.max_clip,
                                      return_log=True)               # [bs, ..., B]

    def _soft_sim_map(self, x):
        """
        See neurite.utils.soft_quantize
        Parameters:
            x [bs, ...]: intensity image. 
        Returns:
            volume with one more dimension [bs, ..., B]
        """
        return ne.utils.soft_quantize(x,
                                      alpha=self.soft_bin_alpha,
                                      bin_centers=self.bin_centers,
                                      nb_bins=self.nb_bins,
                                      min_clip=self.min_clip,
                                      max_clip=self.max_clip,
                                      return_log=False)              # [bs, ..., B]

    def _soft_prob_map(self, x, **kwargs):
        """
        normalize a soft_quantized volume at each voxel, so that each voxel now holds a prob. map
        Parameters:
            x [bs, ..., B]: soft quantized volume
        Returns:
            x [bs, ..., B]: renormalized so that each voxel adds to 1 across last dimension
        """
        x_hist = self._soft_sim_map(x, **kwargs)                      # [bs, ..., B]
        x_hist_sum = K.sum(x_hist, -1, keepdims=True), K.epsilon()   # [bs, ..., B]
        x_prob = x_hist / (x_hist_sum)                               # [bs, ..., B]
        return x_prob




