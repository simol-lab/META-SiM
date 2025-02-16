"""Library for META-SiM models."""
import os
from tqdm.auto import tqdm


CHECKPOINT_PATH = 'encoder-20240111-045226.h5'
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), CHECKPOINT_PATH)
BATCH_SIZE = 16
WINDOW_SIZE = 100


class Model:
    """Class for META-SiM model."""
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        import tensorflow as tf
        # import Keras 2.0
        import tensorflow.keras as keras
        keras.mixed_precision.set_global_policy('mixed_bfloat16')
        from metasim.fret.core.legacy.tf_layers import Attention
        from metasim.fret.core.legacy.tf_layers import Conv
        from metasim.fret.core.legacy.tf_layers import Summary
        from metasim.fret.core.legacy.tf_layers import PrependTaskToken
        from metasim.fret.core.legacy.tf_layers import Embedding
        from metasim.fret.core.legacy.tf_layers import PositionEmbedding
        from metasim.fret.core.legacy.tf_layers import Transformer
        from metasim.fret.core.legacy.tf_layers import Reconstructor
        self.encoder = keras.models.load_model(
            checkpoint_path,
            compile=False,
            custom_objects={
                'PositionEmbedding': PositionEmbedding,
            },
        )

    def _get_stacked_tensors(self, tensors, trim=False):
        """Groups tensors according to their trace's length.

        This function is used to optimize the inference efficienty. Traces of the
        same length can be stored in one tensor for faster inference.
        """
        import tensorflow as tf
        batches = []
        for tensor in tensors:
            if trim and tensor.shape[0] >= WINDOW_SIZE:
                trim_size = tensor.shape[0] // WINDOW_SIZE * WINDOW_SIZE
                tensor = tensor[:trim_size, ...]
            if not batches:
                batches.append([tensor])
                continue
            if batches[-1][-1].shape[0] == tensor.shape[0]:
                batches[-1].append(tensor)
            else:
                batches.append([tensor])
        # stack each batch
        stacked_batches = [tf.stack(batch, axis=0) for batch in batches]
        return stacked_batches

    def encode(self, dataset):
        """Encodes a dataset into a numpy tensor of trace level embeddings.

        This is the main method to use META-SiM model.
        """
        import tensorflow as tf
        if hasattr(self.encoder.layers[-1], 'framewise'):
            # This is a converted encoder
            self.encoder.layers[-1].framewise = False
        tensors = dataset.to_tensors()
        stacked_batches = self._get_stacked_tensors(tensors, trim=True)
        embeddings = []
        with tqdm(total=len(dataset.traces)) as pbar:
            pbar.set_description("Encoding traces in dataset %s" % dataset.title)
            for stacked_batch in stacked_batches:
                embeddings.extend(tf.unstack(
                    self.encoder.predict(
                        stacked_batch, batch_size=BATCH_SIZE, verbose=0,
                    )
                ))
                pbar.update(stacked_batch.shape[0])
        return tf.stack(embeddings, axis=0).numpy()

    def __call__(self, dataset):
        """Alias for encode."""
        return self.encode(dataset)

    def encode_frames(self, dataset):
        """Encodes traces into a list of frame-level embeddings.

        This method is useful when you want to build frame-level downstream models.
        """
        import tensorflow as tf
        self.encoder.layers[-1].framewise = True
        tensors = dataset.to_tensors()
        stacked_batches = self._get_stacked_tensors(tensors)
        embeddings = []
        with tqdm(total=len(dataset.traces)) as pbar:
            pbar.set_description("Encoding traces in dataset %s" % dataset.title)
            for stacked_batch in stacked_batches:
                embeddings.extend(tf.unstack(
                    self.encoder.predict(
                        stacked_batch, batch_size=BATCH_SIZE, verbose=0,
                    )
                ))
                pbar.update(stacked_batch.shape[0])

        # Trims the embeddings' paddings
        # Embeddings can be longer than the trace length due to zero padding of tokenizer.
        for i, tensor in enumerate(tensors):
            embeddings[i] = embeddings[i][:tensor.shape[0], ...].numpy()
        return embeddings

