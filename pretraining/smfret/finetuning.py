"""Library for finetuning pretrained models."""
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.linear_model import LogisticRegression


def encode_traces(trace_set, encoder):
    """Encodes traces into a numpy matrix."""
    with tf.device('CPU:0'):
        embedding = encoder.predict(trace_set.to_tensor(), verbose=0)
    return embedding
  

def finetune_classification_head(encoder, train_sets, epochs, num_classes=2, trace_level=True, lr=1e-3, verbose=0, return_embedding=False, use_sklearn=False, embedding_list=None, l2_regularization=0.0):
    """Finetunes a classification head."""
    if trace_level:
        if embedding_list is None:
            embedding_list = [encode_traces(trace_set, encoder) for trace_set in train_sets]
        label_list = [tf.cast(np.max(trace_set.label, axis=-1), tf.int64) for trace_set in train_sets]
    else:
        if embedding_list is None:
            embedding_list = [tf.reshape(encode_traces(trace_set, encoder), (-1, encoder.output_shape[-1])) for trace_set in train_sets]
        label_list = [tf.cast(tf.reshape(trace_set.label, (-1, )), tf.int64) for trace_set in train_sets]

    embedding = tf.concat(embedding_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    if use_sklearn:
        clf = LogisticRegression(max_iter=5000, tol=1e-6).fit(embedding, label)
        if clf.coef_.shape[0] == 1:
            clf.coef_ = np.concatenate([np.zeros_like(clf.coef_), clf.coef_], axis=0)
            clf.intercept_ = np.concatenate([np.zeros_like(clf.intercept_), clf.intercept_], axis=0)
        num_classes = clf.coef_.shape[0]

    readout = keras.layers.Dense(num_classes, kernel_regularizer=keras.regularizers.l2(l2_regularization))

    readout_model = keras.Sequential([readout])
    readout_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.legacy.Adam(lr),
    )

    #TODO: Customize class weights if necessary.
    class_weights = None
    if use_sklearn:
        readout.build(input_shape=(None, embedding.shape[-1]))
        readout.variables[0].assign(clf.coef_.T)
        readout.variables[1].assign(clf.intercept_)
    else:
        if trace_level:
            readout_model.fit(embedding, label, batch_size=256, epochs=100, verbose=verbose, class_weight=class_weights,)
        else:
            readout_model.fit(embedding, label, batch_size=1024, epochs=2, verbose=verbose, class_weight=class_weights,)

    input = encoder.input
    output = readout(encoder(input))
    classifier = keras.Model(inputs=input, outputs=output)
    
    if return_embedding:
        return classifier, embedding_list
    else:
        return classifier