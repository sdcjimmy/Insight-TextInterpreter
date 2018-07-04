from keras.layers import Dense, Input, Flatten, Dropout, Merge
from keras.layers import concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, trainable=False, extra_conv=True):
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=trainable)
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)


    convs = []
    filter_sizes = [3,4,5]
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)
    l_merge = concatenate(convs, axis=-1)
    print("this is model2")
    #l_merge = Merge(mode='concat', concat_axis=1)(convs) 

# add a 1D convnet with global maxpooling, instead of Yoon Kim model
    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv) 
    if extra_conv==True:
        x = Dropout(0.5)(l_merge)  
    else:
# Original Yoon Kim model
        x = Dropout(0.5)(pool)
    x = Flatten()(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.5)(x)
    preds = Dense(labels_index, activation='softmax')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model