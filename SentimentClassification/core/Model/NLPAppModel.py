import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, SimpleRNN, GRU, Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.layers import Dropout, Activation, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization import FullTokenizer
def buildAppModel(max_fatures, input_length, embed_dim, num_classes, num_GPU, embedding_matrix=None, usingWeight=False):
    model_name = "uncased_L-12_H-768_A-12"
    model_dir = bert.fetch_google_bert_model(model_name, ".models")
    model_ckpt = os.path.join(model_dir, "bert_model.ckpt")

    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
    # use in Keras Model here, and call model.build()
    bert.load_bert_weights(l_bert, model_ckpt)      
    # should be called after model.build()
    
    max_seq_len = 128
    l_input_ids      = Input(shape=(input_length,), dtype='int32')
    l_token_type_ids = Input(shape=(input_length,), dtype='int32')

    # using the default token_type/segment id 0
    output = l_bert(l_input_ids)                              # output: [batch_size, max_seq_len, hidden_size]
    model = Model(inputs=l_input_ids, outputs=output)
    model.build(input_shape=(None, max_seq_len))

    # # provide a custom token_type/segment id as a layer input
    # output = l_bert([l_input_ids, l_token_type_ids])          # [batch_size, max_seq_len, hidden_size]
    # model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
    # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])

    # model = Sequential()

    # if(usingWeight == False):
    #     model.add(Embedding(max_fatures, embed_dim, input_length = input_length))
    # else:
    #     model.add(Embedding(max_fatures, embed_dim, weights = [embedding_matrix], input_length = input_length, trainable = True))

    # model.add(LSTM(128, activation ='tanh'))
    # model.add(Dropout(0.2))

    # model.add(Dense(num_classes, activation = 'softmax'))

    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    compile_model(model)
    # model.compile(loss = 'categorical_crossentropy',
    #         optimizer = 'Adam',
    #         metrics = ['accuracy'])
    return model